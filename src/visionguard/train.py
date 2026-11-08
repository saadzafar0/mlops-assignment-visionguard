"""Train reproducible candidates using only training/validation partitions."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
import yaml

from visionguard.config import PROJECT_ROOT, load_params
from visionguard.metrics import binary_metrics, choose_threshold
from visionguard.models import (
    augmentation,
    build_baseline,
    build_transfer,
    compile_model,
    unfreeze_last_block,
    write_architecture,
)
from visionguard.pipeline import SPLITS, make_dataset, predict_probabilities, read_split
from visionguard.runtime import configure_runtime, tf
from visionguard.tracking import configure_tracking, git_commit


@dataclass
class TrainingSpec:
    name: str
    model: str
    learning_rate: float = 0.001
    dropout: float = 0.3
    augment: bool = True
    unfrozen_layers: int = 4
    logging: str = "manual"
    image_size: int = 128
    batch_size: int = 32
    baseline_epochs: int = 12
    head_epochs: int = 8
    fine_tune_epochs: int = 6
    fine_tune_learning_rate: float = 0.00001
    seed: int = 42


class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(
            {key: float(value) for key, value in (logs or {}).items()}, step=epoch + 1
        )


def data_version() -> str:
    return yaml.safe_load((PROJECT_ROOT / "data/raw.dvc").read_text())["outs"][0]["md5"]


def log_reproducibility(source_commit: str | None = None, seed: int = 42) -> None:
    mlflow.set_tags(
        {
            "git_commit": source_commit or git_commit(),
            "dvc_data_version": data_version(),
            "seed": str(seed),
        }
    )
    for name in ("train", "validation", "test"):
        mlflow.log_artifact(str(SPLITS / f"{name}.csv"), "splits")
    for filename in ("params.yaml", "pyproject.toml", "uv.lock", "data/raw.dvc"):
        mlflow.log_artifact(str(PROJECT_ROOT / filename), "reproducibility")
    packages = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    )
    mlflow.log_text("\n".join(packages) + "\n", "reproducibility/packages.txt")
    for filename in ("pipeline.py", "models.py", "train.py", "metrics.py"):
        mlflow.log_artifact(str(Path(__file__).parent / filename), "source")


def plot_history(history: dict, output: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = int(np.argmin(history["val_loss"])) + 1
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("loss", "accuracy"), strict=True):
        ax.plot(epochs, history[metric], label="training")
        ax.plot(epochs, history[f"val_{metric}"], label="validation")
        ax.axvline(best, linestyle="--", color="black", label=f"best epoch {best}")
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def train_stage(model, train, validation, name, epochs, learning_rate, directory, logging):
    directory.mkdir(parents=True, exist_ok=True)
    compile_model(model, learning_rate)
    write_architecture(model, directory)
    checkpoint = directory / "best.keras"
    params = load_params()["training"]
    with mlflow.start_run(run_name=name, nested=True) as child:
        mlflow.set_tags({"purpose": "visionguard-stage", "stage": name})
        mlflow.log_params({"stage_learning_rate": learning_rate, "maximum_epochs": epochs})
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=params["early_stopping_patience"],
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint), monitor="val_loss", save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=params["lr_patience"],
                min_lr=1e-7,
            ),
        ]
        if logging == "manual":
            callbacks.append(EpochLogger())
        started = time.monotonic()
        history = model.fit(
            train,
            validation_data=validation,
            epochs=epochs,
            callbacks=callbacks,
            shuffle=False,
            verbose=2,
        ).history
        history = {key: [float(value) for value in values] for key, values in history.items()}
        (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        plot_history(history, directory / "curves.png", f"23l-0829_A2_Ops — {name}")
        best_epoch = int(np.argmin(history["val_loss"])) + 1
        result = {
            "stage": name,
            "run_id": child.info.run_id,
            "best_epoch": best_epoch,
            "epochs_completed": len(history["loss"]),
            "best_validation_loss": min(history["val_loss"]),
            "seconds": time.monotonic() - started,
            "checkpoint": str(checkpoint),
        }
        mlflow.log_metrics(
            {"best_epoch": best_epoch, "best_validation_loss": result["best_validation_loss"]}
        )
        for filename in (
            "history.json",
            "curves.png",
            "layers.csv",
            "model_summary.txt",
            "trainable_layers.json",
        ):
            mlflow.log_artifact(str(directory / filename))
    return result


def log_native_model(model, example):
    signature = mlflow.models.infer_signature(example, model(example, training=False).numpy())
    mlflow.tensorflow.log_model(
        model,
        name="model",
        input_example=example,
        signature=signature,
        pip_requirements=[
            f"tensorflow-cpu=={tf.__version__}",
            f"keras=={importlib.metadata.version('keras')}",
            f"numpy=={np.__version__}",
        ],
    )


def save_augmented_batch(dataset, output, seed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images, labels = next(iter(dataset))
    images = augmentation(seed)(images[:8], training=True).numpy()
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for index, ax in enumerate(axes.flat):
        ax.imshow(np.clip(images[index] / 255, 0, 1))
        ax.set_title("AI" if labels[index] else "Real")
        ax.axis("off")
    fig.suptitle("Training-only augmentation — 23l-0829_A2_Ops")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def train_candidate(spec: TrainingSpec, target: str = "dagshub") -> dict:
    if spec.model not in {"baseline", "transfer"}:
        raise ValueError("Model must be baseline or transfer")
    if not spec.name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Run names may contain only letters, numbers, hyphens and underscores")
    output = PROJECT_ROOT / f"reports/generated/training/{spec.name}"
    model_dir = PROJECT_ROOT / f"models/runs/{spec.name}"
    if (output / "result.json").exists():
        raise ValueError(f"Completed candidate already exists: {spec.name}")
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    source_commit = git_commit()
    tf.keras.backend.clear_session()
    configure_runtime(spec.seed)
    configure_tracking(target)
    mlflow.tensorflow.autolog(disable=True)
    if spec.logging == "autolog":
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False, checkpoint=False)
    train = make_dataset("train", spec.image_size, spec.batch_size, spec.seed)
    validation = make_dataset("validation", spec.image_size, spec.batch_size, spec.seed)
    if spec.model == "baseline":
        model = build_baseline(spec.image_size, spec.dropout, spec.augment, spec.seed)
    else:
        model = build_transfer(
            spec.image_size,
            spec.dropout,
            spec.augment,
            spec.seed,
            load_params()["training"]["mobilenet_alpha"],
        )
    with mlflow.start_run(run_name=spec.name) as parent:
        mlflow.set_tags(
            {"purpose": "visionguard-candidate", "approach": spec.model, "logging": spec.logging}
        )
        mlflow.log_params(asdict(spec))
        log_reproducibility(source_commit, spec.seed)
        if spec.augment:
            save_augmented_batch(train, output / "augmented_batch.png", spec.seed)
            mlflow.log_artifact(str(output / "augmented_batch.png"))
        first = "baseline" if spec.model == "baseline" else "frozen_head"
        stages = [
            train_stage(
                model,
                train,
                validation,
                first,
                spec.baseline_epochs if spec.model == "baseline" else spec.head_epochs,
                spec.learning_rate,
                model_dir / first,
                spec.logging,
            )
        ]
        if spec.model == "transfer":
            model = tf.keras.models.load_model(stages[0]["checkpoint"], compile=False)
            unfrozen = unfreeze_last_block(model, spec.unfrozen_layers)
            mlflow.log_dict({"unfrozen_convolutions": unfrozen}, "fine_tuning.json")
            stages.append(
                train_stage(
                    model,
                    train,
                    validation,
                    "fine_tuning",
                    spec.fine_tune_epochs,
                    spec.fine_tune_learning_rate,
                    model_dir / "fine_tuning",
                    spec.logging,
                )
            )
        best_stage = min(stages, key=lambda item: item["best_validation_loss"])
        model = tf.keras.models.load_model(best_stage["checkpoint"], compile=False)
        probabilities = predict_probabilities(model, validation)
        table = read_split("validation")
        table["probability_ai"] = probabilities
        table.to_csv(output / "validation_predictions.csv", index=False)
        threshold = choose_threshold(
            table["label"], probabilities, load_params()["training"]["false_positive_cost"]
        )
        metrics = binary_metrics(table["label"], probabilities, threshold["threshold"])
        model_path = model_dir / "model.keras"
        model.save(model_path)
        result = {
            "spec": asdict(spec),
            "run_id": parent.info.run_id,
            "stages": stages,
            "selected_stage": best_stage["stage"],
            "validation": metrics,
            "threshold_selection": threshold,
            "model_path": str(model_path),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "dvc_data_version": data_version(),
            "git_commit": source_commit,
        }
        mlflow.log_metrics({f"validation_{key}": value for key, value in metrics.items()})
        mlflow.log_param("decision_threshold", threshold["threshold"])
        mlflow.log_artifact(str(output / "validation_predictions.csv"))
        mlflow.log_artifact(str(model_path), "keras_model")
        example = next(iter(validation))[0][:2].numpy()
        log_native_model(model, example)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        mlflow.log_artifact(str(output / "result.json"))
    mlflow.tensorflow.autolog(disable=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", choices=["baseline", "transfer"], required=True)
    parser.add_argument("--target", choices=["local", "dagshub"], default="dagshub")
    parser.add_argument("--logging", choices=["manual", "autolog"], default="manual")
    args = parser.parse_args()
    train_candidate(
        TrainingSpec(name=args.name, model=args.model, logging=args.logging), args.target
    )


if __name__ == "__main__":
    main()
