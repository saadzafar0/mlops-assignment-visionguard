"""Eight warm-started fine-tuning trials, using validation loss and no test data."""

import argparse
import hashlib
import json
from dataclasses import asdict

import keras_tuner as kt
import mlflow
import mlflow.tensorflow
import pandas as pd

from visionguard.config import PROJECT_ROOT, load_params
from visionguard.metrics import binary_metrics, choose_threshold
from visionguard.models import compile_model, unfreeze_last_block, write_architecture
from visionguard.pipeline import make_dataset, predict_probabilities, read_split
from visionguard.runtime import configure_runtime, tf
from visionguard.tracking import configure_tracking, git_commit
from visionguard.train import (
    EpochLogger,
    TrainingSpec,
    data_version,
    log_native_model,
    log_reproducibility,
)


class TrackedSearch(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        with mlflow.start_run(run_name=f"tuner-trial-{trial.trial_id}", nested=True):
            mlflow.set_tags({"purpose": "visionguard-tuning-trial"})
            mlflow.log_params(trial.hyperparameters.values)
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), EpochLogger()]
            histories = super().run_trial(trial, *args, **kwargs)
            for index, history in enumerate(histories):
                mlflow.log_dict(history.history, f"history_{index}.json")
            return histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["local", "dagshub"], default="dagshub")
    args = parser.parse_args()
    configure_runtime()
    configure_tracking(args.target)
    source_commit = git_commit()
    mlflow.tensorflow.autolog(disable=True)
    params = load_params()
    config = params["tuning"]
    reference_path = PROJECT_ROOT / "reports/generated/training/transfer-reference/result.json"
    if not reference_path.exists():
        raise SystemExit("Complete transfer-reference training before tuning.")
    reference = json.loads(reference_path.read_text())
    spec = TrainingSpec(**reference["spec"])
    checkpoint = str(PROJECT_ROOT / "models/runs/transfer-reference/frozen_head/best.keras")
    output = PROJECT_ROOT / "reports/generated/training/transfer-tuned"
    if (output / "result.json").exists():
        raise SystemExit("Completed tuning results already exist; refusing to overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    model_dir = PROJECT_ROOT / "models/runs/transfer-tuned"
    model_dir.mkdir(parents=True, exist_ok=True)
    train = make_dataset("train", spec.image_size, spec.batch_size, spec.seed)
    validation = make_dataset("validation", spec.image_size, spec.batch_size, spec.seed)

    def build(hp):
        tf.keras.utils.set_random_seed(spec.seed)
        model = tf.keras.models.load_model(checkpoint, compile=False)
        model.get_layer("head_dropout").rate = hp.Choice("dropout", config["dropout_rates"])
        unfreeze_last_block(model, hp.Choice("unfrozen_layers", config["unfrozen_layers"]))
        compile_model(model, hp.Choice("learning_rate", config["learning_rates"]))
        return model

    with mlflow.start_run(run_name="transfer-tuned") as parent:
        mlflow.set_tags(
            {"purpose": "visionguard-candidate", "approach": "transfer", "selection": "keras-tuner"}
        )
        log_reproducibility(source_commit, spec.seed)
        mlflow.log_params(
            {
                "max_trials": config["max_trials"],
                "epochs_per_trial": config["epochs_per_trial"],
                "warm_start_checkpoint": str(checkpoint),
            }
        )
        tuner = TrackedSearch(
            build,
            objective="val_loss",
            max_trials=config["max_trials"],
            seed=spec.seed,
            directory=str(PROJECT_ROOT / ".cache/tuner"),
            project_name="23l-0829_A2_Ops",
            overwrite=False,
        )
        tuner.search(
            train,
            validation_data=validation,
            epochs=config["epochs_per_trial"],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=2, restore_best_weights=True
                )
            ],
            shuffle=False,
            verbose=2,
        )
        trials = [
            {
                "trial_id": trial.trial_id,
                "status": trial.status,
                "validation_loss": trial.score,
                **trial.hyperparameters.values,
            }
            for trial in tuner.oracle.trials.values()
        ]
        pd.DataFrame(trials).to_csv(output / "tuner_trials.csv", index=False)
        if sum(row["status"] == "COMPLETED" for row in trials) < 8:
            raise RuntimeError("At least eight successful trials are required.")
        best_hp = tuner.get_best_hyperparameters(1)[0]
        model = tuner.get_best_models(1)[0]
        write_architecture(model, output)
        model_path = model_dir / "model.keras"
        model.save(model_path)
        probabilities = predict_probabilities(model, validation)
        table = read_split("validation")
        table["probability_ai"] = probabilities
        table.to_csv(output / "validation_predictions.csv", index=False)
        threshold = choose_threshold(
            table["label"], probabilities, params["training"]["false_positive_cost"]
        )
        spec.name = "transfer-tuned"
        spec.logging = "manual"
        spec.dropout = best_hp.get("dropout")
        spec.unfrozen_layers = best_hp.get("unfrozen_layers")
        spec.fine_tune_learning_rate = best_hp.get("learning_rate")
        spec.fine_tune_epochs = config["epochs_per_trial"]
        best_trial = tuner.oracle.get_best_trials(1)[0]
        result = {
            "spec": asdict(spec),
            "run_id": parent.info.run_id,
            "stages": [
                {
                    "stage": "tuned_fine_tuning",
                    "best_validation_loss": best_trial.score,
                    "best_epoch": int(best_trial.metrics.get_best_step("val_loss")) + 1,
                }
            ],
            "selected_stage": "tuned_fine_tuning",
            "threshold_selection": threshold,
            "validation": binary_metrics(table["label"], probabilities, threshold["threshold"]),
            "model_path": str(model_path),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "dvc_data_version": data_version(),
            "git_commit": source_commit,
            "tuner_best": best_hp.values,
        }
        mlflow.log_metrics(
            {f"validation_{key}": value for key, value in result["validation"].items()}
        )
        mlflow.log_params(best_hp.values)
        for path in output.iterdir():
            if path.is_file():
                mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(model_path), "keras_model")
        log_native_model(model, next(iter(validation))[0][:2].numpy())
        # Simple descriptive effects, not causal importance estimates.
        effects = {}
        frame = pd.DataFrame(trials)
        for key in ("learning_rate", "dropout", "unfrozen_layers"):
            means = frame.groupby(key)["validation_loss"].mean()
            effects[key] = {
                "mean_loss_by_value": {str(k): v for k, v in means.items()},
                "mean_loss_range": float(means.max() - means.min()),
            }
        (output / "hyperparameter_effects.json").write_text(json.dumps(effects, indent=2) + "\n")
        mlflow.log_dict(effects, "hyperparameter_effects.json")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        mlflow.log_artifact(str(output / "result.json"))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
