"""Task 3.1/3.2: compare tiny Keras models on scikit-learn's built-in digits data."""

import argparse
import importlib.metadata
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/visionguard-matplotlib")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.tensorflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from visionguard.config import PROJECT_ROOT  # noqa: E402
from visionguard.tracking import configure_tracking, git_commit  # noqa: E402


class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(
            {key: float(value) for key, value in (logs or {}).items()}, step=epoch + 1
        )


def make_dataset(images, labels, training, seed):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return dataset.batch(64).with_options(options).prefetch(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["local", "dagshub"], default="local")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.001, 0.01])
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    if args.epochs < 1 or any(rate <= 0 for rate in args.learning_rates):
        parser.error("Epochs and learning rates must be positive.")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    configure_tracking(args.target)
    seed = 42
    digits = load_digits()
    images = digits.images.astype("float32")[..., np.newaxis]
    labels = digits.target.astype("int64")
    indices = np.arange(len(images))
    train_ids, validation_ids = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels
    )
    output = PROJECT_ROOT / f"reports/generated/mlflow_toy/{args.target}"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for learning_rate in args.learning_rates:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        train = make_dataset(images[train_ids], labels[train_ids], True, seed)
        validation = make_dataset(images[validation_ids], labels[validation_ids], False, seed)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(8, 8, 1)),
                tf.keras.layers.Rescaling(1.0 / 16.0),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        with mlflow.start_run(run_name=f"toy-digits-lr-{learning_rate}") as run:
            mlflow.log_params(
                {
                    "dataset": "sklearn_builtin_digits",
                    "learning_rate": learning_rate,
                    "epochs": args.epochs,
                    "batch_size": 64,
                    "hidden_units": 32,
                    "train_images": len(train_ids),
                    "validation_images": len(validation_ids),
                    "seed": seed,
                }
            )
            mlflow.set_tags({"purpose": "mlflow-tool-lab", "git_commit": git_commit()})
            mlflow.log_artifact(__file__, "source")
            history = model.fit(
                train,
                validation_data=validation,
                epochs=args.epochs,
                callbacks=[EpochLogger()],
                shuffle=False,
                verbose=2,
            ).history
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            epochs = np.arange(1, args.epochs + 1)
            for ax, metric in zip(axes, ["loss", "accuracy"], strict=True):
                ax.plot(epochs, history[metric], label="train")
                ax.plot(epochs, history[f"val_{metric}"], label="validation")
                ax.set(xlabel="Epoch", ylabel=metric, title=metric)
                ax.legend()
            fig.suptitle(f"23l-0829_A2_Ops — toy digits, learning rate {learning_rate}")
            fig.tight_layout()
            mlflow.log_figure(fig, "training_curves.png")
            fig.savefig(output / f"curves_lr_{learning_rate}.png", dpi=150)
            plt.close(fig)
            mlflow.log_dict(
                {
                    "train_indices": train_ids.tolist(),
                    "validation_indices": validation_ids.tolist(),
                },
                "split_indices.json",
            )
            example = images[validation_ids[:2]]
            signature = mlflow.models.infer_signature(example, model(example).numpy())
            mlflow.tensorflow.log_model(
                model,
                name="model",
                signature=signature,
                input_example=example,
                pip_requirements=[
                    f"tensorflow-cpu=={tf.__version__}",
                    f"keras=={importlib.metadata.version('keras')}",
                    f"numpy=={np.__version__}",
                ],
            )
            results.append(
                {
                    "run_id": run.info.run_id,
                    "learning_rate": learning_rate,
                    "final_validation_loss": history["val_loss"][-1],
                    "final_validation_accuracy": history["val_accuracy"][-1],
                }
            )
    pd.DataFrame(results).to_csv(output / "comparison.csv", index=False)
    (output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nComparison saved to {output}; these are toy results, not VisionGuard scores.")


if __name__ == "__main__":
    main()
