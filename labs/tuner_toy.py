"""Task 2.7 warm-up: tune learning rate and hidden units on offline digits data."""

import json

import keras_tuner as kt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from visionguard.config import PROJECT_ROOT
from visionguard.runtime import configure_runtime, tf


def main():
    configure_runtime()
    digits = load_digits()
    train_x, val_x, train_y, val_y = train_test_split(
        digits.data.astype("float32") / 16,
        digits.target.astype("int64"),
        test_size=0.2,
        random_state=42,
        stratify=digits.target,
    )

    def dataset(x, y):
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(64).with_options(options)

    def build(hp):
        tf.keras.utils.set_random_seed(42)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input((64,)),
                tf.keras.layers.Dense(hp.Choice("units", [16, 32, 64]), activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            tf.keras.optimizers.Adam(hp.Choice("learning_rate", [0.001, 0.01])),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    tuner = kt.RandomSearch(
        build,
        objective="val_accuracy",
        max_trials=4,
        seed=42,
        directory=str(PROJECT_ROOT / ".cache/tuner"),
        project_name="toy_digits",
        overwrite=False,
    )
    tuner.search(
        dataset(train_x, train_y),
        validation_data=dataset(val_x, val_y),
        epochs=4,
        shuffle=False,
        verbose=2,
    )
    rows = [
        {
            "trial_id": trial.trial_id,
            "validation_accuracy": trial.score,
            **trial.hyperparameters.values,
        }
        for trial in tuner.oracle.trials.values()
    ]
    output = PROJECT_ROOT / "reports/generated/tuner_toy"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "trials.csv", index=False)
    best = tuner.get_best_hyperparameters(1)[0].values
    (output / "best.json").write_text(json.dumps(best, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Best toy settings:", best)


if __name__ == "__main__":
    main()
