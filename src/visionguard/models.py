"""Custom CNN and two-stage ImageNet MobileNetV2 transfer learning."""

import csv
import json
from pathlib import Path

from visionguard.runtime import tf


def augmentation(seed: int = 42):
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=seed + 1),
            tf.keras.layers.RandomZoom(0.1, fill_mode="reflect", seed=seed + 2),
            tf.keras.layers.RandomContrast(0.1, seed=seed + 3),
        ],
        name="augmentation",
    )


def build_baseline(image_size=128, dropout=0.3, augment=True, seed=42):
    inputs = tf.keras.Input((image_size, image_size, 3), name="rgb_pixels")
    x = augmentation(seed)(inputs) if augment else inputs
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="preprocessing")(x)
    for index, filters in enumerate((16, 32, 64), start=1):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"conv_{index}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"normalization_{index}")(x)
        x = tf.keras.layers.ReLU(name=f"relu_{index}")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"pool_{index}")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(dropout, seed=seed, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    return tf.keras.Model(inputs, outputs, name="visionguard_cnn")


def build_transfer(
    image_size=128, dropout=0.3, augment=True, seed=42, alpha=0.5, weights="imagenet"
):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=weights,
        alpha=alpha,
        name="backbone",
    )
    base.trainable = False
    inputs = tf.keras.Input((image_size, image_size, 3), name="rgb_pixels")
    x = augmentation(seed)(inputs) if augment else inputs
    # Exactly equivalent to mobilenet_v2.preprocess_input, without a Lambda layer.
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocessing")(x)
    x = base(x, training=False)  # Preserve ImageNet BatchNorm moving statistics.
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dropout(dropout, seed=seed, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    return tf.keras.Model(inputs, outputs, name="visionguard_mobilenetv2")


def unfreeze_last_block(model, count: int = 4) -> list[str]:
    base = model.get_layer("backbone")
    base.trainable = True
    for layer in base.layers:
        layer.trainable = False
    candidates = [
        layer
        for layer in base.layers
        if (layer.name.startswith("block_16_") or layer.name == "Conv_1")
        and isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
    ]
    if not 1 <= count <= len(candidates):
        raise ValueError(f"Unfrozen layer count must be 1..{len(candidates)}")
    for layer in candidates[-count:]:
        layer.trainable = True
    return [layer.name for layer in candidates[-count:]]


def compile_model(model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def write_architecture(model, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    model.summary(print_fn=lines.append, expand_nested=True)
    (directory / "model_summary.txt").write_text("\n".join(lines) + "\n")
    rows = []

    def visit(layers, prefix=""):
        for layer in layers:
            try:
                shape = str(layer.output.shape)
            except (AttributeError, ValueError):
                shape = "multiple/unspecified"
            rows.append(
                {
                    "layer": prefix + layer.name,
                    "type": type(layer).__name__,
                    "output_shape": shape,
                    "parameters": layer.count_params(),
                    "trainable": layer.trainable,
                }
            )
            if hasattr(layer, "layers"):
                visit(layer.layers, prefix + layer.name + "/")

    visit(model.layers)
    with (directory / "layers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "trainable_layers.json").write_text(
        json.dumps(
            [row["layer"] for row in rows if row["trainable"] and row["parameters"]], indent=2
        )
        + "\n"
    )
