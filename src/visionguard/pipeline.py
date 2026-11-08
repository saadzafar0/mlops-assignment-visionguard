"""One RGB decode/resize path shared by training, evaluation and future serving."""

import hashlib
from pathlib import Path

import pandas as pd

from visionguard.config import PROJECT_ROOT, dataset_root
from visionguard.runtime import tf

SPLITS = PROJECT_ROOT / "data/processed/splits"


def decode_and_resize(encoded, image_size: int):
    image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    image = tf.ensure_shape(image, (None, None, 3))
    image = tf.image.resize_with_pad(image, image_size, image_size, antialias=True)
    return tf.clip_by_value(image, 0.0, 255.0)


def read_split(name: str) -> pd.DataFrame:
    if name not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown partition: {name}")
    table = pd.read_csv(SPLITS / f"{name}.csv")
    if table.empty or not set(table["label"]).issubset({0, 1}):
        raise ValueError(f"Invalid labels or empty {name} split")
    root = dataset_root().resolve()
    for path in table["path"]:
        if not (root / path).resolve().is_relative_to(root):
            raise ValueError("Split path escapes the dataset root")
    return table


def make_dataset(name: str, image_size: int = 128, batch_size: int = 32, seed: int = 42):
    table = read_split(name)
    root = dataset_root()
    paths = [str(root / path) for path in table["path"]]
    labels = table["label"].to_numpy(dtype="float32")
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda path, label: (decode_and_resize(tf.io.read_file(path), image_size), label),
        num_parallel_calls=2,
        deterministic=True,
    )
    # Cache decoded, resized pixels BEFORE shuffling and model-side augmentation.
    digest = hashlib.sha256(
        (SPLITS / f"{name}.csv").read_bytes()
        + Path(__file__).read_bytes()
        + (PROJECT_ROOT / "data/raw.dvc").read_bytes()
    ).hexdigest()[:16]
    cache = PROJECT_ROOT / f".cache/tfdata/{digest}/{name}_{image_size}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    dataset = dataset.cache(str(cache))
    if name == "train":
        dataset = dataset.shuffle(len(table), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    return dataset.batch(batch_size).with_options(options).prefetch(1)


def predict_probabilities(model, dataset):
    """Call the model directly so each evaluation image is inferred exactly once."""
    import numpy as np

    return np.concatenate(
        [model(images, training=False).numpy().reshape(-1) for images, _ in dataset]
    )
