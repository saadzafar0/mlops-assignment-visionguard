"""Configure writable library caches and deterministic, bounded CPU execution."""

import os

from visionguard.config import PROJECT_ROOT

os.environ.setdefault("KERAS_HOME", str(PROJECT_ROOT / ".cache/keras"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/visionguard-matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import tensorflow as tf  # noqa: E402


def configure_runtime(seed: int = 42) -> None:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass  # TensorFlow may already be initialised by a caller or earlier trial.
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
