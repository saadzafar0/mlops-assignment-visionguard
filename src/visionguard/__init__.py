"""VisionGuard, project 23l-0829_A2_Ops."""

import os
from pathlib import Path

# Set these before any entry point imports Keras/TensorFlow via another library.
os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[2] / ".cache/keras"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/visionguard-matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
