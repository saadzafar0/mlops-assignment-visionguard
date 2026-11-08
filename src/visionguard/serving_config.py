"""Environment-only application settings and a pinned model manifest."""

import json
import os
from pathlib import Path

from visionguard.config import PROJECT_ROOT, load_environment

CAVEAT = (
    "Experimental detector. Scores are model estimates, not proof of an image's origin. "
    "Real photographs and human-made artwork can be misclassified."
)


def settings(overrides=None):
    load_environment()
    config = {
        "PROJECT_NAME": os.getenv("PROJECT_NAME", "23l-0829_A2_Ops"),
        "MODEL_PATH": os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models/champion/model.keras")),
        "MODEL_MANIFEST": os.getenv(
            "MODEL_MANIFEST", str(PROJECT_ROOT / "models/champion/manifest.json")
        ),
        "DECISION_THRESHOLD": os.getenv("DECISION_THRESHOLD") or None,
        "INFERENCE_BACKEND": os.getenv("INFERENCE_BACKEND", "local"),
        "INFERENCE_API_URL": os.getenv("INFERENCE_API_URL", ""),
        "INFERENCE_API_KEY": os.getenv("INFERENCE_API_KEY", ""),
        "INFERENCE_TIMEOUT_SECONDS": float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "120")),
        "TFSERVING_URL": os.getenv(
            "TFSERVING_URL", "http://127.0.0.1:8501/v1/models/visionguard:predict"
        ),
        "DATABASE_URL": os.getenv(
            "DATABASE_URL", f"sqlite:///{PROJECT_ROOT / '.cache/history.db'}"
        ),
        "MAX_UPLOAD_BYTES": int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))),
        "MAX_IMAGE_PIXELS": int(os.getenv("MAX_IMAGE_PIXELS", "4000000")),
        "DISPLAY_TIMEZONE": os.getenv("DISPLAY_TIMEZONE", "Asia/Karachi"),
        "API_REQUIRE_KEY": os.getenv("API_REQUIRE_KEY", "true").lower() == "true",
        "DEBUG": False,
        "TESTING": False,
    }
    config.update(overrides or {})
    config["MAX_CONTENT_LENGTH"] = config["MAX_UPLOAD_BYTES"] + 64 * 1024
    return config


def load_manifest(config):
    path = Path(config["MODEL_MANIFEST"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    manifest = json.loads(path.read_text())
    threshold = config["DECISION_THRESHOLD"]
    manifest["threshold"] = float(threshold) if threshold is not None else manifest["threshold"]
    if not 0 <= manifest["threshold"] <= 1:
        raise ValueError("Decision threshold must be between zero and one")
    return manifest
