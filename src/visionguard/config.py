"""Shared paths and configuration; no authentication is performed on import."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def load_params(path: Path | None = None) -> dict:
    with (path or PROJECT_ROOT / "params.yaml").open() as stream:
        return yaml.safe_load(stream)


def dataset_root(value: str | None = None) -> Path:
    load_environment()
    root = Path(value or os.getenv("DATASET_ROOT") or load_params()["data"]["root"])
    return root if root.is_absolute() else PROJECT_ROOT / root
