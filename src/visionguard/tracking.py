"""Select local or DagsHub MLflow tracking without interactive authentication."""

import os
import subprocess

from visionguard.config import PROJECT_ROOT, load_environment


def configure_tracking(target: str = "local") -> str:
    import mlflow

    load_environment()
    project = os.getenv("PROJECT_NAME", "23l-0829_A2_Ops")
    if target == "local":
        storage = PROJECT_ROOT / ".cache/mlflow"
        storage.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{storage / 'mlflow.db'}"
    elif target == "dagshub":
        owner = os.getenv("DAGSHUB_REPO_OWNER", "saadzafar0")
        repo = os.getenv("DAGSHUB_REPO_NAME", "mlops-assignment-visionguard")
        token = os.getenv("MLFLOW_TRACKING_PASSWORD") or os.getenv("DAGSHUB_USER_TOKEN")
        if not token:
            raise ValueError("Set DAGSHUB_USER_TOKEN in the ignored .env before remote tracking.")
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME") or owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        uri = os.getenv("MLFLOW_TRACKING_URI") or f"https://dagshub.com/{owner}/{repo}.mlflow"
    else:
        raise ValueError(f"Unknown tracking target: {target}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME") or project)
    return uri


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    return result.stdout.strip() or "uncommitted"
