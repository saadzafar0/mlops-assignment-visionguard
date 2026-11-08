"""Record exact installed versions and TensorFlow-visible devices (Task 1.1)."""

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime

from visionguard.config import PROJECT_ROOT


def main() -> None:
    import tensorflow as tf

    output = PROJECT_ROOT / "reports/generated/environment"
    output.mkdir(parents=True, exist_ok=True)
    packages = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    )
    (output / "packages.txt").write_text("\n".join(packages) + "\n")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    evidence = {
        "project": "23l-0829_A2_Ops",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": commit.stdout.strip() or "uncommitted",
        "tensorflow": tf.__version__,
        "keras": importlib.metadata.version("keras"),
        "tensorflow_built_with_cuda": tf.test.is_built_with_cuda(),
        "visible_cpus": [device.name for device in tf.config.list_physical_devices("CPU")],
        "visible_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "tensorflow_package": "tensorflow-cpu",
    }
    (output / "environment.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    print(f"\nExact package versions: {output / 'packages.txt'}")


if __name__ == "__main__":
    main()
