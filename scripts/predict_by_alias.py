"""Task 3.6: fetch champion once by registry alias and classify one validation image."""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.tensorflow

from visionguard.config import PROJECT_ROOT, dataset_root
from visionguard.pipeline import decode_and_resize, read_split
from visionguard.runtime import configure_runtime, tf
from visionguard.tracking import configure_tracking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    configure_runtime()
    configure_tracking("dagshub")
    name = "23l-0829_A2_Ops"
    version = mlflow.MlflowClient().get_model_version_by_alias(name, "champion")
    model = mlflow.tensorflow.load_model(f"models:/{name}@champion")
    path = args.image or dataset_root() / read_split("validation").iloc[0]["path"]
    size = int(version.tags["image_size"])
    threshold = float(version.tags["threshold"])
    image = decode_and_resize(tf.io.read_file(str(path)), size)
    probability = float(model(image[None], training=False).numpy().reshape(-1)[0])
    label = "ai" if probability >= threshold else "real"
    result = {
        "registry_model": name,
        "alias": "champion",
        "version": version.version,
        "image": path.name,
        "label": label,
        "probability_ai": probability,
        "threshold": threshold,
        "confidence": probability if label == "ai" else 1 - probability,
    }
    output = PROJECT_ROOT / "reports/generated/registry/alias_prediction.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
