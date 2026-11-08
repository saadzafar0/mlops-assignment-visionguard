"""Register two validation-ranked versions and assign the serving champion alias."""

import json

import mlflow

from visionguard.config import PROJECT_ROOT
from visionguard.evaluate import select_by_validation
from visionguard.tracking import configure_tracking


def main():
    configure_tracking("dagshub")
    selection_file = PROJECT_ROOT / "reports/generated/evaluation/selection.json"
    if not selection_file.exists():
        raise SystemExit("Freeze final validation selections with visionguard.evaluate first.")
    selection = json.loads(selection_file.read_text())
    results = [
        json.loads(path.read_text())
        for path in (PROJECT_ROOT / "reports/generated/training").glob("*/result.json")
    ]
    champion = next(item for item in results if item["spec"]["name"] == selection["champion"])
    others = [
        item
        for item in results
        if item["spec"]["model"] == champion["spec"]["model"]
        and item["run_id"] != champion["run_id"]
    ]
    previous = select_by_validation(others)
    name = "23l-0829_A2_Ops"
    client = mlflow.MlflowClient()
    versions = []
    for result in (previous, champion):
        existing = [
            version
            for version in client.search_model_versions(f"name='{name}'")
            if version.run_id == result["run_id"]
        ]
        if existing:
            version = max(existing, key=lambda item: int(item.version))
        else:
            run = client.get_run(result["run_id"])
            native = [
                client.get_logged_model(item.model_id)
                for item in (run.outputs.model_outputs if run.outputs else [])
            ]
            native = [
                item
                for item in native
                if item.name == "model" and item.source_run_id == result["run_id"]
            ]
            if len(native) != 1:
                raise RuntimeError("Expected one native model attached to the selected source run")
            # Use the explicit MLflow 3 ID: DagsHub's experiment-name search can
            # fail to resolve the legacy runs:/<run>/model convenience URI.
            version = mlflow.register_model(
                f"models:/{native[0].model_id}",
                name,
                tags={
                    "candidate": result["spec"]["name"],
                    "threshold": str(result["threshold_selection"]["threshold"]),
                    "image_size": str(result["spec"]["image_size"]),
                    "model_sha256": result["model_sha256"],
                    "git_commit": result["git_commit"],
                    "dvc_data_version": result["dvc_data_version"],
                },
            )
        versions.append(
            {
                "version": version.version,
                "candidate": result["spec"]["name"],
                "run_id": result["run_id"],
                "source": version.source,
            }
        )
    champion_version = versions[-1]["version"]
    client.set_registered_model_alias(name, "champion", champion_version)
    output = PROJECT_ROOT / "reports/generated/registry"
    output.mkdir(parents=True, exist_ok=True)
    evidence = {
        "registered_model": name,
        "versions": versions,
        "champion_version": champion_version,
        "alias_uri": f"models:/{name}@champion",
    }
    (output / "versions.json").write_text(json.dumps(evidence, indent=2) + "\n")
    manifest_path = PROJECT_ROOT / "models/champion/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        registry_name=name, registry_version=champion_version, registry_alias="champion"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
