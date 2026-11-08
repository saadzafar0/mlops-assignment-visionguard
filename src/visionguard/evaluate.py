"""Freeze validation-selected models/thresholds, then infer the test split once each."""

import hashlib
import json
import shutil

import mlflow
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from visionguard.config import PROJECT_ROOT, dataset_root
from visionguard.metrics import binary_metrics
from visionguard.pipeline import make_dataset, predict_probabilities, read_split
from visionguard.runtime import configure_runtime, tf
from visionguard.tracking import configure_tracking


def select_by_validation(results):
    return max(
        results,
        key=lambda item: (
            item["validation"]["roc_auc"],
            -item["threshold_selection"]["validation_cost"],
        ),
    )


def error_grid(table, output, name):
    import matplotlib.pyplot as plt

    errors = table.loc[table["prediction"] != table["label"]].copy()
    errors["confidence"] = np.where(
        errors["prediction"] == 1, errors["probability_ai"], 1 - errors["probability_ai"]
    )
    errors = errors.sort_values("confidence", ascending=False).head(10)
    errors.to_csv(output / f"{name}_errors.csv", index=False)
    if errors.empty:
        return
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax in axes.flat:
        ax.axis("off")
    for index, (ax, (_, row)) in enumerate(
        zip(axes.flat, errors.iterrows(), strict=False), start=1
    ):
        with Image.open(dataset_root() / row["path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((500, 500))
            ax.imshow(image)
        true_name = "AI" if row["label"] else "Real"
        prediction = "AI" if row["prediction"] else "Real"
        ax.set_title(f"{index}. True {true_name} → {prediction}\np(AI)={row['probability_ai']:.3f}")
    fig.suptitle(f"23l-0829_A2_Ops — {name}: {len(errors)} test errors")
    fig.tight_layout()
    fig.savefig(output / f"{name}_errors.png", dpi=150)
    plt.close(fig)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_runtime()
    configure_tracking("dagshub")
    output = PROJECT_ROOT / "reports/generated/evaluation"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists():
        print("Evaluation already completed; existing predictions were preserved.")
        return
    candidates = [
        json.loads(path.read_text())
        for path in (PROJECT_ROOT / "reports/generated/training").glob("*/result.json")
    ]
    required = {
        "baseline-reference",
        "baseline-lower-lr",
        "baseline-no-augmentation",
        "transfer-reference",
        "transfer-lower-dropout",
        "transfer-fewer-unfrozen",
        "transfer-tuned",
    }
    if not required.issubset({item["spec"]["name"] for item in candidates}):
        raise SystemExit(
            "Finish all six comparisons and eight-trial tuning before test evaluation."
        )
    selected = {
        approach: select_by_validation(
            [item for item in candidates if item["spec"]["model"] == approach]
        )
        for approach in ("baseline", "transfer")
    }
    champion = select_by_validation(list(selected.values()))
    selection = {
        "selection_basis": "validation ROC-AUC, then validation 3*FP+FN cost",
        "models": selected,
        "champion": champion["spec"]["name"],
    }
    selection_path = output / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise SystemExit(
                "Test selection is already locked to different models; refusing to change it."
            )
    else:
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")

    # Publish the validation-selected serving artifact BEFORE seeing test metrics.
    serving = PROJECT_ROOT / "models/champion"
    serving.mkdir(parents=True, exist_ok=True)
    champion_source = PROJECT_ROOT / f"models/runs/{champion['spec']['name']}/model.keras"
    shutil.copyfile(champion_source, serving / "model.keras")
    manifest = {
        "project": "23l-0829_A2_Ops",
        "model_name": champion["spec"]["name"],
        "mlflow_run_id": champion["run_id"],
        "model_sha256": champion["model_sha256"],
        "threshold": champion["threshold_selection"]["threshold"],
        "image_size": champion["spec"]["image_size"],
        "labels": {"0": "real", "1": "ai"},
        "input": "RGB float32 0..255; tf.image.resize_with_pad; model includes normalisation",
        "augmentation": "disabled at inference via training=False",
        "dvc_data_version": champion["dvc_data_version"],
        "git_commit": champion["git_commit"],
        "selection_basis": selection["selection_basis"],
    }
    (serving / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if (
        hashlib.sha256((serving / "model.keras").read_bytes()).hexdigest()
        != manifest["model_sha256"]
    ):
        raise RuntimeError("Serving artifact checksum mismatch")

    all_metrics = {}
    roc_figure, roc_axis = plt.subplots(figsize=(7, 6))
    for approach, result in selected.items():
        spec = result["spec"]
        prediction_path = output / f"{approach}_test_predictions.csv"
        threshold = result["threshold_selection"]["threshold"]
        if prediction_path.exists():
            table = pd.read_csv(prediction_path)  # Resume artifact generation without re-inference.
        else:
            table = read_split("test")
            model_path = PROJECT_ROOT / f"models/runs/{spec['name']}/model.keras"
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != result["model_sha256"]:
                raise RuntimeError("Candidate model checksum differs from its validation result")
            model = tf.keras.models.load_model(model_path, compile=False)
            test = make_dataset("test", spec["image_size"], spec["batch_size"], spec["seed"])
            table["probability_ai"] = predict_probabilities(model, test)
            table["prediction"] = (table["probability_ai"] >= threshold).astype(int)
            temporary = prediction_path.with_suffix(".csv.tmp")
            table.to_csv(temporary, index=False)
            temporary.replace(prediction_path)
        metrics = binary_metrics(table["label"], table["probability_ai"], threshold)
        all_metrics[approach] = {
            "model": spec["name"],
            "run_id": result["run_id"],
            "threshold": threshold,
            **metrics,
        }
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(
            table["label"],
            table["prediction"],
            labels=[0, 1],
            display_labels=["Real", "AI"],
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"{approach} — test confusion matrix")
        fig.tight_layout()
        fig.savefig(output / f"{approach}_confusion_matrix.png", dpi=150)
        plt.close(fig)
        fpr, tpr, _ = roc_curve(table["label"], table["probability_ai"])
        roc_axis.plot(fpr, tpr, label=f"{approach}: AUC={metrics['roc_auc']:.3f}")
        error_grid(table, output, approach)
        with mlflow.start_run(run_id=result["run_id"]):
            mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
            mlflow.log_param("test_threshold", threshold)
            mlflow.set_tag(
                "test_evaluation_policy", "one inference pass; validation-locked selection"
            )
            for path in output.glob(f"{approach}_*"):
                mlflow.log_artifact(str(path), "test_evaluation")
    roc_axis.plot([0, 1], [0, 1], "k--", label="Chance")
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC curves")
    roc_axis.legend()
    roc_figure.tight_layout()
    roc_figure.savefig(output / "roc_curves.png", dpi=150)
    plt.close(roc_figure)
    pd.DataFrame(all_metrics.values()).to_csv(output / "comparison.csv", index=False)
    (output / "evaluation.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    for result in selected.values():
        with mlflow.start_run(run_id=result["run_id"]):
            mlflow.log_artifact(str(output / "roc_curves.png"), "test_evaluation")
            mlflow.log_artifact(str(output / "comparison.csv"), "test_evaluation")
    with mlflow.start_run(run_id=champion["run_id"]):
        mlflow.log_artifact(str(serving / "manifest.json"), "keras_model")
    print(json.dumps(all_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
