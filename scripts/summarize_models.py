"""Generate measured Task 2 tables and a validation comparison chart."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from visionguard.config import PROJECT_ROOT  # noqa: E402


def main():
    folder = PROJECT_ROOT / "reports/generated/training"
    results = [json.loads(path.read_text()) for path in sorted(folder.glob("*/result.json"))]
    rows = []
    stages = []
    for result in results:
        spec = result["spec"]
        rows.append(
            {
                "candidate": spec["name"],
                "validation_auc": result["validation"]["roc_auc"],
                "precision": result["validation"]["precision"],
                "recall": result["validation"]["recall"],
                "f1": result["validation"]["f1"],
                "threshold": result["threshold_selection"]["threshold"],
                "cost_3FP_FN": result["threshold_selection"]["validation_cost"],
                "run_id": result["run_id"],
            }
        )
        for stage in result["stages"]:
            stages.append(
                {
                    "candidate": spec["name"],
                    "stage": stage["stage"],
                    "best_epoch": stage["best_epoch"],
                    "best_validation_loss": stage["best_validation_loss"],
                }
            )
    comparison = pd.DataFrame(rows)
    output = PROJECT_ROOT / "reports/generated/comparison"
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "validation_all_candidates.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    names = comparison["candidate"]
    colours = ["steelblue" if name.startswith("baseline") else "darkorange" for name in names]
    axes[0].barh(names, comparison["validation_auc"], color=colours)
    axes[0].set(xlim=(0, 1), xlabel="Validation ROC-AUC")
    axes[1].barh(names, comparison["cost_3FP_FN"], color=colours)
    axes[1].set(xlabel="Validation error cost (3×FP + FN)")
    fig.suptitle("23l-0829_A2_Ops — validation comparison (test not used)")
    fig.tight_layout()
    fig.savefig(output / "validation_comparison.png", dpi=150)
    plt.close(fig)
    evaluation = json.loads(
        (PROJECT_ROOT / "reports/generated/evaluation/evaluation.json").read_text()
    )
    selection = json.loads(
        (PROJECT_ROOT / "reports/generated/evaluation/selection.json").read_text()
    )
    test = pd.DataFrame(evaluation.values())
    trials = pd.read_csv(folder / "transfer-tuned/tuner_trials.csv")
    effects = json.loads((folder / "transfer-tuned/hyperparameter_effects.json").read_text())
    largest = max(effects, key=lambda key: effects[key]["mean_loss_range"])
    text = [
        "# Task 2 — Measured model results",
        "",
        "Project: **23l-0829_A2_Ops**.",
        "",
        "## Validation comparison",
        "",
        comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Six controlled runs plus one eight-trial tuned candidate were completed. The CNN "
        "variants use manual logging; transfer variants use autologging. Tuner trials use "
        "manual logging in nested runs. Models and thresholds were selected on validation data.",
        "",
        "## Best epochs",
        "",
        pd.DataFrame(stages).to_markdown(index=False, floatfmt=".4f"),
        "",
        "The best epoch is the minimum validation-loss checkpoint within each stage. "
        "Candidate selection uses the declared validation ROC-AUC policy after "
        "checkpoint selection.",
        "",
        "## Keras Tuner",
        "",
        trials.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"The largest descriptive range of grouped mean validation losses was for **{largest}** "
        f"({effects[largest]['mean_loss_range']:.4f}). These grouped differences are exploratory: "
        "the search is small and the other settings are not balanced. Choosing the best trial "
        "can overfit validation noise, so test data stayed untouched until final selection.",
        "",
        "## Locked test evaluation",
        "",
        test.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"The validation-selected serving champion is **{selection['champion']}**. "
        "The test comparison did not change this choice or the validation-chosen thresholds. "
        "Each model inferred the same 147 held-out test images once; prediction CSVs preserve "
        "the outputs. Precision/recall/F1 treat AI as the positive class. False positives mean "
        "real images called AI-generated.",
        "",
        "## Evidence locations",
        "",
        "- Expanded summaries, layer tables, trainability lists, checkpoints, and curves: "
        "`models/runs/<candidate>/<stage>/` and each nested MLflow run.",
        "- Augmented training batches and validation predictions: "
        "`reports/generated/training/<candidate>/`.",
        "- Validation comparison chart: `reports/generated/comparison/validation_comparison.png`.",
        "- Confusion matrices, ROC curves, stored test predictions, and error grids: "
        "`reports/generated/evaluation/`.",
        "- Serving artifact and metadata: `models/champion/model.keras` and `manifest.json`.",
        "",
        "Required terminal/browser screenshots remain outstanding. The figures and recorded "
        "metrics support the final report; inspect the error grids and their accompanying "
        "hypotheses rather than treating accuracy as proof of general-purpose AI detection.",
        "",
    ]
    (PROJECT_ROOT / "docs/task2_results.md").write_text("\n".join(text))
    print("Wrote docs/task2_results.md and validation comparison chart.")


if __name__ == "__main__":
    main()
