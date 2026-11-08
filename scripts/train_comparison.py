"""Run six controlled candidates; resume completed runs without touching test data."""

import argparse
import json
from dataclasses import asdict, replace

import pandas as pd

from visionguard.config import PROJECT_ROOT, load_params
from visionguard.train import TrainingSpec, train_candidate


def planned_candidates():
    params = load_params()
    settings = {
        key: value
        for key, value in params["training"].items()
        if key in TrainingSpec.__dataclass_fields__
    }
    settings["seed"] = params["data"]["seed"]
    baseline = TrainingSpec(name="baseline-reference", model="baseline", **settings)
    transfer = replace(baseline, name="transfer-reference", model="transfer", logging="autolog")
    return [
        baseline,
        replace(baseline, name="baseline-lower-lr", learning_rate=0.0003),
        replace(baseline, name="baseline-no-augmentation", augment=False),
        transfer,
        replace(transfer, name="transfer-lower-dropout", dropout=0.1),
        replace(transfer, name="transfer-fewer-unfrozen", unfrozen_layers=2),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["local", "dagshub"], default="dagshub")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    results = []
    candidates = planned_candidates()
    if args.only and set(args.only) - {item.name for item in candidates}:
        parser.error("Unknown candidate name")
    for spec in candidates:
        if args.only and spec.name not in args.only:
            continue
        result_path = PROJECT_ROOT / f"reports/generated/training/{spec.name}/result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["spec"] != asdict(spec):
                raise ValueError(f"Existing result has different settings: {spec.name}")
            print(f"Using completed candidate: {spec.name}", flush=True)
        else:
            result = train_candidate(spec, args.target)
        results.append(
            {
                "name": spec.name,
                "approach": spec.model,
                "run_id": result["run_id"],
                "validation_roc_auc": result["validation"]["roc_auc"],
                "validation_f1": result["validation"]["f1"],
                "validation_cost": result["threshold_selection"]["validation_cost"],
                "threshold": result["threshold_selection"]["threshold"],
                "selected_stage": result["selected_stage"],
            }
        )
    output = PROJECT_ROOT / "reports/generated/comparison"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output / "validation_comparison.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
