"""Create reproducible stratified splits after removing corrupt/conflicting/duplicate images."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from visionguard.audit import LABELS
from visionguard.config import PROJECT_ROOT, load_params


def create_splits(
    audit_path: Path, output: Path, seed: int, validation_fraction: float, test_fraction: float
) -> dict:
    if not (
        0 < validation_fraction < 1
        and 0 < test_fraction < 1
        and validation_fraction + test_fraction < 1
    ):
        raise ValueError("Validation/test fractions must be positive and sum to less than one.")
    audit = pd.read_csv(audit_path)
    if audit["class"].isna().any() or not set(audit["class"]).issubset(LABELS):
        raise ValueError("Resolve unknown class folders before creating splits.")
    valid = audit.loc[(audit["status"] == "ok") & ~audit["label_conflict"]].copy()
    valid = valid.sort_values("path").drop_duplicates("sha256").reset_index(drop=True)
    if set(valid["class"]) != set(LABELS):
        raise ValueError("Both real and AI classes must remain after filtering.")
    train, remaining = train_test_split(
        valid,
        test_size=validation_fraction + test_fraction,
        random_state=seed,
        stratify=valid["label"],
    )
    validation, test = train_test_split(
        remaining,
        test_size=test_fraction / (validation_fraction + test_fraction),
        random_state=seed,
        stratify=remaining["label"],
    )
    partitions = {"train": train, "validation": validation, "test": test}
    hash_sets = [set(partition["sha256"]) for partition in partitions.values()]
    for index, hashes in enumerate(hash_sets):
        for other in hash_sets[index + 1 :]:
            if hashes & other:
                raise RuntimeError("Content hash overlap detected between partitions.")
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, partition in partitions.items():
        partition.sort_values("path")[["path", "class", "label", "sha256"]].to_csv(
            output / f"{name}.csv", index=False
        )
        counts[name] = {
            "total": len(partition),
            "classes": partition["class"].value_counts().to_dict(),
        }
    summary = {
        "seed": seed,
        "label_mapping": LABELS,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "eligible_unique_images": len(valid),
        "excluded_files": len(audit) - len(valid),
        "hash_overlap": 0,
        "splits": counts,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", type=Path, default=PROJECT_ROOT / "reports/generated/audit/audit.csv"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/splits")
    args = parser.parse_args()
    params = load_params()["data"]
    result = create_splits(
        args.audit,
        args.output,
        params["seed"],
        params["validation_fraction"],
        params["test_fraction"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
