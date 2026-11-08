"""Walk all non-hidden dataset files, decode images, and identify exact duplicates."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from visionguard.config import PROJECT_ROOT, dataset_root, load_params

LABELS = {"real": 0, "ai": 1}


def class_from_path(relative_path: Path, aliases: dict[str, list[str]]) -> str:
    matches = {
        name
        for name, values in aliases.items()
        for part in relative_path.parts[:-1]
        if part.casefold() in {value.casefold() for value in values}
    }
    if len(matches) != 1:
        raise ValueError(f"Expected one class folder, found {sorted(matches)}")
    return matches.pop()


def inspect_image(path: Path, root: Path, aliases: dict[str, list[str]]) -> dict:
    relative = path.relative_to(root)
    row = {
        "path": relative.as_posix(),
        "class": None,
        "label": None,
        "file_size_bytes": None,
        "width": None,
        "height": None,
        "colour_mode": None,
        "format": None,
        "sha256": None,
        "status": "ok",
        "error": "",
    }
    try:
        row["class"] = class_from_path(relative, aliases)
        row["label"] = LABELS[row["class"]]
    except ValueError as error:
        row["status"] = "unknown_class"
        row["error"] = str(error)
    try:
        row["file_size_bytes"] = path.stat().st_size
        with path.open("rb") as stream:
            row["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        with Image.open(path) as image:
            row.update(
                width=image.width,
                height=image.height,
                colour_mode=image.mode,
                format=image.format,
            )
            image.verify()
        # verify() alone does not force JPEG pixel decoding.
        with Image.open(path) as image:
            image.load()
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        row["status"] = "unreadable"
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def audit_dataset(root: Path, output: Path, aliases: dict, workers: int = 4) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not any(p.startswith(".") for p in path.relative_to(root).parts)
    )
    if not files:
        raise ValueError(
            f"No dataset files found in {root}. Download and extract Kaggle data first."
        )
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(lambda path: inspect_image(path, root, aliases), files))
    table = pd.DataFrame(rows)
    first_by_hash = table.dropna(subset=["sha256"]).drop_duplicates("sha256")
    first_paths = first_by_hash.set_index("sha256")["path"]
    table["duplicate_of"] = table["sha256"].map(first_paths)
    table.loc[table["duplicate_of"] == table["path"], "duplicate_of"] = None
    conflicts = table.groupby("sha256")["class"].nunique()
    conflict_hashes = set(conflicts[conflicts > 1].index)
    table["label_conflict"] = table["sha256"].isin(conflict_hashes)
    table.to_csv(output / "audit.csv", index=False)
    table.loc[table["duplicate_of"].notna()].to_csv(output / "duplicates.csv", index=False)
    table.loc[table["status"] != "ok"].to_csv(output / "invalid_files.csv", index=False)
    table.loc[table["label_conflict"]].to_csv(output / "label_conflicts.csv", index=False)

    summary_rows = []
    for name in LABELS:
        all_class = table.loc[table["class"] == name]
        valid = all_class.loc[all_class["status"] == "ok"]
        result = {"class": name, "files": len(all_class), "readable_images": len(valid)}
        for column in ("width", "height", "file_size_bytes"):
            for statistic in ("min", "median", "max"):
                value = getattr(valid[column], statistic)()
                result[f"{column}_{statistic}"] = float(value) if pd.notna(value) else None
        summary_rows.append(result)
    pd.DataFrame(summary_rows).to_csv(output / "class_summary.csv", index=False)
    summary = {
        "project": "23l-0829_A2_Ops",
        "total_files": len(table),
        "readable_images": int((table["status"] == "ok").sum()),
        "unreadable_files": int((table["status"] == "unreadable").sum()),
        "unknown_class_files": int(table["class"].isna().sum()),
        "duplicate_files_beyond_first": int(table["duplicate_of"].notna().sum()),
        "duplicate_groups": int((table.groupby("sha256").size() > 1).sum()),
        "cross_class_duplicate_groups": len(conflict_hashes),
        "classes": summary_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Dataset root; defaults to DATASET_ROOT or params.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/generated/audit")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    summary = audit_dataset(
        dataset_root(args.root), args.output, load_params()["data"]["class_aliases"], args.workers
    )
    print(json.dumps(summary, indent=2))
    if summary["unknown_class_files"]:
        raise SystemExit("Unknown class folders found: review audit.csv and params.yaml aliases.")


if __name__ == "__main__":
    main()
