"""Generate sample grids, class histograms and data-backed figure captions."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/visionguard-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from visionguard.config import PROJECT_ROOT, dataset_root, load_params  # noqa: E402


def make_figures(root: Path, audit_path: Path, output: Path, seed: int) -> None:
    table = pd.read_csv(audit_path)
    valid = table.loc[table["status"] == "ok"]
    if valid.empty:
        raise ValueError("The audit contains no readable images.")
    output.mkdir(parents=True, exist_ok=True)
    captions = ["# Task 1.3: Visual EDA", "", "Project: 23l-0829_A2_Ops", ""]
    fig, axes = plt.subplots(2, 5, figsize=(15, 7), squeeze=False)
    for row_index, name in enumerate(("real", "ai")):
        candidates = valid.loc[(valid["class"] == name) & valid["duplicate_of"].isna()]
        chosen = candidates.sample(n=min(5, len(candidates)), random_state=seed)
        for ax in axes[row_index]:
            ax.axis("off")
        for ax, (_, item) in zip(axes[row_index], chosen.iterrows(), strict=False):
            with Image.open(root / item["path"]) as image:
                thumbnail = ImageOps.exif_transpose(image).convert("RGB")
                thumbnail.thumbnail((512, 512))
                ax.imshow(thumbnail)
            ax.set_title(f"{name} | {int(item['width'])}×{int(item['height'])}")
    fig.suptitle("23l-0829_A2_Ops — seeded image samples by class")
    fig.tight_layout()
    fig.savefig(output / "samples.png", dpi=150)
    plt.close(fig)
    captions.extend(
        [
            "## Figure 1 — Sample images (`samples.png`)",
            "Five readable, nonduplicate images per class are sampled with a fixed seed. "
            "Folder labels are dataset annotations; visual appearance alone does not establish "
            "provenance. Add your own observations about subject matter, image style, and visible "
            "artifacts after inspecting this grid.",
            "",
        ]
    )
    for index, (column, title, divisor, unit) in enumerate(
        [
            ("width", "Image width", 1, "pixels"),
            ("height", "Image height", 1, "pixels"),
            ("file_size_bytes", "File size", 1024, "KiB"),
        ],
        start=2,
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        # Shared bin boundaries make class distributions directly comparable.
        values = valid[column] / divisor
        bins = np.histogram_bin_edges(values, bins=30)
        medians = {}
        for name, colour in (("real", "steelblue"), ("ai", "darkorange")):
            selected = valid.loc[valid["class"] == name, column] / divisor
            ax.hist(selected, bins=bins, alpha=0.55, label=name, color=colour)
            medians[name] = float(selected.median())
        ax.set(xlabel=f"{title} ({unit})", ylabel="Readable image count", title=title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{column}_histogram.png", dpi=150)
        plt.close(fig)
        captions.extend(
            [
                f"## Figure {index} — {title} (`{column}_histogram.png`)",
                f"The real-class median is {medians['real']:.1f} {unit}; the AI-class median is "
                f"{medians['ai']:.1f} {unit}. Shared histogram bins show the distributions before "
                "resizing and exclude unreadable files. Differences between classes may provide "
                "dataset-specific shortcuts, so image dimensions and compression should be "
                "considered when interpreting model accuracy.",
                "",
            ]
        )
    (output / "figure_notes.md").write_text("\n".join(captions))
    print(f"EDA figures and draft captions: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument(
        "--audit", type=Path, default=PROJECT_ROOT / "reports/generated/audit/audit.csv"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/generated/eda")
    args = parser.parse_args()
    make_figures(dataset_root(args.root), args.audit, args.output, load_params()["data"]["seed"])


if __name__ == "__main__":
    main()
