"""Protect the data-quality checks and prevent exact-duplicate split leakage."""

import shutil

import pandas as pd
import pytest
from PIL import Image

from visionguard.audit import audit_dataset, class_from_path
from visionguard.split import create_splits

ALIASES = {"real": ["real"], "ai": ["fake", "ai"]}


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "raw"
    for class_index, name in enumerate(("real", "fake")):
        directory = root / "nested" / name
        directory.mkdir(parents=True)
        for index in range(20):
            Image.new("RGB", (12 + index, 14), (index, class_index * 100, 25)).save(
                directory / f"{index:02d}.png"
            )
    return root


def test_audit_finds_corrupt_files_and_duplicates(dataset, tmp_path):
    shutil.copyfile(dataset / "nested/real/00.png", dataset / "nested/real/copy.png")
    (dataset / "nested/fake/broken.jpg").write_bytes(b"not an image")
    output = tmp_path / "audit"
    result = audit_dataset(dataset, output, ALIASES)
    assert result["total_files"] == 42
    assert result["unreadable_files"] == 1
    assert result["duplicate_files_beyond_first"] == 1
    assert result["cross_class_duplicate_groups"] == 0
    table = pd.read_csv(output / "audit.csv")
    original = table.loc[table["path"] == "nested/real/00.png"].iloc[0]
    assert (original["width"], original["height"], original["colour_mode"]) == (12, 14, "RGB")
    assert len(original["sha256"]) == 64


def test_splits_remove_conflicting_labels_and_do_not_leak(dataset, tmp_path):
    shutil.copyfile(dataset / "nested/real/00.png", dataset / "nested/fake/conflict.png")
    shutil.copyfile(dataset / "nested/real/01.png", dataset / "nested/real/copy.png")
    audit = tmp_path / "audit"
    result = audit_dataset(dataset, audit, ALIASES)
    assert result["cross_class_duplicate_groups"] == 1
    output = tmp_path / "splits"
    summary = create_splits(audit / "audit.csv", output, 42, 0.15, 0.15)
    assert summary["eligible_unique_images"] == 39
    partitions = [pd.read_csv(output / f"{name}.csv") for name in ("train", "validation", "test")]
    combined = pd.concat(partitions)
    assert len(combined) == combined["sha256"].nunique() == 39
    assert all(set(frame["class"]) == {"real", "ai"} for frame in partitions)
    conflicted = pd.read_csv(audit / "label_conflicts.csv")
    assert not set(combined["sha256"]) & set(conflicted["sha256"])
    first = (output / "train.csv").read_bytes()
    create_splits(audit / "audit.csv", output, 42, 0.15, 0.15)
    assert (output / "train.csv").read_bytes() == first


def test_unknown_class_fails_instead_of_guessing(tmp_path):
    with pytest.raises(ValueError, match="Expected one class"):
        class_from_path(tmp_path / "unlabelled/example.png", ALIASES)


def test_empty_dataset_fails(tmp_path):
    with pytest.raises(ValueError, match="No dataset files"):
        audit_dataset(tmp_path, tmp_path / "audit", ALIASES)


def test_unknown_class_prevents_splitting(dataset, tmp_path):
    (dataset / "unlabelled").mkdir()
    Image.new("RGB", (5, 5)).save(dataset / "unlabelled/image.png")
    audit = tmp_path / "audit"
    audit_dataset(dataset, audit, ALIASES)
    with pytest.raises(ValueError, match="unknown class"):
        create_splits(audit / "audit.csv", tmp_path / "splits", 42, 0.15, 0.15)


def test_dot_prefixed_images_are_audited(dataset, tmp_path):
    Image.new("RGB", (7, 9), (23, 45, 67)).save(dataset / "nested/real/.hidden.png")
    output = tmp_path / "audit"
    result = audit_dataset(dataset, output, ALIASES)
    assert result["total_files"] == result["readable_images"] == 41
    table = pd.read_csv(output / "audit.csv")
    assert "nested/real/.hidden.png" in set(table["path"])


def test_format_is_decoded_instead_of_inferred_from_suffix(dataset, tmp_path):
    Image.new("RGB", (7, 9), (14, 24, 34)).save(
        dataset / "nested/real/misleading.webp", format="JPEG"
    )
    output = tmp_path / "audit"
    audit_dataset(dataset, output, ALIASES)
    table = pd.read_csv(output / "audit.csv")
    row = table.loc[table["path"] == "nested/real/misleading.webp"].iloc[0]
    assert row["format"] == "JPEG"
    assert row["status"] == "ok"
