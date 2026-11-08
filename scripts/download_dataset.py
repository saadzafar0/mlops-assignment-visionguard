"""Download Kaggle dataset v1, verify the ZIP and safely extract it (no Git upload)."""

import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDLE = "cashbowman/ai-generated-images-vs-real-images"
VERSION = 1
URL = f"https://www.kaggle.com/api/v1/datasets/download/{HANDLE}?datasetVersionNumber={VERSION}"


def main() -> None:
    archive = ROOT / "data/downloads/ai-generated-images-vs-real-images-v1.zip"
    destination = ROOT / "data/raw"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        print(f"Downloading {HANDLE}, version {VERSION} ...", flush=True)
        try:
            with urllib.request.urlopen(URL, timeout=120) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if not zipfile.is_zipfile(partial):
                raise ValueError("Kaggle did not return a ZIP. Check access in your browser.")
            partial.replace(archive)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite nonempty {destination}; extraction was not started."
        )
    with zipfile.ZipFile(archive) as bundle:
        destination.mkdir(parents=True, exist_ok=True)
        for info in bundle.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
        bad_member = bundle.testzip()
        if bad_member:
            raise ValueError(f"ZIP checksum failed: {bad_member}")
        bundle.extractall(destination)
        extracted_files = len([member for member in bundle.infolist() if not member.is_dir()])
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "project": "23l-0829_A2_Ops",
        "dataset": HANDLE,
        "version": VERSION,
        "download_url": URL,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": digest,
        "archive_bytes": archive.stat().st_size,
        "extracted_files": extracted_files,
    }
    provenance = ROOT / "reports/generated/dataset_source.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Extracted to {destination}")


if __name__ == "__main__":
    main()
