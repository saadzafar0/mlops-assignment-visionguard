"""Fetch the pinned public champion during image build; no DagsHub token is needed."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/champion")
    args = parser.parse_args()
    lock = json.loads((ROOT / "deploy/model.lock.json").read_text())
    base = lock["source_url"].rstrip("/")
    args.output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(base + "/manifest.json", timeout=120) as response:
        manifest = json.load(response)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != lock["manifest_sha256"]:
        raise ValueError("The downloaded manifest does not match the deployment lock")
    model = args.output / "model.keras"
    if not model.exists() or hashlib.sha256(model.read_bytes()).hexdigest() != lock["model_sha256"]:
        temporary = model.with_suffix(".download")
        try:
            with urllib.request.urlopen(base + "/model.keras", timeout=120) as response:
                with temporary.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != lock["model_sha256"]:
                raise ValueError("The downloaded model checksum is incorrect")
            temporary.replace(model)
        finally:
            temporary.unlink(missing_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Downloaded and verified pinned champion version", manifest["registry_version"])


if __name__ == "__main__":
    main()
