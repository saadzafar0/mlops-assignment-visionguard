"""Capture actual curl evidence without exposing the API key in argv or logs."""

import csv
import getpass
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/generated/report"
BASE = "https://mlops-assignment-visionguard.onrender.com"


def curl(path, *, key=None, image=None, method=None):
    arguments = [
        "curl",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "15",
        "--max-time",
        "130",
        "--write-out",
        "\n%{http_code}",
    ]
    configuration = ""
    if key is not None:
        assert "\n" not in key and "\r" not in key
        escaped = key.replace("\\", "\\\\").replace('"', '\\"')
        configuration = f'header = "Authorization: Bearer {escaped}"\n'
        arguments += ["--config", "-"]
    if method:
        arguments += ["--request", method]
    if image:
        arguments += ["--form", f"image=@{image}"]
    arguments.append(BASE + path)
    result = subprocess.run(
        arguments, input=configuration, capture_output=True, text=True, timeout=140
    )
    if result.returncode:
        raise RuntimeError(f"curl failed: {result.stderr}")
    body, status = result.stdout.rsplit("\n", 1)
    try:
        body = json.loads(body)
    except ValueError:
        body = body[:300]
    return int(status), body


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    key = dotenv_values(ROOT / ".env")["INFERENCE_API_KEY"]
    assert key
    rows = list(
        csv.DictReader(
            (ROOT / "reports/generated/training/transfer-tuned/validation_predictions.csv").open()
        )
    )
    selected = [
        next(
            row
            for row in rows
            if row["path"].endswith("10-tips-for-stunning-portrait-photography-7.jpg")
        ),
        rows[0],
    ]
    records = []
    for _attempt in range(8):
        status, body = curl("/health")
        if status == 200 and isinstance(body, dict) and body.get("model_loaded"):
            break
        print(f"Render waking up: HTTP {status}", flush=True)
        time.sleep(10)
    else:
        raise RuntimeError("Render did not become healthy")
    records.append(
        {
            "name": "Model readiness",
            "command": f"curl {BASE}/health",
            "status": status,
            "response": body,
        }
    )
    status, body = curl("/predict", method="POST")
    assert status == 401
    records.append(
        {
            "name": "Missing key",
            "command": f"curl -X POST {BASE}/predict",
            "status": status,
            "response": body,
        }
    )
    for row in selected:
        image = ROOT / "data/raw" / row["path"]
        status, body = curl("/predict", key=key, image=image)
        assert status == 200, (status, body)
        assert body["model_version"] == "2"
        difference = abs(body["probability_ai"] - float(row["probability_ai"]))
        assert difference < 1e-5
        records.append(
            {
                "name": "Real validation image" if row["label"] == "0" else "AI validation image",
                "command": f"curl --config - -F image=@<validation-image> {BASE}/predict",
                "filename": image.name,
                "status": status,
                "response": body,
                "agreement_absolute_error": difference,
            }
        )
    fixtures = ROOT / ".cache/report-inputs"
    fixtures.mkdir(parents=True, exist_ok=True)
    for filename, content, expected in [
        ("wrong.txt", b"not an image", 415),
        ("broken.jpg", b"not a decodable JPEG", 422),
        ("oversized.png", b"x" * (5 * 1024**2 + 1), 413),
    ]:
        image = fixtures / filename
        image.write_bytes(content)
        status, body = curl("/predict", key=key, image=image)
        assert status == expected, (status, body)
        records.append(
            {
                "name": filename,
                "command": f"curl --config - -F image=@{filename} {BASE}/predict",
                "status": status,
                "response": body,
            }
        )
    report = {
        "project": "23l-0829_A2_Ops",
        "username": getpass.getuser(),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "client": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0],
        "authentication": "Bearer key provided privately on curl stdin; not recorded",
        "records": records,
    }
    serialized = json.dumps(report, indent=2)
    assert key not in serialized
    (OUTPUT / "curl_api_checks.json").write_text(serialized + "\n")
    lines = [
        f"Project: {report['project']}",
        f"User: {report['username']}",
        f"Captured UTC: {report['recorded_at_utc']}",
        report["client"],
        "Actual curl output. Authentication header supplied on stdin and omitted here.",
    ]
    for record in records:
        lines += [
            "",
            record["name"],
            "$ " + record["command"],
            "HTTP " + str(record["status"]),
            json.dumps(record["response"], indent=2),
        ]
    transcript = "\n".join(lines) + "\n"
    assert key not in transcript
    (OUTPUT / "curl_transcript.txt").write_text(transcript)
    print("Captured actual curl checks:", [(r["name"], r["status"]) for r in records])


if __name__ == "__main__":
    main()
