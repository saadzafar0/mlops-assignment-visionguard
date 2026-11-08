"""Wait for local model readiness before accepting public inference traffic."""

import json
import time
import urllib.error
import urllib.request

for _attempt in range(90):
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8501/v1/models/visionguard", timeout=2
        ) as response:
            value = json.load(response)
        if any(item["state"] == "AVAILABLE" for item in value["model_version_status"]):
            print("Champion model is ready.", flush=True)
            break
    except (OSError, ValueError, KeyError):
        pass
    time.sleep(1)
else:
    raise SystemExit(
        "Model did not become ready within 90 seconds; inspect TensorFlow Serving logs."
    )
