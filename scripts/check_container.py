"""Check the built inference image under Render Free's 512 MiB memory budget."""

import argparse
import io
import json
import os
import secrets
import subprocess
import sys
import time
import uuid
from pathlib import Path


def probe():
    import requests
    from PIL import Image

    base = "http://127.0.0.1:8000"
    for _attempt in range(90):
        try:
            health = requests.get(base + "/health", timeout=3)
            if health.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("The inference container did not become ready")
    assert health.json()["model_version"] == "2"
    assert os.getuid() == 10001
    assert requests.post(base + "/predict", timeout=5).status_code == 401
    headers = {"Authorization": "Bearer " + os.environ["INFERENCE_API_KEY"]}
    results = []
    for size, image_format in [((24, 16), "PNG"), ((2000, 2000), "JPEG")]:
        stream = io.BytesIO()
        Image.new("RGB", size, (40, 70, 130)).save(stream, format=image_format)
        extension = "png" if image_format == "PNG" else "jpg"
        response = requests.post(
            base + "/predict",
            headers=headers,
            files={"image": ("ci-image." + extension, stream.getvalue())},
            timeout=60,
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert 0 <= result["probability_ai"] <= 1
        assert result["model_version"] == "2"
        results.append({"pixels": size[0] * size[1], "response": result})
    corrupt = requests.post(
        base + "/predict",
        headers=headers,
        files={"image": ("broken.jpg", b"not a JPEG")},
        timeout=5,
    )
    assert corrupt.status_code == 422
    assert "tensorflow" not in sys.modules
    print(
        json.dumps(
            {
                "health": health.json(),
                "uid": os.getuid(),
                "missing_auth_status": 401,
                "corrupt_image_status": 422,
                "predictions": results,
                "memory_peak_bytes": int(Path("/sys/fs/cgroup/memory.peak").read_text()),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="visionguard-inference:ci")
    args = parser.parse_args()
    missing = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", args.image],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert missing.returncode == 64, missing.stderr
    assert "INFERENCE_API_KEY is missing" in missing.stderr
    name = "visionguard-smoke-" + uuid.uuid4().hex[:10]
    environment = {**os.environ, "INFERENCE_API_KEY": secrets.token_hex(32)}
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                name,
                "--network",
                "none",
                "--memory",
                "512m",
                "--memory-swap",
                "512m",
                "--cpus",
                "1",
                "--env",
                "INFERENCE_API_KEY",
                # Reproduce local .env settings accidentally copied to Render.
                "--env",
                "INFERENCE_BACKEND=local",
                "--env",
                "MODEL_MANIFEST=models/champion/manifest.json",
                "--env",
                "TFSERVING_URL=http://127.0.0.1:1/wrong-local-url",
                args.image,
            ],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        completed = subprocess.run(
            ["docker", "exec", "-i", name, "python", "-", "--probe"],
            input=Path(__file__).read_text(),
            capture_output=True,
            text=True,
            timeout=150,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        report = json.loads(completed.stdout)
        state = json.loads(
            subprocess.check_output(["docker", "inspect", "--format", "{{json .State}}", name])
        )
        assert state["Running"] and not state["OOMKilled"]
        report.update(
            memory_limit_bytes=512 * 1024**2,
            missing_key_exit=64,
            oom_killed=False,
            copied_local_settings_overridden=True,
        )
        print(json.dumps(report, indent=2))
    except Exception:
        subprocess.run(["docker", "logs", "--tail", "50", name], check=False)
        raise
    finally:
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, check=False)


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe()
    else:
        main()
