"""Load a model once, or call the configured cloud inference endpoint."""

import base64
import hashlib
import math
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from visionguard.config import PROJECT_ROOT
from visionguard.serving_config import load_manifest


class InferenceUnavailable(RuntimeError):
    pass


def verdict(probability, manifest):
    probability = float(probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise InferenceUnavailable("The inference service returned an invalid score.")
    if not math.isfinite(manifest["threshold"]) or not 0 <= manifest["threshold"] <= 1:
        raise InferenceUnavailable("The inference service returned an invalid decision threshold.")
    label = "ai" if probability >= manifest["threshold"] else "real"
    return {
        "label": label,
        "probability_ai": probability,
        "confidence": probability if label == "ai" else 1 - probability,
        "threshold": manifest["threshold"],
        "model_name": manifest["model_name"],
        "model_version": str(manifest.get("registry_version", "unregistered")),
        "model_sha256": manifest["model_sha256"],
        "image_size": manifest["image_size"],
    }


class KerasPredictor:
    def __init__(self, config):
        from visionguard.runtime import tf

        self.manifest = load_manifest(config)
        path = Path(config["MODEL_PATH"])
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest["model_sha256"]:
            raise ValueError("The model file does not match the champion manifest")
        self.model = tf.keras.models.load_model(path, compile=False)
        self.lock = threading.Lock()

    def predict(self, image):
        from visionguard.preprocessing import decode_and_resize
        from visionguard.runtime import tf

        with self.lock:
            pixels = decode_and_resize(tf.constant(image.content), self.manifest["image_size"])
            probability = self.model(pixels[None], training=False).numpy().reshape(-1)[0]
        return verdict(probability, self.manifest)


class TensorFlowServingPredictor:
    def __init__(self, config):
        self.manifest = load_manifest(config)
        self.url = config["TFSERVING_URL"]

    def ready(self):
        try:
            response = requests.get(self.url.removesuffix(":predict"), timeout=(1, 2))
            response.raise_for_status()
            return any(
                item["state"] == "AVAILABLE" for item in response.json()["model_version_status"]
            )
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return False

    def predict(self, image):
        try:
            response = requests.post(
                self.url,
                json={
                    "signature_name": "serving_default",
                    "instances": [
                        {"image_bytes": {"b64": base64.b64encode(image.content).decode("ascii")}}
                    ],
                },
                timeout=(5, 60),
            )
            response.raise_for_status()
            prediction = response.json()["predictions"][0]
            if isinstance(prediction, dict):
                prediction = prediction["probability_ai"]
            if isinstance(prediction, list):
                prediction = prediction[0]
            return verdict(prediction, self.manifest)
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
            raise InferenceUnavailable("The model service is temporarily unavailable.") from error


class RemotePredictor:
    def __init__(self, config):
        address = config["INFERENCE_API_URL"].rstrip("/")
        parsed = urlparse(address)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("Set INFERENCE_API_URL to a valid HTTP(S) service address")
        self.url = address if address.endswith("/predict") else address + "/predict"
        self.key = config["INFERENCE_API_KEY"]
        if not self.key:
            raise ValueError("Set INFERENCE_API_KEY before using the cloud service")
        self.timeout = config["INFERENCE_TIMEOUT_SECONDS"]

    def predict(self, image):
        try:
            response = requests.post(
                self.url,
                files={
                    "image": (
                        image.filename,
                        image.content,
                        "image/png" if image.format == "PNG" else "image/jpeg",
                    )
                },
                headers={"Authorization": f"Bearer {self.key}"},
                timeout=(5, self.timeout),
            )
            response.raise_for_status()
            result = response.json()
            for key in (
                "label",
                "probability_ai",
                "confidence",
                "threshold",
                "model_name",
                "model_version",
                "model_sha256",
                "image_size",
            ):
                if key not in result:
                    raise ValueError("Missing response field")
            expected = verdict(
                result["probability_ai"],
                {
                    "threshold": float(result["threshold"]),
                    "model_name": result["model_name"],
                    "registry_version": result["model_version"],
                    "model_sha256": result["model_sha256"],
                    "image_size": result["image_size"],
                },
            )
            if result["label"] != expected["label"] or not math.isclose(
                float(result["confidence"]), expected["confidence"], abs_tol=1e-6
            ):
                raise ValueError("Inconsistent verdict")
            return expected
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            raise InferenceUnavailable(
                "The inference service is temporarily unavailable. Please try again in a moment."
            ) from error


def make_predictor(config):
    if config["INFERENCE_BACKEND"] == "local":
        return KerasPredictor(config)
    if config["INFERENCE_BACKEND"] == "remote":
        return RemotePredictor(config)
    if config["INFERENCE_BACKEND"] == "tensorflow-serving":
        return TensorFlowServingPredictor(config)
    raise ValueError("INFERENCE_BACKEND must be local, remote, or tensorflow-serving")


def timed_prediction(predictor, image):
    started = time.monotonic()
    result = predictor.predict(image)
    result["latency_ms"] = round((time.monotonic() - started) * 1000, 2)
    return result
