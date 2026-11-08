"""Stateless, API-key-protected cloud inference. Images stay entirely in memory."""

import hmac
import uuid

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from visionguard.inference import InferenceUnavailable, make_predictor, timed_prediction
from visionguard.serving_config import CAVEAT, settings
from visionguard.uploads import MemoryOnlyRequest, UploadError, validate_upload


def create_app(config=None, predictor=None):
    app = Flask(__name__)
    app.request_class = MemoryOnlyRequest
    app.config.update(settings(config))
    if app.config["API_REQUIRE_KEY"] and not app.config["INFERENCE_API_KEY"]:
        raise ValueError("Set INFERENCE_API_KEY before starting the public inference API")
    engine = predictor if predictor is not None else make_predictor(app.config)
    app.extensions["predictor"] = engine

    @app.before_request
    def authenticate():
        g.request_id = str(uuid.uuid4())
        if request.path == "/predict" and app.config["API_REQUIRE_KEY"]:
            supplied = request.headers.get("Authorization", "")
            expected = f"Bearer {app.config['INFERENCE_API_KEY']}"
            if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
                return jsonify(error="A valid API key is required.", request_id=g.request_id), 401

    @app.get("/")
    def index():
        return jsonify(
            service="VisionGuard",
            project=app.config["PROJECT_NAME"],
            health="/health",
            predict="POST /predict",
            file_field="image",
            authentication="Authorization: Bearer <API key>",
        )

    @app.get("/health")
    def health():
        ready = engine.ready() if hasattr(engine, "ready") else True
        return jsonify(
            status="ok" if ready else "unavailable",
            model_loaded=ready,
            model=engine.manifest["model_name"],
            model_version=str(engine.manifest.get("registry_version", "unregistered")),
        ), 200 if ready else 503

    @app.post("/predict")
    def predict():
        image = validate_upload(
            request.files.get("image"),
            app.config["MAX_UPLOAD_BYTES"],
            app.config["MAX_IMAGE_PIXELS"],
        )
        result = timed_prediction(engine, image)
        app.logger.info(
            "prediction request=%s model=%s version=%s label=%s score=%.6f",
            g.request_id,
            result["model_name"],
            result["model_version"],
            result["label"],
            result["probability_ai"],
        )
        return jsonify(**result, request_id=g.request_id, caveat=CAVEAT)

    @app.errorhandler(UploadError)
    def upload_error(error):
        return jsonify(error=str(error), request_id=g.request_id), error.status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error="The uploaded image exceeds the size limit."), 413

    @app.errorhandler(InferenceUnavailable)
    def unavailable(error):
        app.logger.warning("Inference service unavailable for request %s", g.request_id)
        return jsonify(error=str(error), request_id=g.request_id), 503

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.name), error.code

    @app.errorhandler(Exception)
    def unexpected(error):
        app.logger.exception("Inference request failed")
        return jsonify(
            error="We could not complete this inference request.",
            request_id=getattr(g, "request_id", None),
        ), 503

    return app
