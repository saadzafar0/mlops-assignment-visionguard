"""Flask upload, result, history, JSON and health views with durable database records."""

import time
import uuid
from datetime import UTC
from zoneinfo import ZoneInfo

from flask import Flask, abort, g, jsonify, redirect, render_template, request, url_for
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from visionguard.database import Database
from visionguard.inference import InferenceUnavailable, make_predictor, timed_prediction
from visionguard.serving_config import CAVEAT, settings
from visionguard.uploads import MemoryOnlyRequest, UploadError, validate_upload


def create_app(config=None, predictor=None):
    app = Flask(__name__)
    app.request_class = MemoryOnlyRequest
    app.config.update(settings(config))
    database = Database(app.config["DATABASE_URL"])
    engine = predictor if predictor is not None else make_predictor(app.config)
    app.extensions.update(database=database, predictor=engine)

    @app.before_request
    def start_request():
        g.request_id = str(uuid.uuid4())
        g.started = time.monotonic()

    @app.after_request
    def record_request(response):
        try:
            database.record_request(
                g.request_id,
                request.method,
                request.path,
                response.status_code,
                (time.monotonic() - g.started) * 1000,
            )
        except Exception:
            app.logger.exception("Could not persist the request audit record")
        response.headers["X-Request-ID"] = g.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.context_processor
    def common_context():
        return {
            "caveat": CAVEAT,
            "max_upload_mb": app.config["MAX_UPLOAD_BYTES"] // 1024**2,
            "max_pixels": app.config["MAX_IMAGE_PIXELS"],
            "remote_inference": app.config["INFERENCE_BACKEND"] == "remote",
        }

    @app.template_filter("localtime")
    def localtime(value):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(ZoneInfo(app.config["DISPLAY_TIMEZONE"])).strftime(
            "%d %b %Y · %H:%M:%S"
        )

    @app.get("/")
    def upload_page():
        return render_template("upload.html")

    def run_prediction():
        image = validate_upload(
            request.files.get("image"),
            app.config["MAX_UPLOAD_BYTES"],
            app.config["MAX_IMAGE_PIXELS"],
        )
        result = timed_prediction(engine, image)
        identifier = database.save_prediction(
            g.request_id, image, result, app.config["INFERENCE_BACKEND"]
        )
        return identifier, result

    @app.post("/predict")
    def predict():
        identifier, _ = run_prediction()
        return redirect(url_for("result_page", identifier=identifier), code=303)

    @app.post("/api/predict")
    def predict_json():
        identifier, result = run_prediction()
        return jsonify(**result, id=identifier, caveat=CAVEAT), 201

    @app.get("/result/<identifier>")
    def result_page(identifier):
        result = database.get_prediction(identifier)
        if result is None:
            abort(404)
        return render_template("result.html", result=result)

    @app.get("/history")
    def history():
        page = request.args.get("page", default=1, type=int)
        if page is None or page < 1:
            abort(400)
        rows, total = database.history(page)
        return render_template(
            "history.html", rows=rows, page=page, total=total, has_next=page * 20 < total
        )

    @app.get("/health")
    def health():
        try:
            database.healthy()
        except Exception:
            return jsonify(status="unavailable", database="unavailable"), 503
        return jsonify(
            status="ok", database="ok", inference_backend=app.config["INFERENCE_BACKEND"]
        )

    def error_response(message, code):
        if request.path.startswith("/api/"):
            return jsonify(error=message, request_id=g.request_id), code
        return render_template("error.html", message=message, status=code), code

    @app.errorhandler(UploadError)
    def bad_upload(error):
        return error_response(str(error), error.status)

    @app.errorhandler(InferenceUnavailable)
    def inference_error(error):
        return error_response(str(error), 503)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return error_response("The uploaded image exceeds the size limit.", 413)

    @app.errorhandler(HTTPException)
    def http_error(error):
        return error_response(error.description, error.code)

    @app.errorhandler(Exception)
    def unexpected(error):
        app.logger.exception("Request failed")
        return error_response("We could not complete this request. Please try again shortly.", 503)

    return app
