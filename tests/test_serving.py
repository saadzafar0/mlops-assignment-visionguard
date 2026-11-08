"""Upload safety, durable history, request auditing, and inference API contracts."""

import io

import pytest
import requests
from PIL import Image
from sqlalchemy import func, select

from visionguard.api import create_app as create_api
from visionguard.database import predictions, request_log
from visionguard.inference import InferenceUnavailable, RemotePredictor, verdict
from visionguard.uploads import UploadedImage
from visionguard.webapp import create_app

MANIFEST = {
    "model_name": "test-model",
    "registry_version": "2",
    "threshold": 0.7,
    "image_size": 128,
    "model_sha256": "a" * 64,
}


class FakePredictor:
    manifest = MANIFEST

    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return verdict(0.8, self.manifest)


def png():
    stream = io.BytesIO()
    Image.new("RGB", (24, 16), (20, 40, 60)).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def app():
    return create_app(
        {
            "TESTING": True,
            "DATABASE_URL": "sqlite:///:memory:",
            "INFERENCE_BACKEND": "local",
            "MAX_UPLOAD_BYTES": 1024,
        },
        FakePredictor(),
    )


def count(app, table):
    with app.extensions["database"].engine.connect() as connection:
        return connection.scalar(select(func.count()).select_from(table))


def test_upload_result_history_and_refresh_do_not_repeat_inference(app):
    client = app.test_client()
    response = client.post("/predict", data={"image": (io.BytesIO(png()), "../../portrait.png")})
    assert response.status_code == 303
    assert client.get(response.location).status_code == 200
    assert client.get(response.location).status_code == 200
    history = client.get("/history").get_data(as_text=True)
    assert "portrait.png" in history and "80.0%" in history and "Version 2" in history
    assert app.extensions["predictor"].calls == 1
    assert count(app, predictions) == 1
    assert count(app, request_log) == 4
    with app.extensions["database"].engine.connect() as connection:
        record = connection.execute(select(predictions)).mappings().one()
        assert record["filename"] == "portrait.png"
        assert len(record["image_sha256"]) == 64
        assert record["label"] == "ai" and record["model_version"] == "2"
        assert record["threshold"] == 0.7


@pytest.mark.parametrize(
    ("name", "content", "status", "message"),
    [
        ("file.txt", b"text", 415, "Only JPEG"),
        ("broken.jpg", b"not an image", 422, "could not be decoded"),
        ("large.png", b"x" * 1025, 413, "no larger"),
        ("pretend.jpg", png(), 415, "do not match"),
    ],
)
def test_rejected_uploads_are_friendly_and_audited(app, name, content, status, message):
    response = app.test_client().post("/predict", data={"image": (io.BytesIO(content), name)})
    assert response.status_code == status
    assert message in response.get_data(as_text=True)
    assert app.extensions["predictor"].calls == 0
    assert count(app, predictions) == 0
    assert count(app, request_log) == 1


def test_missing_file_and_excessive_resolution(app):
    client = app.test_client()
    assert client.post("/api/predict").status_code == 400
    app.config["MAX_IMAGE_PIXELS"] = 10
    response = client.post("/api/predict", data={"image": (io.BytesIO(png()), "image.png")})
    assert response.status_code == 422 and "Resize" in response.json["error"]


def test_multipart_file_stream_is_memory_only_even_above_spooling_threshold(app):
    from visionguard.uploads import MemoryOnlyRequest

    request = MemoryOnlyRequest.from_values()
    stream = request._get_file_stream(2_000_000, "image/png", "large.png", 2_000_000)
    stream.write(b"x" * 600_000)
    assert isinstance(stream, io.BytesIO)


def test_history_survives_app_restart(tmp_path):
    config = {"TESTING": True, "DATABASE_URL": f"sqlite:///{tmp_path / 'history.db'}"}
    first = create_app(config, FakePredictor())
    response = first.test_client().post(
        "/api/predict", data={"image": (io.BytesIO(png()), "image.png")}
    )
    assert response.status_code == 201
    first.extensions["database"].engine.dispose()
    restarted = create_app(config, FakePredictor())
    assert "image.png" in restarted.test_client().get("/history").get_data(as_text=True)
    assert count(restarted, predictions) == 1


def test_health_errors_and_empty_history(app):
    client = app.test_client()
    assert client.get("/health").json["database"] == "ok"
    assert "No checks yet" in client.get("/history").get_data(as_text=True)
    assert client.get("/result/unknown").status_code == 404
    assert client.get("/history?page=-1").status_code == 400
    assert count(app, request_log) == 4


def test_cloud_api_authentication_and_prediction():
    predictor = FakePredictor()
    api = create_api(
        {"TESTING": True, "INFERENCE_API_KEY": "test-key", "API_REQUIRE_KEY": True}, predictor
    )
    client = api.test_client()
    assert client.get("/health").status_code == 200
    assert client.post("/predict").status_code == 401
    response = client.post(
        "/predict",
        data={"image": (io.BytesIO(png()), "image.png")},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    assert response.json["label"] == "ai"
    assert response.json["model_version"] == "2"
    assert "caveat" in response.json and "request_id" in response.json
    assert predictor.calls == 1
    assert "test-key" not in response.get_data(as_text=True)


def test_cloud_api_refuses_missing_key():
    with pytest.raises(ValueError, match="Set INFERENCE_API_KEY"):
        create_api({"API_REQUIRE_KEY": True, "INFERENCE_API_KEY": ""}, FakePredictor())


def test_inference_failure_has_no_success_history(app):
    def fail(image):
        raise InferenceUnavailable("The inference service is temporarily unavailable.")

    app.extensions["predictor"].predict = fail
    response = app.test_client().post(
        "/api/predict", data={"image": (io.BytesIO(png()), "image.png")}
    )
    assert response.status_code == 503
    assert count(app, predictions) == 0 and count(app, request_log) == 1


def test_remote_client_sends_key_and_rejects_inconsistent_response(monkeypatch):
    config = {
        "INFERENCE_API_URL": "https://example.com",
        "INFERENCE_API_KEY": "test-key",
        "INFERENCE_TIMEOUT_SECONDS": 120,
    }
    image = UploadedImage("image.png", png(), "b" * 64, "PNG", 24, 16)

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return verdict(0.8, MANIFEST)

    def post(url, **kwargs):
        assert url == "https://example.com/predict"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["timeout"] == (5, 120)
        return Response()

    monkeypatch.setattr(requests, "post", post)
    client = RemotePredictor(config)
    assert client.predict(image)["label"] == "ai"
    monkeypatch.setattr(Response, "json", lambda self: {**verdict(0.8, MANIFEST), "label": "real"})
    with pytest.raises(InferenceUnavailable):
        client.predict(image)


def test_remote_timeout_is_friendly(monkeypatch):
    def timeout(*args, **kwargs):
        raise requests.Timeout("private upstream detail")

    monkeypatch.setattr(requests, "post", timeout)
    client = RemotePredictor(
        {
            "INFERENCE_API_URL": "https://example.com",
            "INFERENCE_API_KEY": "key",
            "INFERENCE_TIMEOUT_SECONDS": 120,
        }
    )
    with pytest.raises(InferenceUnavailable, match="temporarily unavailable"):
        client.predict(UploadedImage("image.png", png(), "b" * 64, "PNG", 24, 16))
