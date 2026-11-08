# Shared CPU runtime: dependencies precede source so builds reuse this large layer.
FROM python:3.12-slim-bookworm AS model-runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src \
    KERAS_HOME=/tmp/keras MPLCONFIGDIR=/tmp/matplotlib TF_CPP_MIN_LOG_LEVEL=2
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements/app.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY deploy/model.lock.json /app/deploy/model.lock.json
COPY scripts/download_champion.py scripts/export_serving_model.py /app/scripts/
COPY src /app/src
RUN python scripts/download_champion.py

FROM model-runtime AS export
RUN python scripts/export_serving_model.py --output /export/1

# Local Flask UI and PostgreSQL Compose stack builds this target.
FROM model-runtime AS web
RUN useradd --uid 10001 --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)"
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "2", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "visionguard.webapp:create_app()"]

FROM tensorflow/serving:2.20.0@sha256:8a208c2322975a1bf04d9c70f2138ef38517bca966d4bd7816ded34e0a9bb257 AS serving

# Default target: small cloud API with no Python TensorFlow/training dependencies.
FROM python:3.12-slim-bookworm AS inference
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src \
    INFERENCE_BACKEND=tensorflow-serving MODEL_MANIFEST=/app/models/serving/manifest.json \
    API_REQUIRE_KEY=true MALLOC_ARENA_MAX=2
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements/web.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=serving /usr/bin/tensorflow_model_server /usr/bin/tensorflow_model_server
COPY --from=export /export /app/models/serving
COPY src /app/src
COPY deploy/start_api.sh deploy/wait_for_model.py /app/deploy/
RUN useradd --uid 10001 --create-home appuser && chmod +x /app/deploy/start_api.sh \
    && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health',timeout=3)"
CMD ["/app/deploy/start_api.sh"]
