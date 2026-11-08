# Tasks 4–5 — Flask inference and database trail

Project: **23l-0829_A2_Ops**. The application serves the registered version 2
champion (`transfer-tuned`) with its validation-selected threshold
**0.7224706411361694**. These notes describe the implementation and measured
checks; screenshots still need to be captured for the report.

## 4.1 — Model lifetime and matching preprocessing

`visionguard.webapp:create_app()` creates one predictor at startup. Local mode
checks the `.keras` file against the SHA-256 in `manifest.json`, loads it with
`compile=False`, and reuses that model. A lock serializes access to the local
Keras model; Gunicorn runs one worker to avoid loading multiple copies.

Training and serving import the same `decode_and_resize` function. It decodes
three channels, ignores animated-image expansion, resizes with preserved aspect
ratio and padding to 128×128, applies antialiasing, and clips pixels to [0, 255].
The MobileNetV2 normalization is already inside the trained model. Serving calls
it with `training=False`, so random training augmentation and dropout are inactive.
Normalizing again or stretching the image would change the input distribution.

Loading the model for every upload would repeat filesystem reads and TensorFlow
initialization, increase latency and memory pressure, and make failures occur in
the request path. Registry resolution and artifact downloads belong to deployment,
not to individual predictions.

Two actual HTTP requests to the native Gunicorn server were compared with the
stored **validation** predictions; no test-set evaluation was repeated:

| Validation image | Decision | p(AI) | Predicted-class score | Absolute difference |
| --- | --- | ---: | ---: | ---: |
| `10-tips-for-stunning-portrait-photography-7.jpg` | Real | 0.304313302 | 0.695686698 | 6.20e-8 |
| `-yellow-kimono-with-a-golden-crown-and-a-very-detailed-Barn-owl-copy-800x800.jpg` | AI | 0.757909596 | 0.757909596 | 5.60e-8 |

The result's label uses the locked threshold, rather than a hard-coded 0.5.
The predicted-class score is `p(AI)` for AI and `1-p(AI)` for real. It is an
uncalibrated model score, not a guarantee of correctness. The interface says so.

## 4.2 — Routes, validation, and temporary data

| Route | Behavior |
| --- | --- |
| `GET /` | Upload page and in-browser image preview |
| `POST /predict` | Validate, infer, save a prediction, then redirect with HTTP 303 |
| `GET /result/<id>` | Read an existing result from the database |
| `GET /history?page=1` | Show 20 persisted results per page |
| `GET /health` | Check database connectivity and report configured backend |
| `POST /api/predict` | Local JSON alternative that also persists history; HTTP 201 |

The separate cloud API exposes `POST /predict` with a Bearer API key and returns
JSON without a database. Its `/health` checks the model server's readiness.

Uploads must contain JPEG or PNG bytes with a matching extension. The server
sanitizes names with `secure_filename`, reads at most 5 MiB plus one byte, verifies
and fully decodes the image with Pillow, and rejects dimensions above 4 million
pixels. The request limit allows a small multipart envelope as well as the image.
Wrong types return 415, oversized files 413, and undecodable images 422. Missing
files return 400. HTML requests receive friendly error pages; API requests receive
JSON errors.

Flask's request class overrides the multipart file stream with `BytesIO`, including
files larger than Werkzeug's usual spooling threshold. Neither Flask nor the
inference service writes uploaded images to disk. Only their sanitized name,
hash, dimensions, and prediction metadata are stored. The browser preview uses a
temporary object URL. POST–redirect–GET prevents refreshing a result page from
repeating model inference or creating another prediction row.

Real HTTP checks confirmed 415 for a text file, 422 for a broken JPEG, and 413 for
an oversized upload, with friendly pages. Automated checks also cover mismatched
content/extensions and excessive resolution.

## 4.3 — HTML and CSS

The shared layout links upload and history. The upload form accepts one image,
shows its preview, and disables the submit button while the request is in flight.
The result page displays the decision, score, threshold marker, model version,
filename, timestamp, and inference time. History reads database rows and links
back to each saved result. Templates escape filenames; CSS supports narrow screens.

Capture the upload, both known-image result pages, populated history, and each
required validation failure in a browser with the localhost address visible.

## 4.4 — Configuration and server

`.env.example` documents model paths, an optional threshold override, database
URL, upload limits, timezone, backend, cloud URL, and API key. The actual `.env`
is ignored. A blank threshold override uses the pinned manifest. Model changes
require a deliberate restart; no remote download occurs during a request.

Run from the repository root with the environment active:

```bash
gunicorn --bind 127.0.0.1:5000 --workers 1 --threads 2 --timeout 180 \
  --access-logfile - --error-logfile - 'visionguard.webapp:create_app()'
```

Debug mode is explicitly off. Gunicorn runs the Flask WSGI application without
the development debugger or automatic reloader. The 180-second worker timeout and
120-second remote read timeout allow for Render's free-service cold start.
The local health endpoint verifies Flask/database availability; a remote model
outage is reported on prediction rather than waking the cloud on each local
health check.

## Task 5 — Durable, queryable records

SQLAlchemy creates two tables on startup:

| Table | Stored information |
| --- | --- |
| `predictions` | UUID, UTC timestamp, sanitized filename, image SHA-256, format/dimensions, decision, p(AI), predicted-class score, threshold, model name/version/SHA, backend, latency |
| `request_log` | Unique request UUID, UTC timestamp, HTTP method/path/status, total duration |

The browser converts UTC timestamps to `Asia/Karachi`. Each successful inference
is committed in a transaction before returning its result. Rejected uploads and
unavailable-model requests create audit rows but no successful prediction row.
All local HTTP requests, including result/history reads and errors, are audited.
Query strings, authorization headers, passwords, and image bytes are not stored.

Native execution uses the persistent `.cache/history.db` SQLite file. A test
creates a prediction, disposes the first application, creates a second application
against the same file, and verifies that history remains. Compose selects
PostgreSQL instead, with a named volume independent of container lifetime and
no externally published database port. Full Compose runtime verification remains
pending on this machine because Docker cannot create bridge interfaces.

Example query without displaying credentials:

```bash
python - <<'PY'
from sqlalchemy import select
from visionguard.database import Database, predictions
from visionguard.serving_config import settings
db = Database(settings()['DATABASE_URL'])
with db.engine.connect() as connection:
    rows = connection.execute(select(
        predictions.c.created_at, predictions.c.filename, predictions.c.label,
        predictions.c.confidence, predictions.c.model_version,
    ).order_by(predictions.c.created_at.desc()).limit(5))
    for row in rows:
        print(row)
PY
```

The complete suite currently has **27 passing tests**. Runtime evidence is in
`reports/generated/serving/local_http_smoke.json`. Capture history after a real
server restart and a database query as report evidence; generated JSON is not a
replacement for the assignment's screenshots.
