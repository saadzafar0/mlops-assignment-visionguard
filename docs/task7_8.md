# Tasks 7–8 — Live API and local Flask integration

Project: **23l-0829_A2_Ops**.

- Model API: <https://mlops-assignment-visionguard.onrender.com>
- Readiness: <https://mlops-assignment-visionguard.onrender.com/health>
- Local Flask interface: <http://127.0.0.1:5000>

The Render service serves the model API. Flask's upload/result/history views and
the persistent database run locally, following the Task 7 instruction. The cloud
has no SQLite history file; its ephemeral disk is not used for user records.

## Task 7 — Deployment and actual API checks

The service builds the repository's default Docker target on Render Free. It
contains a thin Flask API and TensorFlow Serving 2.20.0 with the unchanged
champion graph and preprocessing. The service honors Render's `PORT`, requires a
Bearer API key, and returns the original registry version and model checksum.
Step-by-step deployment, environment settings, and both diagnosed startup
failures are in `deployment.md`.

After the cloud-launcher fix in commit `f91ad04`, the public service returned a
healthy response and actual authenticated inference succeeded. Checks on
**8 November 2026** used two validation images; the test set was not re-evaluated.

| Validation image | Decision | p(AI) | Predicted-class score | Difference from saved validation score |
| --- | --- | ---: | ---: | ---: |
| `10-tips-for-stunning-portrait-photography-7.jpg` | Real | 0.304310948 | 0.695689052 | 2.292e-6 |
| `-yellow-kimono-with-a-golden-crown-and-a-very-detailed-Barn-owl-copy-800x800.jpg` | AI | 0.757909358 | 0.757909358 | 1.820e-7 |

Both responses identified registry version **2**, model `transfer-tuned`, the
locked threshold **0.7224706411361694**, and SHA-256
`a9fd72cfee813fc4dfa5ba49441f5c725a076d24097e9a429999082161f84289`.
The small floating-point differences are within the 1e-5 cross-host comparison
tolerance and do not change either decision.

| Public request | Observed response |
| --- | --- |
| `GET /health` | 200, `status=ok`, `model_loaded=true`, `model_version=2` |
| `POST /predict` without key | 401 |
| Valid real/AI images with key | 200 with decision, scores, and provenance |
| Unsupported text file | 415 with friendly JSON error |
| Corrupt JPEG | 422 with friendly JSON error |
| File larger than 5 MiB | 413 with friendly JSON error |

Results are in `reports/generated/serving/cloud_api_smoke.json`. The source
fixture paths, response request IDs, latency, scores, hashes, and timestamps make
the checks traceable. Credentials and image bytes are absent from the evidence.

The manual GitHub workflow `deployed-api.yml` checks this same public URL from a
hosted runner outside the local network. It requires healthy champion metadata
and HTTP 401 for an unauthenticated prediction. It does not need the inference
key or deployment credentials and is kept separate from the build CI so an idle
or unavailable demo service does not block source-code checks.

For the required **Postman** evidence, import
`deploy/visionguard.postman_collection.json`; the actual base URL is already set.
Store the matching key privately, select validation JPEG/PNG files, and capture
the successful real/AI responses and missing-key failure. Keep the Authorization
value out of screenshots. Automated HTTP checks do not replace this required
Postman demonstration or the report screenshots.

## Task 8 — Local Flask calls the cloud model

The ignored local `.env` now selects `INFERENCE_BACKEND=remote` and the verified
Render URL. The existing Gunicorn workers were gracefully reloaded. Flask's
`/health` reports database `ok` and inference backend `remote`; the key remains
server-side and is not included in browser scripts or HTML.

Two real HTTP uploads to localhost `/predict` returned 303 redirects. Each result
page loaded successfully, and refreshing it did not submit another prediction.
The database stored both decisions with backend **remote**, the cloud model
fingerprint/version, threshold, scores, input hash/dimensions, and latency.
Their POST request IDs also matched the persisted audit records.

| Saved remote result | Decision | p(AI) |
| --- | --- | ---: |
| `a75b8911-684b-40b9-8bef-034da64c6bcd` | Real | 0.304310948 |
| `2ac13b11-c71c-4b1f-ae7d-771a3ef4e276` | AI | 0.757909358 |

After another worker restart, `/history` and both result pages still returned
200 and displayed the same database records. The evidence is
`reports/generated/serving/remote_web_smoke.json`. The history page now contains
these actual cloud-backed results alongside the earlier local-model checks.

Capture the local upload page, multiple cloud-backed result pages, populated
history after restart, and a database query showing `backend=remote` for the
report. Include browser/terminal context and captions. All required screenshots
and the final PDF remain separate deliverables.
