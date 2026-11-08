# Task 6 — Docker, Compose, operations, and CI

Project: **23l-0829_A2_Ops**. This chapter distinguishes local checks from checks
run on GitHub's hosted Docker engine. The local host cannot create bridge-network
interfaces; changing the host kernel is outside the project setup. Standalone
images were tested locally using an isolated network and internal HTTP clients.
The GitHub workflow exercises port mapping and the real Compose bridge network.

## 6.1 — Docker tool lab

Run `bash labs/docker_toy.sh` on a working Docker engine. The lab pulls the official
NGINX image, creates a uniquely named image tag and volume, publishes the HTTP port
on localhost, writes a project page in the mounted directory, reads HTTP responses
and container logs, opens a shell to inspect the user/files, and lists containers
and images. It stops/removes the first container and starts a replacement with the
same volume, verifying that the page remains. It removes only its own container,
volume, and temporary image tag afterward. CI runs the same script before the
project build. Its terminal output includes the project name and UTC timestamp.

- An **image** is the filesystem and startup configuration used to create a container.
- A **container** is a running or stopped instance with isolated processes and a writable layer.
- A **layer** is a cached filesystem change from an image build instruction.
- A **volume** stores data independently of an individual container's writable layer.
- A **port mapping** connects a host address/port to a container's listening port.

Capture the actual run, HTTP response, mount inspection, logs, shell, listings,
and cleanup from the terminal or GitHub job; a script alone is not execution evidence.

## 6.2 — Dockerfile and model delivery

The root Dockerfile uses a slim Python 3.12 base. Dependency requirements are copied
and installed before source, so code-only edits reuse the large package layer.
Runtime dependencies are frozen from the tested environment; model export and
local inference use CPU-only TensorFlow 2.20.0. A GPU package would add unnecessary
CUDA dependencies to these CPU services.

The `web` target runs the Flask UI with the original Keras champion. The default
`inference` target exports its unchanged graph and copies the pinned official C++
TensorFlow Serving executable into a thin Python API image. Both use `appuser`
(UID 10001), Gunicorn with debug disabled, and an HTTP health check. The API
preflight rejects a missing API key before model startup.

The model is built into the image to make deployments self-contained. A build
downloads from an immutable public DagsHub revision and checks the model and
manifest hashes against `deploy/model.lock.json`. No user token is required.
`.dockerignore` excludes `.env`, Git/DVC caches, datasets, training outputs, and
local credentials. Predictions reuse the loaded model; requests do not fetch it.

## 6.3 — Compose stack

`compose.yaml` declares `web` and `db`. The web container connects to PostgreSQL
through hostname `db` on the internal network. PostgreSQL uses the named
`postgres_data` volume and a `pg_isready` health check. `depends_on` requires a
healthy database before starting web. Only the web port is published, on
127.0.0.1; PostgreSQL has no host port binding. The password comes from the ignored
`.env`; the tracked example leaves it blank.

```bash
docker compose up --build -d --wait
docker compose ps
docker compose logs -f web db
docker compose exec web sh
docker compose exec db sh
docker compose stop
docker compose start
```

`docker compose down` removes containers/network while preserving the named data
volume. Adding `--volumes` also deletes the database; use it only for disposable
lab/CI stacks. Normal project operations should retain it.

## 6.4 — Image analysis

The local Python TensorFlow web image measured **2,200,499,661 bytes**. The
multi-stage C++ inference image measured **1,216,319,411 bytes**, a **44.73%**
reduction. They share the model and preprocessing; the stateless API omits the
local UI/history features and Python TensorFlow environment. Layer listings are
in `reports/generated/serving/image_analysis.json`. The largest API layer contains
the model-server executable, and the largest web layer contains Python packages.

Image size is distinct from memory use. The complete API passed an actual 512 MiB
container limit with peak cgroup usage about 325.4 MiB, including test clients.
See `docs/deployment.md` for the measured response and memory table.

## 6.5 — Operations and troubleshooting

CI starts the stack, checks the database by service name, performs a real HTTP
prediction, inspects the volume and absence of public database bindings, then
restarts the database and web separately. It checks readiness and confirms the
saved prediction remains in the history page.

A disposable one-off web container is deliberately given the wrong database
password. The workflow requires its logs to contain `password authentication
failed`; reconnecting with the normal service environment must succeed. It never
changes the real service password or user history. Status and logs are emitted
even after failures, and cleanup removes only the isolated CI stack and its volume.

The first real Render deployment exposed a separate configuration failure: the
model loaded, but the API worker exited because `INFERENCE_API_KEY` was missing
from Render's environment. The diagnosis and remedy are recorded in
`docs/deployment.md`. Do not interpret TensorFlow CPU optimization notices as the
cause of that failure.

## 6.6 — GitHub Actions

`.github/workflows/ci.yml` runs on pushes, pull requests, and manual dispatch:

1. Install Python 3.12 and the locked dependencies through pinned uv 0.12.17.
2. Check Ruff lint/format and run all data, training-policy, and serving tests.
3. Run the Docker warm-up, build the real inference image, and smoke-test it at
   512 MiB, including authentication, image decoding, and missing-key startup.
4. Build/run the real web/PostgreSQL Compose stack, verify network/volume/history
   behavior, exercise the deliberate failure, and clean up its temporary data.

Official checkout/setup actions are pinned by immutable commit. The workflow has
read-only repository permission and cancels superseded runs on the same ref.
It neither retrains models nor re-evaluates the held-out test set. Synthetic images
exercise serving; the model is the existing checksum-verified public champion.
The README badge links to the actual workflow result.

No repository secret is needed for these public downloads. CI generates a fresh,
temporary database password and inference key without printing either. Future
deployment credentials belong in GitHub **Settings → Secrets and variables →
Actions**, referenced through `secrets`, rather than workflow text. The hosted
Render API key belongs in Render's Environment settings and the local ignored
`.env`; it is not required for CI's isolated API.

The isolated [CI toy repository](https://github.com/saadzafar0/23l-0829_A2_Ops-ci-toy)
completed both required runs. An intentionally incorrect subtraction implementation
failed its addition assertion in the [red run](https://github.com/saadzafar0/23l-0829_A2_Ops-ci-toy/actions/runs/42490780163).
Changing subtraction to addition passed on **Python 3.11 and 3.12** in the
[green run](https://github.com/saadzafar0/23l-0829_A2_Ops-ci-toy/actions/runs/42490817253).
`python labs/ci_toy.py` created and published the two commits and waited for each
real workflow result before proceeding. It refuses to overwrite an existing lab
folder or remote repository. Metadata, job results, and the failure log are saved
in `reports/generated/ci_toy/`. Capture both run pages, including their two matrix
jobs, for the report.

The initial project workflow passed linting and all 27 tests but caught a Docker
warm-up readiness race: the first request received `Connection reset by peer`
while NGINX was starting. Retrying only refused connections missed that condition.
The lab now uses bounded `curl --retry-all-errors` retries during startup; its
normal HTTP assertions still fail if the service does not become ready. The
database-failure log assertion uses standard Python, avoiding an extra runner
utility dependency.

The complete project pipeline then
[passed](https://github.com/saadzafar0/mlops-assignment-visionguard/actions/runs/42491527434)
at commit `c251a37e94348c9f20c8e4706932e549d902c5b6`: all 27 tests, the Docker
warm-up, the 512 MiB API smoke, web/database readiness, an actual PostgreSQL-backed
HTTP prediction, internal network/no-public-DB-port checks, deliberate wrong
credentials and recovery, and retained history after restarting both services.
The quality job took 35 seconds; the Docker/Compose job took 3 minutes 1 second.
Actual metadata and logs are versioned in `reports/generated/ci/`. Screenshots of
the green run and relevant step output remain required for the final report.

CI/documentation-only commits may include `[skip render]` to avoid restarting the
cloud API while continuing to run GitHub Actions. See
[Render's skip-deploy documentation](https://render.com/docs/deploys#skipping-an-auto-deploy).

References: [GitHub Actions concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency),
[official checkout action](https://github.com/actions/checkout),
[official setup-uv action](https://github.com/astral-sh/setup-uv).
