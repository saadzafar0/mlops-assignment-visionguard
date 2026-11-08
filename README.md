# VisionGuard — 23l-0829_A2_Ops

AI-generated versus real image classification for MLOps Assignment 2.

- Student: Saad Zafar, **23l-0829**
- Project name: **23l-0829_A2_Ops**
- [GitHub](https://github.com/saadzafar0/mlops-assignment-visionguard)
- [DagsHub](https://dagshub.com/saadzafar0/mlops-assignment-visionguard)
- [MLflow experiments](https://dagshub.com/saadzafar0/mlops-assignment-visionguard.mlflow)
- Live API: pending deployment

Task 1's data audit, EDA, reproducible splits, and local DVC exercise have been run.
The dataset and pipeline outputs have been uploaded to DagsHub DVC storage.
See [measured results](docs/task1_results.md), [Task 1 explanations](docs/task1.md),
and [DagsHub connection checks](docs/tracking_setup.md). Required screenshots and
publishing the toy commits remain outstanding. Model training, the web application,
database, Docker, CI, and deployment are the next phases.

## Environment

Python **3.12** is required. TensorFlow uses the CPU package for a portable local
environment and eventual CPU serving. The environment check distinguishes GPU
visibility from whether this TensorFlow build supports CUDA.

```bash
uv sync --locked
source .venv/bin/activate
cp .env.example .env  # first setup only; preserve an existing .env
python -m visionguard.environment
```

`uv.lock` pins all resolved dependencies. `pyproject.toml` declares the direct
dependencies. The editable package installation lets scripts import `visionguard`
without manually changing `PYTHONPATH`. If the default uv cache is unwritable,
prefix uv commands with `UV_CACHE_DIR=/tmp/visionguard-uv-cache`.

Add your DagsHub token to `DAGSHUB_USER_TOKEN` in `.env`. Credentials must never
appear in source code, report screenshots, Git commits, or Docker images.

## Dataset and Task 1

Use the dataset specified in the assignment:
[cashbowman/ai-generated-images-vs-real-images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images).
The download script fixes the source to version 1, records the ZIP's SHA-256,
checks the archive, and extracts into `data/raw`. It refuses to overwrite existing
data. The archive requires roughly 499 MB; extracted files and DVC cache need
additional space.

```bash
python scripts/download_dataset.py
python -m visionguard.audit
python -m visionguard.eda
python -m visionguard.split
```

`params.yaml` controls class-folder aliases, split fractions, and the seed. Labels
are explicitly **real = 0**, **AI = 1**. Unknown class folders stop the workflow
instead of silently assigning incorrect labels.

The audit produces:

| File under `reports/generated/audit/` | Purpose |
| --- | --- |
| `audit.csv` | Path, class, label, size, dimensions, colour mode, format, SHA-256, decode status, duplicate/conflict flags |
| `class_summary.csv` | Counts and min/median/max dimensions and file sizes by class |
| `duplicates.csv` | Exact duplicate files beyond the first copy |
| `invalid_files.csv` | Corrupt, unreadable, or unrecognised files |
| `label_conflicts.csv` | Identical bytes assigned to both classes |
| `summary.json` | Machine-readable totals |

EDA writes a sample grid, three shared-bin histograms, and draft figure notes to
`reports/generated/eda/`. Inspect the actual images and add your own visual
observations before submitting the report.

Splits are stratified **70% training / 15% validation / 15% test**, with seed 42.
Unreadable files and all copies of conflicting-label images are excluded. Within
a single class, one copy of each SHA-256 remains. Split CSVs contain relative paths
and hashes, making them portable across machines. This prevents exact-duplicate
leakage; perceptually similar images can still occur across splits.

## DVC

Run the standalone CSV exercise before the project workflow:

```bash
python labs/dvc_toy.py
```

It creates a separate Git/DVC repository under ignored `labs/work/`, commits two
CSV versions, pushes both to a local remote, restores the older version, deletes
the working CSV and toy cache, and pulls it back. Evidence is saved under
`reports/generated/dvc_toy/`. This exercise is local; its commits still need to be
published to a separate DagsHub toy repository to meet that part of the PDF.

Configure the real project's DagsHub storage and record its dataset:

```bash
python scripts/configure_dvc.py
dvc add data/raw
dvc repro
dvc push
```

In a restricted environment where DVC's default system cache is unwritable, use
`export DVC_SITE_CACHE_DIR=/tmp/visionguard-dvc-site` before these commands. This
changes DVC's auxiliary index/state location; the project data cache remains in
`.dvc/cache`.

The DagsHub S3 endpoint and remote name are tracked in `.dvc/config`. Tokens go only
into ignored `.dvc/config.local`. Commit `data/raw.dvc`, `dvc.yaml`, `dvc.lock`,
`.dvc/config`, and `.dvc/.gitignore`; keep images and the cache out of Git. Cloning
the Git repo and running `dvc pull` retrieves the versioned data and pipeline outputs.

## MLflow connection check

```bash
python scripts/check_tracking.py --target local
python scripts/check_tracking.py --target dagshub
```

Each command logs a parameter, metric, tag, and small text artifact. These are
connection checks, **not** the required two-model MLflow training lab or the six
VisionGuard experiments. Local tracking uses `.cache/mlflow/mlflow.db`:

```bash
mlflow ui --backend-store-uri "sqlite:///$(pwd)/.cache/mlflow/mlflow.db" --port 5000
```

The actual Task 3.1 training exercise uses scikit-learn's offline 8×8 digits data:

```bash
python labs/mlflow_toy.py --target local
python labs/mlflow_toy.py --target dagshub --learning-rates 0.01
```

Two local runs compare learning rates 0.001 and 0.01 with identical data splits,
initialisation seeds, and six epochs. Each logs per-epoch metrics, a plot, split
indices, and a native MLflow TensorFlow model. Results and evidence instructions
are documented in [the tool labs](labs/README.md).

## Checks

```bash
ruff check .
ruff format --check .
pytest -q
```

The seven data tests cover corruption, exact duplicates, conflicting labels,
unknown folders, dot-prefixed image filenames, decoded formats, deterministic
splits, and absence of shared hashes between splits.

## Submission evidence

The final PDF must be named `23l-0829_VisionGuard.pdf`. Capture real screenshots
with terminal/browser context and captions as work proceeds. Generated CSVs and
plots support the report but do not replace required screenshots.

AI assistance disclosure: OpenAI Codex assisted with repository setup, scripts,
documentation, and validation. Update the final report with the actual scope of
assistance and your own explanations of the submitted work.
