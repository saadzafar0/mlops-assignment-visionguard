# Task 1 — Data understanding and versioning

Project: **23l-0829_A2_Ops**. These are working report notes, not the final PDF.
Actual dataset findings belong in [Task 1 results](task1_results.md) once execution
has completed. Do not report generated examples or smoke-test results as training.

## 1.1 — Isolated environment

`uv sync --locked` installs the dependency versions from `uv.lock` into `.venv`.
`source .venv/bin/activate` selects that environment for subsequent commands.
Python 3.12 was selected to work with the pinned TensorFlow 2.20 CPU package.
The CPU package avoids installing CUDA dependencies for this initial workflow.
GPU training, if added later, requires a separate compatible GPU environment;
a CPU-only build reporting no visible GPU does not establish the machine's
physical GPU inventory.

`python -m visionguard.environment` records the interpreter path, platform,
TensorFlow/Keras versions, CUDA build support, visible devices, current Git commit,
timestamp, and exact installed package versions. Results are written to
`reports/generated/environment/`. The snapshot describes the environment at the
time the command runs; subsequent commits can change the repository revision.

Required evidence: capture the command and its version/device output, with the
terminal path, date/time, and project identity visible.

## 1.2 — Dataset audit

`python scripts/download_dataset.py` retrieves the assignment's Kaggle dataset at
version 1. The source URL, archive size, extracted file count, and SHA-256 are
recorded in `reports/generated/dataset_source.json`.

`python -m visionguard.audit` recursively walks every file under the
dataset root. Class names come from matching ancestor-folder aliases in
`params.yaml`; the script does not infer provenance from filenames. The nearest
folder is not blindly used because some datasets contain nested subject folders.
Ambiguous or unknown class paths are recorded for review and stop the CLI.
Dot-prefixed image filenames are included: this dataset contains two such images,
so treating every hidden file as metadata would silently omit valid samples.

For each file, the script records byte size and streams its contents into SHA-256.
Pillow reads dimensions, colour mode, and format, then verifies the container and
fully decodes pixels. Reopening the image for pixel decoding helps catch files
whose headers are readable but whose image data is truncated. Files are read-only
throughout this procedure.

Equal SHA-256 hashes identify exact byte duplicates. The first path in sorted order
is retained as the reference, and later copies point to it through `duplicate_of`.
An identical image in both classes is marked as a label conflict. This procedure
does not detect recompressed or resized versions of the same picture.

Class statistics use readable, recognised images. They describe the original
dataset, including duplicates, so the reader can distinguish source counts from
the cleaned training population. Duplicate and unreadable counts are reported
separately. No image is deleted from the source dataset.

Required evidence: the per-class counts and min/median/max table, total corrupt
files, duplicate counts, and any label conflicts.

## 1.3 — Visual EDA

`python -m visionguard.eda` uses the audit table to select readable images. It
samples five nonduplicate images per class using seed 42, then creates width,
height, and file-size histograms. Each histogram uses the same bin boundaries for
both classes. File sizes are displayed in KiB (1 KiB = 1024 bytes).

`reports/generated/eda/figure_notes.md` contains draft captions based on the
measured medians. The sample-image caption deliberately asks for personal visual
observations rather than inventing them. Discuss whether resolution, compression,
subject matter, or style could become shortcuts for classification.

Required evidence: sample grid, all three histograms, captions, and two or three
sentences interpreting each figure.

## 1.4 — DVC toy exercise and project application

`python labs/dvc_toy.py` creates an independent Git/DVC repository. Keeping it under
ignored `labs/work/` preserves the main project's history and ensures the toy CSV
cannot be confused with the real training data. Its two commits correspond to the
initial CSV and an added row. A local directory acts as the remote storage.

`git checkout <old_commit>` restores the old pointer, and `dvc checkout` restores
the corresponding data from cache. The script subsequently removes the toy CSV
and toy cache, then runs `dvc pull`, proving the remote can recover the data. The
main branch is restored at the end. Only files created by this toy are removed.
The toy's commits are local until published to a separate DagsHub repository;
publication and screenshot evidence remain required.

For the real dataset, `python scripts/configure_dvc.py` initializes DVC and uses
DagsHub's documented S3-compatible storage endpoint. The remote URL is `s3://dvc`,
with the repository-specific endpoint URL. Credentials are written only to local
configuration, which Git ignores.

`dvc add data/raw` stores the dataset in DVC's content-addressed cache and creates
`data/raw.dvc`. That small YAML pointer contains a content checksum, output path,
size, and file count for a directory. DVC's checksum is separate from the per-image
SHA-256 used in the audit. Git tracks the pointer; DVC storage holds the bytes.
Putting large image collections in Git would enlarge clone/history storage and
make ordinary code collaboration expensive.

`dvc repro` runs the audit, EDA, and split stages in `dvc.yaml`. DVC records input,
code, parameter, and output hashes in `dvc.lock`, and reruns stages when their
declared dependencies change. `dvc push` transfers the cache objects to the remote;
Git commits alone do not upload image data.

In this restricted workspace, DVC's default auxiliary SQLite cache was unwritable.
Setting `DVC_SITE_CACHE_DIR=/tmp/visionguard-dvc-site` resolved that issue while
keeping the actual content-addressed dataset cache in `.dvc/cache`.

Required evidence: initial toy commits, old-version restoration, push/delete/pull
output, project dataset pointer, and the configured remote with secrets hidden.

## Preparing for Task 2

`python -m visionguard.split` creates stratified 70/15/15 splits after dropping
unreadable files, removing all cross-class conflicting hashes, and retaining one
copy of each remaining hash. All splits use the mapping `real=0`, `ai=1`. The split
CSV paths are relative to the dataset root, so training can move to another
machine without changing the lists.

The script verifies that no content hash crosses split boundaries. Tests also
verify deterministic results. Near duplicates and subject overlap can still
remain, so this protection does not imply complete independence of image content.
Test data must not guide augmentation, hyperparameter tuning, threshold selection,
or model choice.

## Sources

- [TensorFlow pip installation](https://www.tensorflow.org/install/pip)
- [DagsHub DVC integration](https://dagshub.com/docs/integration_guide/dvc/)
- [DagsHub MLflow integration](https://dagshub.com/docs/integration_guide/mlflow_tracking/)
