# Task 3 — Experiment tracking and model handoff

Project: **23l-0829_A2_Ops**. See `tracking_setup.md` for connectivity checks and
`labs/README.md` for the completed local/remote Keras practice runs. Their metrics
are separate from the VisionGuard experiments.

## 3.1–3.2 — Local MLflow and DagsHub

The local exercise trained two Keras models on offline digits data, changing only
learning rate (0.001 versus 0.01), and logged configuration, per-epoch metrics,
plots, split indices, and native TensorFlow models. The 0.01 run was repeated on
DagsHub and matched the local 95.00% validation accuracy. Definitions of experiment,
run, parameter, metric, artifact, and tag are in `tracking_setup.md`.

Remote MLflow uses the existing DagsHub repository endpoint. The DagsHub username
and token come from the ignored `.env`; the tracked example contains empty secret
fields. DVC uses the same repository's S3-compatible storage, with credentials
kept in ignored `.dvc/config.local`.

Required evidence: both local run pages and comparison view, the DagsHub project,
remote toy run, and proof that credential files are ignored (show names, not values).

## 3.3 — Manual logging and TensorFlow autologging

CNN stages use a custom Keras callback that sends every epoch's numeric log values
to MLflow with an explicit epoch step. Transfer stages use
`mlflow.tensorflow.autolog(log_models=False, log_datasets=False, checkpoint=False)`;
the surrounding code controls the final model, data artifacts, and checkpoints.

Each candidate is a parent run. Its individual training stages are nested runs:
one baseline stage, or separate frozen-head and fine-tuning stages. This avoids
trying to change a fixed MLflow learning-rate parameter inside the same run when
the optimiser is recompiled for fine-tuning. The parent holds the candidate's
chosen checkpoint, exact validation metrics, decision threshold, source revision,
DVC hash, and eventual test metrics if selected for final evaluation.

Remote metric histories were queried to verify that manual and automatic logging
actually stored the expected epochs:

| Stage | Logging | Recorded `val_loss` values |
| --- | --- | ---: |
| Reference CNN | Manual | 5 |
| Reference MobileNet frozen head | Autologging | 8 |
| Reference MobileNet fine-tuning | Autologging | 6 |

Manual logging gives explicit control over keys and steps, but requires extra code
for optimiser settings, model summaries, and artifacts. Autologging captures
standard training/optimiser information and metrics with less instrumentation.
Neither knows the correct DVC dataset version, product error costs, external split
lists, chosen threshold, or rationale; these are logged explicitly in both modes.

Required evidence: run parameters, metric curves, and provenance tags for both
approaches, with stage/parent relationships visible.

## 3.4 — Artifacts

Every candidate logs the fixed train/validation/test CSVs, package list, lockfile,
configuration, DVC pointer, training source, expanded summaries, layer tables,
trainable-layer lists, learning curves, validation predictions, and final model.
Both a native MLflow TensorFlow model and a directly loadable `.keras` artifact are
saved. The two validation-selected final models also receive confusion matrices,
test prediction CSVs, metric comparisons, error grids, and ROC curves after their
single test inference pass.

Required evidence: the model and evaluation artifacts tabs.

## 3.5 — Controlled comparison

The six planned candidates are:

| Candidate | Reference | Meaningful changed setting |
| --- | --- | --- |
| `baseline-reference` | — | Custom CNN, learning rate 0.001, augmentation |
| `baseline-lower-lr` | Baseline reference | Learning rate 0.0003 |
| `baseline-no-augmentation` | Baseline reference | Disable augmentation |
| `transfer-reference` | — | MobileNetV2, dropout 0.3, four trainable tail convolutions |
| `transfer-lower-dropout` | Transfer reference | Dropout 0.1 |
| `transfer-fewer-unfrozen` | Transfer reference | Two trainable tail convolutions |

The separate eight-trial tuner adds a seventh candidate after selecting its best
fine-tuning settings. The final choice is made from validation metrics before test
inference. ROC-AUC assesses ranking independently of a threshold; the stated 3:1
false-positive cost then chooses a cautious threshold. Actual measured comparison
tables and the selected model are recorded in `task2_results.md` after execution.

Required evidence: select the candidate parent runs in DagsHub's comparison view
and include the validation comparison table with a written selection rationale.

## 3.6 — Model registry and serving handoff

`python scripts/register_models.py` reads the locked validation selection. It
registers the best candidate and a runner-up of the same approach as two versions
of `23l-0829_A2_Ops`, reusing an existing version when the same source run is already
registered. It assigns the winning version the alias `champion`. Version tags
record threshold, input size, source model hash, commit, and DVC data version.

The independent `python scripts/predict_by_alias.py` loads
`models:/23l-0829_A2_Ops@champion`, decodes one validation image with the shared
preprocessing function, and prints the model version, label, `p(AI)`, threshold,
and predicted-class score. An explicit image can be supplied through `--image`.

The future server uses the validated local `models/champion/model.keras` plus its
manifest. A deployment preparation step retrieves the chosen version once or pulls
the DVC artifact; each HTTP request reuses the in-memory model. The web app must
not contact DagsHub per prediction. Alias lookup is a demonstration and deployment
operation, not part of the per-request serving path.

Required evidence: registry page showing at least two versions and `champion`,
then terminal output from the separate alias-loading script.

References:
[MLflow registry and aliases](https://mlflow.org/docs/latest/model-registry/),
[DagsHub MLflow integration](https://dagshub.com/docs/integration_guide/mlflow_tracking/).
