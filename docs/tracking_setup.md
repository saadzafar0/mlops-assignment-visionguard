# MLflow and DagsHub setup

The existing GitHub repository is connected to DagsHub. The assignment's project
name is `23l-0829_A2_Ops`, and the existing repository URL remains
`saadzafar0/mlops-assignment-visionguard`.

`src/visionguard/tracking.py` configures MLflow in two modes:

- **Local:** SQLite metadata under `.cache/mlflow/mlflow.db`, selected explicitly
  even when `.env` contains a remote tracking URI.
- **DagsHub:** the repository's hosted MLflow endpoint, with authentication taken
  from environment variables. A missing token fails with a helpful error instead
  of prompting for browser login or embedding a token in code.

The configuration follows DagsHub's documented manual MLflow authentication.
`MLFLOW_TRACKING_USERNAME` is the account name, and `MLFLOW_TRACKING_PASSWORD`
receives either its explicit environment value or `DAGSHUB_USER_TOKEN`. The
`.env.example` template contains empty credential fields. The actual `.env` and
`.dvc/config.local` are ignored by Git.

Both connectivity checks were executed successfully. Each logged a project
parameter, a numeric metric, a Git-revision tag, and `check.txt` as an artifact.

| Target | Run ID |
| --- | --- |
| Local | `180df5ad5d80433fbd41a2f369d96621` |
| DagsHub | `1091c79211284639930d297970fda080` |

[Open the remote run](https://dagshub.com/saadzafar0/mlops-assignment-visionguard.mlflow/#/experiments/0/runs/1091c79211284639930d297970fda080).

These checks establish parameter/metric/artifact connectivity. They do **not**
meet Task 3.1's two trained Keras runs, Task 3.3's manual/autologging comparison,
Task 3.5's six VisionGuard experiments, or Task 3.6's registry exercise. Those
remain part of the next model-development phase.

The separate [MLflow training lab](../labs/README.md) has since completed both
local Keras runs with learning rates 0.001 and 0.01. See its measured comparison
table and `labs/mlflow_toy.py`; UI screenshots remain required.
The 0.01 run was also repeated successfully on DagsHub, logging all six epochs,
the training plot, split indices, source script, and a native TensorFlow model.
[Open the remote trained run](https://dagshub.com/saadzafar0/mlops-assignment-visionguard.mlflow/#/experiments/0/runs/2f366ee2f8c148b89b249bdb3da7b338).
Its final validation accuracy was 95.00%, matching the local run.

## Terms for the report

- **Experiment:** a named group of related training attempts.
- **Run:** one execution within an experiment, with its own ID and results.
- **Parameter:** a chosen setting, such as a learning rate or image size.
- **Metric:** a measured value, such as validation loss at a particular epoch.
- **Artifact:** a saved output file, such as a plot, model, or split list.
- **Tag:** descriptive metadata used to identify or organise runs, such as the
  source commit or purpose.

## Evidence still needed

Capture the remote experiment/run page with the project context visible. For the
training lab, capture both trained runs and MLflow's comparison view after those
runs exist. Keep credential values out of screenshots.

Reference: [DagsHub MLflow integration](https://dagshub.com/docs/integration_guide/mlflow_tracking/).
