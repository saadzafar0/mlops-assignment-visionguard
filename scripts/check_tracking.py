"""Log a small connectivity check; this is not the Task 3.1 training exercise."""

import argparse

import mlflow

from visionguard.tracking import configure_tracking, git_commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["local", "dagshub"], default="local")
    args = parser.parse_args()
    uri = configure_tracking(args.target)
    with mlflow.start_run(run_name="connectivity-check") as run:
        mlflow.log_param("project", "23l-0829_A2_Ops")
        mlflow.log_metric("connectivity_check", 1.0)
        mlflow.set_tags({"purpose": "connectivity-only", "git_commit": git_commit()})
        mlflow.log_text("Connectivity check succeeded. No model was trained.\n", "check.txt")
        print(f"Tracking URI: {uri}\nRun ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
