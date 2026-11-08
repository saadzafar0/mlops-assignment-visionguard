"""Publish the assignment's isolated red/green two-version GitHub Actions lab."""

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = """name: 23l-0829_A2_Ops CI toy
on:
  push:
  workflow_dispatch:
permissions:
  contents: read
jobs:
  tests:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python: ['3.11', '3.12']
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false
      - uses: actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0 # v7.1.0
        with:
          python-version: ${{ matrix.python }}
      - run: python -m unittest -v
"""
TEST = """import unittest

from arithmetic import add


class AdditionTest(unittest.TestCase):
    def test_two_plus_three(self):
        self.assertEqual(add(2, 3), 5)


if __name__ == '__main__':
    unittest.main()
"""


def command(args, cwd=ROOT, check=True):
    result = subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)
    return result.stdout.strip()


def wait_for_run(repository, commit):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        values = json.loads(
            command(
                [
                    "gh",
                    "run",
                    "list",
                    "--repo",
                    repository,
                    "--commit",
                    commit,
                    "--limit",
                    "1",
                    "--json",
                    "databaseId,status,conclusion,url,headSha",
                ]
            )
        )
        if values and values[0]["status"] == "completed":
            return values[0]
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for the toy workflow at {commit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="saadzafar0")
    parser.add_argument("--repo", default="23l-0829_A2_Ops-ci-toy")
    args = parser.parse_args()
    repository = f"{args.owner}/{args.repo}"
    work = ROOT / "labs/work/ci-toy"
    if work.exists():
        raise SystemExit(f"Refusing to overwrite the existing lab at {work}")
    existing = subprocess.run(
        ["gh", "repo", "view", repository, "--json", "name"],
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0:
        raise SystemExit(f"Refusing to modify the existing repository {repository}")
    output = ROOT / "reports/generated/ci_toy"
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True)
    (work / ".github/workflows").mkdir(parents=True)
    (work / ".github/workflows/ci.yml").write_text(WORKFLOW)
    (work / ".gitignore").write_text("__pycache__/\n")
    (work / "README.md").write_text(
        "# 23l-0829_A2_Ops — CI toy\n\n"
        "Separate Task 6.6 lab: an intentional addition bug produces a red run on "
        "Python 3.11 and 3.12; fixing it produces a green run. No VisionGuard models, "
        "datasets, credentials, or real application failures are involved.\n"
    )
    (work / "test_arithmetic.py").write_text(TEST)
    (work / "arithmetic.py").write_text("def add(left, right):\n    return left - right\n")
    command(["git", "init", "--initial-branch=main"], work)
    for key in ("user.name", "user.email"):
        value = command(["git", "config", key])
        command(["git", "config", key, value], work)
    command(["git", "add", "."], work)
    command(["git", "commit", "-m", "lab: intentionally fail the addition test"], work)
    red_commit = command(["git", "rev-parse", "HEAD"], work)
    command(
        [
            "gh",
            "repo",
            "create",
            repository,
            "--public",
            "--source",
            str(work),
            "--remote",
            "origin",
            "--push",
        ]
    )
    print(f"Published isolated lab: https://github.com/{repository}", flush=True)
    red = wait_for_run(repository, red_commit)
    assert red["conclusion"] == "failure", red
    (output / "red_run.json").write_text(json.dumps(red, indent=2) + "\n")
    (output / "red_log.txt").write_text(
        command(
            [
                "gh",
                "run",
                "view",
                str(red["databaseId"]),
                "--repo",
                repository,
                "--log-failed",
            ]
        )
        + "\n"
    )
    print(f"Intentional red run: {red['url']}", flush=True)
    (work / "arithmetic.py").write_text("def add(left, right):\n    return left + right\n")
    command(["git", "add", "arithmetic.py"], work)
    command(["git", "commit", "-m", "fix: add operands and pass both Python versions"], work)
    green_commit = command(["git", "rev-parse", "HEAD"], work)
    command(["git", "push", "origin", "main"], work)
    green = wait_for_run(repository, green_commit)
    assert green["conclusion"] == "success", green
    details = json.loads(
        command(
            [
                "gh",
                "run",
                "view",
                str(green["databaseId"]),
                "--repo",
                repository,
                "--json",
                "jobs",
            ]
        )
    )
    assert len(details["jobs"]) == 2
    assert all(job["conclusion"] == "success" for job in details["jobs"])
    (output / "green_run.json").write_text(json.dumps(green, indent=2) + "\n")
    (output / "green_jobs.json").write_text(json.dumps(details, indent=2) + "\n")
    print(f"Both Python versions passed: {green['url']}", flush=True)


if __name__ == "__main__":
    main()
