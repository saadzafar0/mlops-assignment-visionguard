"""Run the Task 1.4 CSV exercise in an isolated nested repository with local storage."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    work = ROOT / "labs/work/dvc-toy"
    repo = work / "repo"
    remote = work / "remote"
    if work.exists():
        raise SystemExit(
            f"Refusing to overwrite {work}; use the existing evidence or move it first."
        )
    repo.mkdir(parents=True)
    remote.mkdir()
    transcript = []

    def run(*command: str) -> str:
        shown = " ".join(command).replace(str(ROOT), "<project>")
        print(f"$ {shown}", flush=True)
        result = subprocess.run(command, cwd=repo, capture_output=True, text=True)
        transcript.extend([f"$ {shown}", result.stdout, result.stderr])
        print(result.stdout, end="", flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout.strip()

    dvc = (sys.executable, "-m", "dvc")
    run("git", "init", "--initial-branch=main")
    run("git", "config", "user.name", "Saad Zafar (23l-0829)")
    run("git", "config", "user.email", "saadzafar0@users.noreply.github.com")
    run(*dvc, "init")
    run(*dvc, "config", "core.analytics", "false")
    run(*dvc, "remote", "add", "-d", "local", str(remote))
    csv = repo / "sample.csv"
    version_one = "image,label\nphoto.jpg,real\n"
    csv.write_text(version_one)
    run(*dvc, "add", "sample.csv")
    run("git", "add", ".dvc", ".dvcignore", ".gitignore", "sample.csv.dvc")
    run("git", "commit", "-m", "Track first CSV version with DVC")
    first = run("git", "rev-parse", "HEAD")
    run(*dvc, "push")
    csv.write_text(version_one + "generated.png,ai\n")
    run(*dvc, "add", "sample.csv")
    run("git", "add", "sample.csv.dvc")
    run("git", "commit", "-m", "Track second CSV version with DVC")
    second = run("git", "rev-parse", "HEAD")
    run(*dvc, "push")
    run("git", "checkout", first)
    run(*dvc, "checkout")
    if csv.read_text() != version_one:
        raise RuntimeError("Version-one restoration did not match expected content.")
    transcript.extend(["Restored first version:", csv.read_text()])
    csv.unlink()
    # Remove only this toy's cache as well: pull must really retrieve remote data.
    shutil.rmtree(repo / ".dvc/cache")
    run(*dvc, "pull")
    if csv.read_text() != version_one:
        raise RuntimeError("Remote retrieval did not match expected content.")
    run("git", "checkout", "main")
    # The cache was cleared: the second version must also be retrieved remotely.
    run(*dvc, "pull")
    run("git", "log", "--oneline", "--all")
    evidence = ROOT / "reports/generated/dvc_toy"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "transcript.txt").write_text("\n".join(transcript))
    (evidence / "commits.txt").write_text(f"version_one={first}\nversion_two={second}\n")
    print(f"\nEvidence: {evidence}\nBoth versions remain in {repo} (local Git only).")


if __name__ == "__main__":
    main()
