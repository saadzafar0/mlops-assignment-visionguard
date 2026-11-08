"""Configure DagsHub's S3-compatible DVC storage; secrets stay in config.local."""

import configparser
import io
import os
import subprocess
import sys

from visionguard.config import PROJECT_ROOT, load_environment


def main() -> None:
    load_environment()
    token = os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not token:
        raise SystemExit("Set DAGSHUB_USER_TOKEN in .env first.")
    if not (PROJECT_ROOT / ".dvc").exists():
        subprocess.run([sys.executable, "-m", "dvc", "init"], cwd=PROJECT_ROOT, check=True)
    config_path = PROJECT_ROOT / ".dvc/config"
    config = configparser.ConfigParser()
    config.read(config_path)
    if not config.has_section("core"):
        config.add_section("core")
    config["core"]["remote"] = "dagshub"
    config["core"]["analytics"] = "false"
    section = 'remote "dagshub"'
    if not config.has_section(section):
        config.add_section(section)
    owner = os.getenv("DAGSHUB_REPO_OWNER", "saadzafar0")
    repo = os.getenv("DAGSHUB_REPO_NAME", "mlops-assignment-visionguard")
    config[section]["url"] = "s3://dvc"
    config[section]["endpointurl"] = f"https://dagshub.com/{owner}/{repo}.s3"
    buffer = io.StringIO()
    config.write(buffer)
    config_path.write_text(buffer.getvalue().rstrip() + "\n")
    local_path = PROJECT_ROOT / ".dvc/config.local"
    local = configparser.ConfigParser()
    local.read(local_path)
    if not local.has_section(section):
        local.add_section(section)
    local[section]["access_key_id"] = token
    local[section]["secret_access_key"] = token
    with local_path.open("w") as stream:
        local.write(stream)
    local_path.chmod(0o600)
    print("DVC remote 'dagshub' configured. Credentials are only in ignored .dvc/config.local.")


if __name__ == "__main__":
    main()
