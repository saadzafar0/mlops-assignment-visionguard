"""Freeze minimal serving dependency sets from the project's tested installed versions."""

import importlib.metadata as metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def freeze(roots):
    pending = [(name, "") for name in roots]
    seen = set()
    distributions = {}
    while pending:
        name, extra = pending.pop()
        key = canonicalize_name(name)
        if (key, extra) in seen:
            continue
        seen.add((key, extra))
        distribution = metadata.distribution(name)
        distributions[key] = distribution
        for value in distribution.requires or []:
            requirement = Requirement(value)
            if requirement.marker is None or requirement.marker.evaluate({"extra": extra}):
                pending.append((requirement.name, ""))
                pending.extend((requirement.name, item) for item in requirement.extras)
    return (
        "\n".join(
            sorted(f"{item.metadata['Name']}=={item.version}" for item in distributions.values())
        )
        + "\n"
    )


def main():
    target = ROOT / "requirements"
    target.mkdir(exist_ok=True)
    web = [
        "Flask",
        "gunicorn",
        "Pillow",
        "requests",
        "python-dotenv",
        "PyYAML",
        "SQLAlchemy",
        "psycopg",
        "psycopg-binary",
        "tzdata",
    ]
    model = ["tensorflow-cpu", "keras", "numpy", "python-dotenv", "PyYAML"]
    for name, roots in [("web", web), ("model", model), ("app", web + model)]:
        (target / f"{name}.txt").write_text(freeze(roots))
    print("Wrote exact web, model-export and complete local-app runtime requirements.")


if __name__ == "__main__":
    main()
