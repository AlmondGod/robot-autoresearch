from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--split", default="pretrain", choices=["pretrain", "target", "real"])
    parser.add_argument("--source", default="human", choices=["human", "mimicgen", "mg", "mg_5x5", "mg_5x1"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Answer yes to the RoboCasa download confirmation prompt.")
    args = parser.parse_args()

    if importlib.util.find_spec("robocasa") is None:
        raise ModuleNotFoundError(
            "RoboCasa is not installed in this environment. "
            "Install/clone RoboCasa first, then rerun this downloader."
        )

    cmd = [sys.executable, "-m", "robocasa.scripts.download_datasets"]
    if args.all:
        cmd.append("--all")
    else:
        tasks = args.tasks or _tasks_from_manifest(Path(args.manifest))
        if tasks:
            cmd.extend(["--tasks", *tasks])
        cmd.extend(["--split", args.split, "--source", args.source])
    if args.overwrite:
        cmd.append("--overwrite")

    subprocess.run(cmd, check=True, input="y\n" if args.yes else None, text=True)


def _tasks_from_manifest(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    return [task["robocasa_task"] for task in payload["tasks"]]


if __name__ == "__main__":
    main()
