from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ARCHIVE = Path("runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl")
DEFAULT_MANIFEST = Path("data/robocasa5/manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the World Model Evaluator task assets.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--task-alias", default="OpenDrawer")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    payload = verify_assets(Path(args.manifest), Path(args.archive), str(args.task_alias))
    print(json.dumps(payload, indent=2, sort_keys=True))


def verify_assets(manifest_path: Path, archive_path: Path, task_alias: str) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)

    manifest = json.loads(manifest_path.read_text())
    tasks = {task["alias"]: task for task in manifest["tasks"]}
    if task_alias not in tasks:
        raise ValueError(f"{task_alias!r} not in {manifest_path}")
    dataset_root = Path(tasks[task_alias]["dataset_path"])
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    rows = [json.loads(line) for line in archive_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{archive_path} has no candidate rows")

    trace_count = 0
    missing: list[str] = []
    for row in rows:
        eval_path = Path(row.get("eval_path", ""))
        if eval_path and not eval_path.is_absolute():
            eval_path = Path.cwd() / eval_path
        if not eval_path.exists():
            missing.append(str(eval_path))
            continue
        eval_payload = json.loads(eval_path.read_text())
        for detail in eval_payload.get("details", []):
            trace_path = Path(detail.get("trace_path", ""))
            if trace_path and not trace_path.is_absolute():
                trace_path = Path.cwd() / trace_path
            if trace_path.exists():
                trace_count += 1
            else:
                missing.append(str(trace_path))

    if missing:
        raise FileNotFoundError(f"missing {len(missing)} trace/eval files, first={missing[0]}")

    return {
        "task": "world_model_evaluator",
        "manifest": str(manifest_path),
        "archive": str(archive_path),
        "task_alias": task_alias,
        "dataset_root": str(dataset_root),
        "candidate_rows": len(rows),
        "trace_files": trace_count,
        "verified": True,
    }


if __name__ == "__main__":
    main()
