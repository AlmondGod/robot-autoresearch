from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.robocasa_dataset import DEFAULT_VIEWS, build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/robocasa5/manifest.json")
    parser.add_argument(
        "--task-map",
        default="",
        help="Optional JSON file mapping aliases to local RoboCasa task names.",
    )
    parser.add_argument("--split", default="pretrain", choices=["pretrain", "target", "real"])
    parser.add_argument("--source", default="human", choices=["human", "mg", "mg_5x5", "mg_5x1"])
    parser.add_argument("--policy-demos-per-task", type=int, default=50)
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--verify-exists", action="store_true")
    args = parser.parse_args()

    task_map = json.loads(Path(args.task_map).read_text()) if args.task_map else None
    out = Path(args.out)
    manifest = build_manifest(
        out.parent,
        split=args.split,
        source=args.source,
        policy_demos_per_task=args.policy_demos_per_task,
        views=args.views,
        verify_exists=args.verify_exists,
        task_map=task_map,
    )
    print(json.dumps({k: manifest[k] for k in ["split", "source", "task_count", "total_available_demos", "total_selected_demos"]}, indent=2))
    print(out)


if __name__ == "__main__":
    main()

