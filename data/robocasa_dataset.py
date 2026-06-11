from __future__ import annotations

import json
from pathlib import Path


DEFAULT_VIEWS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
]


DEFAULT_TASKS = [
    {
        "task_id": 0,
        "alias": "OpenDrawer",
        "robocasa_task": "OpenDrawer",
        "description": "Open a kitchen drawer.",
    },
    {
        "task_id": 1,
        "alias": "CloseDrawer",
        "robocasa_task": "CloseDrawer",
        "description": "Close a kitchen drawer.",
    },
    {
        "task_id": 2,
        "alias": "PickPlaceCounterToStove",
        "robocasa_task": "PickPlaceCounterToStove",
        "description": "Pick an object from the counter and place it on the stove.",
    },
    {
        "task_id": 3,
        "alias": "TurnOffStove",
        "robocasa_task": "TurnOffStove",
        "description": "Turn a stove knob off.",
    },
    {
        "task_id": 4,
        "alias": "PickPlaceCounterToCabinet",
        "robocasa_task": "PickPlaceCounterToCabinet",
        "description": "Pick an object from the counter and place it into a cabinet region.",
    },
]


def build_manifest(
    out_dir: Path,
    *,
    split: str = "pretrain",
    source: str = "human",
    policy_demos_per_task: int = 50,
    views: list[str] | None = None,
    verify_exists: bool = False,
    task_map: dict[str, str] | None = None,
) -> dict:
    views = views or list(DEFAULT_VIEWS)
    tasks = []
    for task in DEFAULT_TASKS:
        entry = dict(task)
        if task_map and entry["alias"] in task_map:
            entry["robocasa_task"] = task_map[entry["alias"]]
        meta = _get_ds_meta(entry["robocasa_task"], split=split, source=source)
        dataset_path = Path(meta["path"])
        stats = inspect_lerobot_dataset(dataset_path, expected_views=views)
        registered_demos = _registered_demo_count(meta["filter_key"])
        if verify_exists and not stats["exists"]:
            raise FileNotFoundError(f"missing dataset for {entry['robocasa_task']}: {dataset_path}")
        entry.update(
            {
                "dataset_path": str(dataset_path),
                "split": split,
                "source": source,
                "horizon": meta["horizon"],
                "filter_key": meta["filter_key"],
                "exists": stats["exists"],
                "registered_demos": registered_demos,
                "available_demos": stats["num_episodes"],
                "selected_demos": min(policy_demos_per_task, registered_demos or stats["num_episodes"] or policy_demos_per_task),
                "available_views": stats["available_views"],
            }
        )
        tasks.append(entry)

    manifest = {
        "benchmark": "robocasa5",
        "suite": "robocasa",
        "split": split,
        "source": source,
        "policy_demos_per_task": policy_demos_per_task,
        "views": views,
        "action_dim": 7,
        "tasks": tasks,
        "task_count": len(tasks),
        "total_registered_demos": sum(int(task["registered_demos"]) for task in tasks),
        "total_available_demos": sum(int(task["available_demos"]) for task in tasks),
        "total_selected_demos": sum(int(task["selected_demos"]) for task in tasks),
        "notes": [
            "Built from the RoboCasa registry via robocasa.utils.dataset_registry_utils.get_ds_meta.",
            "Datasets are expected in LeRobot format under the local RoboCasa dataset base path.",
            "selected_demos may be smaller than available_demos for policy training.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def inspect_lerobot_dataset(dataset_path: Path, expected_views: list[str] | None = None) -> dict:
    expected_views = expected_views or []
    meta_dir = dataset_path / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"

    exists = dataset_path.exists()
    num_episodes = 0
    features = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            num_episodes = int(info.get("total_episodes") or info.get("num_episodes") or 0)
            features = info.get("features") or {}
        except (ValueError, TypeError):
            num_episodes = 0
            features = {}
    if num_episodes == 0 and episodes_path.exists():
        num_episodes = sum(1 for line in episodes_path.read_text().splitlines() if line.strip())

    available_views = []
    for view in expected_views:
        if f"observation.images.{view}" in features:
            available_views.append(view)

    if not available_views:
        for key in sorted(features):
            if key.startswith("observation.images."):
                available_views.append(key.removeprefix("observation.images."))

    return {
        "exists": exists,
        "num_episodes": num_episodes,
        "available_views": available_views,
    }


def _get_ds_meta(task: str, *, split: str, source: str) -> dict:
    try:
        from robocasa.utils.dataset_registry_utils import get_ds_meta
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RoboCasa is not installed. Install/clone RoboCasa so "
            "`robocasa.utils.dataset_registry_utils.get_ds_meta` is importable."
        ) from exc

    meta = get_ds_meta(task=task, split=split, source=source)
    if meta is None:
        raise ValueError(f"no dataset metadata registered for task={task} split={split} source={source}")
    return meta


def _registered_demo_count(filter_key: str) -> int:
    prefix = filter_key.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return 0
