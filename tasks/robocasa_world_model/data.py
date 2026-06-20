from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "data" / "robocasa5" / "manifest.json"
DEFAULT_SPLIT = ROOT / "data" / "autorobobench" / "robocasa_bc5_splits.json"
DEFAULT_POLICY_SET = ROOT / "data" / "autorobobench" / "robocasa_world_model_policy_set.json"


@dataclass
class TransitionData:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    progress: np.ndarray
    next_progress: np.ndarray
    reward: np.ndarray
    success: np.ndarray
    task_id: np.ndarray
    episode_id: np.ndarray
    frame_idx: np.ndarray

    def __len__(self) -> int:
        return int(self.state.shape[0])


def load_transition_data(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    split_path: str | Path = DEFAULT_SPLIT,
    train_episodes_per_task: int = 20,
    val_episodes_per_task: int = 5,
    task_aliases: set[str] | None = None,
    frame_stride: int = 1,
) -> tuple[TransitionData, TransitionData, list[dict[str, Any]]]:
    manifest = json.loads(Path(manifest_path).read_text())
    split = json.loads(Path(split_path).read_text())
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    aliases = task_aliases or set()
    train_parts = []
    val_parts = []
    summary = []
    for split_task in split["tasks"]:
        alias = str(split_task["alias"])
        if aliases and alias not in aliases:
            continue
        task_id = int(split_task["task_id"])
        dataset_root = Path(manifest_tasks[alias]["dataset_path"])
        train_ids = [int(x) for x in split_task["train_episode_ids"][: int(train_episodes_per_task)]]
        val_ids = [int(x) for x in split_task["val_episode_ids"][: int(val_episodes_per_task)]]
        train_count = _append_episodes(train_parts, dataset_root, train_ids, task_id, int(frame_stride))
        val_count = _append_episodes(val_parts, dataset_root, val_ids, task_id, int(frame_stride))
        summary.append(
            {
                "alias": alias,
                "task_id": task_id,
                "dataset_path": str(dataset_root),
                "train_episode_ids": train_ids,
                "val_episode_ids": val_ids,
                "train_transitions": int(train_count),
                "val_transitions": int(val_count),
            }
        )
    return _concat(train_parts), _concat(val_parts), summary


def _append_episodes(parts: list[dict[str, np.ndarray]], dataset_root: Path, episode_ids: list[int], task_id: int, frame_stride: int) -> int:
    count = 0
    for episode_id in episode_ids:
        part = load_episode_transitions(dataset_root, int(episode_id), int(task_id), frame_stride=max(1, frame_stride))
        if part["state"].shape[0] > 0:
            parts.append(part)
            count += int(part["state"].shape[0])
    return count


def load_episode_transitions(dataset_root: Path, episode_id: int, task_id: int, *, frame_stride: int = 1) -> dict[str, np.ndarray]:
    episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
    frame = pd.read_parquet(episode_path)
    state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    action = episode_actions(dataset_root, episode_id, frame).astype(np.float32)
    n = min(len(state), len(action))
    if n <= 1:
        return _empty_part(state_dim=state.shape[-1] if state.ndim == 2 else 1, action_dim=action.shape[-1] if action.ndim == 2 else 1)
    rows = np.arange(0, n - 1, max(1, frame_stride), dtype=np.int32)
    progress = rows.astype(np.float32) / max(1, n - 1)
    next_progress = (rows + 1).astype(np.float32) / max(1, n - 1)
    reward = _episode_reward(frame, rows, n)
    success = _episode_success(frame, rows, n)
    return {
        "state": state[rows].astype(np.float32),
        "action": action[rows].astype(np.float32),
        "next_state": state[rows + 1].astype(np.float32),
        "progress": progress[:, None].astype(np.float32),
        "next_progress": next_progress[:, None].astype(np.float32),
        "reward": reward[:, None].astype(np.float32),
        "success": success[:, None].astype(np.float32),
        "task_id": np.full((len(rows),), int(task_id), dtype=np.int64),
        "episode_id": np.full((len(rows),), int(episode_id), dtype=np.int32),
        "frame_idx": rows.astype(np.int32),
    }


def _episode_reward(frame: pd.DataFrame, rows: np.ndarray, n: int) -> np.ndarray:
    for key in ("next.reward", "reward"):
        if key in frame:
            values = np.asarray(frame[key].to_numpy(), dtype=np.float32).reshape(-1)
            return values[np.minimum(rows + 1, len(values) - 1)]
    reward = np.zeros((len(rows),), dtype=np.float32)
    if len(reward):
        reward[-1] = 1.0
    return reward


def _episode_success(frame: pd.DataFrame, rows: np.ndarray, n: int) -> np.ndarray:
    for key in ("next.success", "success", "is_success"):
        if key in frame:
            values = np.asarray(frame[key].to_numpy(), dtype=np.float32).reshape(-1)
            return values[np.minimum(rows + 1, len(values) - 1)]
    success = np.zeros((len(rows),), dtype=np.float32)
    if len(success):
        success[-1] = 1.0
    return success


def episode_actions(dataset_root: Path, episode_id: int, frame: pd.DataFrame | None = None) -> np.ndarray:
    if frame is None:
        episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
        frame = pd.read_parquet(episode_path)
    if "action" in frame:
        return np.stack(frame["action"].to_numpy()).astype(np.float32)
    try:
        from autorobobench.robocasa_runtime import ensure_robocasa_runtime

        ensure_robocasa_runtime()
        import robocasa.utils.lerobot_utils as LU

        return LU.get_episode_actions(dataset_root, episode_id).astype(np.float32)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "episode parquet has no action column and RoboCasa is not importable for lerobot_utils fallback"
        ) from exc


def _concat(parts: list[dict[str, np.ndarray]]) -> TransitionData:
    if not parts:
        return TransitionData(
            state=np.zeros((0, 1), dtype=np.float32),
            action=np.zeros((0, 1), dtype=np.float32),
            next_state=np.zeros((0, 1), dtype=np.float32),
            progress=np.zeros((0, 1), dtype=np.float32),
            next_progress=np.zeros((0, 1), dtype=np.float32),
            reward=np.zeros((0, 1), dtype=np.float32),
            success=np.zeros((0, 1), dtype=np.float32),
            task_id=np.zeros((0,), dtype=np.int64),
            episode_id=np.zeros((0,), dtype=np.int32),
            frame_idx=np.zeros((0,), dtype=np.int32),
        )
    return TransitionData(**{key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]})


def _empty_part(state_dim: int, action_dim: int) -> dict[str, np.ndarray]:
    return {
        "state": np.zeros((0, int(state_dim)), dtype=np.float32),
        "action": np.zeros((0, int(action_dim)), dtype=np.float32),
        "next_state": np.zeros((0, int(state_dim)), dtype=np.float32),
        "progress": np.zeros((0, 1), dtype=np.float32),
        "next_progress": np.zeros((0, 1), dtype=np.float32),
        "reward": np.zeros((0, 1), dtype=np.float32),
        "success": np.zeros((0, 1), dtype=np.float32),
        "task_id": np.zeros((0,), dtype=np.int64),
        "episode_id": np.zeros((0,), dtype=np.int32),
        "frame_idx": np.zeros((0,), dtype=np.int32),
    }


def mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    return mean, np.maximum(std, 1e-6).astype(np.float32)


def normalize_data(data: TransitionData, stats: dict[str, np.ndarray]) -> TransitionData:
    return TransitionData(
        state=((data.state - stats["state_mean"]) / stats["state_std"]).astype(np.float32),
        action=((data.action - stats["action_mean"]) / stats["action_std"]).astype(np.float32),
        next_state=((data.next_state - stats["state_mean"]) / stats["state_std"]).astype(np.float32),
        progress=data.progress.astype(np.float32),
        next_progress=data.next_progress.astype(np.float32),
        reward=data.reward.astype(np.float32),
        success=data.success.astype(np.float32),
        task_id=data.task_id.astype(np.int64),
        episode_id=data.episode_id.astype(np.int32),
        frame_idx=data.frame_idx.astype(np.int32),
    )


def make_stats(train: TransitionData) -> dict[str, np.ndarray]:
    state_mean, state_std = mean_std(np.concatenate([train.state, train.next_state], axis=0))
    action_mean, action_std = mean_std(train.action)
    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
