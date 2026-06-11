from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DemoRef:
    task_id: int
    task_name: str
    dataset_path: str
    demo_key: str
    split: str
    paired: bool


IMAGE_KEYS = [
    "agentview_rgb",
    "robot0_agentview_left_image",
    "robot0_agentview_image",
    "image",
    "rgb",
]
PROPRIO_KEYS = ["robot_states", "proprio", "states", "joint_states"]
ACTION_KEYS = ["actions", "action"]
WRIST_IMAGE_KEYS = [
    "eye_in_hand_rgb",
    "robot0_eye_in_hand_image",
    "wrist_rgb",
]
INSTRUCTION_VOCAB_SIZE = 512
INSTRUCTION_LENGTH = 16


def find_libero_dataset_dir(root: Path, suite: str) -> Path:
    candidates = [
        root / "datasets" / suite,
        root / "libero" / "datasets" / suite,
        root / "LIBERO" / "datasets" / suite,
        root / suite,
        root,
    ]
    for candidate in candidates:
        if candidate.exists() and list(candidate.glob("*.hdf5")):
            return candidate
    raise FileNotFoundError(
        f"Could not find {suite} HDF5 files. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def build_manifest(
    libero_root: Path,
    out_dir: Path,
    suite: str = "libero_object",
    task_count: int = 5,
    video_task_count: int | None = None,
    paired_demos_per_task: int = 10,
    video_repeat_factor: int = 1,
    seed: int = 0,
    task_names: list[str] | None = None,
) -> dict[str, Any]:
    h5py = _h5py()
    dataset_dir = find_libero_dataset_dir(libero_root, suite)
    video_task_count = task_count if video_task_count is None else video_task_count
    all_files = sorted(dataset_dir.glob("*.hdf5"))
    if task_names:
        by_name = {path.stem.removesuffix("_demo"): path for path in all_files}
        missing = [name for name in task_names if name not in by_name]
        if missing:
            raise ValueError(f"missing {suite} task demos: {missing}")
        files = [by_name[name] for name in task_names]
        video_task_count = len(files)
        task_count = min(task_count, len(files))
    else:
        files = all_files[:video_task_count]
    if len(files) < task_count:
        raise ValueError(f"expected at least {task_count} {suite} task files in {dataset_dir}")
    if len(files) < video_task_count:
        raise ValueError(f"expected at least {video_task_count} {suite} task files in {dataset_dir}")

    rng = np.random.default_rng(seed)
    refs: list[DemoRef] = []
    tasks = []
    for task_id, path in enumerate(files):
        task_name = path.stem
        with h5py.File(path, "r") as handle:
            demo_keys = sorted(_demo_group(handle).keys())
        if not demo_keys:
            raise ValueError(f"no demos found in {path}")
        order = np.asarray(demo_keys, dtype=object)
        rng.shuffle(order)
        train_cut = max(1, int(0.8 * len(order)))
        val_cut = max(train_cut + 1, int(0.9 * len(order))) if len(order) > 2 else len(order)
        split_for_key = {}
        for idx, demo_key in enumerate(order):
            if idx < train_cut:
                split = "train"
            elif idx < val_cut:
                split = "val"
            else:
                split = "test"
            split_for_key[str(demo_key)] = split
        by_split = {"train": [], "val": [], "test": []}
        for demo_key in order:
            by_split[split_for_key[str(demo_key)]].append(str(demo_key))
        paired_keys = set()
        if task_id < task_count:
            train_pair_count = max(1, int(0.8 * paired_demos_per_task))
            val_pair_count = max(1, int(0.1 * paired_demos_per_task))
            test_pair_count = max(1, paired_demos_per_task - train_pair_count - val_pair_count)
            paired_keys = set(
                by_split["train"][:train_pair_count]
                + by_split["val"][:val_pair_count]
                + by_split["test"][:test_pair_count]
            )
        for demo_key in demo_keys:
            refs.append(
                DemoRef(
                    task_id=task_id,
                    task_name=task_name,
                    dataset_path=str(path),
                    demo_key=demo_key,
                    split=split_for_key[demo_key],
                    paired=demo_key in paired_keys,
                )
            )
        if task_id < task_count:
            tasks.append({"task_id": task_id, "task_name": task_name, "dataset_path": str(path)})

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": suite,
        "task_count": len(tasks),
        "video_task_count": len(files),
        "paired_demos_per_task": paired_demos_per_task,
        "video_repeat_factor": video_repeat_factor,
        "tasks": tasks,
        "video_tasks": [
            {"task_id": task_id, "task_name": path.stem, "dataset_path": str(path)}
            for task_id, path in enumerate(files)
        ],
        "demos": [asdict(ref) for ref in refs],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def materialize_shards(
    manifest_path: Path,
    out_dir: Path,
    image_size: int = 64,
    max_transitions_per_demo: int = 120,
    history: int = 4,
    action_horizon: int = 4,
    video_repeat_factor: int | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    video_repeat_factor = int(video_repeat_factor or manifest.get("video_repeat_factor", 1))
    if video_repeat_factor < 1:
        raise ValueError("video_repeat_factor must be >= 1")
    rows = []
    paired_rows = []
    for ref in manifest["demos"]:
        frames, wrist_frames, proprio, actions = load_demo_arrays(
            Path(ref["dataset_path"]),
            ref["demo_key"],
            image_size=image_size,
            max_transitions=max_transitions_per_demo,
        )
        if len(frames) < 2:
            continue
        row = {
            "task_id": int(ref["task_id"]),
            "split": ref["split"],
            "frames": frames[:-1],
            "wrist_frames": wrist_frames[:-1],
            "next_frames": frames[1:],
        }
        rows.append(row)
        if ref["paired"] and actions is not None:
            paired_rows.extend(
                _paired_rows_for_demo(
                    task_id=int(ref["task_id"]),
                    task_name=ref["task_name"],
                    split=ref["split"],
                    frames=frames,
                    wrist_frames=wrist_frames,
                    proprio=proprio if proprio is not None else np.zeros((len(frames), 1), dtype=np.float32),
                    actions=actions,
                    history=history,
                    action_horizon=action_horizon,
                )
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = manifest.get("name") or out_dir.name
    video_path = out_dir / f"{prefix}_video.npz"
    paired_path = out_dir / f"{prefix}_paired.npz"
    _save_transition_rows(video_path, rows, include_actions=False, repeat_factor=video_repeat_factor)
    _save_transition_rows(paired_path, paired_rows, include_actions=True)
    raw_video_demos = len(rows)
    summary = {
        "video_path": str(video_path),
        "paired_path": str(paired_path),
        "raw_video_demos": raw_video_demos,
        "video_repeat_factor": video_repeat_factor,
        "video_demos": raw_video_demos * video_repeat_factor,
        "paired_samples": len(paired_rows),
        "paired_demos": sum(1 for ref in manifest["demos"] if ref["paired"]),
        "image_size": image_size,
        "history": history,
        "action_horizon": action_horizon,
    }
    (out_dir / "shards.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def load_video_npz(path: Path, split: str = "train") -> dict[str, np.ndarray]:
    data = np.load(path)
    if "sample_index" in data.files:
        virtual_split = data["virtual_split"]
        mask = virtual_split == split
        sample_index = data["sample_index"][mask]
        out: dict[str, Any] = {
            "frames": IndexedArray(data["frames"], sample_index),
            "wrist_frames": IndexedArray(data["wrist_frames"], sample_index),
            "next_frames": IndexedArray(data["next_frames"], sample_index),
            "task_id": IndexedArray(data["task_id"], sample_index),
            "split": virtual_split[mask],
            "sample_index": sample_index,
            "raw_size": np.asarray(len(data["frames"]), dtype=np.int64),
            "video_repeat_factor": data["video_repeat_factor"],
        }
        return out
    mask = data["split"] == split
    masked = {"frames", "wrist_frames", "next_frames", "task_id", "split"}
    return {key: data[key][mask] if key in masked else data[key] for key in data.files}


def load_paired_npz(path: Path, split: str = "train") -> dict[str, np.ndarray]:
    data = np.load(path)
    mask = data["split"] == split
    masked = {"frames", "wrist_frames", "next_frames", "proprio", "actions", "task_id", "split", "instruction_tokens"}
    return {key: data[key][mask] if key in masked else data[key] for key in data.files}


def load_demo_arrays(
    dataset_path: Path,
    demo_key: str,
    image_size: int,
    max_transitions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    h5py = _h5py()
    with h5py.File(dataset_path, "r") as handle:
        demo = _demo_group(handle)[demo_key]
        image_arr = _read_first_available(demo, IMAGE_KEYS)
        wrist_arr = _read_first_available(demo, WRIST_IMAGE_KEYS, required=False)
        proprio = _read_first_available(demo, PROPRIO_KEYS, required=False)
        actions = _read_first_available(demo, ACTION_KEYS, required=False)
    frames = _resize_uint8_images(image_arr[: max_transitions + 1], image_size)
    if wrist_arr is None:
        wrist_frames = frames.copy()
    else:
        wrist_frames = _resize_uint8_images(wrist_arr[: max_transitions + 1], image_size)
    proprio_arr = None if proprio is None else np.asarray(proprio[: max_transitions + 1], dtype=np.float32)
    action_arr = None if actions is None else np.asarray(actions[:max_transitions], dtype=np.float32)
    return frames, wrist_frames, proprio_arr, action_arr


class IndexedArray:
    def __init__(self, base: np.ndarray, index: np.ndarray):
        self.base = base
        self.index = np.asarray(index, dtype=np.int64)
        self.shape = (len(self.index), *base.shape[1:])
        self.dtype = base.dtype

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: Any) -> np.ndarray:
        return self.base[self.index[item]]

    def max(self) -> Any:
        return self.base[self.index].max()


def _save_transition_rows(path: Path, rows: list[dict[str, Any]], include_actions: bool, repeat_factor: int = 1) -> None:
    if not rows:
        raise ValueError(f"no rows to save for {path}")
    if include_actions:
        frames = np.stack([row["frames"] for row in rows], axis=0)
        next_frames = np.stack([row["next_frames"][0] for row in rows], axis=0)
        task_id = np.asarray([row["task_id"] for row in rows], dtype=np.int64)
        split = np.asarray([row["split"] for row in rows])
    else:
        frames = np.concatenate([row["frames"] for row in rows], axis=0)
        next_frames = np.concatenate([row["next_frames"] for row in rows], axis=0)
        task_id = np.concatenate([np.full(len(row["frames"]), row["task_id"], dtype=np.int64) for row in rows])
        split = np.concatenate([np.full(len(row["frames"]), row["split"]) for row in rows])
    payload: dict[str, np.ndarray] = {
        "frames": frames,
        "next_frames": next_frames,
        "task_id": task_id,
        "split": split,
    }
    if "wrist_frames" in rows[0]:
        if include_actions:
            payload["wrist_frames"] = np.stack([row["wrist_frames"] for row in rows], axis=0)
        else:
            payload["wrist_frames"] = np.concatenate([row["wrist_frames"] for row in rows], axis=0)
    if not include_actions and repeat_factor > 1:
        base_count = len(payload["frames"])
        payload["sample_index"] = np.tile(np.arange(base_count, dtype=np.int64), repeat_factor)
        payload["virtual_split"] = np.tile(payload["split"], repeat_factor)
        payload["video_repeat_factor"] = np.asarray(repeat_factor, dtype=np.int64)
    if include_actions:
        payload["proprio"] = np.stack([row["proprio"] for row in rows], axis=0)
        payload["actions"] = np.stack([row["actions"] for row in rows], axis=0)
        payload["instruction_tokens"] = np.stack([row["instruction_tokens"][0] for row in rows], axis=0)
        payload["proprio_mean"] = payload["proprio"].reshape(-1, payload["proprio"].shape[-1]).mean(axis=0).astype(np.float32)
        payload["proprio_std"] = (payload["proprio"].reshape(-1, payload["proprio"].shape[-1]).std(axis=0) + 1e-6).astype(np.float32)
        payload["action_mean"] = payload["actions"].reshape(-1, payload["actions"].shape[-1]).mean(axis=0).astype(np.float32)
        payload["action_std"] = (payload["actions"].reshape(-1, payload["actions"].shape[-1]).std(axis=0) + 1e-6).astype(np.float32)
    np.savez_compressed(path, **payload)


def _paired_rows_for_demo(
    task_id: int,
    task_name: str,
    split: str,
    frames: np.ndarray,
    wrist_frames: np.ndarray,
    proprio: np.ndarray,
    actions: np.ndarray,
    history: int,
    action_horizon: int,
) -> list[dict[str, Any]]:
    rows = []
    transition_count = min(len(frames) - 1, len(actions))
    instruction = tokenize_instruction(task_name)
    for idx in range(transition_count):
        hist_indices = np.clip(np.arange(idx - history + 1, idx + 1), 0, transition_count)
        action_indices = np.clip(np.arange(idx, idx + action_horizon), 0, transition_count - 1)
        rows.append(
            {
                "task_id": task_id,
                "split": split,
                "frames": frames[hist_indices],
                "wrist_frames": wrist_frames[hist_indices],
                "next_frames": frames[min(idx + 1, len(frames) - 1)][None],
                "proprio": proprio[hist_indices],
                "actions": actions[action_indices],
                "instruction_tokens": instruction[None],
            }
        )
    return rows


def tokenize_instruction(text: str, length: int = INSTRUCTION_LENGTH, vocab_size: int = INSTRUCTION_VOCAB_SIZE) -> np.ndarray:
    words = text.replace("_demo", "").replace("_", " ").split()
    tokens = np.zeros(length, dtype=np.int64)
    for idx, word in enumerate(words[:length]):
        tokens[idx] = 1 + (sum((i + 1) * ord(ch) for i, ch in enumerate(word)) % (vocab_size - 1))
    return tokens


def _demo_group(handle: Any) -> Any:
    if "data" in handle:
        return handle["data"]
    return handle


def _read_first_available(group: Any, names: list[str], required: bool = True) -> np.ndarray | None:
    for name in names:
        found = _find_dataset(group, name)
        if found is not None:
            return np.asarray(found)
    if required:
        raise KeyError(f"none of {names} found in demo keys {list(group.keys())}")
    return None


def _find_dataset(group: Any, suffix: str) -> Any | None:
    if suffix in group:
        return group[suffix]
    for key, value in group.items():
        if hasattr(value, "keys"):
            found = _find_dataset(value, suffix)
            if found is not None:
                return found
        elif key.endswith(suffix):
            return value
    return None


def _resize_uint8_images(images: np.ndarray, image_size: int) -> np.ndarray:
    from PIL import Image

    arr = np.asarray(images)
    if arr.ndim != 4:
        raise ValueError(f"expected images as NHWC/NCHW, got shape {arr.shape}")
    if arr.shape[1] in {1, 3}:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    out = []
    for image in arr:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil = Image.fromarray(image[..., :3])
        pil = pil.resize((image_size, image_size), Image.Resampling.BILINEAR)
        out.append(np.asarray(pil, dtype=np.uint8))
    return np.stack(out, axis=0)


def _h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install LIBERO data dependencies with: pip install -e '.[libero]'") from exc
    return h5py
