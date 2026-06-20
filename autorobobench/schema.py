from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrackSpec:
    id: str
    name: str
    weight: float
    phase: int
    primary_metric: str
    starter_metric: float
    reference_metric: float
    higher_is_better: bool
    score_weights: dict[str, float]
    required_result_keys: tuple[str, ...]
    allowed_edit_globs: tuple[str, ...]
    immutable_globs: tuple[str, ...]
    task_spec: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, root: Path | None = None) -> "TrackSpec":
        payload = _expand_task_spec(payload, root=root)
        score_weights = {
            str(key): float(value)
            for key, value in payload.get("score_weights", {}).items()
        }
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            weight=float(payload["weight"]),
            phase=int(payload.get("phase", 1)),
            primary_metric=str(payload["primary_metric"]),
            starter_metric=float(payload.get("starter_metric", 0.0)),
            reference_metric=float(payload.get("reference_metric", 1.0)),
            higher_is_better=bool(payload.get("higher_is_better", True)),
            score_weights=score_weights,
            required_result_keys=tuple(str(item) for item in payload.get("required_result_keys", score_weights.keys())),
            allowed_edit_globs=tuple(str(item) for item in payload.get("allowed_edit_globs", ())),
            immutable_globs=tuple(str(item) for item in payload.get("immutable_globs", ())),
            task_spec=str(payload.get("task_spec", "")),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class SuiteSpec:
    version: str
    tracks: tuple[TrackSpec, ...]
    total_points: float = 100.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, root: Path | None = None) -> "SuiteSpec":
        tracks = tuple(TrackSpec.from_dict(item, root=root) for item in payload["tracks"])
        return cls(
            version=str(payload["version"]),
            tracks=tracks,
            total_points=float(payload.get("total_points", sum(track.weight for track in tracks))),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "SuiteSpec":
        import json

        resolved = Path(path).resolve()
        return cls.from_dict(json.loads(resolved.read_text()), root=resolved.parent.parent)


def _expand_task_spec(payload: dict[str, Any], *, root: Path | None) -> dict[str, Any]:
    task_spec = payload.get("task_spec")
    if not task_spec:
        return payload
    if root is None:
        root = Path(".")

    import json

    task_path = _resolve_task_path(str(task_spec), root)
    task = json.loads(task_path.read_text())
    benchmark = dict(task.get("benchmark", {}))
    metrics = task.get("metrics", {})

    expanded: dict[str, Any] = {
        "id": task.get("id", payload.get("id")),
        "name": task.get("name", payload.get("name", task.get("id", payload.get("id")))),
        "phase": benchmark.get("phase", task.get("phase", payload.get("phase", 1))),
        "primary_metric": benchmark.get("primary_metric", metrics.get("primary", payload.get("primary_metric"))),
        "starter_metric": benchmark.get("starter_metric", task.get("starter_metric", payload.get("starter_metric", 0.0))),
        "reference_metric": benchmark.get("reference_metric", task.get("reference_metric", payload.get("reference_metric", 1.0))),
        "higher_is_better": benchmark.get("higher_is_better", payload.get("higher_is_better", True)),
        "score_weights": benchmark.get("score_weights", payload.get("score_weights", {})),
        "required_result_keys": benchmark.get("required_result_keys", payload.get("required_result_keys", ())),
        "allowed_edit_globs": benchmark.get("allowed_edit_globs", payload.get("allowed_edit_globs", ())),
        "immutable_globs": benchmark.get("immutable_globs", payload.get("immutable_globs", ())),
        "notes": benchmark.get("notes", payload.get("notes", "")),
        "task_spec": str(task_spec),
    }
    expanded.update(payload)
    expanded["task_spec"] = str(task_spec)
    return expanded


def _resolve_task_path(task_spec: str, root: Path) -> Path:
    task = Path(task_spec)
    candidates = [
        task if task.is_absolute() else root / task,
        task if task.is_absolute() else root.parent / task,
        task if task.is_absolute() else Path.cwd() / task,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()
