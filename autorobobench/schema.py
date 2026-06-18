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
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrackSpec":
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
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class SuiteSpec:
    version: str
    tracks: tuple[TrackSpec, ...]
    total_points: float = 100.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SuiteSpec":
        return cls(
            version=str(payload["version"]),
            tracks=tuple(TrackSpec.from_dict(item) for item in payload["tracks"]),
            total_points=float(payload.get("total_points", 100.0)),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "SuiteSpec":
        import json

        return cls.from_dict(json.loads(Path(path).read_text()))
