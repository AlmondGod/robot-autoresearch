from __future__ import annotations

import math
from typing import Any

from autorobobench.schema import SuiteSpec, TrackSpec


def normalized_progress(
    value: float,
    starter: float,
    reference: float,
    *,
    higher_is_better: bool = True,
) -> float:
    if math.isclose(starter, reference):
        return 0.0
    raw = (value - starter) / (reference - starter)
    if not higher_is_better:
        raw = (starter - value) / (starter - reference)
    return max(0.0, min(1.0, raw))


def score_suite(spec: SuiteSpec, results: dict[str, Any]) -> dict[str, Any]:
    result_tracks = results.get("tracks", results)
    track_scores = []
    total = 0.0
    max_total = 0.0
    for track in spec.tracks:
        max_total += track.weight
        payload = result_tracks.get(track.id)
        if payload is None:
            track_scores.append(_missing_track(track))
            continue
        scored = score_track(track, payload)
        total += scored["points"]
        track_scores.append(scored)
    return {
        "version": spec.version,
        "score": total,
        "max_score": max_total,
        "normalized_score": total / max_total if max_total > 0 else 0.0,
        "tracks": track_scores,
    }


def score_track(track: TrackSpec, result: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in track.required_result_keys if key not in result]
    components: dict[str, float] = {}
    weighted_sum = 0.0
    weight_sum = 0.0
    for key, weight in track.score_weights.items():
        value = _component_value(track, result, key)
        components[key] = value
        weighted_sum += float(weight) * value
        weight_sum += float(weight)
    task_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    task_score = max(0.0, min(1.0, task_score))
    integrity = float(result.get("reproducibility_integrity", 1.0))
    if missing:
        integrity = min(integrity, 0.5)
    points = track.weight * task_score
    return {
        "track_id": track.id,
        "name": track.name,
        "phase": track.phase,
        "points": points,
        "max_points": track.weight,
        "task_score": task_score,
        "components": components,
        "missing_required_keys": missing,
        "primary_metric": track.primary_metric,
        "primary_metric_value": result.get(track.primary_metric),
        "primary_metric_progress": _primary_progress(track, result),
        "reproducibility_integrity": integrity,
    }


def _component_value(track: TrackSpec, result: dict[str, Any], key: str) -> float:
    if key == "normalized_primary_progress":
        return _primary_progress(track, result)
    value = result.get(key, 0.0)
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _primary_progress(track: TrackSpec, result: dict[str, Any]) -> float:
    value = result.get(track.primary_metric)
    if value is None:
        return 0.0
    return normalized_progress(
        float(value),
        track.starter_metric,
        track.reference_metric,
        higher_is_better=track.higher_is_better,
    )


def _missing_track(track: TrackSpec) -> dict[str, Any]:
    return {
        "track_id": track.id,
        "name": track.name,
        "phase": track.phase,
        "points": 0.0,
        "max_points": track.weight,
        "task_score": 0.0,
        "components": {},
        "missing_required_keys": list(track.required_result_keys),
        "primary_metric": track.primary_metric,
        "primary_metric_value": None,
        "primary_metric_progress": 0.0,
        "reproducibility_integrity": 0.0,
    }
