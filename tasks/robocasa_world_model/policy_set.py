from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def discover_policy_runs(runs_root: str | Path = "runs/autorobobench/robocasa_bc5") -> list[dict[str, Any]]:
    """Discover real RoboCasa BC-5 policy checkpoints with existing real eval JSONs."""
    root = Path(runs_root)
    policies = []
    for eval_path in sorted(root.glob("*/eval_10_per_task_local.json")):
        payload = json.loads(eval_path.read_text())
        checkpoint = Path(str(payload.get("checkpoint", "")))
        if not checkpoint.exists():
            checkpoint = eval_path.parent / "policy_best.pt"
        if not checkpoint.exists():
            continue
        name = eval_path.parent.name
        policies.append(
            {
                "name": name,
                "checkpoint": str(checkpoint),
                "inference": str(payload.get("inference", "tasks.robocasa_bc5.inference")),
                "real_eval_json": str(eval_path),
                "real_success_rate": float(payload.get("success_rate", 0.0)),
                "ood": _is_ood_name(name),
            }
        )
    return policies


def _is_ood_name(name: str) -> bool:
    lowered = name.lower()
    # Treat policies outside the base 5-minute single-backbone runs as OOD
    # policy-family examples for the correlation set.
    return any(token in lowered for token in ("ensemble", "recede", "full_history", "history", "mini", "trajectory"))

