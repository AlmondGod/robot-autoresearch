from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RoboCasa world-model posttraining artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--eval-json", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "visualize"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = _resolve(args.metrics_json, run_dir, "metrics.json")
    if metrics_path is None:
        metrics_path = _resolve(args.metrics_json, run_dir, "train_metrics.json")
    eval_path = _resolve(args.eval_json, run_dir, "eval*.json")
    metrics = _read_json(metrics_path)
    eval_payload = _read_json(eval_path)
    summary = _summary(run_dir, metrics_path, eval_path, metrics, eval_payload)
    (out_dir / "wm_posttraining_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "wm_posttraining_summary.svg").write_text(_summary_svg(summary), encoding="utf-8")
    summary["outputs"] = {
        "summary_json": str(out_dir / "wm_posttraining_summary.json"),
        "summary_svg": str(out_dir / "wm_posttraining_summary.svg"),
    }
    (out_dir / "wm_posttraining_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _summary(
    run_dir: Path,
    metrics_path: Path | None,
    eval_path: Path | None,
    metrics: dict[str, Any],
    eval_payload: dict[str, Any],
) -> dict[str, Any]:
    history = metrics.get("history", []) if metrics else []
    return {
        "task": "robocasa_world_model_posttraining",
        "run_dir": str(run_dir),
        "metrics_json": "" if metrics_path is None else str(metrics_path),
        "eval_json": "" if eval_path is None else str(eval_path),
        "success_rate": eval_payload.get("success_rate"),
        "successes": eval_payload.get("successes"),
        "episodes": eval_payload.get("episodes"),
        "best_val_policy_improvement_score": metrics.get("best_val_policy_improvement_score"),
        "val_wm_objective": _metric(metrics, history, "val_wm_objective", "best_val_wm_objective"),
        "val_action_mse_normalized": _metric(
            metrics,
            history,
            "val_action_mse_normalized",
            "best_val_action_mse_normalized",
        ),
        "val_init_anchor_mse": _metric(metrics, history, "val_init_anchor_mse", "best_val_init_anchor_mse"),
        "steps_completed": metrics.get("steps_completed"),
        "history_tail": history[-8:],
    }


def _summary_svg(summary: dict[str, Any]) -> str:
    bars = [
        ("real eval success", summary.get("success_rate")),
        ("WM objective", summary.get("val_wm_objective")),
        ("policy improvement", summary.get("best_val_policy_improvement_score")),
    ]
    width, height = 900, 300
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#111">World-model posttraining</text>',
        f'<text x="24" y="68" font-family="Arial" font-size="12" fill="#666">{_esc(summary.get("metrics_json", ""))}</text>',
    ]
    for idx, (name, value) in enumerate(bars):
        y = 118 + idx * 52
        score = _as_float(value)
        lines.append(f'<text x="24" y="{y}" font-family="Arial" font-size="14" fill="#111">{_esc(name)}</text>')
        lines.append(f'<rect x="230" y="{y - 16}" width="500" height="18" fill="#e5e7eb"/>')
        lines.append(f'<rect x="230" y="{y - 16}" width="{500 * score:.1f}" height="18" fill="#0891b2"/>')
        lines.append(f'<text x="750" y="{y}" font-family="Arial" font-size="13" fill="#333">{_fmt(value)}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _last_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in reversed(rows):
        if isinstance(row, dict) and key in row:
            return row[key]
    return None


def _metric(metrics: dict[str, Any], history: list[dict[str, Any]], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
        value = _last_value(history, key)
        if value is not None:
            return value
    return None


def _resolve(value: str, run_dir: Path, pattern: str) -> Path | None:
    if value:
        path = Path(value)
        return path if path.exists() else None
    exact = run_dir / pattern
    if "*" not in pattern and exact.exists():
        return exact
    candidates = sorted(run_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _as_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
