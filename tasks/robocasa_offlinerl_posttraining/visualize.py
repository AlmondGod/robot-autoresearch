from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RoboCasa offline-RL posttraining artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--eval-json", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "visualize"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = _resolve(args.metrics_json, run_dir, "metrics.json")
    eval_path = _resolve(args.eval_json, run_dir, "eval*.json")
    metrics = _read_json(metrics_path)
    eval_payload = _read_json(eval_path)
    summary = _summary(run_dir, metrics_path, eval_path, metrics, eval_payload)
    (out_dir / "offlinerl_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "offlinerl_summary.svg").write_text(_summary_svg(summary), encoding="utf-8")
    summary["outputs"] = {
        "summary_json": str(out_dir / "offlinerl_summary.json"),
        "summary_svg": str(out_dir / "offlinerl_summary.svg"),
    }
    (out_dir / "offlinerl_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _summary(
    run_dir: Path,
    metrics_path: Path | None,
    eval_path: Path | None,
    metrics: dict[str, Any],
    eval_payload: dict[str, Any],
) -> dict[str, Any]:
    source_counts = metrics.get("source_counts", {}) if metrics else {}
    bad_weight = float(metrics.get("bad_sample_weight", 0.0) or 0.0)
    correction_weight = float(metrics.get("correction_weight", 0.0) or 0.0)
    return {
        "task": "robocasa_offlinerl_posttraining",
        "run_dir": str(run_dir),
        "metrics_json": "" if metrics_path is None else str(metrics_path),
        "eval_json": "" if eval_path is None else str(eval_path),
        "success_rate": eval_payload.get("success_rate"),
        "successes": eval_payload.get("successes"),
        "episodes": eval_payload.get("episodes"),
        "source_counts": source_counts,
        "assigned_advantages": {"demo": 1.0, "bad_rollout": -1.0, "correction": 1.0},
        "sample_weights": {
            "demo": 1.0,
            "bad_rollout": bad_weight,
            "correction": correction_weight,
        },
        "bad_action_noise": metrics.get("bad_action_noise"),
        "correction_fraction": metrics.get("correction_fraction"),
        "experience_multiplier": metrics.get("experience_multiplier"),
        "best_val_action_mse_normalized": metrics.get("best_val_action_mse_normalized"),
        "final_val_action_mse_normalized": metrics.get("final_val_action_mse_normalized"),
        "history_tail": metrics.get("history", [])[-5:] if metrics else [],
    }


def _summary_svg(summary: dict[str, Any]) -> str:
    counts = summary.get("source_counts", {})
    weights = summary.get("sample_weights", {})
    adv = summary.get("assigned_advantages", {})
    rows = ["demo", "bad_rollout", "correction"]
    max_count = max([int(counts.get(row, 0) or 0) for row in rows] + [1])
    width, height = 960, 360
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#111">Offline-RL posttraining</text>',
        f'<text x="24" y="68" font-family="Arial" font-size="14" fill="#333">success={_fmt(summary.get("success_rate"))} val_mse={_fmt(summary.get("best_val_action_mse_normalized"))}</text>',
        f'<text x="24" y="94" font-family="Arial" font-size="12" fill="#666">{_esc(summary.get("metrics_json", ""))}</text>',
    ]
    for idx, name in enumerate(rows):
        y = 140 + idx * 58
        count = int(counts.get(name, 0) or 0)
        bar = 420 * count / max_count
        color = "#16a34a" if adv.get(name, 0.0) > 0 else "#dc2626"
        lines.append(f'<text x="24" y="{y}" font-family="Arial" font-size="14" fill="#111">{_esc(name)}</text>')
        lines.append(f'<rect x="190" y="{y - 16}" width="420" height="20" fill="#e5e7eb"/>')
        lines.append(f'<rect x="190" y="{y - 16}" width="{bar:.1f}" height="20" fill="{color}"/>')
        lines.append(f'<text x="630" y="{y}" font-family="Arial" font-size="13" fill="#333">n={count} advantage={_fmt(adv.get(name))} weight={_fmt(weights.get(name))}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


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


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
