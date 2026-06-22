from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    payload = visualize_run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main_with_defaults(**defaults: str) -> None:
    argv = list(sys.argv)
    present = {_arg_key(arg) for arg in argv[1:] if arg.startswith("--")}
    for key, value in reversed(list(defaults.items())):
        flag = f"--{key.replace('_', '-')}"
        if flag not in present and value:
            argv[1:1] = [flag, value]
    old = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old


def visualize_run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "visualize"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_path = _resolve_json(args.eval_json, run_dir, "eval*.json")
    history_path = _resolve_json(args.history_json, run_dir, "history.json")
    metrics_path = _resolve_json(args.metrics_json, run_dir, "metrics.json")

    eval_payload = _read_json(eval_path)
    history = _read_json(history_path, default=[])
    metrics = _read_json(metrics_path, default={})

    render_payload = None
    if args.render:
        render_payload = _run_render_eval(args, run_dir, out_dir)

    summary = _summary(
        task_name=str(args.task_name),
        run_dir=run_dir,
        eval_path=eval_path,
        history_path=history_path,
        metrics_path=metrics_path,
        eval_payload=eval_payload,
        history=history if isinstance(history, list) else history.get("history", []),
        metrics=metrics if isinstance(metrics, dict) else {},
        render_payload=render_payload,
    )
    svg_path = out_dir / "eval_summary.svg"
    json_path = out_dir / "eval_summary.json"
    svg_path.write_text(_summary_svg(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["outputs"] = {"summary_json": str(json_path), "summary_svg": str(svg_path)}
    if render_payload:
        summary["outputs"]["render_json"] = str(render_payload.get("out", ""))
        summary["outputs"]["render_dir"] = str(render_payload.get("render_dir", ""))
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize BC-style RoboCasa train/eval results.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--task-name", default="robocasa_bc5")
    parser.add_argument("--eval-json", default="")
    parser.add_argument("--history-json", default="")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--render", action="store_true", help="Run a tiny render eval and save videos under visualize/render.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--eval-script", default="tasks/robocasa_bc5/eval.py")
    parser.add_argument("--inference", default="tasks.robocasa_bc5.inference")
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--eval-episodes-per-task", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--commit-steps", type=int, default=16)
    parser.add_argument("--render-episodes-per-task", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _run_render_eval(args: argparse.Namespace, run_dir: Path, out_dir: Path) -> dict[str, str]:
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "policy_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"cannot render without checkpoint: {checkpoint}")
    render_dir = out_dir / "render"
    render_out = out_dir / "render_eval.json"
    cmd = [
        sys.executable,
        str(args.eval_script),
        "--manifest",
        str(args.manifest),
        "--split",
        str(args.split),
        "--inference",
        str(args.inference),
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(render_out),
        "--eval-episodes-per-task",
        str(int(args.eval_episodes_per_task)),
        "--max-steps",
        str(int(args.max_steps)),
        "--commit-steps",
        str(int(args.commit_steps)),
        "--render-dir",
        str(render_dir),
        "--render-episodes-per-task",
        str(int(args.render_episodes_per_task)),
        "--device",
        str(args.device),
    ]
    subprocess.run(cmd, check=True)
    return {"cmd": " ".join(cmd), "out": str(render_out), "render_dir": str(render_dir)}


def _summary(
    *,
    task_name: str,
    run_dir: Path,
    eval_path: Path | None,
    history_path: Path | None,
    metrics_path: Path | None,
    eval_payload: Any,
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
    render_payload: dict[str, str] | None,
) -> dict[str, Any]:
    per_task = eval_payload.get("per_task", {}) if isinstance(eval_payload, dict) else {}
    details = eval_payload.get("details", []) if isinstance(eval_payload, dict) else []
    success_rate = float(eval_payload.get("success_rate", 0.0)) if isinstance(eval_payload, dict) else 0.0
    episodes = int(eval_payload.get("episodes", 0)) if isinstance(eval_payload, dict) else 0
    successes = int(eval_payload.get("successes", round(success_rate * episodes))) if isinstance(eval_payload, dict) else 0
    task_rows = [
        {
            "task": str(name),
            "success_rate": float(row.get("success_rate", 0.0)),
            "successes": int(row.get("successes", 0)),
            "episodes": int(row.get("episodes", 0)),
        }
        for name, row in sorted(per_task.items())
        if isinstance(row, dict)
    ]
    failure_examples = [
        {
            "task_alias": row.get("task_alias"),
            "episode_id": row.get("episode_id"),
            "steps": row.get("steps"),
        }
        for row in details
        if isinstance(row, dict) and not bool(row.get("success", False))
    ][:8]
    return {
        "task": task_name,
        "run_dir": str(run_dir),
        "eval_json": "" if eval_path is None else str(eval_path),
        "history_json": "" if history_path is None else str(history_path),
        "metrics_json": "" if metrics_path is None else str(metrics_path),
        "success_rate": success_rate,
        "successes": successes,
        "episodes": episodes,
        "per_task": task_rows,
        "failure_examples": failure_examples,
        "last_history": history[-5:] if history else [],
        "best_val_loss": _best_history_value(history, "val_loss"),
        "best_val_action_mse_normalized": _best_history_value(history, "val_action_mse_normalized"),
        "steps_completed": int(metrics.get("steps_completed", 0)) if isinstance(metrics, dict) else 0,
        "render": render_payload or {},
    }


def _summary_svg(summary: dict[str, Any]) -> str:
    rows = summary.get("per_task", [])
    width = 980
    height = max(300, 170 + 42 * max(1, len(rows)))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="36" font-family="Arial" font-size="24" font-weight="700" fill="#111">{_esc(summary["task"])} eval</text>',
        f'<text x="24" y="66" font-family="Arial" font-size="15" fill="#333">success {summary["successes"]}/{summary["episodes"]} = {summary["success_rate"]:.3f}</text>',
        f'<text x="24" y="92" font-family="Arial" font-size="12" fill="#666">{_esc(summary.get("eval_json", ""))}</text>',
    ]
    x0, bar_x, y0 = 24, 310, 132
    bar_w = 520
    if rows:
        for idx, row in enumerate(rows):
            y = y0 + idx * 42
            rate = max(0.0, min(1.0, float(row["success_rate"])))
            lines.extend(
                [
                    f'<text x="{x0}" y="{y}" font-family="Arial" font-size="14" fill="#111">{_esc(row["task"])}</text>',
                    f'<rect x="{bar_x}" y="{y - 15}" width="{bar_w}" height="18" rx="2" fill="#edf2f7"/>',
                    f'<rect x="{bar_x}" y="{y - 15}" width="{bar_w * rate:.1f}" height="18" rx="2" fill="#2563eb"/>',
                    f'<text x="{bar_x + bar_w + 14}" y="{y}" font-family="Arial" font-size="13" fill="#111">{row["successes"]}/{row["episodes"]} ({rate:.2f})</text>',
                ]
            )
    else:
        lines.append(f'<text x="{x0}" y="{y0}" font-family="Arial" font-size="14" fill="#666">No per-task eval rows found.</text>')
    failures = summary.get("failure_examples", [])
    if failures:
        y = height - 78
        lines.append(f'<text x="24" y="{y}" font-family="Arial" font-size="13" font-weight="700" fill="#111">First failures</text>')
        text = "; ".join(f"{row.get('task_alias')} ep{row.get('episode_id')} {row.get('steps')} steps" for row in failures[:4])
        lines.append(f'<text x="24" y="{y + 22}" font-family="Arial" font-size="12" fill="#555">{_esc(text)}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _resolve_json(value: str, run_dir: Path, pattern: str) -> Path | None:
    if value:
        path = Path(value)
        return path if path.exists() else None
    exact = run_dir / pattern
    if "*" not in pattern and exact.exists():
        return exact
    candidates = sorted(run_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _read_json(path: Path | None, default: Any = None) -> Any:
    if path is None or not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())


def _best_history_value(history: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in history if isinstance(row, dict) and row.get(key) is not None]
    return min(values) if values else None


def _arg_key(arg: str) -> str:
    return arg.split("=", 1)[0]


def _esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
