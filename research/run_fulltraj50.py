from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "runs/libero/fulltraj50"
ARCHIVE = OUT_ROOT / "archive_v2.jsonl"
TRAIN = ROOT / "eval/train_full_trajectory_bc.py"


BASE = {
    "manifest": "data/libero_easy1_task0/manifest.json",
    "horizon": 160,
    "episodes_per_task": 5,
    "steps": 2500,
    "batch_size": 32,
    "lr": 3e-4,
    "device": "cuda",
    "train_demos": 40,
    "width": 256,
    "dropout": 0.0,
    "loss": "mse",
    "temporal_decay": 1.0,
    "front_weight": 1.0,
    "tail_weight": 1.0,
    "image_noise": 0.0,
    "proprio_noise": 0.0,
    "action_noise": 0.0,
    "action_smooth": 0.0,
    "weight_decay": 0.0,
}


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    completed = _completed()
    for idx, spec in enumerate(_experiments(), start=1):
        if idx in completed:
            continue
        run_dir = OUT_ROOT / f"exp_{idx:03d}"
        cfg = {**BASE, **{key: value for key, value in spec.items() if key != "name"}}
        cfg["out_dir"] = str(run_dir)
        cmd = _cmd(cfg)
        record = {
            "idx": idx,
            "commit": _commit(),
            "change": spec["name"],
            "name": spec["name"],
            "config": cfg,
            "cmd": cmd,
        }
        print(json.dumps({"starting": idx, "name": spec["name"]}), flush=True)
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            record["metrics"] = metrics
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
        _append(record)
        if record["status"] == "ok":
            metrics = record["metrics"]
            print(
                json.dumps(
                    {
                        "finished": idx,
                        "success_rate": metrics.get("success_rate"),
                        "val_action_mse": metrics.get("val_action_mse"),
                    }
                ),
                flush=True,
            )


def _cmd(cfg: dict) -> list[str]:
    args = [sys.executable, str(TRAIN)]
    for key, value in cfg.items():
        if key in {"name"}:
            continue
        flag = "--" + key.replace("_", "-")
        args.extend([flag, str(value)])
    return args


def _append(record: dict) -> None:
    with ARCHIVE.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _commit() -> str:
    fallback = os.environ.get("ROBOT_AUTORESEARCH_COMMIT")
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return fallback or "unknown"


def _completed() -> set[int]:
    if not ARCHIVE.exists():
        return set()
    out = set()
    for line in ARCHIVE.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("status") == "ok":
            out.add(int(payload["idx"]))
    return out


def _experiments() -> list[dict]:
    specs = [
        {"name": "baseline rerun seed0"},
        {"name": "huber loss", "loss": "huber"},
        {"name": "l1 loss", "loss": "l1"},
        {"name": "early action weight 2x", "front_weight": 2.0},
        {"name": "early action weight 4x", "front_weight": 4.0},
        {"name": "tail action weight 2x", "tail_weight": 2.0},
        {"name": "tail action weight 4x", "tail_weight": 4.0},
        {"name": "temporal decay 0.995", "temporal_decay": 0.995},
        {"name": "temporal decay 0.99", "temporal_decay": 0.99},
        {"name": "temporal decay 0.98", "temporal_decay": 0.98},
        {"name": "best decay 0.98 + smooth 1e-4", "temporal_decay": 0.98, "action_smooth": 1e-4},
        {"name": "best decay 0.98 + smooth 1e-3", "temporal_decay": 0.98, "action_smooth": 1e-3},
        {"name": "best decay 0.98 + smooth 1e-2", "temporal_decay": 0.98, "action_smooth": 1e-2},
        {"name": "best decay 0.98 + front 2x", "temporal_decay": 0.98, "front_weight": 2.0},
        {"name": "best decay 0.98 + front 4x", "temporal_decay": 0.98, "front_weight": 4.0},
        {"name": "best decay 0.98 + front 6x", "temporal_decay": 0.98, "front_weight": 6.0},
        {"name": "best decay 0.98 + tail 2x", "temporal_decay": 0.98, "tail_weight": 2.0},
        {"name": "best decay 0.98 + tail 4x", "temporal_decay": 0.98, "tail_weight": 4.0},
        {"name": "best decay 0.98 + front 4x tail 2x", "temporal_decay": 0.98, "front_weight": 4.0, "tail_weight": 2.0},
        {"name": "best decay 0.98 + front 4x tail 4x", "temporal_decay": 0.98, "front_weight": 4.0, "tail_weight": 4.0},
        {"name": "best decay 0.98 + width 384", "temporal_decay": 0.98, "width": 384},
        {"name": "best decay 0.98 + width 512", "temporal_decay": 0.98, "width": 512},
        {"name": "best decay 0.98 + width 768", "temporal_decay": 0.98, "width": 768},
        {"name": "best decay 0.98 + dropout 0.05", "temporal_decay": 0.98, "dropout": 0.05},
        {"name": "best decay 0.98 + weight decay 1e-4", "temporal_decay": 0.98, "weight_decay": 1e-4},
        {"name": "best decay 0.98 + image noise 0.01", "temporal_decay": 0.98, "image_noise": 0.01},
        {"name": "best decay 0.98 + proprio noise 0.005", "temporal_decay": 0.98, "proprio_noise": 0.005},
        {"name": "best decay 0.98 + image proprio noise", "temporal_decay": 0.98, "image_noise": 0.01, "proprio_noise": 0.005},
        {"name": "best decay 0.98 + action noise 0.005", "temporal_decay": 0.98, "action_noise": 0.005},
        {"name": "best decay 0.98 + lr 1e-4", "temporal_decay": 0.98, "lr": 1e-4},
        {"name": "best decay 0.98 + lr 2e-4", "temporal_decay": 0.98, "lr": 2e-4},
        {"name": "best decay 0.98 + lr 5e-4", "temporal_decay": 0.98, "lr": 5e-4},
        {"name": "best decay 0.98 + batch 16", "temporal_decay": 0.98, "batch_size": 16},
        {"name": "best decay 0.98 + batch 64", "temporal_decay": 0.98, "batch_size": 64},
        {"name": "best decay 0.98 + train demos 45", "temporal_decay": 0.98, "train_demos": 45},
        {"name": "best decay 0.98 + train demos 49", "temporal_decay": 0.98, "train_demos": 49},
        {"name": "best decay 0.98 + seed 1 split", "temporal_decay": 0.98, "seed": 1},
        {"name": "best decay 0.98 + seed 2 split", "temporal_decay": 0.98, "seed": 2},
        {"name": "best decay 0.98 + seed 3 split", "temporal_decay": 0.98, "seed": 3},
        {"name": "best decay 0.98 + huber", "temporal_decay": 0.98, "loss": "huber"},
        {"name": "best decay 0.98 + l1", "temporal_decay": 0.98, "loss": "l1"},
        {"name": "near best decay 0.975", "temporal_decay": 0.975},
        {"name": "near best decay 0.985", "temporal_decay": 0.985},
        {"name": "near best decay 0.97", "temporal_decay": 0.97},
        {"name": "best decay 0.98 + steps 5000", "temporal_decay": 0.98, "steps": 5000},
        {"name": "best decay 0.98 + steps 5000 lr 1e-4", "temporal_decay": 0.98, "steps": 5000, "lr": 1e-4},
        {"name": "best decay 0.98 + steps 5000 width 512", "temporal_decay": 0.98, "steps": 5000, "width": 512},
        {"name": "best decay 0.98 + robust combo", "temporal_decay": 0.98, "image_noise": 0.01, "proprio_noise": 0.005, "action_noise": 0.005, "dropout": 0.05},
        {"name": "best decay 0.98 + front 4x smooth 1e-3", "temporal_decay": 0.98, "front_weight": 4.0, "action_smooth": 1e-3},
        {"name": "best decay 0.98 + confirm 10 eval", "temporal_decay": 0.98, "episodes_per_task": 10},
    ]
    return specs[:50]


if __name__ == "__main__":
    main()
