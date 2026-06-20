from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from autorobobench.robocasa_runtime import ensure_robocasa_runtime


ensure_robocasa_runtime()


FROZEN_MANIFEST = "data/autorobobench/robocasa_stand_mixer_peak_manifest.json"
FROZEN_SPLIT = "data/autorobobench/robocasa_stand_mixer_peak_splits.json"
FROZEN_VIDEO_POOL = "data/autorobobench/robocasa_stand_mixer_peak_video_pool.json"


def main() -> None:
    _default("--manifest", FROZEN_MANIFEST)
    _default("--split", FROZEN_SPLIT)
    _default("--video-pool", FROZEN_VIDEO_POOL)
    _default("--out-dir", "runs/autorobobench/robocasa_stand_mixer_peak/bc_base")
    _default("--train-episodes-per-task", "80")
    _default("--val-episodes-per-task", "10")
    _default("--chunk-horizon", "16")
    _default("--frame-stride", "1")
    _default("--steps", "800")
    _default("--batch-size", "128")
    _default("--width", "256")
    _default("--dropout", "0.03")
    _default("--image-noise", "0.004")
    _default("--proprio-noise", "0.004")
    _default("--chunk-decay", "0.82")
    _default("--action-smooth", "0.0005")
    _default("--progress-scale", "750")
    _default("--eval-commit-steps", "8")

    from train.train_autorobobench_robocasa_bc5 import main as train_main

    train_main()


def _default(flag: str, value: str | None) -> None:
    if flag in sys.argv:
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


if __name__ == "__main__":
    main()
