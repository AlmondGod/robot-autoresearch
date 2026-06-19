from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from autorobobench.robocasa_runtime import ensure_robocasa_runtime


ensure_robocasa_runtime()

from train.train_robocasa_mini_video_world_model import main  # noqa: E402


def _default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


def _flag_default(flag: str) -> None:
    if flag not in sys.argv:
        sys.argv.append(flag)


def _repeated_default(flag: str, values: tuple[str, ...]) -> None:
    if flag not in sys.argv:
        for value in values:
            sys.argv.extend([flag, value])


if __name__ == "__main__":
    _default("--out-dir", "runs/autorobobench/world_model_evaluator/mini_video_world_model")
    _default("--frame-stride", "4")
    _default("--train-demos-per-task", "80")
    _default("--steps", "1000")
    _default("--batch-size", "128")
    _default("--latent-dim", "512")
    _default("--width", "512")
    _default("--dynamics-kind", "mlp")
    _flag_default("--condition-on-robocasa-task-index")
    _repeated_default("--val-episode-id", ("87", "92", "93", "94", "98", "100", "101"))
    main()
