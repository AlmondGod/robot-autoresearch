from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from autorobobench.robocasa_runtime import ensure_robocasa_runtime


ensure_robocasa_runtime()

from train.train_robocasa_tiny_evaluator import main  # noqa: E402


def _default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _default("--out-dir", "runs/autorobobench/world_model_evaluator/tiny_open_drawer")
    _default("--task-alias", "OpenDrawer")
    _default("--frame-stride", "4")
    _default("--train-demos-per-task", "80")
    _default("--steps", "3000")
    _default("--latent-dim", "256")
    _default("--width", "512")
    main()
