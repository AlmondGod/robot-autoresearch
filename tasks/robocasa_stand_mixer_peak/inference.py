from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from autorobobench.robocasa_runtime import ensure_robocasa_runtime


ensure_robocasa_runtime()

from tasks.robocasa_bc5.inference import act, load_policy  # noqa: E402,F401


__all__ = ["act", "load_policy"]
