from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_bc5.eval import main as robocasa_bc5_eval_main


def main() -> None:
    out_path = _arg_value("--out")
    if not any(arg == "--inference" or arg.startswith("--inference=") for arg in sys.argv):
        sys.argv.extend(["--inference", "tasks.robocasa_recap_offline.inference"])
    robocasa_bc5_eval_main()
    if out_path:
        out = Path(out_path)
        if out.exists():
            payload = json.loads(out.read_text())
            payload["track"] = "robocasa_recap_offline"
            out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _arg_value(flag: str) -> str | None:
    for idx, arg in enumerate(sys.argv):
        if arg == flag and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


if __name__ == "__main__":
    main()
