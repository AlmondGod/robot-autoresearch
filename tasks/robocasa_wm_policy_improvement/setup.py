from __future__ import annotations

import importlib.util
import json
from pathlib import Path

DEFAULT_MANIFEST = Path("data/autorobobench/robocasa_faucet_peak_manifest.json")
DEFAULT_SPLIT = Path("data/autorobobench/robocasa_faucet_peak_splits.json")
DEFAULT_POLICY_CHECKPOINT = Path(
    "data/autorobobench/pretrained_policies/robocasa_faucet_direct_bc_all_data_5min_seed0.pt"
)


def main() -> None:
    checks = {
        "task": "robocasa_wm_policy_improvement",
        "torch_available": importlib.util.find_spec("torch") is not None,
        "robocasa_bc5_inference_available": importlib.util.find_spec("tasks.robocasa_bc5.inference") is not None,
        "world_model_available": importlib.util.find_spec("tasks.robocasa_world_model.model") is not None,
        "visual_world_model_available": importlib.util.find_spec("tasks.robocasa_visual_world_model.model") is not None,
        "manifest_exists": DEFAULT_MANIFEST.exists(),
        "split_exists": DEFAULT_SPLIT.exists(),
        "default_policy_checkpoint_exists": DEFAULT_POLICY_CHECKPOINT.exists(),
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    missing = [key for key, ok in checks.items() if key != "task" and not ok]
    if missing:
        raise SystemExit(f"missing requirements: {', '.join(missing)}")


if __name__ == "__main__":
    main()
