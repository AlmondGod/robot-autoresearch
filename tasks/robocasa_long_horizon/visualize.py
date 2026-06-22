from __future__ import annotations

from tasks.robocasa_bc5.visualize import main_with_defaults


if __name__ == "__main__":
    main_with_defaults(
        task_name="robocasa_long_horizon",
        eval_script="tasks/robocasa_long_horizon/eval.py",
        inference="tasks.robocasa_long_horizon.inference",
        manifest="data/autorobobench/robocasa_long_horizon_manifest.json",
        split="data/autorobobench/robocasa_long_horizon_splits.json",
        max_steps="750",
        commit_steps="8",
    )
