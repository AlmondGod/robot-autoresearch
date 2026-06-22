from __future__ import annotations

from tasks.robocasa_bc5.visualize import main_with_defaults


if __name__ == "__main__":
    main_with_defaults(
        task_name="robocasa_faucet_peak",
        eval_script="tasks/robocasa_faucet_peak/eval.py",
        inference="tasks.robocasa_faucet_peak.inference",
        manifest="data/autorobobench/robocasa_faucet_peak_manifest.json",
        split="data/autorobobench/robocasa_faucet_peak_splits.json",
        max_steps="750",
        commit_steps="8",
    )
