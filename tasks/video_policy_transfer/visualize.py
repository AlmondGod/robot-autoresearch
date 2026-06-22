from __future__ import annotations

from tasks.robocasa_bc5.visualize import main_with_defaults


if __name__ == "__main__":
    main_with_defaults(
        task_name="video_policy_transfer",
        eval_script="tasks/video_policy_transfer/eval.py",
        inference="tasks.video_policy_transfer.inference",
        manifest="data/robocasa5/manifest.json",
        split="data/autorobobench/video_policy_transfer_splits.json",
        max_steps="260",
        commit_steps="16",
    )
