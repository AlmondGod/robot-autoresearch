from __future__ import annotations

from tasks.robocasa_bc5.visualize import main_with_defaults


if __name__ == "__main__":
    main_with_defaults(
        task_name="robocasa_choose_measuring_cup_language",
        eval_script="tasks/robocasa_choose_measuring_cup_language/eval.py",
        inference="tasks.robocasa_choose_measuring_cup_language.inference",
        manifest="data/autorobobench/robocasa_choose_measuring_cup_language_manifest.json",
        split="data/autorobobench/robocasa_choose_measuring_cup_language_splits.json",
        max_steps="900",
        commit_steps="8",
    )
