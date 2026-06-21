# robocasa_long_horizon Instructions

Write outputs under `runs/autorobobench/robocasa_long_horizon/<run>/`. Do not
edit eval files or split files for scored runs.

## Task

- Optimize one policy for sequential tasks:
  `PickPlaceCounterToStove`, `TurnOffStove`, `PickPlaceCounterToCabinet`.
- Metric: final success plus subgoal progress.
- Default eval: 10 episodes/task, max 750 steps, commit 8.

## Train

```bash
python3 tasks/robocasa_long_horizon/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_long_horizon_splits.json \
  --out-dir runs/autorobobench/robocasa_long_horizon/<run> \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_long_horizon/eval.py \
  --checkpoint runs/autorobobench/robocasa_long_horizon/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_long_horizon/<run>/eval_10_per_task.json \
  --device cuda
```
