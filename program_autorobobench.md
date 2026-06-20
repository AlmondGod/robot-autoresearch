# AutoroboBench Agent Instructions

You are running an AutoroboBench RoboCasa research loop.

Goal:

```text
Improve the robot-learning system under the fixed benchmark budget.
```

Primary score comes from evaluator reruns, not self-reported metrics.

## Rules

- Do not edit files matched by the active track's `immutable_globs` in
  `configs/autorobobench_v0.json`.
- Do not read hidden eval files, canary files, or answer files.
- Do not use network access unless the active track explicitly allows fixed
  external-data corpora.
- Keep an experiment ledger with change, commit, budget, metrics, decision, and
  notes.
- Commit only accepted improvements.

## Tracks

### RoboCasa BC-5

Improve the policy on the five seed RoboCasa tasks.

Executable files:

- split: `data/autorobobench/robocasa_bc5_splits.json`
- setup: `tasks/robocasa_bc5/setup.py`
- train: `tasks/robocasa_bc5/train.py`
- inference: `tasks/robocasa_bc5/inference.py`
- eval: `tasks/robocasa_bc5/eval.py`

### Long-Horizon RoboCasa

Improve compositional manipulation and recovery on the sequential seed tasks.

### Video Data to Policy Transfer

Use scarce paired-action demos plus action-free RoboCasa videos to improve the
closed-loop policy.

## Smoke Test

```bash
python -m autorobobench.cli describe --config configs/autorobobench_v0.json
python -m autorobobench.cli score \
  --config configs/autorobobench_v0.json \
  --results examples/autorobobench_v0_results.json
```
