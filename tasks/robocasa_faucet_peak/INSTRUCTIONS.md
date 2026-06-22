# robocasa_faucet_peak Instructions

Keep scored runs under 300 seconds. Write outputs under
`runs/autorobobench/robocasa_faucet_peak/<run>/`. Do not edit eval files or
split files for scored runs.

## Task

- Optimize one policy for `TurnOnSinkFaucet`.
- Metric: single-task reliability. Report success out of 100.
- Data: task-specific trajectories are allowed. Generic video-only pool is
  allowed. Test-time demo replay is not allowed for learned-policy claims.
- Current learned base: `robocasa_faucet_direct_bc_all_data_5min_seed0`,
  6/10 success, reported as 60/100 normalized.

## Train

```bash
python3 tasks/robocasa_bc5/train.py \
  --manifest data/autorobobench/robocasa_faucet_peak_manifest.json \
  --split data/autorobobench/robocasa_faucet_peak_splits.json \
  --out-dir runs/autorobobench/robocasa_faucet_peak/<run> \
  --policy-kind bc \
  --chunk-horizon 32 \
  --progress-conditioning \
  --progress-scale 750 \
  --eval-commit-steps 8 \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_bc5/eval_parallel.py \
  --manifest data/autorobobench/robocasa_faucet_peak_manifest.json \
  --split data/autorobobench/robocasa_faucet_peak_splits.json \
  --inference tasks.robocasa_bc5.inference \
  --checkpoint runs/autorobobench/robocasa_faucet_peak/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_faucet_peak/<run>/eval_10.json \
  --eval-episodes-per-task 10 \
  --max-steps 400 \
  --commit-steps 8 \
  --workers 5 \
  --device cuda
```

## Visualize

Summarize eval/training outputs under `<run>/visualize/`. Add `--render` to also save eval videos.

```bash
python3 tasks/robocasa_faucet_peak/visualize.py \
  --run-dir runs/autorobobench/robocasa_faucet_peak/<run>
```
