# robocasa_recap_offline Instructions

Write outputs under `runs/autorobobench/robocasa_recap_offline/<run>/`. Do not
edit eval files or split files for scored runs.

## Task

- Optimize `PickPlaceCounterToStandMixer` from demonstrations plus offline
  experience: failed rollouts, corrections, or other saved rollouts.
- Metric: rollout success.
- Do not use test-time demos.
- Current measured result: 0/100.

## Train

```bash
python3 tasks/robocasa_recap_offline/train.py \
  --manifest data/autorobobench/robocasa_stand_mixer_peak_manifest.json \
  --split data/autorobobench/robocasa_stand_mixer_peak_splits.json \
  --out-dir runs/autorobobench/robocasa_recap_offline/<run> \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_recap_offline/eval.py \
  --checkpoint runs/autorobobench/robocasa_recap_offline/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_recap_offline/<run>/eval_10.json \
  --eval-episodes-per-task 10 \
  --device cuda
```
