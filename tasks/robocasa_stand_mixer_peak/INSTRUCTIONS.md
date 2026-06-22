# robocasa_stand_mixer_peak Instructions

Keep scored runs under 300 seconds. Write outputs under
`runs/autorobobench/robocasa_stand_mixer_peak/<run>/`. Do not edit eval files or
split files for scored runs.

## Task

- Optimize one policy for `PickPlaceCounterToStandMixer`.
- Metric: single-task rollout success.
- Data: task-specific action demos and generic video-only pool are allowed.
- Current measured learned policies are 0/100.

## Train

```bash
python3 tasks/robocasa_stand_mixer_peak/train.py \
  --manifest data/autorobobench/robocasa_stand_mixer_peak_manifest.json \
  --split data/autorobobench/robocasa_stand_mixer_peak_splits.json \
  --out-dir runs/autorobobench/robocasa_stand_mixer_peak/<run> \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_stand_mixer_peak/eval.py \
  --checkpoint runs/autorobobench/robocasa_stand_mixer_peak/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_stand_mixer_peak/<run>/eval_10.json \
  --eval-episodes-per-task 10 \
  --device cuda
```

## Visualize

Summarize eval/training outputs under `<run>/visualize/`. Add `--render` to also save eval videos.

```bash
python3 tasks/robocasa_stand_mixer_peak/visualize.py \
  --run-dir runs/autorobobench/robocasa_stand_mixer_peak/<run>
```
