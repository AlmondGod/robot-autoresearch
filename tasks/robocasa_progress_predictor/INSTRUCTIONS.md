# robocasa_progress_predictor Instructions

Keep scored runs under 300 seconds. Write outputs under
`runs/autorobobench/robocasa_progress_predictor/<run>/`. Do not edit eval files
or split files for scored runs.

## Task

- Train an auxiliary progress regressor on BC5 state/action/task data.
- Metric: validation R2, MAE, RMSE, baseline RMSE.
- This is not a rollout-success benchmark.

## Train

```bash
python3 -m tasks.robocasa_progress_predictor.train \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/robocasa_bc5_splits.json \
  --out-dir runs/autorobobench/robocasa_progress_predictor/<run> \
  --max-train-seconds 300 \
  --device cuda
```
