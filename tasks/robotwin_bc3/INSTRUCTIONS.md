# robotwin_bc3 Instructions

Keep scored runs under 300 seconds. Write outputs under
`runs/autorobobench/robotwin_bc3/<run>/`. Do not edit eval files or split files
for scored runs.

## Task

- Train one RoboTwin policy for `blocks_ranking_rgb`, `place_a2b_left`,
  `place_object_basket`.
- This is not RoboCasa. Use RoboTwin data and eval scripts.
- Metric: success rate for online eval; heldout action MSE for offline eval.

## Train

```bash
python3 tasks/robotwin_bc3/train.py \
  --split data/autorobobench/robotwin_bc3_splits.json \
  --output-dir runs/autorobobench/robotwin_bc3/<run> \
  --device cuda
```

## Eval

```bash
python3 tasks/robotwin_bc3/eval.py \
  --checkpoint runs/autorobobench/robotwin_bc3/<run>/policy_best.pt \
  --out runs/autorobobench/robotwin_bc3/<run>/eval.json \
  --device cuda
```
