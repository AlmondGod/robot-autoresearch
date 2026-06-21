# robocasa_wm_policy_improvement Instructions

Keep scored runs under 300 seconds. Write outputs under
`runs/autorobobench/robocasa_wm_policy_improvement/<run>/`. Do not edit eval
files or split files for scored runs.

## Task

- Start from a differentiable BC5-compatible policy.
- Use a frozen world model to improve the policy offline.
- Keep BC loss, init-policy anchor, and action penalty active. Real simulator
  success is final; WM objective alone is not enough.
- Supported policy modes: temporal chunk BC, temporal chunk flow, sequence flow.
- Unsupported for v0: trajectory banks, history policies, frozen VLM feature
  cache policies.
- Current checked-in WM-compatible base:
  `data/autorobobench/pretrained_policies/robocasa_faucet_direct_bc_all_data_5min_seed0.pt`.

## Train

```bash
python3 tasks/robocasa_wm_policy_improvement/train.py \
  --manifest <manifest.json> \
  --split <split.json> \
  --policy-checkpoint <base_policy.pt> \
  --world-model-checkpoint <world_model.pt> \
  --out-dir runs/autorobobench/robocasa_wm_policy_improvement/<run> \
  --max-train-seconds 300 \
  --device cuda
```

## Eval

```bash
python3 tasks/robocasa_bc5/eval_parallel.py \
  --manifest <manifest.json> \
  --split <split.json> \
  --inference tasks.robocasa_bc5.inference \
  --checkpoint runs/autorobobench/robocasa_wm_policy_improvement/<run>/policy_best.pt \
  --out runs/autorobobench/robocasa_wm_policy_improvement/<run>/eval.json \
  --eval-episodes-per-task 10 \
  --workers 5 \
  --device cuda
```
