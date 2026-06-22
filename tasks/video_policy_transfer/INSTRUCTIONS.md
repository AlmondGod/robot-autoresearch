# video_policy_transfer Instructions

Write outputs under `runs/autorobobench/video_policy_transfer/<run>/`. Do not
edit eval files or split files for scored runs.

## Task

- Train one policy from scarce paired action demos plus RGB-only video.
- Tasks use BC5 task set.
- Data: two paired action demos/task plus video-only pool.
- Metric: rollout success and paired-action efficiency.
- Current smoke evals are 0/100.

## Train

```bash
python3 tasks/video_policy_transfer/train.py \
  --manifest data/robocasa5/manifest.json \
  --split data/autorobobench/video_policy_transfer_splits.json \
  --video-pool data/autorobobench/video_policy_transfer_video_pool.json \
  --out-dir runs/autorobobench/video_policy_transfer/<run> \
  --device cuda
```

## Eval

```bash
python3 tasks/video_policy_transfer/eval.py \
  --checkpoint runs/autorobobench/video_policy_transfer/<run>/policy_best.pt \
  --out runs/autorobobench/video_policy_transfer/<run>/eval.json \
  --device cuda
```

## Visualize

Summarize eval/training outputs under `<run>/visualize/`. Add `--render` to also save eval videos.

```bash
python3 tasks/video_policy_transfer/visualize.py \
  --run-dir runs/autorobobench/video_policy_transfer/<run>
```
