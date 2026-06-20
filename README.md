# robot-autoresearch

Compact AutoroboBench harness for RoboCasa robot-learning research loops.

The repo is now trimmed to the pieces needed to run and score the current
RoboCasa benchmark profile:

- benchmark CLI and scoring utilities in `autorobobench/`
- RoboCasa task packages in `tasks/`
- frozen public split/data metadata in `data/autorobobench/` and
  `data/robocasa5/manifest.json`
- shared RoboCasa dataset/runtime helpers in `data/` and `autorobobench/`
- retained policy/training code in `models/` and `train/`

Old LIBERO, DROID, OSCAR, learned-world-model evaluator, scratch research, and
generated run artifacts have been removed.

## Tracks

`configs/autorobobench_v0.json` defines the executable profile:

| Track | Package | Public data |
| --- | --- | --- |
| RoboCasa BC-5 | `tasks/robocasa_bc5/` | `data/autorobobench/robocasa_bc5_splits.json` |
| Long-Horizon RoboCasa | `tasks/robocasa_long_horizon/` | `data/autorobobench/robocasa_long_horizon_splits.json` |
| Video Data to Policy Transfer | `tasks/video_policy_transfer/` | `data/autorobobench/video_policy_transfer_splits.json` and `data/autorobobench/video_policy_transfer_video_pool.json` |
| RoboCasa World Model Policy Correlation | `tasks/robocasa_world_model/` | `data/autorobobench/robocasa_world_model_policy_set.json` |

## Setup

Install the package and RoboCasa dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[robocasa,plot]"
```

The task runners use the local `third_party/robocasa` and `third_party/robosuite`
checkouts when present. If they are absent, install equivalent RoboCasa and
robosuite packages before running simulator evaluation.

Create or verify the RoboCasa seed manifest:

```bash
python data/make_robocasa5.py --out data/robocasa5/manifest.json
python data/download_robocasa.py --manifest data/robocasa5/manifest.json --split pretrain --source human
python data/make_robocasa5.py --out data/robocasa5/manifest.json --verify-exists
```

## Smoke Checks

Inspect and score the benchmark contract:

```bash
python -m autorobobench.cli describe --config configs/autorobobench_v0.json
python -m autorobobench.cli score \
  --config configs/autorobobench_v0.json \
  --results examples/autorobobench_v0_results.json
python -m autorobobench.cli hash-manifest \
  --config configs/autorobobench_v0.json \
  --out runs/autorobobench/v0_immutable_hashes.json
```

Run a tiny BC-5 training/eval pass:

```bash
python tasks/robocasa_bc5/setup.py --verify

python tasks/robocasa_bc5/train.py \
  --out-dir runs/autorobobench/robocasa_bc5/baseline \
  --train-episodes-per-task 4 \
  --val-episodes-per-task 2 \
  --steps 200

python tasks/robocasa_bc5/eval.py \
  --policy runs/autorobobench/robocasa_bc5/baseline/policy_best.pt \
  --out runs/autorobobench/robocasa_bc5/baseline/eval_success.json \
  --eval-episodes-per-task 1
```

Run the long-horizon wrapper:

```bash
python tasks/robocasa_long_horizon/setup.py --verify
python tasks/robocasa_long_horizon/train.py --steps 200
```

Run the scarce-action video-transfer wrapper:

```bash
python tasks/video_policy_transfer/setup.py --verify
python tasks/video_policy_transfer/train.py --max-train-seconds 300
```
