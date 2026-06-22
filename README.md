# robot-autoresearch

Compact AutoRoboBench harness for RoboCasa robot-learning research loops.

The source tree is intentionally small:

- `benchmark.json`: benchmark suites, tracks, weights, metrics, and task specs
- `setup.py`: universal installer/verifier, generated metadata writer, scorer, and hasher
- `tasks/`: task-owned setup, train, inference, eval, visualize, model, and instruction files
- `data/`: local generated benchmark metadata plus shipped pretrained policy artifacts
- `docs/`: task descriptions and baseline notes

`configs/`, `examples/`, repo-level `models/`, repo-level `train/`, and the
`autorobobench` Python package were removed. Task implementations own their
training/model code directly.

## Install

From a fresh checkout:

```bash
git clone <repo-url> robot-autoresearch
cd robot-autoresearch
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[robocasa,plot]"
python setup.py
```

`python setup.py` writes generated JSON metadata under `data/`, validates the
benchmark spec, checks imports, and runs metadata-only task setup. It does not
require the full RoboCasa datasets.

To download referenced RoboCasa tasks:

```bash
python setup.py --download-robocasa --yes
```

To verify mounted or synced datasets:

```bash
python setup.py --verify
```

## Benchmark Commands

Inspect the main suite:

```bash
python setup.py --describe-benchmark --suite autorobobench_v0
```

Score a result file:

```bash
python setup.py --score-results path/to/results.json --suite autorobobench_v0
```

Hash immutable benchmark files:

```bash
python setup.py --hash-manifest --suite autorobobench_v0 --out runs/autorobobench/v0_hashes.json
```

Additional suite keys are `visual_world_model_v0` and
`world_model_posttraining_v0`.

## Tracks

The active task packages are:

| Track | Package | Main RoboCasa task/data |
| --- | --- | --- |
| RoboCasa BC-5 | `tasks/robocasa_bc5/` | `OpenCabinet`, `CloseDrawer`, `CloseFridge`, `TurnOffStove`, `PickPlaceCounterToCabinet` |
| Long-Horizon Microwave | `tasks/robocasa_long_horizon/` | `PickPlaceCounterToMicrowave` |
| Video Data to Policy Transfer | `tasks/video_policy_transfer/` | BC-5 demos plus RGB-only video pool |
| RoboCasa World Model | `tasks/robocasa_world_model/` | BC-5 transition and policy-ranking world model |
| Choose Measuring Cup Language | `tasks/robocasa_choose_measuring_cup_language/` | measuring-cup language variants |
| Visual World Model | `tasks/robocasa_visual_world_model/` | BC-5 next-frame prediction |
| World-Model Posttraining | `tasks/robocasa_world_model_posttraining/` | `PickPlaceCounterToMicrowave` policy improvement |
| Offline-RL Posttraining | `tasks/robocasa_offlinerl_posttraining/` | `PickPlaceCounterToMicrowave` policy improvement |
| Faucet Peak | `tasks/robocasa_faucet_peak/` | `TurnOnSinkFaucet` |
| Stand Mixer Peak | `tasks/robocasa_stand_mixer_peak/` | `PickPlaceCounterToStandMixer` |

Each task owns its `setup.py`, `train.py`, `inference.py`, `eval.py`,
`visualize.py`, `task.json`, and `INSTRUCTIONS.md`. Visualizers write compact
JSON/SVG summaries, and optional render artifacts where supported, under
`runs/autorobobench/<task>/<run>/visualize/`.

## Local Outputs

`runs/` is local-only and recreated by training/eval commands. Generated JSON
metadata in `data/` is also local-only; `setup.py` recreates it from embedded
benchmark metadata. Shipped policy checkpoint artifacts live under
`data/autorobobench/pretrained_policies/`.

## Smoke Checks

Tiny BC-5 train/eval:

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

Long-horizon wrapper:

```bash
python tasks/robocasa_long_horizon/setup.py --verify
python tasks/robocasa_long_horizon/train.py --steps 200
```

Video-transfer wrapper:

```bash
python tasks/video_policy_transfer/setup.py --verify
python tasks/video_policy_transfer/train.py --max-train-seconds 300
```
