# robot-autoresearch

How do we assess whether models are good at AI robotics research, not just AI
research in the abstract?

Robotics research is not just making a loss curve go down. A useful robotics
research agent has to improve real closed-loop behavior, generalize out of
distribution in small-data regimes, adapt data that is not perfectly matched to
the task, use broad video/internet-style data when it helps, and push reliability
toward deployment-level performance.

This repo is an autoresearch loop for that problem. Like Karpathy-style
autoresearch, it is meant to be cheap to run, fast to iterate, and small enough
for one GPU. Instead of hand-editing policy code yourself, point an AI agent at a
task folder and let it run the loop: read instructions, edit the task-owned
training and inference files, train for the fixed budget, evaluate in RoboCasa,
visualize rollouts or model predictions, and keep changes that improve the
score.

The benchmark is intentionally focused on directions that matter for scaling
robotics into deployment:

- out-of-distribution generalization
- using the data that is actually available, including RGB-only video data
- high reliability on repetitive tasks
- small, installable, single-GPU experiments with fast feedback

![World model autoresearch progress](docs/assets/world_model_autoresearch_progress.png)

This repository is the research/reference version of the benchmark. It contains
the current best-known RoboCasa task implementations and baseline results. A
separate Docker harness can be built from this repo when the goal is clean
external-agent evaluation with isolated train/eval containers.

## How It Works

The repo is kept small around two ideas:

- `setup.py` is the universal setup, metadata, suite, scoring, and hashing
  entrypoint.
- `tasks/<task>/` owns everything an agent needs for one benchmark task:
  `setup.py`, `train.py`, `inference.py`, `eval.py`, `visualize.py`,
  `task.json`, and `INSTRUCTIONS.md`.

By design, benchmark training is time-budgeted. The core policy tasks use a
5-minute training cap for scored experiments unless a task explicitly says
otherwise. The main metric is simulator success rate; world-model tasks are
scored by held-out transition/visual accuracy and policy-ranking correlation.

Generated metadata under `data/` is recreated by `python setup.py`. Local runs
and visualizations go under `runs/` and are not committed.

## Task Files

Each task is deliberately self-contained:

- `INSTRUCTIONS.md`: task-specific operating instructions for agents.
- `setup.py`: verifies local metadata and datasets for that task.
- `train.py`: the main editable research surface. Agents change architectures,
  data loading, losses, optimization, and hyperparameters here.
- `inference.py`: the policy or world-model interface used by eval. Agents edit
  this when model loading or action generation changes.
- `eval.py`: fixed evaluator wrapper for the task. Scored runs should treat this
  as read-only.
- `visualize.py`: editable artifact viewer. It writes summaries, plots, videos,
  or rollout/world-model diagnostics under the run directory.
- `task.json`: task metadata, default train/eval settings, metrics, and file
  permissions.

For BC-style tasks, `visualize.py` summarizes eval success and can render
rollouts. For world-model tasks, it compares predicted dynamics or visual
rollouts against actual held-out data. For offline-RL/posttraining tasks, it
shows the assigned rewards, sample weights, and whether the learned update
improves real simulator success.

## Quick Start

Requirements: Python 3.10+, a CUDA GPU for real RoboCasa training/eval, and the
RoboCasa/robosuite dependencies.

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
embedded benchmark suites, checks imports, and runs metadata-only task setup. It
does not require the full RoboCasa datasets.

To download referenced RoboCasa tasks:

```bash
python setup.py --download-robocasa --yes
```

To verify mounted or synced datasets:

```bash
python setup.py --verify
```

## Running An Agent

Spin up Codex/Claude/etc. in this repo and point it at a task instruction file,
for example:

```text
Read tasks/robocasa_long_horizon/INSTRUCTIONS.md and try to improve the policy.
Keep training under the task budget, run eval, visualize the result, and only
keep changes that improve the task metric.
```

The task instructions give the exact train/eval/visualize commands. Agents
should normally edit only files inside the active task folder plus
`program_autorobobench.md` if they are improving the research procedure.

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

Each task owns its train/eval/inference code. Visualizers write compact JSON/SVG
summaries, and optional render artifacts where supported, under
`runs/autorobobench/<task>/<run>/visualize/`.

## Project Structure

```text
setup.py                       installer/verifier, suite metadata, scorer, hasher
program_autorobobench.md       high-level agent research instructions
tasks/                         task-owned setup/train/inference/eval/visualize code
data/                          generated metadata plus shipped pretrained policy artifacts
docs/                          task descriptions and baseline notes
runs/                          local-only training, eval, and visualization outputs
```

`configs/`, `examples/`, repo-level `models/`, repo-level `train/`, and the
`autorobobench` Python package were removed. Task implementations own their
training/model code directly.

## Design Choices

Task-owned files. Agents work inside a task folder instead of a large shared
framework. This keeps diffs reviewable and makes each benchmark task executable
on its own.

Fixed training budget. Most scored policy experiments are capped at 5 minutes.
That makes changes comparable within the same compute environment and rewards
fast, reliable improvements.

Real simulator metrics. Policy tasks are judged by RoboCasa rollout success, not
just train loss. World-model tasks use held-out transition/visual metrics and
policy-ranking correlation.

Visualization-first debugging. Every task has `visualize.py`, so agents can
inspect eval summaries, rollout videos, world-model predictions, or offline-RL
reward assignments under the run directory.

Local outputs stay local. `runs/` is recreated by training/eval commands.
Generated JSON metadata in `data/` is also local-only; `setup.py` recreates it
from embedded benchmark metadata. Shipped policy checkpoint artifacts live under
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
