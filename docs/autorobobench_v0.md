# AutoroboBench v0

AutoroboBench is a benchmark of robotics research loops.

The question is:

```text
Given a fixed robot-learning repo, data budget, compute budget, and hidden eval,
can an agent make the robot system better?
```

It is not primarily a benchmark of final robot skill. It scores autonomous
research progress: reading the repo, choosing experiments, editing code, running
bounded trials, keeping improvements, and surviving a clean hidden rerun.

## Tracks

`configs/autorobobench_v0.json` defines six tracks and a 100 point score.

| ID | Track | Phase | Points | Primary metric |
| --- | --- | ---: | ---: | --- |
| `robocasa_bc5` | RoboCasa BC-5 | 1 | 15 | hidden held-out success |
| `robocasa_long_horizon` | Long-Horizon RoboCasa | 1 | 20 | full success plus subgoal progress |
| `world_model_evaluator` | World Model Evaluator | 1 | 20 | policy-ranking correlation and speedup |
| `external_data_wall` | External Data / Data Wall | 2 | 15 | value of fixed external robot data |
| `sim_rl_improvement` | Sim RL Improvement | 3 | 15 | success AUC under sim budget |
| `transfer_robust_language` | Transfer / Robustness / Language | 3 | 15 | hidden robust success |

Phase 1 is the launchable v0 profile for this repo: BC-5, Long-Horizon, and
World Model Evaluator. Phases 2 and 3 are defined in the score file so the suite
can grow without changing the scoring contract.

## Scoring

Each track result is a JSON object with metrics in `[0, 1]`, plus the track's
primary metric. For primary metrics with known starter/reference targets, the
score can use:

```text
normalized_progress =
  clip((agent_metric - starter_metric) / (reference_metric - starter_metric), 0, 1)
```

The suite score is:

```text
AutoroboBench score = sum(track_weight * track_score)
```

where `track_score` is a weighted component average from the config.

Run a local smoke score:

```bash
python -m autorobobench.cli score \
  --config configs/autorobobench_v0.json \
  --results examples/autorobobench_v0_results.json
```

## Anti-Cheating Contract

Final scoring should be done by organizer rerun, not by self-reported metrics.

Protocol:

- public train/dev seeds for debugging
- hidden seeds, objects, layouts, tasks, and canary files for final eval
- immutable evaluator and data-prep files
- file hashes for hidden/eval/config artifacts
- network disabled by default, except fixed external-data tasks
- submit patch and run log, not claimed score
- statistical eval over multiple seeds
- reproducibility/integrity penalties for metric fabrication or eval tampering

Create a local immutable-file hash manifest:

```bash
python -m autorobobench.cli hash-manifest \
  --config configs/autorobobench_v0.json \
  --out runs/autorobobench/v0_immutable_hashes.json
```

## Current Repo Mapping

The current repo already has the ingredients for the Phase 1 launch profile:

- RoboCasa seed tasks: `data/robocasa5/manifest.json`
- BC and policy loops: `train.py`, `train/`, `models/`, `research/`
- World model evaluator traces:
  `runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl`
- Ranking metric implementation: `eval/eval_world_model_ranking.py`

The World Model Evaluator track should be scored by decision usefulness:
correlation with true sim success, ranking accuracy, calibration, and measured
speedup. Pixel/video prediction loss is diagnostic only.
