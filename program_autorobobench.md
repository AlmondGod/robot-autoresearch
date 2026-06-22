# AutoRoboBench Agent Instructions

You are running an AutoRoboBench RoboCasa research loop.

Goal:

```text
Improve the robot-learning system under the fixed benchmark budget.
```

Primary score comes from evaluator reruns, not self-reported metrics.

## Rules

- Read the active task's `task.json` and `INSTRUCTIONS.md` first.
- Do not edit files matched by the active track's `immutable_globs`.
- Do not read hidden eval files, canary files, or answer files.
- Do not use network access unless the active track explicitly allows fixed
  external-data corpora.
- Keep experiment outputs under `runs/`.
- Commit only accepted improvements.

## Benchmark Metadata

Use `benchmark.json` for suite membership and `python setup.py` for
measurement:

```bash
python setup.py --describe-benchmark --suite autorobobench_v0
python setup.py --score-results path/to/results.json --suite autorobobench_v0
```

Generated manifests, splits, video pools, policy registries, and eval metadata
are written under `data/` by `python setup.py`.
