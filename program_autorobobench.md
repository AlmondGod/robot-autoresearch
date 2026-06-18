# AutoroboBench Agent Instructions

You are running an AutoroboBench v0 research loop.

Goal:

```text
Improve the robot-learning system under the fixed benchmark budget.
```

Primary score comes from hidden evaluator reruns, not self-reported metrics.

## Rules

- Do not edit files matched by the active track's `immutable_globs` in
  `configs/autorobobench_v0.json`.
- Do not read hidden eval files, canary files, or answer files.
- Do not use network access unless the active track explicitly allows fixed
  external-data corpora.
- Keep a clear experiment ledger with change, commit, train budget, metrics,
  accepted/rejected decision, and notes.
- Prefer changes that can survive a clean rerun.
- Commit only accepted improvements.

## Phase 1 Tracks

### RoboCasa BC-5

Improve the BC/VLM policy on the five seed RoboCasa tasks.

Good experiment families:

- action chunking and temporal ensembling
- flow or diffusion action heads
- better language/task conditioning
- image augmentation and view dropout
- balanced multitask sampling
- auxiliary progress/value losses
- validation-gated early stopping

### Long-Horizon RoboCasa

Improve compositional manipulation and recovery.

Good experiment families:

- subgoal prediction
- progress-value heads
- open-loop chunk plus closed-loop correction
- failure recovery policy
- task decomposition from language

### World Model Evaluator

Train a learned evaluator that ranks policy candidates faster than simulator
rollouts.

Good experiment families:

- trace-conditioned latent dynamics
- progress/success calibration
- held-out candidate splits
- speed/accuracy tradeoffs
- ranking loss instead of only pixel or latent loss

The World Model Evaluator track is judged by policy-ranking usefulness, not
visual fidelity alone.

## Required Ledger Fields

Each experiment row should contain:

```json
{
  "experiment": 1,
  "commit": "abc1234",
  "track": "world_model_evaluator",
  "change": "trace-calibrate VAE evaluator on fold A",
  "train_budget_seconds": 300,
  "metrics": {
    "wm_spearman": 0.72,
    "checkpoint_ranking_accuracy": 0.8,
    "wm_speedup_score": 0.6,
    "calibration_score": 0.55
  },
  "accepted": true,
  "notes": "Improves held-out candidate ranking."
}
```

## Scoring Smoke Test

```bash
python -m autorobobench.cli describe --config configs/autorobobench_v0.json
python -m autorobobench.cli score \
  --config configs/autorobobench_v0.json \
  --results examples/autorobobench_v0_results.json
```
