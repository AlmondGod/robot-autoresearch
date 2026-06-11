# RoboCasa Learned Evaluator Track

Core thesis:

```text
Can a fast learned world-model evaluator replace most slow robot simulator
rollouts during robot policy search?
```

The launchable claim is not that the world model predicts video perfectly. The
claim is that it ranks policy variants well enough that autoresearch can spend
real RoboCasa simulator rollouts only on finalists.

## Benchmark

Use a 5-task RoboCasa seed benchmark:

| id | task | role |
| --- | --- | --- |
| 0 | `OpenDrawer` | articulated drawer opening |
| 1 | `CloseDrawer` | articulated drawer closing |
| 2 | `PickPlaceCounterToStove` | mug/object pick-place |
| 3 | `TurnOffStove` | knob manipulation |
| 4 | `PickPlaceCounterToCabinet` | region/container-style pick-place |

This benchmark is built from the official RoboCasa dataset registry. By default
it uses `split=pretrain` and `source=human`, because all five chosen tasks are
available there and `CloseDrawer` does not expose a target-human release in the
official registry snapshot used here.

If a local RoboCasa installation exposes slightly different canonical names,
`data/make_robocasa5.py` is the single place to map aliases to exact task ids.
The eval and judge code should consume the generated manifest, not hardcode task
names.

## Policy Baseline

Policy v0 should be deliberately simple:

- inputs: 2 RGB views, proprio, task language/id
- model: diffusion policy or small transformer BC
- output: action chunk, initially 8-16 actions
- data: 50 policy demos per task from 100 available pretrain-human demos
- metric: closed-loop RoboCasa success per task

Target before using the evaluator in the loop:

```text
Average RoboCasa-5 success > 0.40, ideally > 0.60.
```

## Learned Evaluator

The evaluator should be small, batched, latent-space, low-res, and
success-predictive.

Use Cosmos or another pretrained video model only as a visual prior or
initialization. Do not use full-resolution video diffusion as the fast evaluator.

Minimal model:

- latent encoder: small VAE, image tokenizer, or frozen video/Cosmos encoder
- dynamics: action-conditioned transformer over latent states
- heads:
  - next-latent prediction
  - success probability
  - progress/value score
  - failure classifier

Inputs:

- initial observation
- task language/id
- candidate policy action sequence or rollout prefix

Outputs:

- predicted success probability
- progress score
- optional predicted future latent/video

## Calibration Before Autoresearch

Do not require the autoresearch loop to work before the evaluator exists.

1. Train many policy checkpoints:
   - undertrained
   - overfit
   - different learning rates
   - different data amounts
   - different augmentations
   - different action horizons
2. Run real RoboCasa sim eval for all checkpoints.
3. Train the evaluator on demo plus rollout data to predict success/progress and
   next latent state.
4. Hold out unseen policy checkpoints.
5. Test whether evaluator scores rank held-out checkpoints like real sim
   success.
6. Only then put the evaluator inside the autoresearch loop.

## Autoresearch Loop

For each iteration:

1. Generate 20-100 candidate policy variants.
2. Run 100-1000 evaluator samples per candidate, batched on GPU.
3. Select the top 3-5 candidates by learned score.
4. Run real RoboCasa sim only on those finalists.
5. Append the finalist sim rollouts to evaluator training data.

The evaluator may reject most candidates cheaply. The simulator remains the
ground truth judge.

## Required Plots

1. Policy improvement:
   - x-axis: autoresearch iteration
   - y-axis: real RoboCasa success
2. Evaluator correlation:
   - x-axis: learned predicted success
   - y-axis: actual sim success
   - report Pearson, Spearman, and top-k hit rate
3. Speedup:
   - x-axis: eval method
   - y-axis: rollouts per second
   - claim speedup only if measured

Good enough:

```text
Spearman rho > 0.5: interesting
Spearman rho > 0.7: strong
top-5 learned candidates contain the best sim candidate: compelling
learned evaluator is 10x-100x faster than RoboCasa sim eval: launchable
```

## Minimum Claim

```text
I built a tiny robot autoresearch loop on RoboCasa. It trains policies on 5
kitchen tasks, uses a learned world model to cheaply rank policy variants, and
only calls the simulator for finalists. The learned score correlates with real
task success and speeds up evaluation.
```
