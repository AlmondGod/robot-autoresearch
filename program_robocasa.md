# RoboCasa Evaluator Autoresearch Program

You are a robotics research agent improving policy search on RoboCasa-5.

## Objective

Use a learned world-model evaluator to reduce slow simulator rollouts during
policy search.

Primary ground-truth metric:

- real RoboCasa closed-loop `success_rate`

Primary evaluator metric:

- Spearman rank correlation between learned evaluator score and real sim success
  on held-out policy checkpoints

Secondary metrics:

- Pearson correlation
- top-k hit rate
- learned evaluator rollouts/second
- simulator rollouts/second
- speedup ratio

## Fixed Benchmark

Use the generated manifest:

```text
data/robocasa5/manifest.json
```

The initial 5 task aliases are:

- `OpenDrawer`
- `CloseDrawer`
- `PickPlaceCounterToStove`
- `TurnOffStove`
- `PickPlaceCounterToCabinet`

If exact RoboCasa task ids differ locally, update the manifest generation step,
not the evaluator or judge.

## Editable Surface

For policy candidate experiments, edit policy training code and configs only.

Do not edit:

- RoboCasa task definitions during an experiment
- real simulator success criteria
- correlation metric code
- top-k judge code
- speed measurement code

## Evaluation Contract

Each policy candidate row must record:

- `candidate_id`
- `commit`
- `change`
- `policy_checkpoint`
- `learned_score`
- `learned_eval_seconds`
- `learned_eval_rollouts`
- `sim_success`
- `sim_eval_seconds`
- `sim_eval_rollouts`

The learned evaluator may rank candidates and choose finalists. It may not
replace final real simulator judging.

## Research Rule

Optimize for ranking quality, not perfect video prediction.

The right question is:

```text
Can the learned evaluator choose policies that win under actual RoboCasa rollout?
```
