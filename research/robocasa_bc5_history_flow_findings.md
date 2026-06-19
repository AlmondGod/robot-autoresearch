# RoboCasa BC-5 History Flow Findings

Date: 2026-06-19

## Change

Tested a history-conditioned flow-matching decoder as an alternative to the history-ACT action decoder.

The policy uses:

- current RGB agent+wrist views
- previous RGB agent+wrist views
- current proprio
- previous proprio
- proprio delta
- task id
- rectified-flow action decoder
- 32-step action chunks

Eval code was unchanged. The policy receives no test-time demonstrations.

## Run

Run directory:

`runs/autorobobench/robocasa_bc5_history_flow/histflow_h32_w256_s16_3min_seed0`

Training setup:

- 80 train demos/task
- 10 val demos/task
- chunk horizon: 32
- history stride: 16
- width: 256
- transformer depth: 2
- action depth: 2
- residual flow source: BC head
- BC auxiliary weight: 2.0
- optimizer cap: 180 seconds

Metrics:

| metric | value |
| --- | ---: |
| optimizer steps | 1021 |
| train seconds | 184.3 |
| best val action MSE | 0.3597 |
| quick eval, commit 16 | 0/10 |
| quick eval, commit 32 | 0/10 |

Quick eval means first two frozen eval episodes for each of the five BC-5 tasks, `max_steps=260`.

## Interpretation

This did not improve over history-ACT.

History-ACT previous result:

- final val action MSE: 0.2613
- quick eval: 2/10

History-flow result:

- best val action MSE: 0.3597
- quick eval: 0/10 at both commit 16 and commit 32

The flow decoder regressed the only task history-ACT solved on this quick slice (`CloseDrawer`: 2/2 -> 0/2). Under the same short training budget, the flow decoder appears harder to optimize than the direct ACT-style action decoder.

## Follow-Up

Do not promote this checkpoint. If revisiting flow matching, likely better options are:

- longer training budget
- pretrained/direct ACT checkpoint initialization before flow fine-tuning
- smaller flow residual weight and checkpoint selection on direct closed-loop eval
- diffusion/flow only for late chunk refinement, not the whole action decoder
