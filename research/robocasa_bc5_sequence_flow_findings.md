# RoboCasa BC-5 Sequence-Flow Policy Findings

Date: 2026-06-18

## Change

Added a benchmark-compliant closed-loop sequence policy:

- vision encoder over agent+wrist RGB
- proprio encoder
- task embedding
- transformer context encoder
- transformer action-chunk decoder
- rectified-flow action matching
- optional BC-head action output with `flow_steps=0`

Eval code was not changed. The policy still exposes the normal `load_policy` / `act` interface and receives no test-time demos.

## Runs

| run | setup | train seconds | best val action MSE | quick eval |
| --- | --- | ---: | ---: | ---: |
| `seqflow_h32_w256_residual_3min_seed0` | horizon 32, residual flow from BC, flow eval | 181.9 | 0.3489 | 0/10 |
| `seqflow_h32_w256_residual_3min_seed0` | same checkpoint, BC-head eval ablation | - | mini-val BC MSE 0.2948 | 0/10 |
| `seqflow_h16_w256_bcdom_3min_seed1` | horizon 16, BC-dominant loss, BC-head eval | 184.0 | 0.2423 | 0/10 |

Quick eval means first two frozen eval episodes for each of the five BC-5 tasks, `max_steps=260`, `commit_steps=16`.

## Interpretation

The sequence transformer can improve offline action MSE under the 3-minute training budget, but that did not transfer to closed-loop success. The best offline run reached `0.2423` normalized val MSE, beating the earlier CNN BC baseline's `0.3174`, yet closed-loop success stayed at `0/10`.

The first residual-flow run showed that the flow sampler was worse than the BC head on held-out chunks:

- BC head MSE: `0.2948`
- residual-flow sample MSE: `0.3369`

So this change is useful infrastructure but not an accepted policy improvement. The next policy experiment should focus on rollout robustness rather than lower one-step chunk MSE: temporal action ensembling, shorter receding-horizon commits, history-conditioned ACT, or task-specific progress/contact auxiliary losses.
