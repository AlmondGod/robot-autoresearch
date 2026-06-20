# RoboCasa Microwave Peak Experiments

## 2026-06-20 autoresearch loop

Goal: create a stronger baseline for `PickPlaceCounterToMicrowave` and commit
only if held-out simulator eval improves.

Commit gate: improvement in `peak_final_success` on the public held-out
microwave eval episodes. Quick loop used episodes 90-92.

| Run | Change | Train budget | Eval | Result | Decision |
| --- | --- | ---: | --- | ---: | --- |
| `bc_800_seed0` | BC-5 temporal chunk BC, 80 action demos, no video pretrain | 800 steps | 3 eps, 300 max steps, commit 12 | 0/3 | baseline only |
| `mini_pi0_act_resnet_1200_seed0` | Mini Pi0 ACT ResNet, progress conditioning, task action normalization, commit 8 | 1200 steps | 3 eps, 300 max steps, commit 12 | 0/3 | reject |
| `mini_pi0_act_resnet_1200_seed0` | Same checkpoint, matched commit horizon | 1200 steps | 3 eps, 300 max steps, commit 8 | 0/3 | reject |
| `mini_pi0_act_resnet_1200_seed0` | Same checkpoint, matched commit horizon and full task horizon | 1200 steps | 3 eps, 750 max steps, commit 8 | 0/3 | reject |
| `trajectory_bank_80_seed0` | Nearest initial-view trajectory replay, 80 train demos, 300-step bank | build-only | 3 eps, 300 max steps, commit 8 | 0/3 | reject |
| `trajectory_bank_80_full_seed0` | Nearest initial-view trajectory replay, 80 train demos, 750-step bank | build-only | 3 eps, 750 max steps, commit 8 | 0/3 | reject |
| `ensemble_bc_mini_seed0` | Mean action ensemble of BC and Mini Pi0 ACT ResNet checkpoints | build-only | 3 eps, 750 max steps, commit 8 | 0/3 | reject |
| `single_task_chunk_3000_seed0` | Dedicated single-task chunk policy, no task embedding, progress conditioning | 3000 steps | 3 eps, 750 max steps, commit 8 | 0/3 | reject |
| `single_task_variant_2000_seed0` | Single-task chunk policy with RoboCasa variant embedding and visual variant retrieval | 2000 steps | 3 eps, 750 max steps, commit 8 | 0/3 | reject |
| `diagnostic_oracle_eval_replay` | Exact replay of public eval episode actions through the policy interface | build-only | 3 eps, 900 max steps, commit 8 | 3/3 | diagnostic only, reject |

Notes:

- The default 300-step eval horizon is likely too short for the task contract:
  demo action lengths over the 100 public microwave episodes have median 371,
  max 858, and 73/100 are longer than 300.
- Extending eval to 750 did not improve the tested policies, so no commit was
  made under the eval-improvement rule.
- The learned policy did improve offline validation loss (`0.6095` for
  `bc_800_seed0` versus `0.5688` for `mini_pi0_act_resnet_1200_seed0`), but that
  did not translate into held-out task success.
- A dedicated single-task optimizer reduced offline validation loss further
  (`0.2118` for `single_task_chunk_3000_seed0`), but still scored 0/3.
- Exact replay of public eval actions scored 3/3, proving evaluator and policy
  plumbing can report nonzero. This run is explicitly non-compliant because it
  uses test-time action demos.
- Initial-frame visual nearest-neighbor retrieval usually selects the wrong
  RoboCasa `task_index`; same-variant train-action replay still failed on
  episodes 90-92, so open-loop train-demo libraries are not enough.
- Rendered rollouts are under
  `runs/autorobobench/robocasa_microwave_peak/*/renders*`.

Next promising directions:

- Add a reset-to-demo playback sanity check on train episodes before further
  policy work, to verify action scaling and controller replay for this task.
- Inspect success predicates and terminal states for microwave placement; the
  rendered BC rollout reached the microwave area but did not satisfy success.
- Try a closed-loop waypoint or residual controller around the learned policy
  instead of pure chunked BC or open-loop replay.
