# Full Trajectory BC-160 Experiments

Goal: improve the learned `initial observation -> 160-step action trajectory` policy on `libero_easy1_task0`.

Baseline:

- Run: `runs/libero/easy1_task0_full_traj_bc_h160`
- Train demos: 40
- Val demos: 10
- Horizon: 160
- Val action MSE: 0.044803
- Closed-loop success: 1/10

Constraints:

- No demo retrieval algorithm at test time.
- The policy must output actions from the current initial observation/proprio.
- Experiments may use demonstration data for supervised training, augmentation, weighting, regularization, and architecture changes.

Experiment log lives at:

- Remote: `runs/libero/fulltraj50/archive_v2.jsonl`
- Local copy after sync: `runs/libero/fulltraj50/archive_v2.jsonl`

Note: `archive.jsonl` contains 50 invalid CLI attempts from the first launch, caused by passing the experiment `name` metadata into `train_full_trajectory_bc.py`. Those are not counted as experiments.

Families to explore:

- Loss shaping: Huber/L1, early-action emphasis, tail emphasis, temporal decay.
- Robustness: image/proprio/action noise, dropout, weight decay.
- Capacity: wider/narrower visual and proprio encoders.
- Data splits/seeds: different held-out demos and train-demo counts.
- Smoothness: output trajectory smoothness regularization.
- Optimization: learning rate, batch size, longer/shorter fits.

Valid v2 results:

| idx | change | val_action_mse | success_rate | notes |
| --- | --- | ---: | ---: | --- |
| 1 | baseline rerun seed0 | 0.037648 | 0.40 | 2/5 successes under screening eval |
| 2 | huber loss | 0.043508 | 0.00 | Retried after interrupt; 0/5 successes |
| 3 | l1 loss | 0.034449 | 0.00 | Lower val MSE but no rollout success |
| 4 | early action weight 2x | 0.039900 | 0.00 | Early-timestep emphasis did not help at 2x |
| 5 | early action weight 4x | 0.037141 | 0.40 | Tied best success, slightly better MSE than baseline |
| 6 | tail action weight 2x | 0.033630 | 0.20 | Lowest MSE so far, partial rollout success |
| 7 | tail action weight 4x | 0.037929 | 0.40 | Tied best success, worse MSE than early 4x |
| 8 | temporal decay 0.995 | 0.036313 | 0.00 | Recency/early horizon decay hurt success |
| 9 | temporal decay 0.99 | 0.035684 | 0.00 | Lower MSE but no success |
| 10 | temporal decay 0.98 | 0.036128 | 0.60 | New best; remaining queue switched to variants around this config |

Queue policy update:

- Experiments 1-10 were a broad initial sweep.
- From experiment 11 onward, variants build on the current best config: `temporal_decay=0.98`.
- Old broad-sweep variants after 10 were replaced before completion, so experiment 11+ tests are local search around the best-so-far policy rather than independent baseline perturbations.
