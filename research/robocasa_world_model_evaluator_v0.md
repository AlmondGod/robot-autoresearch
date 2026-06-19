# RoboCasa World-Model Evaluator v0

Goal:
- Replace most slow RoboCasa simulator evaluations with a fast learned evaluator.
- Use the simulator only to validate top candidates and measure evaluator correlation.

Long-term architecture:
- Teacher/init: Cosmos-Predict2.5 Video2World or another internet-video world model.
- Finetune target: RoboCasa-5 action-conditioned rollouts.
- Distilled evaluator: small latent dynamics model with progress, success, and latent prediction heads.

Implemented v0:
- `models/robocasa_tiny_evaluator.py`
  - Encodes two RGB camera views, proprio, and task id into a latent.
  - Predicts action-conditioned next latent and next proprio.
  - Predicts progress and success probability from latent state.
- `train/train_robocasa_tiny_evaluator.py`
  - Trains from RoboCasa demonstration videos/actions.
  - Treats late successful demo frames as positive success labels.
  - Saves normalization stats so candidate policies can be scored offline.
- `eval/eval_robocasa_tiny_evaluator_correlation.py`
  - Scores existing autoresearch policy candidates with imagined latent rollouts.
  - Writes learned-vs-sim candidate records.
  - Produces Pearson/Spearman/top-k ranking metrics and a correlation SVG.

Smoke result:
- Run: `runs/robocasa/world_evaluator/tiny_opendrawer_task0_stable_smoke`
- Data: OpenDrawer task-index 0, 80 train demos, held-out episodes 87/92/93/94/98/100/101.
- Training: 150 steps, frame stride 8, 957 train transitions, 158 validation transitions.
- Offline validation:
  - val loss: 0.1889
  - progress MAE: 0.0568
  - success-label accuracy: 0.968
- Candidate ranking over 38 archived task-0 policies:
  - learned rollout speed: ~741 imagined rollouts/sec
  - Pearson correlation with RoboCasa sim success: -0.307
  - Spearman correlation: -0.044
  - top-5 hit: 0.0

Interpretation:
- The evaluator is fast enough for the thesis.
- The current score is not useful yet because it only scores action chunks from the initial observation.
- Many archive rows differ only in commit horizon or temporal ensembling; those changes affect closed-loop execution but not the first predicted action chunk.
- Next v0.1 should record actual action traces during RoboCasa sim eval and train/score on those traces.

Current limitation:
- This v0 is a fast evaluator scaffold, not yet a full visual world model.
- It imagines in learned latent space using predicted policy action chunks from initial observations.
- It does not yet decode video frames or use Cosmos features.

Next pass:
- Add a Cosmos feature cache interface:
  - raw RGB frames -> Cosmos latent/features
  - train the tiny evaluator on Cosmos latents instead of its own small CNN latent
- Record policy rollout action traces during real sim eval.
  - This lets the evaluator score full closed-loop action sequences, not just initial action chunks.
- Validate correlation on a larger candidate set and across RoboCasa-5, not only OpenDrawer task-index 0.

v0.1 trace-evaluator result:
- Added action trace recording to `eval/eval_robocasa_policy_ensemble.py`.
  - Each sim eval can now write one `.npz` per episode with executed actions and success flags.
- Added trace scoring to `eval/eval_robocasa_tiny_evaluator_correlation.py`.
  - `--prefer-action-traces` scores the actual executed action sequence instead of only the first predicted chunk.
  - `--invert-learned-score` supports the observed calibration where the raw head acts like a failure/risk score on traces.
- Re-evaluated experiments 29-38 with action traces:
  - trace archive: `runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl`
  - calibrated correlation plot: `runs/robocasa/world_evaluator/trace_eval_frontier/correlation_trace_calibrated.svg`
- Trace-based calibrated evaluator:
  - n: 10 candidates
  - learned speed: ~243 imagined rollouts/sec at 260 imagined steps
  - Pearson correlation with sim success: 0.995
  - Spearman correlation: 0.931
  - top-5 hit: 1.0

Interpretation:
- Full action traces fix the main v0 failure mode.
- The tiny evaluator is now useful as a cheap pre-filter over candidate rollouts, as long as candidates provide planned/executed action traces.
- This is still not a no-sim replacement for policy evaluation because the trace currently comes from sim eval. The next step is to score proposed policy action traces before full sim validation, then run RoboCasa only on top-ranked candidates.

v0.2 VAE-latent visual world model:
- Replaced the bolt-on latent-to-RGB decoder with a VAE-style visual latent model in `models/robocasa_tiny_evaluator.py`.
  - Encoder: two RGB views, proprio, and task id -> Gaussian latent.
  - Decoder: latent and task id -> reconstructed left-view and wrist/right-view RGB.
  - Dynamics: latent, action, and task id -> next latent and next proprio.
  - Heads: progress and success/risk prediction.
- Added:
  - `train/train_robocasa_vae_world_model.py`
  - `train/train_robocasa_vae_trace_calibrator.py`
  - `eval/render_robocasa_vae_rgb_rollout.py`
- Base visual training:
  - run: `runs/robocasa/world_evaluator/vae_opendrawer_task0_smoke`
  - data: OpenDrawer task-index 0, 80 train demos, held-out episodes 87/92/93/94/98/100/101
  - train samples: 957
  - val samples: 158
  - train time: 23.5 sec on local MPS/CPU-auto setup
  - best val loss: 0.6006
  - val RGB PSNR: 14.62 dB
  - val RGB MSE: 0.0345
  - progress MAE: 0.0529
  - success-label accuracy: 0.962
- Trace calibration:
  - run: `runs/robocasa/world_evaluator/vae_opendrawer_task0_trace_calibrated_h260`
  - initialized from the base VAE checkpoint
  - frozen encoder and decoder, so reconstruction quality is preserved
  - trained only dynamics/progress/success scoring on archived policy action traces
  - rollout horizon: 260 steps
  - best trace loss: 0.0316
  - train time: 127.4 sec
- Candidate ranking with calibrated VAE score:
  - archive: `runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl`
  - plot: `runs/robocasa/world_evaluator/vae_opendrawer_task0_trace_calibrated_h260/correlation_trace_calibrated.svg`
  - n: 10 candidates
  - raw success head behaves like a failure/risk score, so evaluation uses `--invert-learned-score`
  - Pearson correlation with RoboCasa sim success: 0.9979
  - Spearman correlation: 0.9309
  - top-5 hit: 1.0
  - best learned candidate id: 32
  - best sim candidate id: 32
- Decoded rollout:
  - video: `runs/robocasa/world_evaluator/vae_opendrawer_task0_trace_calibrated_h260/vae_decoded_world_rollout_exp034_ep87.mp4`
  - source trace final sim success: true

Interpretation:
- VAE latents improve reconstruction over the previous bolt-on decoder, but the decoded video is still blurry and not yet high-fidelity enough to be a strong visual simulator.
- The same latent model can be trace-calibrated into a very strong candidate ranker on the current archive.
- The correlation result is not a held-out claim yet: the trace archive used for calibration is also the archive used for the reported ranking measurement.
- Next required step is a proper held-out candidate split: calibrate on one set of policy traces, then score unseen candidate traces before running RoboCasa sim validation.

v0.3 flow-matching next-RGB trial:
- Added a compact conditional rectified-flow decoder:
  - `models/robocasa_flow_rgb.py`
  - `train/train_robocasa_flow_next_rgb.py`
  - `eval/render_robocasa_flow_rgb_rollout.py`
- Goal:
  - improve the blurry VAE visual rollout by training a flow decoder for next-frame prediction.
  - keep the VAE encoder/dynamics/evaluator frozen so the existing correlation result is not disturbed.
- Trial A: noise -> next RGB.
  - run: `runs/robocasa/world_evaluator/flow_next_rgb_opendrawer_task0_smoke`
  - best step: 200
  - validation sampled PSNR: 11.31 dB
  - result: rejected; much worse than the VAE decoder baseline.
- Trial B: residual flow from VAE predicted RGB -> next RGB.
  - run: `runs/robocasa/world_evaluator/flow_next_rgb_opendrawer_task0_residual`
  - best step: 100
  - best validation sampled PSNR: 14.33 dB
  - final validation sampled PSNR: 14.16 dB
  - result: rejected; close to but still below the VAE decoder baseline of 14.62 dB.
- Visual comparison:
  - `runs/robocasa/world_evaluator/flow_next_rgb_opendrawer_task0_residual/real_vs_vae_vs_flow_exp034_ep87.mp4`

Interpretation:
- Flow matching did not improve reconstruction quality in this small setup.
- Residual flow is much better than noise flow, but it mostly preserves the VAE blur and sometimes adds noise/contrast rather than useful detail.
- A likely next attempt is not a tiny pixel-space flow head, but a stronger spatial decoder: skip-connected latent U-Net, patch-token diffusion/flow over VQ tokens, or training the encoder/decoder and dynamics jointly on multi-step video prediction.

v0.4 RGB refiner trial:
- Added a conditional U-Net-style deterministic refiner:
  - `models/robocasa_rgb_refiner.py`
  - `train/train_robocasa_rgb_refiner.py`
  - `eval/render_robocasa_rgb_refiner_rollout.py`
- Setup:
  - frozen VAE encoder/dynamics/decoder
  - input: VAE predicted next RGB, predicted next latent, normalized action, task id
  - output: residual-refined next RGB
- Run:
  - `runs/robocasa/world_evaluator/rgb_refiner_opendrawer_task0`
  - train samples: 957
  - val samples: 158
  - train time: 325.7 sec
  - refiner size: 3.55M params
  - VAE + refiner size: 7.02M params
- Result:
  - prior VAE next-frame PSNR on the same validation target: 14.62 dB
  - best refiner PSNR: 14.67 dB at step 100
  - final refiner PSNR: 14.35 dB at step 800
  - accepted as a marginal early-stopped visual improvement, but not a strong qualitative improvement.
- Visual comparison:
  - `runs/robocasa/world_evaluator/rgb_refiner_opendrawer_task0/real_vs_vae_vs_refiner_exp034_ep87.mp4`

Interpretation:
- A stronger spatial decoder can improve the metric, but the improvement is tiny with the current data/model setup.
- The refiner overfits quickly: train loss keeps improving while validation PSNR degrades after step 100.
- Better next steps are early stopping by default, stronger augmentation, more video data, or joint multi-step latent/video training rather than only refining one-step predicted frames.

v0.5 scale and data-use trial:
- Question:
  - Does visual world-model quality improve from scaling model size and using more available video data?
- Previous baseline:
  - run: `runs/robocasa/world_evaluator/vae_opendrawer_task0_smoke`
  - data: OpenDrawer RoboCasa task-index 0 only
  - train samples: 957
  - model: latent_dim 128, width 256
  - size: 3.47M params, 13.9 MB fp32
  - validation PSNR: 14.62 dB
- Scale-only trial:
  - run: `runs/robocasa/world_evaluator/vae_opendrawer_task0_scaled_w512_z256`
  - data: same OpenDrawer task-index 0 subset
  - model: latent_dim 256, width 512
  - observed peak: 15.20 dB at step 200 in one 600-step run
  - repeated 200-step run: 14.62 dB
  - interpretation: scale alone was not robust on the small subset.
- Scale + more OpenDrawer data:
  - run: `runs/robocasa/world_evaluator/vae_opendrawer_allvariants_scaled_w512_z256`
  - data: all OpenDrawer variants, not just task-index 0
  - train samples: 2433
  - val samples: 158
  - model: latent_dim 256, width 512
  - size: 6.33M params, 25.3 MB fp32
  - validation PSNR: 15.09 dB
  - prior task-index-0 baseline PSNR: 14.62 dB
  - accepted as a real visual-quality improvement.
- Visual comparison:
  - `runs/robocasa/world_evaluator/vae_opendrawer_allvariants_scaled_w512_z256/real_vs_smallvae_vs_scaledvae_exp034_ep87.mp4`

Data-use status:
- Local RoboCasa-5 manifest contains 537 demos total:
  - OpenDrawer: 102 demos
  - CloseDrawer: 110 demos
  - PickPlaceCounterToStove: 108 demos
  - TurnOffStove: 109 demos
  - PickPlaceCounterToCabinet: 108 demos
- The accepted v0.5 run uses all OpenDrawer demos except the held-out validation episodes.
- It still does not use the full RoboCasa-5 pool of 537 demos.
- Next scaling step should train on all five tasks with task conditioning, then report both aggregate PSNR and OpenDrawer-specific held-out PSNR.

v0.6 RoboCasa-5 all-task VAE:
- Goal:
  - train the same scaled VAE on all five local RoboCasa tasks instead of only OpenDrawer.
- Run:
  - `runs/robocasa/world_evaluator/vae_robocasa5_all_scaled_w512_z256`
  - data: all five RoboCasa-5 datasets
  - held out: episodes 87/92/93/94/98/100/101 per task
  - train samples: 13,824
  - val samples: 985
  - model: latent_dim 256, width 512
  - task conditioning: raw RoboCasa task_index embedding, 68 ids
  - size: 6.33M params, 25.3 MB fp32
  - train time: 40.4 sec
- Aggregate validation:
  - PSNR: 15.70 dB
  - RGB MSE: 0.0269
  - progress MAE: 0.1406
  - success-label accuracy: 0.912
- Per-task validation PSNR:
  - OpenDrawer: 16.23 dB, 158 val samples
  - CloseDrawer: 15.98 dB, 162 val samples
  - PickPlaceCounterToStove: 15.24 dB, 204 val samples
  - TurnOffStove: 16.35 dB, 270 val samples
  - PickPlaceCounterToCabinet: 14.80 dB, 191 val samples
- Visual comparison:
  - `runs/robocasa/world_evaluator/vae_robocasa5_all_scaled_w512_z256/real_vs_vae_scaling_data_exp034_ep87.mp4`

Interpretation:
- Using all five tasks is the strongest visual result so far.
- OpenDrawer held-out PSNR improves from 15.09 dB in the OpenDrawer-only scaled run to 16.23 dB in the all-task run.
- Multi-task video data helps more than the earlier flow/refiner heads.
- Visuals are still blurry, so the next meaningful improvement should target spatial detail: larger image resolution, patch/VQ-token prediction, skip-connected decoder, or multi-step video training.

v0.7 additional scaling attempts:
- Question:
  - After the RoboCasa-5 data win, does simply scaling temporal density or model size improve further?
- Current best before these runs:
  - run: `runs/robocasa/world_evaluator/vae_robocasa5_all_scaled_w512_z256`
  - frame stride: 8
  - model: latent_dim 256, width 512
  - size: 6.33M params
  - train samples: 13,824
  - val samples: 985
  - aggregate PSNR: 15.70 dB
- Attempt A: more temporal data + larger model.
  - run: `runs/robocasa/world_evaluator/vae_robocasa5_stride4_w768_z512`
  - frame stride: 4
  - model: latent_dim 512, width 768
  - size: 10.43M params
  - train samples: 27,397
  - val samples: 1,954
  - train time: 71.8 sec
  - aggregate PSNR: 15.55 dB
  - result: rejected; more temporal samples did not beat the stride-8 baseline.
- Attempt B: larger model only.
  - run: `runs/robocasa/world_evaluator/vae_robocasa5_stride8_w1024_z512`
  - frame stride: 8
  - model: latent_dim 512, width 1024
  - size: 14.01M params
  - train samples: 13,824
  - val samples: 985
  - train time: 67.1 sec
  - aggregate PSNR: 15.46 dB
  - result: rejected; larger capacity alone did not beat the 6.33M baseline.

Interpretation:
- The scaling law is not monotonic in this tiny setup.
- Multi-task data diversity was high leverage; denser adjacent frames and more parameters were not.
- Current bottleneck is likely decoder/object-detail modeling, objective choice, or resolution, not raw VAE parameter count.
- Practical next step before more sweeps: cache preprocessed transition tensors so experiments do not repeatedly decode all videos.

v0.8 longer training trial:
- Question:
  - Was the RoboCasa-5 VAE undertrained at 500 steps?
- Run:
  - `runs/robocasa/world_evaluator/vae_robocasa5_all_scaled_w512_z256_5min`
  - same config as current best v0.6:
    - frame stride: 8
    - latent_dim: 256
    - width: 512
    - train samples: 13,824
    - val samples: 985
  - steps: 4000
  - train time: 283.6 sec
- PSNR checkpoints:
  - step 500: 15.55 dB
  - step 1000: 15.42 dB
  - step 1500: 15.19 dB
  - step 2000: 15.24 dB
  - step 2500: 15.12 dB
  - step 3000: 15.04 dB
  - step 3500: 15.07 dB
  - step 4000: 14.88 dB
- Result:
  - rejected for visual quality.
  - prior v0.6 best at 500 steps remains better: 15.70 dB.

Interpretation:
- Longer training lowered train loss and improved weighted validation loss until around step 1000, but visual PSNR peaked early and then decayed.
- The current weighted objective is not aligned enough with visual reconstruction quality for long training.
- Next visual-quality run should use PSNR/reconstruction-based checkpoint selection, higher reconstruction weight, or a staged objective: first train VAE reconstruction, then add dynamics/progress/success heads.

v0.9 narrower holdout / more-training-data split:
- Question:
  - Is the 7-episode validation split too far out of distribution, and can we improve by moving most held-out episodes into training?
- Run:
  - `runs/robocasa/world_evaluator/vae_robocasa5_moretrain_holdout101_w512_z256`
  - same model/config as v0.6:
    - frame stride: 8
    - latent_dim: 256
    - width: 512
  - validation: only episode 101 per task
  - added to train: former held-out episodes 87/92/93/94/98/100 per task
  - train samples: 14,688
  - val samples: 121
  - train time: 57.2 sec
- Direct episode-101 comparison:
  - v0.6 checkpoint trained with 7 held-out episodes: 15.97 dB on episode 101
  - new holdout-101-only checkpoint: 15.60 dB on episode 101
- Result:
  - rejected.
  - adding the nearby former validation episodes to train did not improve the remaining episode-101 holdout.

Interpretation:
- The validation issue is not solved by simply moving more local held-out episodes into training.
- The stronger hypothesis is objective/checkpoint mismatch or insufficient visual decoder detail, not only validation distribution shift.
- Next data-side improvement should use genuinely new demonstrations/tasks/views, not just reshuffling this same local 537-demo pool.

v0.10 held-out BC-5 policy correlation smoke:
- Question:
  - Does the world-model evaluator correlate with real sim success on held-out policy candidates across the five RoboCasa seed tasks, rather than fitting the same OpenDrawer trace frontier?
- Implementation:
  - `tasks/robocasa_bc5/eval.py` can now save action traces with `--trace-dir`.
  - `tasks/world_model_evaluator/eval.py` can score mixed-task archives and uses `split=test` as the primary metric when present.
  - `tasks/world_model_evaluator/build_archive.py` builds split-labeled candidate archives from traced eval JSONs.
- Archive:
  - `runs/autorobobench/world_model_evaluator/bc5_heldout_traces/archive_bc5_heldout.jsonl`
  - 5 tasks: OpenDrawer, CloseDrawer, PickPlaceCounterToStove, TurnOffStove, PickPlaceCounterToCabinet.
  - 4 candidate policies, 40 total traces.
  - held-out test split: 3 candidates x 5 tasks x 2 eval episodes/task.
  - held-out sim success rates: 0.0, 0.1, 0.2.
- Best zero-shot evaluator tested:
  - checkpoint: `runs/robocasa/world_evaluator/vae_robocasa5_all_scaled_w512_z256_5min/vae_world_model_best.pt`
  - readout: deterministic 260-step trace replay, learned score = `1 - raw`.
  - held-out test Spearman: 1.000
  - held-out test Pearson: 0.899
  - speedup proxy: 22.3x
- Important caveat:
  - This is a smoke benchmark, not final evidence. The test split has only 3 policy candidates and 10 eval episodes/candidate.
  - Next step is scale to at least 10-20 held-out policies and 10 eval episodes/task before making claims.

v0.11 RoboCasa-only evaluator baselines:
- Question:
  - If we drop DROID/external video and use only RoboCasa data, what are reasonable world-model evaluator baselines and held-out correlation scores?
- Archive:
  - `runs/autorobobench/world_model_evaluator/bc5_heldout_traces/archive_bc5_heldout.jsonl`
  - 5 tasks, 4 total candidate policies, 3 held-out test policies.
  - Test sim success rates: 0.0, 0.1, 0.2.
- Baselines:

| evaluator | training | readout | test Spearman | test Pearson | speedup |
|---|---:|---|---:|---:|---:|
| VAE w512 z256 500-step | RoboCasa-5 VAE | 1-raw | 0.000 | 0.000 | 22.0x |
| VAE w512 z256 5-min | RoboCasa-5 VAE | 1-raw | 1.000 | 0.899 | 22.3x |
| VAE w768 z512 stride4 | RoboCasa-5 VAE | raw | -0.500 | -0.736 | 18.8x |
| VAE w768 z512 stride4 | RoboCasa-5 VAE | 1-raw | 0.500 | 0.736 | 20.4x |
| VAE w1024 z512 stride8 | RoboCasa-5 VAE | 1-raw | 0.500 | 0.000 | 19.8x |
| mixed trace calibrated w512 z256 | train split traces only | 1-raw | 1.000 | 0.875 | 22.1x |

- New trainer:
  - `train/train_robocasa_mixed_trace_calibrator.py`
  - trains on `split=train` / `split=calibration` traces only.
  - supports mixed task aliases by reading dataset roots from `data/robocasa5/manifest.json`.
- Best current RoboCasa-only baseline:
  - `runs/robocasa/world_evaluator/vae_robocasa5_all_scaled_w512_z256_5min/vae_world_model_best.pt`
  - test Spearman 1.000, Pearson 0.899 on the current small held-out split.
- Caveat:
  - The split is still too small. This gives us a good implementation baseline, not yet a robust scientific result.

v0.12 scaled RoboCasa-only held-out archive:
- Goal:
  - Re-test world-model evaluator correlation on a larger RoboCasa-only archive with more held-out policy candidates and less noisy real-sim labels.
- Archive:
  - `runs/autorobobench/world_model_evaluator/bc5_heldout_scaled/archive.jsonl`
  - 12 candidates total.
  - Held-out test split: 10 learned policies.
  - Calibration split: 2 trajectory-oracle policies.
  - Real sim labels: 10 episodes per task x 5 tasks = 50 traces per candidate.
- Held-out learned-policy sim success:

| candidate | split | real success |
|---|---|---:|
| seqflow_weak | test | 0.02 |
| starter_bc5 | test | 0.10 |
| history_act | test | 0.06 |
| rgb16_train80 | test | 0.18 |
| rgb16_trainval90 | test | 0.18 |
| histflow_3min | test | 0.02 |
| histflow_5min | test | 0.04 |
| histflow_progress | test | 0.02 |
| seqflow_residual | test | 0.04 |
| actinit_res005 | test | 0.08 |
| trajectory_oracle_exact | calibration | 0.78 |
| trajectory_oracle_all | calibration | 0.78 |

- Evaluator results on the 10 held-out test candidates:

| evaluator | readout | test Spearman | test Pearson | top-5 hit | speedup |
|---|---|---:|---:|---:|---:|
| VAE w512 z256 5-min | 1-raw | -0.562 | -0.902 | 0.0 | 27.8x |
| VAE w768 z512 stride4 | 1-raw | 0.840 | 0.928 | 1.0 | 27.2x |
| VAE w1024 z512 stride8 | 1-raw | -0.599 | -0.712 | 0.0 | 26.7x |
| mixed calibrated from stride4 | 1-raw | 0.272 | 0.253 | 1.0 | 28.2x |

- Best current baseline:
  - `runs/robocasa/world_evaluator/vae_robocasa5_stride4_w768_z512/vae_world_model_best.pt`
  - test Spearman: 0.840
  - test Pearson: 0.928
  - top-5 hit: 1.0
  - speedup proxy: 27.2x
  - plot: `runs/autorobobench/world_model_evaluator/bc5_heldout_scaled/world_corr_vae_robocasa5_stride4_w768_z512.svg`

Interpretation:
- The prior small-archive result was not robust: the w512/z256 5-minute evaluator flips negative on the scaled archive.
- The stride-4 w768/z512 VAE is the first robust RoboCasa-only evaluator baseline on this larger held-out split.
- Calibration on only two oracle candidates overfits the high-success anchors and hurts held-out ranking, even though it preserves top-5 hit rate.
- The useful baseline is therefore zero-shot trace scoring from the stride-4 VAE, not the oracle-calibrated head.
- Next step:
  - Add more diverse calibration candidates, not only high-success oracles.
  - Score proposed policy traces before full sim validation, then run RoboCasa only on the top-ranked candidates.
