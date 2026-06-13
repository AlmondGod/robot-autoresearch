# Finding: Multi-Task Video Data Helped the World Model

Question:
- For a tiny RoboCasa visual world model, is it better to add small architecture heads or simply train on more related robot video?

Minimal setup:
- Model: same scaled VAE world model.
- Architecture: 6.33M params, latent_dim 256, width 512.
- Input: two 64x64 RGB views, proprio, action, task id.
- Output: reconstructed / predicted RGB latent rollout frames.
- Metric: held-out RGB PSNR.
- Held-out split: episodes 87/92/93/94/98/100/101.

Results:

| Training data | Train samples | Val samples | OpenDrawer PSNR | Aggregate PSNR |
| --- | ---: | ---: | ---: | ---: |
| OpenDrawer task-index 0 only | 957 | 158 | 14.62 dB | - |
| All OpenDrawer variants | 2,433 | 158 | 15.09 dB | - |
| RoboCasa-5 all tasks | 13,824 | 985 | 16.23 dB | 15.70 dB |

Per-task PSNR for RoboCasa-5:
- OpenDrawer: 16.23 dB
- CloseDrawer: 15.98 dB
- PickPlaceCounterToStove: 15.24 dB
- TurnOffStove: 16.35 dB
- PickPlaceCounterToCabinet: 14.80 dB

Interpretation:
- Multi-task robot video data improved the visual model more than the flow/refiner heads.
- OpenDrawer improved by +1.61 dB over the original small-data baseline.
- The setup was intentionally minimal: same VAE family, same resolution, same training code, just more related task video plus task conditioning.
- The videos are still blurry, but the direction is clear: data diversity is currently the highest-leverage improvement.

Takeaway:
- Before adding complex world-model machinery, scale the offline robot video corpus across related manipulation tasks.
