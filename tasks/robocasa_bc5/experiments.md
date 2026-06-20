# RoboCasa BC5 Experiments

## SmolVLM2 frozen vision backbone

Date: 2026-06-20

Question: does `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` improve BC5 when used as a frozen VLM backbone for the existing chunked BC/flow policy?

Setup:
- Data: 5 RoboCasa BC5 tasks, 4 train demos/task, 2 val demos/task.
- Training: 5 minute action-head budget, `chunk_horizon=16`, `frame_stride=2`, `width=256`, `action_depth=3`.
- Eval: normalized validation action MSE plus 1 closed-loop episode/task locally.

| Backbone | Frozen params | Trainable params | Cache train/val | Best val MSE | Final val MSE | Closed-loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolVLM2-500M vision tower | 507.5M | 7.7M | 22.3s / 10.3s | 0.4785 | 0.4941 | 1/5 |
| CLIP ViT-B/32 | 151.3M | 7.0M | 7.1s / 2.4s | 0.4921 | 0.4990 | 1/5 |

Result: SmolVLM2 gave a small offline MSE improvement over CLIP, but did not improve the tiny closed-loop sample; both solved only `CloseDrawer`.

Follow-up closed-loop eval with 10 episodes/task:

| Backbone | OpenDrawer | CloseDrawer | PickPlaceCounterToStove | TurnOffStove | PickPlaceCounterToCabinet | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolVLM2-500M vision tower | 0/10 | 6/10 | 0/10 | 0/10 | 0/10 | 6/50 |
| CLIP ViT-B/32 | 0/10 | 6/10 | 0/10 | 0/10 | 0/10 | 6/50 |

Conclusion: the larger SmolVLM2 frozen vision tower improves validation MSE slightly, but this did not transfer to measured closed-loop success under the 5-minute training budget. The current benchmark signal is dominated by `CloseDrawer`; the other four tasks remain unsolved by both policies.
