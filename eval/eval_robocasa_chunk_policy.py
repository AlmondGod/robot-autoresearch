from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eval.render_robocasa_chunk_policy import _rollout_closed_loop
from eval.train_temporal_chunk_bc_robocasa import RoboCasaTemporalChunkBC
from train.common import device_from_arg

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-id", action="append", type=int, required=True)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--commit-steps", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    checkpoint = torch.load(args.policy, map_location=device, weights_only=False)
    model = RoboCasaTemporalChunkBC(
        proprio_dim=int(checkpoint["proprio_dim"]),
        chunk_horizon=int(checkpoint["chunk_horizon"]),
        action_dim=int(checkpoint["action_dim"]),
        task_count=int(checkpoint["task_count"]),
        width=int(checkpoint.get("width", 512)),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    details = []
    for episode_id in args.episode_id:
        _, success, steps = _rollout_closed_loop(
            dataset_root=Path(args.dataset_root),
            episode_idx=int(episode_id),
            model=model,
            checkpoint=checkpoint,
            device=device,
            camera=str(args.camera),
            width=64,
            height=64,
            max_steps=int(args.max_steps),
            commit_steps=int(args.commit_steps),
            clip_actions=True,
        )
        details.append({"episode_id": int(episode_id), "success": bool(success), "steps": int(steps)})
        print(json.dumps(details[-1]), flush=True)

    payload = {
        "policy": args.policy,
        "episodes": len(details),
        "successes": sum(int(row["success"]) for row in details),
        "success_rate": sum(int(row["success"]) for row in details) / max(1, len(details)),
        "commit_steps": int(args.commit_steps),
        "details": details,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
