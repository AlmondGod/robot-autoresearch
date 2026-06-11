from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.libero_dataset import load_paired_npz, load_video_npz, tokenize_instruction
from models.inverse_dynamics import TinyInverseDynamics
from models.tokenizer import TinyVQTokenizer, images_to_tensor
from train.common import device_from_arg, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-data", default="data/libero_object5/libero_object5_video.npz")
    parser.add_argument("--paired-data", default="data/libero_object5/libero_object5_paired.npz")
    parser.add_argument("--manifest", default="data/libero_object5/manifest.json")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--inverse", required=True)
    parser.add_argument("--out-dir", default="runs/libero/dreamzero_pseudo")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    video = load_video_npz(Path(args.video_data), split="train")
    paired = load_paired_npz(Path(args.paired_data), split="train")
    manifest = json.loads(Path(args.manifest).read_text())
    task_names = {int(task["task_id"]): task["task_name"] for task in manifest.get("video_tasks", manifest["tasks"])}

    tok_ckpt = torch.load(args.tokenizer, map_location=device)
    tokenizer = TinyVQTokenizer(tok_ckpt["codebook_size"], tok_ckpt["embed_dim"]).to(device)
    tokenizer.load_state_dict(tok_ckpt["state_dict"])
    tokenizer.eval()

    inv_ckpt = torch.load(args.inverse, map_location=device)
    action_dim = int(inv_ckpt["action_dim"])
    proprio_dim = int(inv_ckpt["proprio_dim"])
    inverse = TinyInverseDynamics(
        vocab_size=int(inv_ckpt["vocab_size"]),
        action_dim=action_dim,
        proprio_dim=proprio_dim,
    ).to(device)
    inverse.load_state_dict(inv_ckpt["state_dict"])
    inverse.eval()

    task_count = int(np.asarray(paired["task_id"]).max()) + 1
    video_task_id = np.asarray(video["task_id"][:], dtype=np.int64)
    selected = np.flatnonzero(video_task_id < task_count)[: int(args.max_samples)]
    count = len(selected)
    if count == 0:
        raise ValueError("no video samples matched the paired task ID range")
    proprio_mean = np.asarray(paired["proprio"], dtype=np.float32).reshape(-1, proprio_dim).mean(axis=0)

    frames_out = []
    wrist_out = []
    next_out = []
    proprio_out = []
    actions_out = []
    task_out = []
    instruction_out = []

    with torch.no_grad():
        for start in range(0, count, args.batch_size):
            end = min(count, start + args.batch_size)
            idx = selected[start:end]
            frames = np.asarray(video["frames"][idx])
            wrist = np.asarray(video.get("wrist_frames", video["frames"])[idx])
            next_frames = np.asarray(video["next_frames"][idx])
            task_id = np.asarray(video["task_id"][idx], dtype=np.int64)
            proprio = np.repeat(proprio_mean[None], end - start, axis=0).astype(np.float32)

            z = tokenizer.encode_indices(images_to_tensor(frames).to(device)).reshape(end - start, -1)
            z_next = tokenizer.encode_indices(images_to_tensor(next_frames).to(device)).reshape(end - start, -1)
            prop_t = torch.as_tensor(proprio, dtype=torch.float32, device=device)
            task_t = torch.as_tensor(task_id, dtype=torch.long, device=device)
            pred, _ = inverse(z, z_next, prop_t, task_t)
            action = pred.cpu().numpy().astype(np.float32)

            frames_out.append(np.repeat(frames[:, None], args.history, axis=1))
            wrist_out.append(np.repeat(wrist[:, None], args.history, axis=1))
            next_out.append(next_frames)
            proprio_out.append(np.repeat(proprio[:, None], args.history, axis=1))
            actions_out.append(np.repeat(action[:, None], args.action_horizon, axis=1))
            task_out.append(task_id)
            instruction_out.append(
                np.stack([tokenize_instruction(task_names.get(int(tid), f"task_{int(tid)}")) for tid in task_id], axis=0)
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pseudo_paired.npz"
    split = np.full(count, "train")
    np.savez_compressed(
        out_path,
        frames=np.concatenate(frames_out, axis=0),
        wrist_frames=np.concatenate(wrist_out, axis=0),
        next_frames=np.concatenate(next_out, axis=0),
        proprio=np.concatenate(proprio_out, axis=0),
        actions=np.concatenate(actions_out, axis=0),
        task_id=np.concatenate(task_out, axis=0),
        instruction_tokens=np.concatenate(instruction_out, axis=0),
        split=split,
    )
    write_metrics(
        out_dir,
        {
            "pseudo_path": str(out_path),
            "pseudo_samples": count,
            "method": "dreamzero_pseudo",
            "checkpoint": str(out_path),
        },
    )
    print(out_path)


if __name__ == "__main__":
    main()
