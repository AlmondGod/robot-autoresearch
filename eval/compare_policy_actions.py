from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data.libero_dataset import load_paired_npz
from models.policy import TinyBCPolicy
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--data", default="data/libero_easy5/libero_easy5_paired.npz")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    checkpoint = torch.load(args.policy, map_location=device)
    data = load_paired_npz(Path(args.data), split=args.split)
    n = min(args.max_samples, len(data["frames"]))
    if n <= 0:
        raise ValueError(f"no samples found for split {args.split}")

    policy = TinyBCPolicy(
        action_dim=int(checkpoint["action_dim"]),
        proprio_dim=int(checkpoint["proprio_dim"]),
        n_embd=int(checkpoint.get("n_embd", 128)),
        action_horizon=int(checkpoint.get("action_horizon", 1)),
        max_history=max(int(checkpoint.get("history", 1)), 1),
        policy_kind=str(checkpoint.get("policy_kind", "bc")),
        flow_steps=int(checkpoint.get("flow_steps", 8)),
    ).to(device)
    policy.load_state_dict(checkpoint["state_dict"], strict=False)
    policy.eval()

    images = torch.as_tensor(data["frames"][:n], dtype=torch.float32, device=device)
    wrist = torch.as_tensor(data.get("wrist_frames", data["frames"])[:n], dtype=torch.float32, device=device)
    proprio = torch.as_tensor(data["proprio"][:n], dtype=torch.float32, device=device)
    proprio = (proprio - _ckpt_tensor(checkpoint, "proprio_mean", device)) / _ckpt_tensor(checkpoint, "proprio_std", device)
    task_id = torch.as_tensor(data["task_id"][:n], dtype=torch.long, device=device)
    instruction = torch.as_tensor(data["instruction_tokens"][:n], dtype=torch.long, device=device)

    with torch.no_grad():
        if str(checkpoint.get("policy_kind", "bc")) == "flow":
            pred_norm = policy.sample_flow(
                images,
                proprio,
                task_id,
                wrist_images=wrist,
                instruction_tokens=instruction,
                steps=int(checkpoint.get("flow_steps", 8)),
            )
        else:
            pred_norm, _ = policy(images, proprio, task_id, wrist_images=wrist, instruction_tokens=instruction)

    action_mean = _ckpt_tensor(checkpoint, "action_mean", device)
    action_std = _ckpt_tensor(checkpoint, "action_std", device)
    pred = (pred_norm * action_std + action_mean).cpu().numpy()
    actual = np.asarray(data["actions"][:n], dtype=np.float32)
    err = pred - actual
    mse_by_dim = (err.reshape(-1, err.shape[-1]) ** 2).mean(axis=0)
    mae_by_dim = np.abs(err.reshape(-1, err.shape[-1])).mean(axis=0)
    pred_std = pred.reshape(-1, pred.shape[-1]).std(axis=0)
    actual_std = actual.reshape(-1, actual.shape[-1]).std(axis=0)

    out_path = Path(args.out or Path(args.policy).with_name(f"action_compare_{args.split}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": n,
        "split": args.split,
        "mse": float((err**2).mean()),
        "mae": float(np.abs(err).mean()),
        "mse_by_dim": mse_by_dim.tolist(),
        "mae_by_dim": mae_by_dim.tolist(),
        "pred_std_by_dim": pred_std.tolist(),
        "actual_std_by_dim": actual_std.tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_preview(out_path.with_suffix(".csv"), pred, actual, max_rows=min(32, n))
    print(json.dumps({"out": str(out_path), **payload}, indent=2, sort_keys=True))


def _ckpt_tensor(checkpoint: dict, key: str, device: torch.device) -> torch.Tensor:
    value = checkpoint[key]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device, dtype=torch.float32)


def _write_preview(path: Path, pred: np.ndarray, actual: np.ndarray, max_rows: int) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        dim = pred.shape[-1]
        writer.writerow(["sample", "horizon", *[f"pred_{i}" for i in range(dim)], *[f"actual_{i}" for i in range(dim)]])
        for sample_idx in range(max_rows):
            for horizon_idx in range(pred.shape[1]):
                writer.writerow([sample_idx, horizon_idx, *pred[sample_idx, horizon_idx], *actual[sample_idx, horizon_idx]])


if __name__ == "__main__":
    main()
