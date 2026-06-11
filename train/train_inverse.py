from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from data.libero_dataset import load_paired_npz
from models.inverse_dynamics import TinyInverseDynamics
from models.tokenizer import TinyVQTokenizer, images_to_tensor
from train.common import batches, device_from_arg, save_checkpoint, write_metrics


def _last_step(values):
    return values[:, -1] if getattr(values, "ndim", 0) == 3 else values


def _first_action(values):
    return values[:, 0] if getattr(values, "ndim", 0) == 3 else values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/libero_object5/libero_object5_paired.npz")
    parser.add_argument("--tokenizer", default="runs/libero/tokenizer/tokenizer.pt")
    parser.add_argument("--out-dir", default="runs/libero/inverse")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    tok_ckpt = torch.load(args.tokenizer, map_location=device)
    tokenizer = TinyVQTokenizer(tok_ckpt["codebook_size"], tok_ckpt["embed_dim"]).to(device)
    tokenizer.load_state_dict(tok_ckpt["state_dict"])
    tokenizer.eval()
    train = load_paired_npz(Path(args.data), split="train")
    val = load_paired_npz(Path(args.data), split="val")
    action_dim = int(train["actions"].shape[-1])
    proprio_dim = int(train["proprio"].shape[-1])
    model = TinyInverseDynamics(tok_ckpt["codebook_size"], action_dim=action_dim, proprio_dim=proprio_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    started = time.time()
    last_loss = None
    for step, idx in enumerate(batches(len(train["frames"]), args.batch_size, args.steps), start=1):
        with torch.no_grad():
            z = tokenizer.encode_indices(images_to_tensor(train["frames"][idx]).to(device)).reshape(len(idx), -1)
            z_next = tokenizer.encode_indices(images_to_tensor(train["next_frames"][idx]).to(device)).reshape(len(idx), -1)
        proprio = torch.as_tensor(_last_step(train["proprio"][idx]), dtype=torch.float32, device=device)
        actions = torch.as_tensor(_first_action(train["actions"][idx]), dtype=torch.float32, device=device)
        task_id = torch.as_tensor(train["task_id"][idx], dtype=torch.long, device=device)
        _, loss = model(z, z_next, proprio, task_id, actions=actions)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
            print(f"step={step} inverse_loss={last_loss:.6f} elapsed_s={time.time() - started:.1f}", flush=True)
        if args.max_train_seconds > 0 and time.time() - started >= args.max_train_seconds:
            print(f"stopping_at_step={step} elapsed_s={time.time() - started:.1f}", flush=True)
            break

    with torch.no_grad():
        n = min(len(val["frames"]), 256)
        z = tokenizer.encode_indices(images_to_tensor(val["frames"][:n]).to(device)).reshape(n, -1)
        z_next = tokenizer.encode_indices(images_to_tensor(val["next_frames"][:n]).to(device)).reshape(n, -1)
        proprio = torch.as_tensor(_last_step(val["proprio"][:n]), dtype=torch.float32, device=device)
        actions = torch.as_tensor(_first_action(val["actions"][:n]), dtype=torch.float32, device=device)
        task_id = torch.as_tensor(val["task_id"][:n], dtype=torch.long, device=device)
        _, val_loss = model(z, z_next, proprio, task_id, actions=actions)

    out_dir = Path(args.out_dir)
    ckpt = save_checkpoint(
        out_dir,
        "inverse.pt",
        model,
        {"vocab_size": tok_ckpt["codebook_size"], "action_dim": action_dim, "proprio_dim": proprio_dim},
    )
    write_metrics(
        out_dir,
        {
            "action_mse": float(val_loss.cpu()),
            "last_train_loss": last_loss,
            "checkpoint": str(ckpt),
            "train_seconds": time.time() - started,
        },
    )
    print(out_dir / "metrics.json")


if __name__ == "__main__":
    main()
