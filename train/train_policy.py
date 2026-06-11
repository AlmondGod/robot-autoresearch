from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from data.libero_dataset import load_paired_npz
from models.policy import TinyBCPolicy
from train.common import batches, device_from_arg, save_checkpoint, write_metrics


def _normalization(values) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    flat = tensor.reshape(-1, tensor.shape[-1])
    return flat.mean(dim=0), flat.std(dim=0).clamp_min(1e-6)


def _norm_tensor(values, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    return (tensor - mean.to(device)) / std.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/libero_object5/libero_object5_paired.npz")
    parser.add_argument("--aux-data", default="")
    parser.add_argument("--aux-weight", type=float, default=0.0)
    parser.add_argument("--out-dir", default="runs/libero/bc_policy")
    parser.add_argument("--method", default="bc")
    parser.add_argument("--policy-kind", choices=["bc", "flow"], default="bc")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--chunk-decay", type=float, default=1.0)
    parser.add_argument("--image-noise", type=float, default=0.0)
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--history-dropout", type=float, default=0.0)
    parser.add_argument("--wrist-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--flow-sigma", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    train = load_paired_npz(Path(args.data), split="train")
    val = load_paired_npz(Path(args.data), split="val")
    aux_train = load_paired_npz(Path(args.aux_data), split="train") if args.aux_data else None
    action_dim = int(train["actions"].shape[-1])
    action_horizon = int(train["actions"].shape[1]) if train["actions"].ndim == 3 else 1
    proprio_dim = int(train["proprio"].shape[-1])
    history = int(train["frames"].shape[1]) if train["frames"].ndim == 5 else 1
    proprio_mean, proprio_std = _normalization(train["proprio"])
    action_mean, action_std = _normalization(train["actions"])
    model = TinyBCPolicy(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        n_embd=args.n_embd,
        action_horizon=action_horizon,
        max_history=max(history, 1),
        policy_kind=args.policy_kind,
        flow_steps=args.flow_steps,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    aux_batches = (
        batches(len(aux_train["frames"]), args.batch_size, args.steps, seed=1)
        if aux_train is not None and args.aux_weight > 0
        else None
    )

    started = time.time()
    last_loss = None
    for step, idx in enumerate(batches(len(train["frames"]), args.batch_size, args.steps), start=1):
        images = _augment_images(torch.as_tensor(train["frames"][idx], dtype=torch.float32, device=device), args.image_noise)
        wrist_images = _augment_images(
            torch.as_tensor(train.get("wrist_frames", train["frames"])[idx], dtype=torch.float32, device=device),
            args.image_noise,
        )
        images = _drop_history(images, args.history_dropout)
        wrist_images = images if _drop_now(args.wrist_dropout) else _drop_history(wrist_images, args.history_dropout)
        proprio = _norm_tensor(train["proprio"][idx], proprio_mean, proprio_std, device)
        actions = _norm_tensor(train["actions"][idx], action_mean, action_std, device)
        if args.action_noise > 0:
            actions = actions + args.action_noise * torch.randn_like(actions)
        task_id = torch.as_tensor(train["task_id"][idx], dtype=torch.long, device=device)
        instruction_tokens = torch.as_tensor(train["instruction_tokens"][idx], dtype=torch.long, device=device)
        loss = _policy_loss(
            model,
            args.policy_kind,
            args.loss,
            args.chunk_decay,
            args.flow_sigma,
            images,
            proprio,
            task_id,
            wrist_images,
            instruction_tokens,
            actions,
        )
        if aux_train is not None and args.aux_weight > 0:
            assert aux_batches is not None
            aux_idx = next(aux_batches)
            aux_loss = _batch_loss(
                model,
                args,
                aux_train,
                aux_idx,
                proprio_mean,
                proprio_std,
                action_mean,
                action_std,
                device,
            )
            loss = loss + args.aux_weight * aux_loss
        opt.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        last_loss = float(loss.detach().cpu())
        if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
            elapsed = time.time() - started
            print(f"step={step} loss={last_loss:.6f} elapsed_s={elapsed:.1f}", flush=True)
        if args.max_train_seconds > 0 and time.time() - started >= args.max_train_seconds:
            print(f"stopping_at_step={step} elapsed_s={time.time() - started:.1f}", flush=True)
            break

    with torch.no_grad():
        n = min(len(val["frames"]), 256)
        images = torch.as_tensor(val["frames"][:n], dtype=torch.float32, device=device)
        wrist_images = torch.as_tensor(val.get("wrist_frames", val["frames"])[:n], dtype=torch.float32, device=device)
        proprio = _norm_tensor(val["proprio"][:n], proprio_mean, proprio_std, device)
        actions = _norm_tensor(val["actions"][:n], action_mean, action_std, device)
        task_id = torch.as_tensor(val["task_id"][:n], dtype=torch.long, device=device)
        instruction_tokens = torch.as_tensor(val["instruction_tokens"][:n], dtype=torch.long, device=device)
        if args.policy_kind == "flow":
            pred = model.sample_flow(
                images,
                proprio,
                task_id,
                wrist_images=wrist_images,
                instruction_tokens=instruction_tokens,
                steps=args.flow_steps,
            )
            val_loss = _chunk_loss(pred, actions, mode="mse", decay=1.0)
        else:
            _, val_loss = model(
                images,
                proprio,
                task_id,
                wrist_images=wrist_images,
                instruction_tokens=instruction_tokens,
                actions=actions,
            )

    out_dir = Path(args.out_dir)
    ckpt = save_checkpoint(
        out_dir,
        "policy.pt",
        model,
        {
            "action_dim": action_dim,
            "proprio_dim": proprio_dim,
            "action_horizon": action_horizon,
            "history": history,
            "n_embd": args.n_embd,
            "policy_kind": args.policy_kind,
            "flow_steps": args.flow_steps,
            "flow_sigma": args.flow_sigma,
            "proprio_mean": proprio_mean,
            "proprio_std": proprio_std,
            "action_mean": action_mean,
            "action_std": action_std,
        },
    )
    write_metrics(
        out_dir,
        {
            "bc_loss": float(val_loss.cpu()),
            "last_train_loss": last_loss,
            "checkpoint": str(ckpt),
            "device": str(device),
            "method": args.method,
            "policy_kind": args.policy_kind,
            "n_embd": args.n_embd,
            "loss": args.loss,
            "chunk_decay": args.chunk_decay,
            "image_noise": args.image_noise,
            "action_noise": args.action_noise,
            "history_dropout": args.history_dropout,
            "wrist_dropout": args.wrist_dropout,
            "aux_data": args.aux_data,
            "aux_weight": args.aux_weight,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "flow_steps": args.flow_steps,
            "flow_sigma": args.flow_sigma,
            "steps": args.steps,
            "train_seconds": time.time() - started,
        },
    )
    print(out_dir / "metrics.json")


def _batch_loss(
    model: TinyBCPolicy,
    args: argparse.Namespace,
    data: dict,
    idx,
    proprio_mean: torch.Tensor,
    proprio_std: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    images = _augment_images(torch.as_tensor(data["frames"][idx], dtype=torch.float32, device=device), args.image_noise)
    wrist_images = _augment_images(
        torch.as_tensor(data.get("wrist_frames", data["frames"])[idx], dtype=torch.float32, device=device),
        args.image_noise,
    )
    images = _drop_history(images, args.history_dropout)
    wrist_images = images if _drop_now(args.wrist_dropout) else _drop_history(wrist_images, args.history_dropout)
    proprio = _norm_tensor(data["proprio"][idx], proprio_mean, proprio_std, device)
    actions = _norm_tensor(data["actions"][idx], action_mean, action_std, device)
    if args.action_noise > 0:
        actions = actions + args.action_noise * torch.randn_like(actions)
    task_id = torch.as_tensor(data["task_id"][idx], dtype=torch.long, device=device)
    instruction_tokens = torch.as_tensor(data["instruction_tokens"][idx], dtype=torch.long, device=device)
    return _policy_loss(
        model,
        args.policy_kind,
        args.loss,
        args.chunk_decay,
        args.flow_sigma,
        images,
        proprio,
        task_id,
        wrist_images,
        instruction_tokens,
        actions,
    )


def _policy_loss(
    model: TinyBCPolicy,
    policy_kind: str,
    loss_mode: str,
    chunk_decay: float,
    flow_sigma: float,
    images: torch.Tensor,
    proprio: torch.Tensor,
    task_id: torch.Tensor,
    wrist_images: torch.Tensor,
    instruction_tokens: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    if policy_kind == "flow":
        return _flow_matching_loss(
            model,
            images,
            proprio,
            task_id,
            wrist_images,
            instruction_tokens,
            actions,
            sigma=flow_sigma,
        )
    pred, _ = model(
        images,
        proprio,
        task_id,
        wrist_images=wrist_images,
        instruction_tokens=instruction_tokens,
    )
    return _chunk_loss(pred, actions, mode=loss_mode, decay=chunk_decay)


def _chunk_loss(pred: torch.Tensor, actions: torch.Tensor, mode: str, decay: float) -> torch.Tensor:
    actions = actions.unsqueeze(1) if actions.ndim == 2 else actions
    horizon = min(actions.shape[1], pred.shape[1])
    if mode == "huber":
        per = torch.nn.functional.smooth_l1_loss(pred[:, :horizon], actions[:, :horizon], reduction="none")
    else:
        per = torch.nn.functional.mse_loss(pred[:, :horizon], actions[:, :horizon], reduction="none")
    if decay != 1.0:
        weights = torch.as_tensor([decay**idx for idx in range(horizon)], dtype=per.dtype, device=per.device)
        weights = weights / weights.mean().clamp_min(1e-6)
        per = per * weights.reshape(1, horizon, 1)
    return per.mean()


def _flow_matching_loss(
    model: TinyBCPolicy,
    images: torch.Tensor,
    proprio: torch.Tensor,
    task_id: torch.Tensor,
    wrist_images: torch.Tensor,
    instruction_tokens: torch.Tensor,
    actions: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    actions = actions.unsqueeze(1) if actions.ndim == 2 else actions
    noise = torch.randn_like(actions) * sigma
    t = torch.rand((actions.shape[0],), dtype=actions.dtype, device=actions.device)
    view_t = t.reshape(-1, 1, 1)
    action_t = (1.0 - view_t) * noise + view_t * actions
    target_velocity = actions - noise
    obs_h = model.encode_obs(
        images,
        proprio,
        task_id,
        wrist_images=wrist_images,
        instruction_tokens=instruction_tokens,
    )
    pred_velocity = model.flow_velocity(obs_h, action_t, t)
    return torch.nn.functional.mse_loss(pred_velocity, target_velocity)


def _augment_images(images: torch.Tensor, noise: float) -> torch.Tensor:
    if noise <= 0:
        return images
    images = images + torch.randn_like(images) * (255.0 * noise)
    return images.clamp(0.0, 255.0)


def _drop_history(images: torch.Tensor, prob: float) -> torch.Tensor:
    if prob <= 0 or images.ndim != 5 or images.shape[1] <= 1:
        return images
    mask = torch.rand((images.shape[0], images.shape[1], 1, 1, 1), device=images.device) < prob
    mask[:, -1] = False
    return torch.where(mask, images[:, -1:].expand_as(images), images)


def _drop_now(prob: float) -> bool:
    return prob > 0 and bool(torch.rand(()) < prob)


if __name__ == "__main__":
    main()
