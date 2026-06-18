from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import imageio.v3 as iio
import torch
import torch.distributed as dist
from PIL import Image

import worldsim._ext.imaginaire.utils.distributed
from inference._core import load_video_np, prepare_batch_skeleton
from worldsim._ext.imaginaire.utils import misc
from worldsim._src.utils.model_loader import load_model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints")
    parser.add_argument("--case", default="agibot_465")
    parser.add_argument(
        "--asset-dir",
        default="",
        help="Directory containing rgb.mp4, gripper_scenario.mp4, and caption.pickle. Overrides --case.",
    )
    parser.add_argument(
        "--asset-root",
        default="",
        help="Directory containing multiple OSCAR asset dirs. If set, training samples across all children with rgb.mp4.",
    )
    parser.add_argument("--start-frame", type=int, default=91)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--load-adapter-dir", default="", help="Existing PEFT adapter to load before training/sampling.")
    parser.add_argument("--sample-only", action="store_true", help="Skip training and only render a sample.")
    parser.add_argument("--sample-all", action="store_true", help="In sample-only mode, render every asset instead of just the first.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for --sample-all.")
    parser.add_argument("--eval-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/finetune_agibot_overfit")
    args = parser.parse_args()

    torch.backends.cuda.preferred_linalg_library(backend="cusolver")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.enable_grad(True)
    worldsim._ext.imaginaire.utils.distributed.init()

    try:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        assets = _find_assets(args)
        if not assets:
            raise ValueError("no OSCAR assets found")
        eval_asset = assets[0]

        model, _ = load_model_from_checkpoint(
            experiment_name="cosmos2_robot_plus_human_v2_70f",
            checkpoint_path=args.checkpoint,
            enable_fsdp=False,
            config_file="worldsim/_src/configs/agibot_control/config.py",
        )
        model.net = model.add_lora(
            model.net,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        )
        if args.load_adapter_dir:
            _load_adapter_state_direct(model.net, Path(args.load_adapter_dir))
            for param in model.net.parameters():
                param.requires_grad_(not args.sample_only)
        model.config.text_encoder_config.compute_online = False
        model.eval() if args.sample_only else model.train()
        trainable = [param for param in model.net.parameters() if param.requires_grad]

        text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        losses: list[float] = []
        if not args.sample_only:
            optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
            for step in range(args.steps):
                asset = assets[step % len(assets)]
                batch = _load_batch(model, asset, args, text_cache, random_crop=True)
                optimizer.zero_grad(set_to_none=True)
                _, loss = model.training_step(batch, step)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                print(json.dumps({"step": step + 1, "loss": loss_value}), flush=True)

        metrics = {
            "case": args.case,
            "asset_count": len(assets),
            "assets": [str(asset) for asset in assets],
            "steps": args.steps,
            "lr": args.lr,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "load_adapter_dir": args.load_adapter_dir,
            "sample_only": bool(args.sample_only),
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "losses": losses,
            "first_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
        if not args.sample_only and hasattr(model.net, "save_pretrained"):
            model.net.save_pretrained(out_dir / "adapter")
        elif not args.sample_only:
            torch.save(model.net.state_dict(), out_dir / "net_state.pt")

        sample_assets = assets if args.sample_all else [eval_asset]
        if args.max_samples:
            sample_assets = sample_assets[: int(args.max_samples)]
        sample_outputs = []
        with torch.no_grad():
            for sample_idx, sample_asset in enumerate(sample_assets):
                base_batch = _load_batch(model, sample_asset, args, text_cache, random_crop=False)
                batch = _clone_batch(base_batch)
                _, x0, _ = model.get_data_and_condition(batch)
                sample = model.generate_samples_from_batch(
                    batch,
                    guidance=6.0,
                    seed=42 + sample_idx,
                    state_shape=x0.shape[1:],
                    n_sample=x0.shape[0],
                    num_steps=args.eval_steps,
                    is_negative_prompt=True,
                    shift=5.0,
                )
                decoded = model.decode(sample)
                stem = "finetuned_sample" if len(sample_assets) == 1 else f"{sample_asset.name}_sample"
                sample_path = out_dir / f"{stem}.mp4"
                ref_path = out_dir / f"{sample_asset.name}_reference.mp4" if len(sample_assets) > 1 else out_dir / "reference.mp4"
                _save_video(decoded, sample_path, fps=15)
                reference_rgb = load_video_np(
                    sample_asset / "rgb.mp4", args.start_frame, args.num_frames, args.height, args.width
                )
                _save_reference(reference_rgb, ref_path, fps=15)
                sample_outputs.append({"asset": str(sample_asset), "sample": str(sample_path), "reference": str(ref_path)})
                print(json.dumps({"sampled": sample_outputs[-1]}), flush=True)
        metrics["sample_outputs"] = sample_outputs
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(json.dumps({"out_dir": str(out_dir), **metrics}, indent=2, sort_keys=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _load_prompt(path: Path) -> str:
    caption = pickle.load(open(path, "rb"))
    if isinstance(caption, str):
        return caption
    if isinstance(caption, dict) and "caption" in caption:
        return str(caption["caption"])
    return str(caption)


def _load_adapter_state_direct(net, adapter_dir: Path) -> None:
    from safetensors.torch import load_file

    path = adapter_dir / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(path)
    saved = load_file(str(path))
    target = net.state_dict()
    mapped = {}
    for key, value in saved.items():
        candidates = [
            key,
            key.replace(".lora_A.weight", ".lora_A.default.weight"),
            key.replace(".lora_B.weight", ".lora_B.default.weight"),
        ]
        for candidate in candidates:
            if candidate in target:
                mapped[candidate] = value.to(dtype=target[candidate].dtype)
                break
    missing = len(saved) - len(mapped)
    if missing:
        print(json.dumps({"adapter_load_warning": "some adapter tensors were not matched", "missing": missing}), flush=True)
    result = net.load_state_dict(mapped, strict=False)
    print(
        json.dumps(
            {
                "adapter_loaded": str(path),
                "saved_tensors": len(saved),
                "mapped_tensors": len(mapped),
                "missing_keys": len(result.missing_keys),
                "unexpected_keys": len(result.unexpected_keys),
            }
        ),
        flush=True,
    )


def _find_assets(args: argparse.Namespace) -> list[Path]:
    if args.asset_root:
        root = Path(args.asset_root)
        return sorted(path for path in root.iterdir() if (path / "rgb.mp4").exists() and (path / "gripper_scenario.mp4").exists())
    if args.asset_dir:
        return [Path(args.asset_dir)]
    return [Path(args.checkpoint) / "assets" / args.case]


def _load_batch(
    model,
    asset: Path,
    args: argparse.Namespace,
    text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    random_crop: bool,
) -> dict:
    prompt = _load_prompt(asset / "caption.pickle")
    start_frame = _choose_start_frame(asset / "rgb.mp4", args, random_crop=random_crop)
    rgb = load_video_np(asset / "rgb.mp4", start_frame, args.num_frames, args.height, args.width)
    skel = load_video_np(asset / "gripper_scenario.mp4", start_frame, args.num_frames, args.height, args.width)
    return _make_batch(model, rgb, skel, prompt, args, text_cache)


def _choose_start_frame(path: Path, args: argparse.Namespace, *, random_crop: bool) -> int:
    if not random_crop:
        return int(args.start_frame)
    frame_count = _video_frame_count(path)
    max_start = max(0, frame_count - int(args.num_frames))
    if max_start <= 0:
        return 0
    return random.randint(0, max_start)


def _video_frame_count(path: Path) -> int:
    metadata_path = path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            if "frames" in metadata:
                return int(metadata["frames"])
        except Exception:
            pass
    try:
        meta = iio.immeta(path)
        if "nframes" in meta and meta["nframes"] not in (None, float("inf")):
            return int(meta["nframes"])
    except Exception:
        pass
    count = 0
    for _ in iio.imiter(path):
        count += 1
    return count


def _make_batch(
    model,
    rgb,
    skel,
    prompt: str,
    args: argparse.Namespace,
    text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict:
    batch = prepare_batch_skeleton(
        rgb_frames=rgb,
        condition_frames=skel,
        caption=prompt,
        num_frames=args.num_frames,
        fps=15.0,
        height=args.height,
        width=args.width,
    )
    batch = misc.to(batch, **model.tensor_kwargs)
    embed_dtype = model.tensor_kwargs.get("dtype", torch.bfloat16)
    if prompt not in text_cache:
        cond_emb = model.text_encoder.compute_text_embeddings_online(
            {"ai_caption": batch["ai_caption"], "images": None}, "ai_caption"
        )
        neg_emb = model.text_encoder.compute_text_embeddings_online(
            {"ai_caption": [""], "images": None}, "ai_caption"
        )
        text_cache[prompt] = (cond_emb.to(dtype=embed_dtype), neg_emb.to(dtype=embed_dtype))
    cond_emb, neg_emb = text_cache[prompt]
    batch["t5_text_embeddings"] = cond_emb.to(dtype=embed_dtype)
    batch["t5_text_mask"] = torch.ones(cond_emb.shape[0], cond_emb.shape[1], device="cuda", dtype=embed_dtype)
    batch["neg_t5_text_embeddings"] = neg_emb.to(dtype=embed_dtype)
    batch["neg_t5_text_mask"] = batch["t5_text_mask"]
    return batch


def _clone_batch(batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.clone() if torch.is_tensor(value) else value
    return out


def _save_video(decoded: torch.Tensor, path: Path, fps: int) -> None:
    video = decoded[0].detach().float().cpu().clamp(-1, 1)
    frames = ((video.permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
    iio.imwrite(path, frames, fps=fps, codec="libx264")


def _save_reference(rgb, path: Path, fps: int) -> None:
    iio.imwrite(path, rgb, fps=fps, codec="libx264")


if __name__ == "__main__":
    main()
