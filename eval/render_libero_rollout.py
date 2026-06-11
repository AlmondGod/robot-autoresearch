from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data.libero_dataset import tokenize_instruction
from eval.eval_libero_success import _RolloutBuffers, _ckpt_array, _wrist_image
from models.policy import TinyBCPolicy
from train.common import device_from_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="runs/libero/real_bc_all5/policy.pt")
    parser.add_argument("--manifest", default="data/libero_object5/manifest.json")
    parser.add_argument("--libero-root", default="third_party/LIBERO")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--replan-every", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="runs/libero/real_bc_all5/rollout_task0_ep0.mp4")
    args = parser.parse_args()
    render_rollout(args)


def render_rollout(args: argparse.Namespace) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render MP4 video")

    config_path = Path(".libero_config").resolve()
    if config_path.exists():
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_path))
    if not Path(args.libero_root).exists():
        raise FileNotFoundError(f"LIBERO checkout not found: {args.libero_root}")

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    device = device_from_arg(args.device)
    checkpoint = torch.load(args.policy, map_location=device)
    action_dim = int(checkpoint["action_dim"])
    proprio_dim = int(checkpoint["proprio_dim"])
    action_horizon = int(checkpoint.get("action_horizon", 1))
    history = int(checkpoint.get("history", 1))
    n_embd = int(checkpoint.get("n_embd", 128))
    policy_kind = str(checkpoint.get("policy_kind", "bc"))
    flow_steps = int(checkpoint.get("flow_steps", 8))
    policy = TinyBCPolicy(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        n_embd=n_embd,
        action_horizon=action_horizon,
        max_history=max(history, 1),
        policy_kind=policy_kind,
        flow_steps=flow_steps,
    ).to(device)
    policy.load_state_dict(checkpoint["state_dict"], strict=False)
    policy.eval()

    manifest = json.loads(Path(args.manifest).read_text())
    task_names = [task["task_name"].removesuffix("_demo") for task in manifest["tasks"]]
    if args.task_id < 0 or args.task_id >= len(task_names):
        raise ValueError(f"task-id must be in [0, {len(task_names) - 1}]")
    task_name = task_names[args.task_id]
    suite = manifest.get("suite", "libero_object")
    benchmark = get_benchmark(suite)(0)
    name_to_task_id = {task.name: idx for idx, task in enumerate(benchmark.tasks)}
    benchmark_task = benchmark.get_task(name_to_task_id[task_name])
    env_args = {
        "bddl_file_name": os.path.join(
            get_libero_path("bddl_files"),
            benchmark_task.problem_folder,
            benchmark_task.bddl_file,
        ),
        "camera_heights": args.image_size,
        "camera_widths": args.image_size,
    }
    init_states = torch.load(
        os.path.join(get_libero_path("init_states"), benchmark_task.problem_folder, benchmark_task.init_states_file),
        weights_only=False,
    )

    frames: list[np.ndarray] = []
    env = OffScreenRenderEnv(**env_args)
    done = False
    try:
        env.reset()
        obs = env.set_init_state(init_states[args.episode_id % init_states.shape[0]])
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(action_dim, dtype=np.float32))
        rollout = _RolloutBuffers(history=max(history, 1))
        rollout.append(obs)
        action_queue: list[np.ndarray] = []
        frames.append(_compose_frame(obs, task_name, 0, done, None))
        for step_idx in range(1, args.max_steps + 1):
            if not action_queue:
                action_queue = _policy_action_chunk(policy, rollout, task_name, args.task_id, checkpoint, device, args.replan_every)
            action = action_queue.pop(0)
            obs, reward, done, info = env.step(action)
            rollout.append(obs)
            frames.append(_compose_frame(obs, task_name, step_idx, done, float(reward), info=info))
            if done:
                break
    finally:
        env.close()

    if frames:
        frames.extend([frames[-1].copy() for _ in range(args.fps)])
    _write_mp4(frames, Path(args.out), args.fps, ffmpeg)
    print(json.dumps({"out": args.out, "frames": len(frames), "success": bool(done), "task": task_name}, indent=2))


def _policy_action_chunk(
    policy: TinyBCPolicy,
    rollout: _RolloutBuffers,
    task_name: str,
    task_id: int,
    checkpoint: dict,
    device: torch.device,
    replan_every: int = 0,
) -> list[np.ndarray]:
    agent, wrist, proprio = rollout.arrays()
    proprio = (proprio - _ckpt_array(checkpoint, "proprio_mean")) / _ckpt_array(checkpoint, "proprio_std")
    instruction = tokenize_instruction(task_name)
    with torch.no_grad():
        images = torch.as_tensor(agent[None], dtype=torch.float32, device=device)
        proprio_t = torch.as_tensor(proprio[None], dtype=torch.float32, device=device)
        task_t = torch.as_tensor([task_id], dtype=torch.long, device=device)
        wrist_t = torch.as_tensor(wrist[None], dtype=torch.float32, device=device)
        instruction_t = torch.as_tensor(instruction[None], dtype=torch.long, device=device)
        if str(checkpoint.get("policy_kind", "bc")) == "flow":
            action_chunk = policy.sample_flow(
                images,
                proprio_t,
                task_t,
                wrist_images=wrist_t,
                instruction_tokens=instruction_t,
                steps=int(checkpoint.get("flow_steps", 8)),
            )
        else:
            action_chunk, _ = policy(
                images,
                proprio_t,
                task_t,
                wrist_images=wrist_t,
                instruction_tokens=instruction_t,
            )
    actions = action_chunk[0].cpu().numpy()
    actions = actions * _ckpt_array(checkpoint, "action_std") + _ckpt_array(checkpoint, "action_mean")
    out = [np.asarray(action, dtype=np.float32) for action in actions]
    if replan_every > 0:
        out = out[:replan_every]
    return out


def _compose_frame(obs: dict, task_name: str, step_idx: int, done: bool, reward: float | None, info: dict | None = None) -> np.ndarray:
    agent = np.asarray(obs["agentview_image"], dtype=np.uint8)
    wrist = np.asarray(_wrist_image(obs), dtype=np.uint8)
    scale = 5
    agent_img = Image.fromarray(agent).resize((agent.shape[1] * scale, agent.shape[0] * scale), Image.Resampling.NEAREST)
    wrist_img = Image.fromarray(wrist).resize((wrist.shape[1] * scale, wrist.shape[0] * scale), Image.Resampling.NEAREST)
    pad = 12
    header_h = 42
    width = agent_img.width + wrist_img.width + 3 * pad
    height = header_h + agent_img.height + 2 * pad
    image = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    status_color = (38, 150, 78) if done else (190, 55, 45)
    draw.rectangle([0, 0, width, 8], fill=status_color)
    reward_text = "" if reward is None else f" | reward {reward:.3f}"
    info_text = ""
    if info:
        info_text = f" | success {bool(info.get('success', done))}"
    draw.text((pad, 16), f"{task_name} | step {step_idx:03d}{reward_text}{info_text}", fill=(20, 20, 20))
    draw.text((pad, header_h - 2), "agentview", fill=(60, 60, 60))
    draw.text((2 * pad + agent_img.width, header_h - 2), "wrist", fill=(60, 60, 60))
    image.paste(agent_img, (pad, header_h + pad))
    image.paste(wrist_img, (2 * pad + agent_img.width, header_h + pad))
    return np.asarray(image, dtype=np.uint8)


def _write_mp4(frames: list[np.ndarray], out: Path, fps: int, ffmpeg: str) -> None:
    if not frames:
        raise ValueError("no frames to render")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for idx, frame in enumerate(frames):
            _write_ppm(tmp_path / f"frame_{idx:04d}.ppm", frame)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_path / "frame_%04d.ppm"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _write_ppm(path: Path, image: np.ndarray) -> None:
    h, w, _ = image.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        handle.write(image.tobytes())


if __name__ == "__main__":
    main()
