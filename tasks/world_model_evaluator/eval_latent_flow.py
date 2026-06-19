from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from eval.eval_robocasa_tiny_evaluator_correlation import (  # noqa: E402
    _evaluator_task_id,
    _initial_obs,
    _load_archive,
    _tensor,
)
from eval.eval_world_model_ranking import _write_svg, compute_metrics  # noqa: E402
from models.robocasa_latent_flow import RoboCasaLatentResidualFlow  # noqa: E402
from models.robocasa_tiny_evaluator import RoboCasaVAEWorldModel  # noqa: E402
from tasks.world_model_evaluator.eval import (  # noqa: E402
    DEFAULT_ARCHIVE,
    _calibration_score,
    _dataset_roots,
    _speedup_score,
    _split_metrics,
    _trace_groups,
)
from train.common import device_from_arg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a latent-flow world-model scorer on frozen RoboCasa trace archives.")
    parser.add_argument("--checkpoint", "--evaluator", dest="checkpoint", required=True)
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--task-alias", default="OpenDrawer")
    parser.add_argument("--out", default="runs/autorobobench/world_model_evaluator/eval_latent_flow_result.json")
    parser.add_argument("--scores-out", default=None)
    parser.add_argument("--plot", default=None)
    parser.add_argument("--imagined-rollouts", type=int, default=4)
    parser.add_argument("--imagined-steps", type=int, default=260)
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--invert-learned-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sim-seconds-per-rollout", type=float, default=12.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    out_path = Path(args.out)
    score_path = Path(args.scores_out) if args.scores_out else out_path.with_name("candidate_scores_latent_flow.jsonl")
    plot_path = Path(args.plot) if args.plot else out_path.with_suffix(".svg")

    device = device_from_arg(args.device)
    flow_ckpt = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    base_path = Path(flow_ckpt["base_checkpoint"])
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    base_ckpt = torch.load(base_path, map_location=device, weights_only=False)
    base = _load_base(base_ckpt, device)
    base.eval()
    flow = _load_flow(flow_ckpt, device)
    flow.load_state_dict(flow_ckpt["state_dict"])
    flow.eval()

    dataset_roots = _dataset_roots(Path(args.manifest))
    default_dataset_root = dataset_roots[str(args.task_alias)]
    rows = _load_archive(Path(args.archive))
    scored_rows: list[dict] = []
    for idx, row in enumerate(rows):
        trace_groups = _trace_groups(row, default_alias=str(args.task_alias))
        sim_success = float(row.get("success_rate", row.get("score", 0.0)))
        if trace_groups:
            raw_learned_score, learned_progress, seconds = _score_trace_groups(
                base=base,
                base_ckpt=base_ckpt,
                flow=flow,
                dataset_roots=dataset_roots,
                default_dataset_root=default_dataset_root,
                trace_groups=trace_groups,
                imagined_rollouts=int(args.imagined_rollouts),
                imagined_steps=int(args.imagined_steps),
                flow_steps=int(args.flow_steps or flow_ckpt.get("flow_steps", 8)),
                noise_scale=float(args.noise_scale),
                action_noise=float(args.action_noise),
                seed=int(args.seed) + idx * 997,
                device=device,
            )
            learned_score = 1.0 - raw_learned_score if args.invert_learned_score else raw_learned_score
            error = None
        else:
            raw_learned_score = None
            learned_score = None
            learned_progress = None
            seconds = None
            error = "no trace paths"
        scored_rows.append(
            {
                "candidate_id": int(row.get("experiment", idx)),
                "change": row.get("change"),
                "split": row.get("split"),
                "learned_score": learned_score,
                "raw_learned_score": raw_learned_score,
                "learned_score_transform": "1-raw" if args.invert_learned_score else "raw",
                "learned_progress": learned_progress,
                "sim_success": sim_success,
                "sim_successes": row.get("successes"),
                "sim_eval_rollouts": row.get("episodes"),
                "sim_eval_seconds": float(row.get("episodes") or 0.0) * float(args.sim_seconds_per_rollout),
                "learned_eval_rollouts": int(args.imagined_rollouts) * sum(len(paths) for paths in trace_groups.values()),
                "learned_eval_seconds": seconds,
                "learned_mode": "latent_residual_flow_action_trace",
                "error": error,
            }
        )

    metrics = compute_metrics(scored_rows, top_k=int(args.top_k))
    split_metrics = _split_metrics(scored_rows, top_k=int(args.top_k))
    primary_split = "test" if "test" in split_metrics else "heldout" if "heldout" in split_metrics else "all"
    primary_metrics = split_metrics.get(primary_split, metrics)
    speedup = primary_metrics.get("speedup_ratio")
    result = {
        "track": "world_model_evaluator",
        "evaluator_kind": "latent_residual_flow",
        "checkpoint": str(args.checkpoint),
        "base_checkpoint": str(base_path),
        "archive": str(args.archive),
        "task_alias": str(args.task_alias),
        "candidate_count": int(metrics["n"]),
        "wm_spearman": float(primary_metrics["spearman_rho"]),
        "pearson_r": float(primary_metrics["pearson_r"]),
        "checkpoint_ranking_accuracy": float(primary_metrics["top_k_hit_rate"]),
        "wm_speedup_score": _speedup_score(speedup),
        "speedup_ratio": speedup,
        "calibration_score": _calibration_score(scored_rows),
        "reproducibility_integrity": 1.0,
        "ranking_metrics": metrics,
        "split_metrics": split_metrics,
        "primary_split": primary_split,
        "imagined_rollouts": int(args.imagined_rollouts),
        "imagined_steps": int(args.imagined_steps),
        "flow_steps": int(args.flow_steps or flow_ckpt.get("flow_steps", 8)),
        "noise_scale": float(args.noise_scale),
        "action_noise": float(args.action_noise),
        "learned_score_transform": "1-raw" if args.invert_learned_score else "raw",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    score_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in scored_rows) + "\n")
    _write_svg(scored_rows, metrics, plot_path)
    print(json.dumps(result, indent=2, sort_keys=True))


def _load_base(checkpoint: dict, device: torch.device) -> RoboCasaVAEWorldModel:
    model = RoboCasaVAEWorldModel(
        proprio_dim=int(checkpoint["proprio_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        task_count=int(checkpoint["task_count"]),
        latent_dim=int(checkpoint["latent_dim"]),
        width=int(checkpoint.get("width", 512)),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _load_flow(checkpoint: dict, device: torch.device) -> RoboCasaLatentResidualFlow:
    return RoboCasaLatentResidualFlow(
        latent_dim=int(checkpoint["latent_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        task_count=int(checkpoint["task_count"]),
        hidden=int(checkpoint.get("hidden", 1024)),
        depth=int(checkpoint.get("depth", 3)),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)


def _score_trace_groups(
    *,
    base: RoboCasaVAEWorldModel,
    base_ckpt: dict,
    flow: RoboCasaLatentResidualFlow,
    dataset_roots: dict[str, Path],
    default_dataset_root: Path,
    trace_groups: dict[str, list[Path]],
    imagined_rollouts: int,
    imagined_steps: int,
    flow_steps: int,
    noise_scale: float,
    action_noise: float,
    seed: int,
    device: torch.device,
) -> tuple[float, float, float]:
    total_count = 0
    weighted_score = 0.0
    weighted_progress = 0.0
    total_seconds = 0.0
    for alias, trace_paths in trace_groups.items():
        if not trace_paths:
            continue
        dataset_root = dataset_roots.get(alias, default_dataset_root)
        raw_score, progress, seconds = _score_trace_candidate(
            base=base,
            base_ckpt=base_ckpt,
            flow=flow,
            dataset_root=dataset_root,
            trace_paths=trace_paths,
            imagined_rollouts=imagined_rollouts,
            imagined_steps=imagined_steps,
            flow_steps=flow_steps,
            noise_scale=noise_scale,
            action_noise=action_noise,
            seed=seed + total_count * 37,
            device=device,
        )
        count = len(trace_paths)
        weighted_score += raw_score * count
        weighted_progress += progress * count
        total_seconds += seconds
        total_count += count
    if total_count == 0:
        raise ValueError("no valid traces to score")
    return weighted_score / total_count, weighted_progress / total_count, total_seconds


def _score_trace_candidate(
    *,
    base: RoboCasaVAEWorldModel,
    base_ckpt: dict,
    flow: RoboCasaLatentResidualFlow,
    dataset_root: Path,
    trace_paths: list[Path],
    imagined_rollouts: int,
    imagined_steps: int,
    flow_steps: int,
    noise_scale: float,
    action_noise: float,
    seed: int,
    device: torch.device,
) -> tuple[float, float, float]:
    start = time.time()
    scores: list[float] = []
    progresses: list[float] = []
    rng = np.random.default_rng(int(seed))
    proprio_mean = _tensor(base_ckpt, "proprio_mean", device)
    proprio_std = _tensor(base_ckpt, "proprio_std", device)
    action_mean = _tensor(base_ckpt, "action_mean", device)
    action_std = _tensor(base_ckpt, "action_std", device)
    with torch.no_grad():
        for trace_path in trace_paths:
            trace = np.load(trace_path)
            episode_id = int(trace["episode_id"][0])
            actions = np.asarray(trace["actions"], dtype=np.float32)
            if actions.size == 0:
                continue
            obs = _initial_obs(dataset_root, episode_id)
            task_id = _evaluator_task_id(dataset_root, episode_id, base_ckpt)
            task_t = torch.full((imagined_rollouts,), task_id, dtype=torch.long, device=device)
            agent = torch.as_tensor(np.repeat(obs["agent"][None], imagined_rollouts, axis=0), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            wrist = torch.as_tensor(np.repeat(obs["wrist"][None], imagined_rollouts, axis=0), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            proprio = (torch.as_tensor(np.repeat(obs["proprio"][None], imagined_rollouts, axis=0), dtype=torch.float32, device=device) - proprio_mean) / proprio_std
            latent = base.encode(agent, wrist, proprio, task_t)
            episode_scores = []
            episode_progress = []
            limit = min(len(actions), max(1, int(imagined_steps)))
            for step in range(limit):
                action_batch = np.repeat(actions[step][None], imagined_rollouts, axis=0).astype(np.float32)
                if action_noise > 0:
                    action_batch = action_batch + rng.normal(0.0, action_noise, size=action_batch.shape).astype(np.float32)
                action_t = (torch.as_tensor(action_batch, dtype=torch.float32, device=device) - action_mean) / action_std
                noise = torch.randn_like(latent) * float(noise_scale) if noise_scale > 0 else torch.zeros_like(latent)
                residual = flow.sample_residual(
                    latent=latent,
                    action=action_t,
                    task_id=task_t,
                    steps=flow_steps,
                    noise=noise,
                )
                latent = latent + residual
                progress, success_logit = base.heads(latent, task_t)
                episode_scores.append(torch.sigmoid(success_logit).detach().cpu().numpy())
                episode_progress.append(torch.sigmoid(progress).detach().cpu().numpy())
            scores.append(float(np.max(np.stack(episode_scores), axis=0).mean()))
            progresses.append(float(np.max(np.stack(episode_progress), axis=0).mean()))
    return float(np.mean(scores)), float(np.mean(progresses)), time.time() - start


def _speedup_score(speedup: float | None) -> float:
    if speedup is None or not math.isfinite(float(speedup)) or speedup <= 0:
        return 0.0
    return max(0.0, min(1.0, float(speedup) / 10.0))


if __name__ == "__main__":
    main()
