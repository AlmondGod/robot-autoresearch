from __future__ import annotations

import argparse
import json
import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path


TRAIN_FILE = Path("train.py")
RUNS_ROOT = Path("runs/libero/autoresearch50")
LEDGER = RUNS_ROOT / "ledger.jsonl"
judge = None


BASE = {
    "METHOD": "bc",
    "CHANGE": "BC 256 baseline",
    "STRATEGY_KIND": "bc",
    "POLICY_KIND": "bc",
    "BATCH_SIZE": 64,
    "LR": 3e-4,
    "N_EMBD": 256,
    "LOSS": "mse",
    "CHUNK_DECAY": 1.0,
    "IMAGE_NOISE": 0.0,
    "ACTION_NOISE": 0.0,
    "HISTORY_DROPOUT": 0.0,
    "WRIST_DROPOUT": 0.0,
    "WEIGHT_DECAY": 0.01,
    "GRAD_CLIP": 0.0,
    "FLOW_STEPS": 8,
    "FLOW_SIGMA": 1.0,
    "VIDEO_AUX": False,
    "TOKENIZER_CODEBOOK": 128,
    "TOKENIZER_EMBD": 64,
    "WORLD_LAYERS": 4,
    "WORLD_HEADS": 4,
    "WORLD_EMBD": 128,
}


def main() -> None:
    global judge
    judge = _load_judge()
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--max-train-seconds", type=float, default=300.0)
    parser.add_argument("--baseline", default="")
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    original = TRAIN_FILE.read_text()
    best_train_text = original
    baseline = Path(args.baseline) if args.baseline else None
    best_metrics = _read_metrics(baseline) if baseline else None
    strategies = _strategies()
    for offset in range(args.iterations):
        idx = args.start_index + offset
        strategy = strategies[idx % len(strategies)]
        run_dir = RUNS_ROOT / f"iter_{idx:03d}"
        _write_train_file(original, strategy)
        started = time.time()
        status = "completed"
        error = ""
        try:
            _run(
                [
                    sys.executable,
                    "train.py",
                    "--out-dir",
                    str(run_dir),
                    "--max-train-seconds",
                    str(args.max_train_seconds),
                ]
            )
        except subprocess.CalledProcessError as exc:
            status = "failed"
            error = str(exc)

        accepted = False
        decision = {"accepted": False, "reason": status}
        metrics = _read_metrics(run_dir) if (run_dir / "metrics.json").exists() else {}
        if status == "completed":
            if baseline is None:
                accepted = True
                baseline = run_dir
                best_metrics = metrics
                best_train_text = TRAIN_FILE.read_text()
                decision = {"accepted": True, "reason": "initialized_baseline"}
            else:
                decision = judge(baseline, run_dir)
                accepted = bool(decision["accepted"])
                if accepted:
                    baseline = run_dir
                    best_metrics = metrics
                    best_train_text = TRAIN_FILE.read_text()

        entry = {
            "iteration": idx,
            "status": status,
            "accepted": accepted,
            "decision": decision,
            "run_dir": str(run_dir),
            "elapsed_seconds": time.time() - started,
            "strategy": strategy,
            "metrics": metrics,
            "best_metrics": best_metrics,
            "error": error,
        }
        _append_jsonl(LEDGER, entry)
        (run_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True)) if run_dir.exists() else None
        print(json.dumps(entry, indent=2, sort_keys=True), flush=True)

        if accepted:
            _commit_acceptance(idx, strategy)
        else:
            TRAIN_FILE.write_text(best_train_text)


def _strategies() -> list[dict]:
    specs = [
        ("BC 256 baseline", {}),
        ("BC 384 wider policy", {"N_EMBD": 384}),
        ("BC 192 smaller faster policy", {"N_EMBD": 192}),
        ("BC huber action loss", {"LOSS": "huber"}),
        ("BC first action weighted chunk loss", {"CHUNK_DECAY": 0.6}),
        ("BC mild image noise augmentation", {"IMAGE_NOISE": 0.015}),
        ("BC mild action denoising", {"ACTION_NOISE": 0.03}),
        ("BC history dropout", {"HISTORY_DROPOUT": 0.25}),
        ("BC wrist dropout", {"WRIST_DROPOUT": 0.35}),
        ("BC lower weight decay", {"WEIGHT_DECAY": 0.001}),
        ("BC no weight decay", {"WEIGHT_DECAY": 0.0}),
        ("BC grad clip 1.0", {"GRAD_CLIP": 1.0}),
        ("BC lower LR", {"LR": 1e-4}),
        ("BC higher LR", {"LR": 6e-4}),
        ("BC batch 32", {"BATCH_SIZE": 32}),
        ("BC batch 96", {"BATCH_SIZE": 96}),
        ("Flow-ish BC huber plus action denoise", {"LOSS": "huber", "ACTION_NOISE": 0.05, "CHUNK_DECAY": 0.8}),
        ("Flow-ish BC strong action denoise", {"LOSS": "huber", "ACTION_NOISE": 0.1, "GRAD_CLIP": 1.0}),
        ("Augmented BC image plus history dropout", {"IMAGE_NOISE": 0.02, "HISTORY_DROPOUT": 0.2}),
        ("Augmented BC image plus wrist dropout", {"IMAGE_NOISE": 0.02, "WRIST_DROPOUT": 0.3}),
        ("Regularized BC dropout combo", {"HISTORY_DROPOUT": 0.15, "WRIST_DROPOUT": 0.25, "WEIGHT_DECAY": 0.001}),
        ("Wider huber BC", {"N_EMBD": 384, "LOSS": "huber"}),
        ("Wider first-action BC", {"N_EMBD": 384, "CHUNK_DECAY": 0.65}),
        ("Wider augmented BC", {"N_EMBD": 384, "IMAGE_NOISE": 0.015, "WEIGHT_DECAY": 0.001}),
        ("Smaller huber BC", {"N_EMBD": 192, "LOSS": "huber"}),
        ("Smaller higher LR BC", {"N_EMBD": 192, "LR": 6e-4}),
        ("Video aux tokenizer world baseline", {"VIDEO_AUX": True, "STRATEGY_KIND": "video_aux"}),
        ("Video aux bigger world", {"VIDEO_AUX": True, "STRATEGY_KIND": "video_aux", "WORLD_LAYERS": 6, "WORLD_EMBD": 160}),
        ("Video aux larger codebook", {"VIDEO_AUX": True, "STRATEGY_KIND": "video_aux", "TOKENIZER_CODEBOOK": 256}),
        ("Video aux compact tokenizer", {"VIDEO_AUX": True, "STRATEGY_KIND": "video_aux", "TOKENIZER_CODEBOOK": 64, "TOKENIZER_EMBD": 48}),
        ("Video aux plus huber BC", {"VIDEO_AUX": True, "STRATEGY_KIND": "video_aux", "LOSS": "huber"}),
        ("Video aux plus wider BC", {"VIDEO_AUX": True, "STRATEGY_KIND": "video_aux", "N_EMBD": 384}),
        ("Video aux plus first-action BC", {"VIDEO_AUX": True, "STRATEGY_KIND": "video_aux", "CHUNK_DECAY": 0.6}),
        ("Video aux plus action denoise", {"VIDEO_AUX": True, "STRATEGY_KIND": "video_aux", "ACTION_NOISE": 0.04, "LOSS": "huber"}),
        ("World-model policy proxy wider world", {"VIDEO_AUX": True, "STRATEGY_KIND": "world_model_policy", "WORLD_LAYERS": 8, "WORLD_EMBD": 192, "N_EMBD": 384}),
        ("World-model policy proxy compact BC", {"VIDEO_AUX": True, "STRATEGY_KIND": "world_model_policy", "WORLD_LAYERS": 6, "N_EMBD": 192, "LOSS": "huber"}),
        ("Transformer VLM proxy wide BC", {"STRATEGY_KIND": "vlm_bc", "N_EMBD": 512, "BATCH_SIZE": 32, "LR": 1e-4}),
        ("Transformer VLM proxy wide huber", {"STRATEGY_KIND": "vlm_bc", "N_EMBD": 512, "BATCH_SIZE": 32, "LR": 1e-4, "LOSS": "huber"}),
        ("Transformer VLM proxy wide augmented", {"STRATEGY_KIND": "vlm_bc", "N_EMBD": 512, "BATCH_SIZE": 32, "IMAGE_NOISE": 0.015, "WEIGHT_DECAY": 0.001}),
        ("Low LR first-action BC", {"LR": 1e-4, "CHUNK_DECAY": 0.55}),
        ("High LR huber clipped BC", {"LR": 6e-4, "LOSS": "huber", "GRAD_CLIP": 1.0}),
        ("Batch 96 low LR BC", {"BATCH_SIZE": 96, "LR": 1e-4}),
        ("Batch 32 high LR clipped BC", {"BATCH_SIZE": 32, "LR": 6e-4, "GRAD_CLIP": 1.0}),
        ("Denoise plus image aug BC", {"ACTION_NOISE": 0.04, "IMAGE_NOISE": 0.015, "LOSS": "huber"}),
        ("Denoise plus first-action BC", {"ACTION_NOISE": 0.04, "CHUNK_DECAY": 0.6, "LOSS": "huber"}),
        ("History robust first-action BC", {"HISTORY_DROPOUT": 0.25, "CHUNK_DECAY": 0.6}),
        ("Wrist robust first-action BC", {"WRIST_DROPOUT": 0.35, "CHUNK_DECAY": 0.6}),
        ("Wider robust denoise BC", {"N_EMBD": 384, "LOSS": "huber", "ACTION_NOISE": 0.04, "GRAD_CLIP": 1.0}),
        ("Compact robust augmented BC", {"N_EMBD": 192, "LOSS": "huber", "IMAGE_NOISE": 0.02, "HISTORY_DROPOUT": 0.2}),
        ("Video inverse proxy full stack", {"VIDEO_AUX": True, "STRATEGY_KIND": "video_inverse_proxy", "TOKENIZER_CODEBOOK": 256, "WORLD_LAYERS": 6, "LOSS": "huber", "ACTION_NOISE": 0.04}),
        ("Flow matching action policy", {"METHOD": "flow_matching", "STRATEGY_KIND": "flow_matching", "POLICY_KIND": "flow", "FLOW_STEPS": 16}),
        ("Flow matching wrist dropout", {"METHOD": "flow_matching", "STRATEGY_KIND": "flow_matching", "POLICY_KIND": "flow", "FLOW_STEPS": 16, "WRIST_DROPOUT": 0.35}),
    ]
    strategies = []
    for change, updates in specs:
        strategy = dict(BASE)
        strategy.update(updates)
        strategy["CHANGE"] = change
        strategies.append(strategy)
    return strategies


def _write_train_file(template: str, strategy: dict) -> None:
    text = template
    for key, value in strategy.items():
        literal = repr(value)
        text = re.sub(rf"^{key} = .*$", f"{key} = {literal}", text, flags=re.MULTILINE)
    TRAIN_FILE.write_text(text)


def _read_metrics(run_dir: Path | None) -> dict | None:
    if run_dir is None:
        return None
    path = run_dir / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _commit_acceptance(idx: int, strategy: dict) -> None:
    if subprocess.run(["git", "diff", "--quiet", "--", str(TRAIN_FILE)]).returncode == 0:
        return
    subprocess.run(["git", "add", str(TRAIN_FILE)], check=True)
    msg = f"Accept LIBERO iter {idx:03d}: {strategy['CHANGE']}"
    subprocess.run(["git", "commit", "-m", msg], check=True)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _load_judge():
    spec = importlib.util.spec_from_file_location("libero_judge", Path("research/judge.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load research/judge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.judge


if __name__ == "__main__":
    main()
