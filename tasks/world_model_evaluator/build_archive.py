from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a world-evaluator candidate archive from traced BC-5 eval JSONs.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate spec in split,name,eval_json form. Example: test,bc5_starter,runs/.../eval.json",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for idx, spec in enumerate(args.candidate):
        split, name, eval_path = _parse_candidate(spec)
        payload_path = Path(eval_path)
        if not payload_path.exists():
            raise FileNotFoundError(payload_path)
        payload = json.loads(payload_path.read_text())
        details = payload.get("details", [])
        trace_count = sum(1 for detail in details if detail.get("trace_path"))
        if trace_count == 0:
            raise ValueError(f"{payload_path} has no detail trace_path entries")
        rows.append(
            {
                "experiment": idx,
                "split": split,
                "change": name,
                "checkpoint": payload.get("checkpoint"),
                "eval_path": str(payload_path),
                "episodes": int(payload.get("episodes", len(details))),
                "successes": int(payload.get("successes", 0)),
                "success_rate": float(payload.get("success_rate", 0.0)),
                "score": float(payload.get("success_rate", 0.0)),
                "trace_count": int(trace_count),
                "per_task": payload.get("per_task", {}),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    print(json.dumps({"out": str(out), "candidates": len(rows)}, indent=2, sort_keys=True))


def _parse_candidate(spec: str) -> tuple[str, str, str]:
    parts = spec.split(",", 2)
    if len(parts) != 3:
        raise ValueError(f"candidate must be split,name,eval_json; got {spec!r}")
    split, name, eval_path = [part.strip() for part in parts]
    if not split or not name or not eval_path:
        raise ValueError(f"empty field in candidate spec {spec!r}")
    return split, name, eval_path


if __name__ == "__main__":
    main()
