from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="runs/robocasa/autoresearch_task0/archive.jsonl")
    parser.add_argument("--plot", default="runs/robocasa/autoresearch_task0/progress.svg")
    args = parser.parse_args()

    archive = Path(args.archive)
    rows = []
    if archive.exists():
        rows = [json.loads(line) for line in archive.read_text().splitlines() if line.strip()]
    _write_svg(rows, Path(args.plot))
    print(json.dumps({"archive": str(archive), "experiments": len(rows), "plot": args.plot}, indent=2))


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_svg(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 420
    left, right, top, bottom = 70, 30, 35, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    vals = [float(row.get("score", 0.0)) for row in rows] or [0.0]
    lo, hi = min(vals + [0.0]), max(vals + [1.0])
    if hi - lo < 1e-9:
        hi = lo + 1.0

    def point(i: int, y: float) -> tuple[float, float]:
        x = left + (plot_w * i / max(1, len(rows) - 1))
        yy = top + plot_h * (1.0 - (y - lo) / (hi - lo))
        return x, yy

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="Arial" font-size="18">RoboCasa Autoresearch Progress</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">Experiment #</text>',
        f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">Eval success rate</text>',
    ]
    for tick in range(6):
        yv = lo + (hi - lo) * tick / 5
        _, yy = point(0, yv)
        parts.append(f'<line x1="{left-4}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{yv:.2f}</text>')
    if rows:
        coords = [point(i, float(row.get("score", 0.0))) for i, row in enumerate(rows)]
        parts.append('<polyline fill="none" stroke="#238b45" stroke-width="2" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in coords) + '"/>')
        for i, row in enumerate(rows):
            x, y = coords[i]
            color = "#2ecc71" if row.get("accepted") else "#bbb"
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="#333"/>')
            label = str(row.get("change", ""))[:36]
            parts.append(f'<text x="{x+7:.1f}" y="{y-7:.1f}" font-family="Arial" font-size="10" fill="#333">{_esc(label)}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts))


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
