from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="RoboCasa BC Training Loss")
    args = parser.parse_args()

    history = json.loads(Path(args.history).read_text())
    if not history:
        raise ValueError("history is empty")
    steps = [int(row["step"]) for row in history]
    losses = [float(row["train_loss"]) for row in history]
    _write_svg(steps, losses, Path(args.out), args.title)


def _write_svg(steps: list[int], losses: list[float], out: Path, title: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    width = 1000
    height = 520
    left = 84
    right = 28
    top = 52
    bottom = 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x = min(steps)
    max_x = max(steps)
    min_y = min(losses)
    max_y = max(losses)
    if max_x == min_x:
        max_x = min_x + 1
    if max_y == min_y:
        max_y = min_y + 1.0

    def x_px(xv: float) -> float:
        return left + plot_w * (xv - min_x) / (max_x - min_x)

    def y_px(yv: float) -> float:
        return top + plot_h * (1.0 - (yv - min_y) / (max_y - min_y))

    points = " ".join(f"{x_px(x):.2f},{y_px(y):.2f}" for x, y in zip(steps, losses))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="24">{_esc(title)}</text>',
    ]
    for i in range(5):
        gx = left + plot_w * i / 4
        gy = top + plot_h * i / 4
        parts.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" stroke="#e8e8e8"/>')
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + plot_w}" y2="{gy:.1f}" stroke="#e8e8e8"/>')
        xv = min_x + (max_x - min_x) * i / 4
        yv = max_y - (max_y - min_y) * i / 4
        parts.append(f'<text x="{gx:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="13" fill="#444">{xv:.0f}</text>')
        parts.append(f'<text x="{left - 10}" y="{gy + 4:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#444">{yv:.4f}</text>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>',
            f'<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{points}"/>',
            f'<circle cx="{x_px(steps[-1]):.2f}" cy="{y_px(losses[-1]):.2f}" r="4.5" fill="#1f77b4"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle" font-family="Arial" font-size="14">training step</text>',
            f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="14">train loss</text>',
            '</svg>',
        ]
    )
    out.write_text("\n".join(parts))


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
