# -*- coding: utf-8 -*-
"""
Save side-by-side grayscale PNGs (lift vs random same |A|) into docs/images/.

Uses Pillow only (no matplotlib) so it runs cleanly alongside NumPy 2.x.

From repo root:

    python scripts/export_mask_heatmaps.py --n 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from behrend import random_mask_same_density  # noqa: E402
from corner_lift import corner_free_lift_from_behrend  # noqa: E402


def main() -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise SystemExit("Install Pillow: pip install Pillow") from e

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="docs/images/mask-lift-vs-random.png")
    args = p.parse_args()

    rng = __import__("random").Random(args.seed)
    lift = corner_free_lift_from_behrend(args.n)
    rnd = random_mask_same_density((args.n, args.n), int(lift.sum()), rng)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    def to_gray(a: np.ndarray) -> Image.Image:
        g = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(g, mode="L")

    margin = 8
    title_h = 28
    w = args.n
    h = args.n
    gap = 6
    img_w = margin * 2 + w * 2 + gap
    img_h = margin * 2 + title_h + h
    canvas = Image.new("L", (img_w, img_h), color=255)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    cnt = int(lift.sum())
    draw.text((margin, margin), f"Corner-free lift |A|={cnt}", fill=0, font=font)
    draw.text((margin + w + gap, margin), f"Random same |A| seed={args.seed}", fill=0, font=font)

    y0 = margin + title_h
    canvas.paste(to_gray(lift), (margin, y0))
    canvas.paste(to_gray(rnd), (margin + w + gap, y0))

    draw.rectangle([0, 0, img_w - 1, img_h - 1], outline=0)
    canvas.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
