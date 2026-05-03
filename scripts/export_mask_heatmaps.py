# -*- coding: utf-8 -*-
"""
High-resolution side-by-side PNG: corner-free lift vs uniform random (same |A|).

Uses Pillow only. Use --scale (pixels per matrix cell) so the README can show
a modest width without upscaling a tiny bitmap.

From repo root:

    python scripts/export_mask_heatmaps.py --n 16 --scale 14
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
    p.add_argument("--scale", type=int, default=14, help="pixels per cell (larger = sharper PNG)")
    p.add_argument("--out", type=str, default="docs/images/mask-lift-vs-random.png")
    args = p.parse_args()

    rng = __import__("random").Random(args.seed)
    lift = corner_free_lift_from_behrend(args.n)
    rnd = random_mask_same_density((args.n, args.n), int(lift.sum()), rng)
    s = max(1, args.scale)

    def upsample(a: np.ndarray) -> Image.Image:
        g = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
        small = Image.fromarray(g, mode="L")
        w, h = small.size
        return small.resize((w * s, h * s), Image.Resampling.NEAREST)

    img_l = upsample(lift)
    img_r = upsample(rnd)
    gap = max(12, s)
    title_h = max(44, s * 3)
    margin = max(16, s)
    W = margin * 2 + img_l.width + gap + img_r.width
    H = margin * 2 + title_h + max(img_l.height, img_r.height)

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", max(16, s))
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(16, s))
        except OSError:
            font = ImageFont.load_default()

    cnt = int(lift.sum())
    draw.text((margin, margin), f"Corner-free lift |A|={cnt}", fill=(0, 0, 0), font=font)
    draw.text((margin + img_l.width + gap, margin), f"Uniform random same |A| (seed={args.seed})", fill=(0, 0, 0), font=font)
    y0 = margin + title_h
    canvas.paste(img_l.convert("RGB"), (margin, y0))
    canvas.paste(img_r.convert("RGB"), (margin + img_l.width + gap, y0))
    draw.rectangle([0, 0, W - 1, H - 1], outline=(80, 80, 80))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, "PNG", optimize=True)
    print(f"Wrote {out} ({W}x{H})")


if __name__ == "__main__":
    main()
