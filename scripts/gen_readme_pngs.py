# -*- coding: utf-8 -*-
"""
Build PNG versions of README figures using Pillow only (no Cairo/Inkscape).

GitHub often fails to render <img src="*.svg"> in READMEs; PNG is reliable.
Run from repo root:

    python scripts/gen_readme_pngs.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"


def _font(size: int = 16):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def save(w: int, h: int, draw_fn, name: str) -> None:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), (250, 250, 250))
    dr = ImageDraw.Draw(im)
    draw_fn(dr, w, h)
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "PNG", optimize=True)
    print("Wrote", path)


def fig_corner_to_ap(dr, w: int, h: int) -> None:
    font = _font(15)
    font_b = _font(18)
    dr.text((16, 8), "Corner in grid -> 3-term AP on line", fill=(11, 61, 109), font=font_b)
    dr.text((16, 36), "Lift z=x+2y; corner maps to z, z+d, z+2d", fill=(51, 51, 51), font=font)
    # axes
    dr.line([(50, 200), (240, 200)], fill=(68, 68, 68), width=2)
    dr.line([(50, 200), (50, 90)], fill=(68, 68, 68), width=2)
    dr.ellipse((108, 165, 132, 189), fill=(198, 40, 40))
    dr.ellipse((188, 165, 212, 189), fill=(198, 40, 40))
    dr.ellipse((108, 105, 132, 129), fill=(198, 40, 40))
    dr.line([(120, 177), (200, 177)], fill=(21, 101, 192), width=3)
    dr.line([(120, 177), (120, 117)], fill=(21, 101, 192), width=3)
    dr.line([(270, 130), (360, 130)], fill=(85, 85, 85), width=2)
    dr.polygon([(368, 130), (355, 123), (355, 137)], fill=(85, 85, 85))
    dr.line([(400, 130), (640, 130)], fill=(68, 68, 68), width=2)
    for cx in (460, 520, 580):
        dr.ellipse((cx - 10, 118, cx + 10, 142), fill=(46, 125, 50))


def fig_k22(dr, w: int, h: int) -> None:
    font = _font(15)
    font_b = _font(18)
    dr.text((16, 8), "K(2,2) / box norm G(2,2)", fill=(11, 61, 109), font=font_b)
    dr.text((16, 34), "Four factors f(x1,y1)f(x1,y2)f(x2,y1)f(x2,y2)", fill=(34, 34, 34), font=font)
    dr.ellipse((90, 90, 136, 136), fill=(21, 101, 192), outline=(255, 255, 255), width=2)
    dr.ellipse((90, 170, 136, 216), fill=(21, 101, 192), outline=(255, 255, 255), width=2)
    dr.ellipse((420, 90, 466, 136), fill=(198, 40, 40), outline=(255, 255, 255), width=2)
    dr.ellipse((420, 170, 466, 216), fill=(198, 40, 40), outline=(255, 255, 255), width=2)
    for y1, y2 in [(113, 113), (113, 203), (203, 113), (203, 203)]:
        dr.line([(136, y1), (420, y2)], fill=(144, 164, 174), width=2)


def fig_k2k(dr, w: int, h: int) -> None:
    font = _font(14)
    font_b = _font(18)
    dr.text((16, 8), "K(2,k) / norm G(2,k)", fill=(11, 61, 109), font=font_b)
    dr.text((16, 34), "Product over j of f(x1,yj)*f(x2,yj)", fill=(34, 34, 34), font=font)
    dr.ellipse((70, 110, 110, 150), fill=(21, 101, 192), outline=(255, 255, 255), width=2)
    dr.ellipse((70, 200, 110, 240), fill=(21, 101, 192), outline=(255, 255, 255), width=2)
    xs = [320, 380, 440, 500, 560]
    for x in xs:
        dr.ellipse((x, 70, x + 32, 102), fill=(239, 108, 0), outline=(255, 255, 255), width=1)
    for x in xs:
        dr.line([(110, 130), (x + 16, 86)], fill=(120, 144, 156), width=1)
        dr.line([(110, 220), (x + 16, 86)], fill=(120, 144, 156), width=1)


def fig_balanced(dr, w: int, h: int) -> None:
    font = _font(16)
    font_b = _font(18)
    dr.text((16, 10), "Balanced f_A = 1_A - alpha", fill=(11, 61, 109), font=font_b)
    box = (20, 48, w - 20, h - 16)
    if hasattr(dr, "rounded_rectangle"):
        dr.rounded_rectangle(box, radius=8, outline=(249, 168, 37), width=2, fill=(255, 248, 225))
    else:
        dr.rectangle(box, outline=(249, 168, 37), width=2, fill=(255, 248, 225))
    dr.text((36, 70), "alpha = E[1_A] = |A|/|Omega|;  E[f_A]=0", fill=(34, 34, 34), font=font)
    dr.text((36, 100), "Proof uses f_A; demos may clip for nonnegative MC", fill=(34, 34, 34), font=font)


def fig_mc_exact(dr, w: int, h: int) -> None:
    font = _font(15)
    font_b = _font(18)
    dr.text((16, 8), "Monte Carlo vs exact", fill=(11, 61, 109), font=font_b)
    if hasattr(dr, "rounded_rectangle"):
        dr.rounded_rectangle((24, 52, 360, 200), radius=10, outline=(25, 118, 210), width=2, fill=(227, 242, 253))
        dr.rounded_rectangle((380, 52, w - 24, 200), radius=10, outline=(46, 125, 50), width=2, fill=(232, 245, 233))
    else:
        dr.rectangle((24, 52, 360, 200), outline=(25, 118, 210), width=2, fill=(227, 242, 253))
        dr.rectangle((380, 52, w - 24, 200), outline=(46, 125, 50), width=2, fill=(232, 245, 233))
    dr.text((44, 88), "MC: scalable, noisy sparse", fill=(34, 34, 34), font=font)
    dr.text((400, 88), "Exact: zero noise, tuple cap", fill=(34, 34, 34), font=font)


def fig_density(dr, w: int, h: int) -> None:
    font = _font(15)
    font_b = _font(18)
    dr.text((16, 8), "best_rectangle_lift (demo)", fill=(11, 61, 109), font=font_b)
    ox, oy = 40, 50
    cs = 36
    for i in range(5):
        for j in range(4):
            x0, y0 = ox + i * cs, oy + j * cs
            light = (i, j) in [(3, 2), (4, 2), (3, 3), (4, 3)]
            fill = (255, 204, 128) if light else (236, 239, 241)
            dr.rectangle([x0, y0, x0 + cs - 1, y0 + cs - 1], fill=fill, outline=(176, 190, 197))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    save(720, 280, fig_corner_to_ap, "fig-corner-to-ap.png")
    save(760, 300, fig_k22, "fig-k22-bipartite.png")
    save(820, 300, fig_k2k, "fig-k2k-bipartite.png")
    save(760, 200, fig_balanced, "fig-balanced-indicator.png")
    save(780, 240, fig_mc_exact, "fig-mc-vs-exact.png")
    save(760, 260, fig_density, "fig-density-increment.png")
    print("Done.")


if __name__ == "__main__":
    main()
