# Figure assets (`docs/images/`)

Paths are relative to the **repository root**.

## README gallery (PNG — what GitHub displays)

| # | PNG file | Generator |
|---|----------|-----------|
| 1 | `fig-corner-to-ap.png` | `python scripts/gen_readme_pngs.py` |
| 2 | `fig-k22-bipartite.png` | same |
| 3 | `fig-k2k-bipartite.png` | same |
| 4 | `fig-balanced-indicator.png` | same |
| 5 | `fig-mc-vs-exact.png` | same |
| 6 | `fig-density-increment.png` | same |
| 7 | `mask-lift-vs-random.png` | `python scripts/export_mask_heatmaps.py --n 16 --scale 14` |

**Why PNG in the README?** GitHub’s Markdown renderer often shows **broken icons** for `<img src="…svg">`. PNGs are reliable; SVGs here are optional sources for hand-editing (see `fig-*.svg`).

## Optional SVG sources

Matching `fig-*.svg` files use UTF-8 + ASCII labels. Regenerate PNGs after editing SVGs (or edit `scripts/gen_readme_pngs.py` instead).
