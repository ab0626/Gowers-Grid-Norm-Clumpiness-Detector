# Figure assets (`docs/images/`)

All paths are relative to the **repository root** (for example `docs/images/fig-k22-bipartite.svg`).

| # | File | Description |
|---|------|-------------|
| 1 | `fig-corner-to-ap.svg` | Corner \((x,y),(x+d,y),(x,y+d)\) maps to \(z,z+d,z+2d\) under \(z=x+2y\). |
| 2 | `fig-k22-bipartite.svg` | Complete bipartite \(K_{2,2}\) / box norm \(G(2,2)\). |
| 3 | `fig-k2k-bipartite.svg` | Complete bipartite \(K_{2,k}\) / norm \(G(2,k)\). |
| 4 | `fig-balanced-indicator.svg` | Balanced indicator \(f_A=\mathbf 1_A-\alpha\). |
| 5 | `fig-mc-vs-exact.svg` | Monte Carlo vs exact enumeration branches. |
| 6 | `fig-density-increment.svg` | Heuristic rectangle / density increment. |
| 7 | `mask-lift-vs-random.png` | **Generated** — run `python scripts/export_mask_heatmaps.py` from repo root. |

Vector figures are hand-authored SVG. The PNG is not required for tests to pass except the export smoke test, which writes to a temp path; for a **pretty README clone**, run the heatmap script once so `mask-lift-vs-random.png` exists here.
