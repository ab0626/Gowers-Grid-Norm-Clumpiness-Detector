# -*- coding: utf-8 -*-
"""
Empirical "phase transition" sweep: ‖1_A‖_{G(k,ℓ)} vs k for matched-density pairs.

Compares a Behrend raster mask to a uniform random mask with the same |A| (same α).
This is a *toy diagnostic* in the spirit of §2.2 / higher G(2,k) parameters — not a
proof of a density increment or a corner-free vs random separation (Behrend is AP-free,
not corner-free).

Requires matplotlib (requirements-dev.txt). Headless-safe (Agg backend).

Usage (from repo root):

    python scripts/phase_transition_sweep.py --n 10 --l 2 --k-max 5 --out docs/images/phase-transition-behrend-vs-random.png
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from behrend import behrend_matrix, random_mask_same_density  # noqa: E402
from grid_norm import (  # noqa: E402
    DEFAULT_EXACT_AUTO_MAX_TUPLES,
    exact_grid_norm_tuple_count,
    grid_norm,
    resolve_grid_norm_method,
    suggested_grid_size_from_density,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot ‖·‖_{G(k,ℓ)} vs k: Behrend vs random (matched |A|).")
    p.add_argument("--n", type=int, default=10, help="grid side n×n")
    p.add_argument("--l", type=int, default=2, help="right parameter ℓ in K_{k,ℓ}")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=200_000, help="MC samples when auto picks MC")
    p.add_argument("--max-tuples", type=int, default=25_000_000)
    p.add_argument("--exact-auto-max", type=int, default=DEFAULT_EXACT_AUTO_MAX_TUPLES)
    p.add_argument("--out", type=str, default="docs/images/phase-transition-behrend-vs-random.png")
    p.add_argument("--csv", type=str, default="", help="optional path to write k,behrend,random rows")
    args = p.parse_args()

    if args.k_min < 1 or args.k_max < args.k_min:
        raise SystemExit("need 1 ≤ k_min ≤ k_max")

    rng = random.Random(args.seed)
    n = args.n
    l = args.l
    B = behrend_matrix(n)
    cnt = int(B.sum())
    R = random_mask_same_density((n, n), cnt, rng)
    alpha = float(B.mean())
    k_sug, _ = suggested_grid_size_from_density(alpha)

    ks = list(range(args.k_min, args.k_max + 1))
    yb: list[float] = []
    yr: list[float] = []
    resolved_tags: list[str] = []

    for k in ks:
        cnt_t = exact_grid_norm_tuple_count(n, n, k, l)
        rslv = resolve_grid_norm_method(
            n,
            n,
            k,
            l,
            method="auto",
            max_tuples=args.max_tuples,
            exact_auto_max_tuples=args.exact_auto_max,
        )
        resolved_tags.append(f"{k}:{rslv}")
        w: list[str] = []
        yb.append(
            grid_norm(
                B,
                k,
                l,
                rng=rng,
                n_samples=args.samples,
                method="auto",
                max_tuples=args.max_tuples,
                exact_auto_max_tuples=args.exact_auto_max,
                out_warnings=w,
            )
        )
        if w:
            for line in w:
                print(f"[k={k} Behrend] {line}", file=sys.stderr)
        w2: list[str] = []
        yr.append(
            grid_norm(
                R,
                k,
                l,
                rng=rng,
                n_samples=args.samples,
                method="auto",
                max_tuples=args.max_tuples,
                exact_auto_max_tuples=args.exact_auto_max,
                out_warnings=w2,
            )
        )
        if w2:
            for line in w2:
                print(f"[k={k} random] {line}", file=sys.stderr)

    crossover = None
    for i, k in enumerate(ks):
        if yb[i] > yr[i] * 1.02:  # small separation band
            crossover = k
            break

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    ax.plot(ks, yb, "o-", label="Behrend raster ‖1_B‖", color="#1565c0", linewidth=2, markersize=6)
    ax.plot(ks, yr, "s--", label="Uniform random ‖1_R‖ (same |A|)", color="#c62828", linewidth=2, markersize=6)
    ax.axvline(k_sug, color="#6a1b9a", linestyle=":", linewidth=1.5, label=f"paper-scale k≈⌈ln(1/α)⌉ = {k_sug}")
    ax.set_xlabel("k (left side of K_{k,ℓ})")
    ax.set_ylabel(f"‖1_A‖_{{G(k,{l})}}")
    ax.set_title(
        f"Empirical phase-style sweep (n={n}, ℓ={l}, α≈{alpha:.4f}, |A|={cnt})\n"
        "Toy: Behrend structure vs noise — not corner-free vs random"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.resolve()}")
    print(f"  resolved methods per k: {', '.join(resolved_tags)}")
    if crossover is not None:
        print(f"  first k with Behrend > random (×1.02): k = {crossover}")
    else:
        print("  no clear Behrend > random separation in this band (try larger n or different seed)")

    if args.csv:
        cp = Path(args.csv)
        cp.parent.mkdir(parents=True, exist_ok=True)
        with cp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["k", "behrend", "random", "resolved_method"])
            for i, k in enumerate(ks):
                w.writerow([k, yb[i], yr[i], resolved_tags[i].partition(":")[2]])


if __name__ == "__main__":
    main()
