"""
Exact G(2, k) grid-norm "microscope" — no Monte Carlo noise.

For f on an n×m grid, ‖f‖_{G(2,k)} is the (2k)-th root of

    E_{x1,x2,y1,…,yk}  ∏_{j=1}^k f(x1,yj) f(x2,yj),

i.e. the K_{2,k} specialization of the complete bipartite formula in grid_norm.py.
This matches the paper's ‖·‖_{G(2,k)} family (e.g. Theorem 3.5, Lemma 5.11 scale).

Use on *small* grids only: enumeration cost is Θ(n^2 m^k).
"""

from __future__ import annotations

import argparse
import random
from typing import Iterable, List, Tuple

import numpy as np

from behrend import random_mask_same_density
from corner_lift import corner_free_lift_from_behrend
from grid_norm import exact_grid_norm_tuple_count, grid_norm_exact, suggested_grid_size_from_density


def scan_exact_g2k(
    f: np.ndarray,
    k_values: Iterable[int],
    *,
    max_tuples: int = 25_000_000,
) -> List[Tuple[int, float, int]]:
    """
    Returns list of (k, ‖f‖_{G(2,k)}, tuple_count) for each k in k_values
    that fits under ``max_tuples``. Skips k with infeasible budget.
    """
    f = np.asarray(f, dtype=np.float64)
    n, m = f.shape
    out: List[Tuple[int, float, int]] = []
    for k in k_values:
        if k < 1:
            continue
        tc = exact_grid_norm_tuple_count(n, m, 2, k)
        if tc > max_tuples:
            continue
        val = grid_norm_exact(f, 2, k, max_tuples=max_tuples)
        out.append((k, val, tc))
    return out


def _default_k_values(alpha: float, k_cap: int = 8) -> List[int]:
    k_sug, _ = suggested_grid_size_from_density(alpha)
    hi = min(max(k_sug, 3), k_cap)
    return list(range(2, hi + 1))


def run_microscope(
    n: int = 8,
    *,
    seed: int = 0,
    max_tuples: int = 25_000_000,
    k_cap: int = 8,
) -> None:
    rng = random.Random(seed)
    lift = corner_free_lift_from_behrend(n)
    rnd = random_mask_same_density((n, n), int(lift.sum()), rng)
    alpha = float(lift.mean())
    ks = _default_k_values(alpha, k_cap=k_cap)

    print(f"Exact G(2,k) microscope  (n={n}, α={alpha:.5f}, |A|={int(lift.sum())}, max_tuples={max_tuples})")
    print(f"  paper-scale k≈ln(1/α) ≈ {suggested_grid_size_from_density(alpha)[0]}; scanning k in {ks}")
    print()
    print(f"{'k':>3}  {'tuples':>12}  {'‖1_lift‖':>14}  {'‖1_rand‖':>14}")
    print("-" * 52)

    for k in ks:
        tc = exact_grid_norm_tuple_count(n, n, 2, k)
        if tc > max_tuples:
            print(f"{k:>3}  {tc:>12}  {'(skip)':>14}  {'(skip)':>14}")
            continue
        nl = grid_norm_exact(lift, 2, k, max_tuples=max_tuples)
        nr = grid_norm_exact(rnd, 2, k, max_tuples=max_tuples)
        print(f"{k:>3}  {tc:>12}  {nl:>14.8f}  {nr:>14.8f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Exact G(2,k) grid norms (lift vs random, same |A|).")
    p.add_argument("--n", type=int, default=8, help="grid side length (keep small; cost ~ n^{2+k})")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tuples", type=int, default=25_000_000, help="enumeration safety cap")
    p.add_argument("--k-cap", type=int, default=8, help="largest k to try (still subject to cap)")
    args = p.parse_args()
    run_microscope(n=args.n, seed=args.seed, max_tuples=args.max_tuples, k_cap=args.k_cap)


if __name__ == "__main__":
    main()
