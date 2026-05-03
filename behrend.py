"""
Behrend's classical AP-free subset of {0, …, M−1} (sphere in digit space),
embedded into an n×n grid for comparison with `grid_norm`.

This is independent of the corners theorem (Behrend rules out *arithmetic*
progressions in Z, not corner patterns in G×G), but it gives a strongly
structured sparse set at fixed density—useful as an extremal "clumpy"
sanity check alongside `random_binary_matrix`.
"""

from __future__ import annotations

import argparse
import math
import random
from itertools import product
from typing import List, Optional, Tuple

import numpy as np

from grid_norm import (
    PIPE_FORMAT,
    grid_norm,
    suggested_grid_size_from_density,
    write_grid_norm_pipe_v1,
)


def _value_from_digits(digits: Tuple[int, ...], q: int) -> int:
    return sum(d * (q**i) for i, d in enumerate(digits))


def behrend_indices(M: int) -> List[int]:
    """
    Largest known Behrend-type shell inside {0, …, M−1} by brute search over
    small (d, m) with base q = 2d−1 and digits in {0, …, d−1}.

    Returns sorted distinct integers in [0, M).
    """
    if M <= 1:
        return [0] if M == 1 else []
    best: List[int] = []
    max_d = max(3, int(math.ceil(math.sqrt(math.log(M + 2.0)))) + 6)
    for d in range(2, max_d):
        q = 2 * d - 1
        m = 0
        while q ** (m + 1) <= M:
            m += 1
        if m < 1:
            continue
        if d**m > 2_000_000:
            continue
        shells: dict[int, List[Tuple[int, ...]]] = {}
        for digits in product(range(d), repeat=m):
            r = sum(x * x for x in digits)
            shells.setdefault(r, []).append(digits)
        for r, tuples in shells.items():
            vals = []
            for dig in tuples:
                v = _value_from_digits(dig, q)
                if v < M:
                    vals.append(v)
            if len(vals) > len(best):
                best = vals
    if not best:
        return [0]
    return sorted(set(best))


def behrend_matrix(n: int) -> np.ndarray:
    """n×n binary matrix: 1 at (t // n, t % n) for t in Behrend subset of [0, n²)."""
    M = n * n
    idx = behrend_indices(M)
    a = np.zeros((n, n), dtype=np.float64)
    for t in idx:
        if 0 <= t < M:
            a[t // n, t % n] = 1.0
    return a


def random_mask_same_density(shape: Tuple[int, int], count: int, rng: random.Random) -> np.ndarray:
    """Uniform random |A| = count subset of an n×m grid."""
    n, m = shape
    nm = n * m
    count = max(0, min(count, nm))
    flat = [0.0] * nm
    pos = list(range(nm))
    rng.shuffle(pos)
    for p in pos[:count]:
        flat[p] = 1.0
    return np.array(flat, dtype=np.float64).reshape(n, m)


def compare_behrend_vs_random(
    n: int = 10,
    *,
    seed: int = 1,
    n_samples: int = 400_000,
) -> Tuple[float, float, float, int, int]:
    """
    Returns (‖1_Behrend‖_{G(ks,ks)}, ‖1_random‖_{G(ks,ks)}, α, |A|, ks).

    For very sparse A, ‖·‖_{G(k,k)} MC estimates need many samples; we cap k at 3
    but force k=2 when α is small so the K_{k,k} product is not almost always zero.
    """
    rng = random.Random(seed)
    B = behrend_matrix(n)
    cnt = int(B.sum())
    alpha = float(B.mean())
    rnd = random_mask_same_density((n, n), cnt, rng)
    ks, _ = suggested_grid_size_from_density(alpha)
    ks = min(ks, 3)
    if alpha < 0.08:
        ks = min(ks, 2)
    nb = grid_norm(B, ks, ks, rng=rng, n_samples=n_samples, method="auto")
    nr = grid_norm(rnd, ks, ks, rng=rng, n_samples=n_samples, method="auto")
    return nb, nr, alpha, cnt, ks


def _print_compare(n: int = 10) -> None:
    nb, nr, alpha, cnt, ks = compare_behrend_vs_random(n=n)
    print("Behrend (AP-free subset of [0,n²), raster-embedded in n×n) vs random |A|")
    print(f"  grid {n}×{n}, |A|={cnt}, α={alpha:.5f}, MC samples=400k, G({ks},{ks})")
    print(f"  ‖1_Behrend‖ ≈ {nb:.5f}   ‖1_random‖ ≈ {nr:.5f}")


def _main() -> None:
    ap = argparse.ArgumentParser(description="Behrend shell vs random (grid norms)")
    ap.add_argument("-n", "--n", type=int, default=10, help="grid side for n×n Behrend raster")
    ap.add_argument(
        "--export-pipe",
        type=str,
        metavar="PATH",
        help=f"write Behrend n×n mask as {PIPE_FORMAT} JSON and exit",
    )
    args = ap.parse_args()
    if args.export_pipe:
        b = behrend_matrix(args.n)
        write_grid_norm_pipe_v1(
            args.export_pipe,
            b,
            meta={"source": "behrend_matrix", "n": args.n, "|A|": int(b.sum())},
        )
        print(f"Wrote {args.export_pipe!r} ({args.n}×{args.n}, |A|={int(b.sum())})")
    else:
        _print_compare(n=args.n)


if __name__ == "__main__":
    _main()
