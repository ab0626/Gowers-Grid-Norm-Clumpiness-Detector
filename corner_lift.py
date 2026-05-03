"""
Corner-free lift A ⊆ [0,n-1]^2 from an AP-free set S ⊆ Z:

    (x,y) ∈ A   ⇔   x + 2y ∈ S   (integer sum, no modular reduction).

If z, z+d, z+2d ∈ S with d ≠ 0 then z = x+2y gives (x,y), (x+d,y), (x,y+d) ∈ A,
so any 3-term arithmetic progression in S would yield a corner in A. Hence
integer AP-free S ⇒ A has no corners in the usual sense on this coordinate range.

This closes the loop suggested in the README: combine Behrend-type S with the
standard linear lift used in corners↔progressions reductions (cf. paper §1 / §2).
"""

from __future__ import annotations

import math
import random
from typing import Sequence, Set, Tuple

import numpy as np

from behrend import behrend_indices, random_mask_same_density
from grid_norm import grid_norm, suggested_grid_size_from_density


def lift_mask_x_plus_2y(n: int, S: Set[int] | Sequence[int]) -> np.ndarray:
    """
    n×n binary matrix: A[x,y] = 1 iff (x + 2*y) ∈ S as integers (x,y ∈ {0,…,n-1}).

    Require S ⊆ {0, …, 3n-3} so that every value x+2y lies in the ambient interval
    where S lives (max x+2y = (n-1) + 2(n-1) = 3n-3).
    """
    Sset = set(S)
    a = np.zeros((n, n), dtype=np.float64)
    for x in range(n):
        for y in range(n):
            t = x + 2 * y
            if t in Sset:
                a[x, y] = 1.0
    return a


def behrend_line_for_lift(n: int) -> set[int]:
    """AP-free S ⊂ {0,…,3n-4} from Behrend construction on M = 3n-2 values."""
    M = max(3 * n - 2, 1)
    return set(behrend_indices(M))


def corner_free_lift_from_behrend(n: int) -> np.ndarray:
    """Corner-free (integer-grid) mask using Behrend AP-free S and x+2y ∈ S."""
    return lift_mask_x_plus_2y(n, behrend_line_for_lift(n))


def has_corner_in_grid(mask: np.ndarray) -> bool:
    """
    Brute-force search for (x,y,d) with d ≠ 0 and x,x+d,y,y+d all in [0,n-1]
    with mask[x,y] = mask[x+d,y] = mask[x,y+d] = 1.
    """
    a = np.asarray(mask, dtype=np.float64)
    n, m = a.shape
    if n != m:
        raise ValueError("only implemented for square masks")
    for x in range(n):
        for y in range(n):
            if a[x, y] < 0.5:
                continue
            for d in range(-(n - 1), n):
                if d == 0:
                    continue
                x2, y2 = x + d, y + d
                if 0 <= x2 < n and 0 <= y2 < n:
                    if a[x2, y] >= 0.5 and a[x, y2] >= 0.5:
                        return True
    return False


def compare_corner_free_lift_vs_random(
    n: int = 10,
    *,
    seed: int = 2,
    n_samples: int = 400_000,
    force_g22: bool = True,
) -> Tuple[float, float, float, int, int]:
    """
    Returns (‖1_A‖_{G(ks,ks)}, ‖1_random‖_{G(ks,ks)}, α, |A|, ks) with |A| matched.

    Default ``force_g22=True``: use the box norm G(2,2) (paper's initial obstruction
    norm) for a stable Monte Carlo signal at modest α.
    """
    rng = random.Random(seed)
    lift = corner_free_lift_from_behrend(n)
    cnt = int(lift.sum())
    alpha = float(lift.mean())
    rnd = random_mask_same_density((n, n), cnt, rng)
    if force_g22:
        ks = 2
    else:
        ks, _ = suggested_grid_size_from_density(alpha)
        ks = min(ks, 3)
        if alpha < 0.08:
            ks = min(ks, 2)
    nl = grid_norm(lift, ks, ks, rng=rng, n_samples=n_samples)
    nr = grid_norm(rnd, ks, ks, rng=rng, n_samples=n_samples)
    return nl, nr, alpha, cnt, ks


def _main() -> None:
    n = 10
    lift = corner_free_lift_from_behrend(n)
    cor = has_corner_in_grid(lift)
    nl, nr, alpha, cnt, ks = compare_corner_free_lift_vs_random(n=n)
    print("Corner-free lift: A[x,y]=1 iff x+2y in Behrend AP-free S ⊂ {0,…,3n-3}")
    print(f"  n={n}, |A|={cnt}, α={alpha:.5f}, brute corner found? {cor}")
    print(f"  G({ks},{ks}) box norm, MC 400k: ‖1_lift‖ ≈ {nl:.5f}   ‖1_random‖ ≈ {nr:.5f}")


if __name__ == "__main__":
    _main()
