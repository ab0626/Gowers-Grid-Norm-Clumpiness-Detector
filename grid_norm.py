"""
Gowers (arithmetic) grid norm ‖f‖_{G(k,ℓ)} for f on a finite grid Ω₁ × Ω₂.

As in Jabber–Liu–Lovett–Ostuni–Sawhney (arXiv:2504.07006), for the complete
bipartite graph K_{k,ℓ} with left vertices [k] and right vertices [ℓ],

    ‖f‖_{G(k,ℓ)}^{kℓ}
        = E_{x₁,…,x_k ∈ Ω₁,  y₁,…,y_ℓ ∈ Ω₂}
            ∏_{i=1}^k ∏_{j=1}^ℓ f(x_i, y_j).

For [0,1]-valued f this is a multilinear average (a "box norm" when k=ℓ=2).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def suggested_grid_size_from_density(alpha: float) -> Tuple[int, int]:
    """
    Paper-scale parameter k ≈ log(1/α) (natural log; see §2 overview and Lemma 5.11
    with p = Ω(log(1/(α δ_D))/ε⁴) for ‖·‖_{G(2,p)}).

    Returns (k, ℓ) with k = max(2, ⌈ln(1/α)⌉) and ℓ = k.
    """
    a = max(float(alpha), 1e-300)
    k = max(2, int(math.ceil(math.log(1.0 / a))))
    return k, k


def _product_over_grid(
    f: np.ndarray,
    xs: Sequence[int],
    ys: Sequence[int],
) -> float:
    """∏_{i,j} f[xs[i], ys[j]] for f shape (n_rows, n_cols)."""
    p = 1.0
    for i in xs:
        for j in ys:
            p *= float(f[i, j])
    return p


def grid_norm_power_kl(
    f: np.ndarray,
    k: int,
    l: int,
    *,
    rng: Optional[random.Random] = None,
    n_samples: int = 50_000,
    method: str = "mc",
) -> float:
    """
    Estimate ‖f‖_{G(k,ℓ)}^{kℓ} = E[∏_{i,j} f(x_i, y_j)].

    method:
      - "mc": Monte Carlo (default), unbiased for the expectation inside the kℓ-th root
      - "exact": exact enumeration O(n^k m^l); only feasible for tiny grids
    """
    if k < 1 or l < 1:
        raise ValueError("k and l must be positive")
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError("f must be a 2D matrix")
    n, m = f.shape
    rng = rng or random.Random()

    if method == "exact":
        return _grid_norm_power_exact(f, k, l)

    acc = 0.0
    for _ in range(n_samples):
        xs = [rng.randrange(n) for _ in range(k)]
        ys = [rng.randrange(m) for _ in range(l)]
        acc += _product_over_grid(f, xs, ys)
    return acc / n_samples


def grid_norm(
    f: np.ndarray,
    k: int,
    l: int,
    *,
    rng: Optional[random.Random] = None,
    n_samples: int = 50_000,
    method: str = "mc",
) -> float:
    """‖f‖_{G(k,ℓ)} = (E[∏ f(x_i,y_j)])^{1/(kℓ)}."""
    power = grid_norm_power_kl(f, k, l, rng=rng, n_samples=n_samples, method=method)
    if power < 0:
        power = 0.0
    return power ** (1.0 / (k * l))


def exact_grid_norm_tuple_count(n_rows: int, n_cols: int, k: int, l: int) -> int:
    """Number of (x₁,…,x_k, y₁,…,y_ℓ) tuples in the K_{k,ℓ} grid-norm average."""
    if k < 0 or l < 0:
        raise ValueError("k and l must be nonnegative")
    return (n_rows**k) * (n_cols**l)


def _grid_norm_power_exact(f: np.ndarray, k: int, l: int) -> float:
    """Brute-force expectation over independent uniform x's and y's."""
    n, m = f.shape
    from itertools import product

    total = 0.0
    count = 0
    for xs in product(range(n), repeat=k):
        for ys in product(range(m), repeat=l):
            total += _product_over_grid(f, xs, ys)
            count += 1
    return total / count


def grid_norm_exact(
    f: np.ndarray,
    k: int,
    l: int,
    *,
    max_tuples: int = 25_000_000,
) -> float:
    """
    Exact ‖f‖_{G(k,ℓ)} via full enumeration. Raises ``ValueError`` if
    ``n^k m^ℓ > max_tuples`` (default 25M) to avoid accidental blow-ups.
    """
    f = np.asarray(f, dtype=np.float64)
    n, m = f.shape
    cnt = exact_grid_norm_tuple_count(n, m, k, l)
    if cnt > max_tuples:
        raise ValueError(
            f"exact grid norm would enumerate {cnt} tuples (> max_tuples={max_tuples}); "
            f"shrink the grid or (k,ℓ)."
        )
    return grid_norm(f, k, l, method="exact")


def balanced_indicator(mask: np.ndarray) -> np.ndarray:
    """
    Balanced {0,1} indicator: f = 1_A - α with α = |A|/|Ω| (mean zero on the grid).
    Values in [-α, 1-α]; grid norm is still well-defined but can be negative inside
    the kℓ-product expectation; we clip contributions in MC only for stability
    when demonstrating clumpiness of *nonnegative* patterns — see `balanced_nonneg_clip`.
    """
    mask = np.asarray(mask, dtype=np.float64)
    alpha = float(mask.mean())
    return mask - alpha


def balanced_nonneg_clip(f_bal: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Shift and scale to [0,1] for MC stability when using nonnegative grid norms."""
    lo = float(f_bal.min())
    hi = float(f_bal.max())
    if hi - lo < eps:
        return np.full_like(f_bal, 0.5)
    return (f_bal - lo) / (hi - lo)


@dataclass
class DensityIncrement:
    """A sub-rectangle where f correlates above global mean (empirical 'zoom-in')."""

    row_start: int
    row_end: int  # exclusive
    col_start: int
    col_end: int  # exclusive
    density_in_rect: float
    global_mean: float
    lift: float  # density_in_rect / global_mean


def best_rectangle_lift(
    mask: np.ndarray,
    min_rows: int = 2,
    min_cols: int = 2,
) -> DensityIncrement:
    """
    Heuristic density increment: maximize |A ∩ R|/|R| among subrectangles R,
    reporting lift over global density α = E[1_A]. This mimics 'zooming in' on
    the region that carries excess mass (cf. sifting / Lemma 3.3 in the paper).
    """
    a = np.asarray(mask, dtype=np.float64)
    n, m = a.shape
    alpha = float(a.mean())
    best: Optional[DensityIncrement] = None
    prefix = np.zeros((n + 1, m + 1), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            prefix[i + 1, j + 1] = (
                a[i, j] + prefix[i, j + 1] + prefix[i + 1, j] - prefix[i, j]
            )

    def rect_sum(r0: int, r1: int, c0: int, c1: int) -> float:
        return prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0]

    for r0 in range(n):
        for r1 in range(r0 + min_rows, n + 1):
            for c0 in range(m):
                for c1 in range(c0 + min_cols, m + 1):
                    area = (r1 - r0) * (c1 - c0)
                    dens = rect_sum(r0, r1, c0, c1) / area
                    lift = dens / alpha if alpha > 1e-15 else float("inf")
                    cand = DensityIncrement(r0, r1, c0, c1, dens, alpha, lift)
                    if best is None or cand.lift > best.lift:
                        best = cand
    assert best is not None
    return best


def random_binary_matrix(n: int, m: int, alpha: float, rng: random.Random) -> np.ndarray:
    """Bernoulli(α) entries — typically low ‖·‖_{G(k,ℓ)} for moderate k,ℓ."""
    return (np.array([[rng.random() < alpha for _ in range(m)] for _ in range(n)], dtype=np.float64))


def clumped_matrix(
    n: int,
    m: int,
    alpha: float,
    rng: random.Random,
    *,
    clump_frac: float = 0.2,
    clump_density: float = 0.92,
) -> np.ndarray:
    """
    Nearly the same global density α as a Bernoulli grid, but a small block is much denser
    (structured / corner-prone mass), which raises ‖1_A‖_{G(k,ℓ)} for moderate k,ℓ.
    """
    base = random_binary_matrix(n, m, alpha, rng)
    cr = max(2, int(n * clump_frac))
    cc = max(2, int(m * clump_frac))
    out = base.copy()
    for i in range(n - cr, n):
        for j in range(m - cc, m):
            out[i, j] = 1.0 if rng.random() < clump_density else 0.0
    # Thin the complement slightly so global α stays comparable to `base`
    target_ones = int(round(alpha * n * m))
    flat = out.ravel()
    ones = [i for i, v in enumerate(flat) if v > 0.5]
    zeros = [i for i, v in enumerate(flat) if v <= 0.5]
    current = len(ones)
    if current > target_ones:
        rng.shuffle(ones)
        for idx in ones:
            if current <= target_ones:
                break
            r, c = divmod(idx, m)
            if r >= n - cr and c >= m - cc:
                continue
            flat[idx] = 0.0
            current -= 1
    elif current < target_ones:
        rng.shuffle(zeros)
        for idx in zeros:
            if current >= target_ones:
                break
            r, c = divmod(idx, m)
            if r >= n - cr and c >= m - cc:
                continue
            flat[idx] = 1.0
            current += 1
    return flat.reshape(n, m)


def demo(
    n: int = 16,
    m: int = 16,
    alpha: float = 0.05,
    seed: int = 0,
    n_samples: int = 80_000,
) -> None:
    rng = random.Random(seed)
    k_sug, ell_sug = suggested_grid_size_from_density(alpha)
    # G(2,2) is the classical "box" norm; paper also uses G(2,p) with p ≳ log(1/(α δ_D)).
    k, ell = 2, 2

    rnd = random_binary_matrix(n, m, alpha, rng)
    clp = clumped_matrix(n, m, alpha, rng)

    # Paper uses nonnegative indicators on containers; ‖1_A‖_{G(k,ℓ)} detects clumping.
    nr1 = grid_norm(rnd, k, ell, rng=rng, n_samples=n_samples)
    nc1 = grid_norm(clp, k, ell, rng=rng, n_samples=n_samples)
    ks, es = min(k_sug, 3), min(ell_sug, 3)
    if ks != k or es != ell:
        nr_s = grid_norm(rnd, ks, es, rng=rng, n_samples=n_samples)
        nc_s = grid_norm(clp, ks, es, rng=rng, n_samples=n_samples)
    else:
        nr_s = nc_s = float("nan")
    # Balanced f = 1_A − α is signed; we report ‖·‖ on affine clip to [0,1] for stable MC.
    f_r = np.clip(balanced_nonneg_clip(balanced_indicator(rnd)), 0.0, 1.0)
    f_c = np.clip(balanced_nonneg_clip(balanced_indicator(clp)), 0.0, 1.0)
    nr = grid_norm(f_r, k, ell, rng=rng, n_samples=n_samples)
    nc = grid_norm(f_c, k, ell, rng=rng, n_samples=n_samples)

    inc_r = best_rectangle_lift(rnd)
    inc_c = best_rectangle_lift(clp)

    print("Gowers grid norm demo (arXiv:2504.07006-style K_{k,ℓ} averaging)")
    print(f"  grid {n}x{m}, target density α={alpha:.3f}")
    print(
        f"  paper-scale suggestion k ≈ log(1/α): ({k_sug},{ell_sug}); "
        f"demo computes ‖·‖_G(2,2) (box norm)"
    )
    print(f"  ‖1_A‖_G({k},{ell})  random ≈ {nr1:.5f}   clump ≈ {nc1:.5f}")
    if not math.isnan(nr_s):
        print(f"  ‖1_A‖_G({ks},{es}) (paper-scale k≈log 1/α) random ≈ {nr_s:.5f}   clump ≈ {nc_s:.5f}")
    print(f"  ‖clip(1_A−α)‖_G({k},{ell}) random ≈ {nr:.5f}   clump ≈ {nc:.5f}")
    print()
    print("Heuristic rectangle 'density increment' (best lift vs global α):")
    print(f"  random: rows [{inc_r.row_start},{inc_r.row_end}), cols [{inc_r.col_start},{inc_r.col_end}), "
          f"density={inc_r.density_in_rect:.3f}, lift={inc_r.lift:.2f}x")
    print(f"  clump:  rows [{inc_c.row_start},{inc_c.row_end}), cols [{inc_c.col_start},{inc_c.col_end}), "
          f"density={inc_c.density_in_rect:.3f}, lift={inc_c.lift:.2f}x")


if __name__ == "__main__":
    demo()
