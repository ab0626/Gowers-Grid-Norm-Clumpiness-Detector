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

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PIPE_FORMAT = "grid_norm_pipe_v1"
"""JSON tag consumed by ``load_grid_norm_pipe_v1`` / ``write_grid_norm_pipe_v1``."""

DEFAULT_EXACT_AUTO_MAX_TUPLES = 10**7
"""If ``n^k m^ℓ`` is at most this (and ≤ ``max_tuples``), ``method='auto'`` picks exact."""

MIN_EFFECTIVE_MC_PRODUCTS = 30.0
"""Heuristic floor for ``n_samples × α^{kℓ}`` before emitting an MC variance advisory."""


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
    max_tuples: int = 25_000_000,
    exact_auto_max_tuples: int = DEFAULT_EXACT_AUTO_MAX_TUPLES,
    out_warnings: Optional[List[str]] = None,
) -> float:
    """
    Estimate ‖f‖_{G(k,ℓ)}^{kℓ} = E[∏_{i,j} f(x_i, y_j)].

    method:
      - ``"mc"``: Monte Carlo (default), unbiased for the expectation inside the kℓ-th root
      - ``"exact"``: exact enumeration O(n^k m^ℓ); only feasible for tiny grids
      - ``"auto"``: exact iff ``n^k m^ℓ ≤ min(max_tuples, exact_auto_max_tuples)``, else MC
    """
    if k < 1 or l < 1:
        raise ValueError("k and l must be positive")
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError("f must be a 2D matrix")
    n, m = f.shape
    rng = rng or random.Random()

    resolved = resolve_grid_norm_method(
        n, m, k, l, method=method, max_tuples=max_tuples, exact_auto_max_tuples=exact_auto_max_tuples
    )
    if resolved == "exact":
        cnt = exact_grid_norm_tuple_count(n, m, k, l)
        if cnt > max_tuples:
            raise ValueError(
                f"auto/exact grid norm would enumerate {cnt} tuples (> max_tuples={max_tuples})"
            )
        return _grid_norm_power_exact(f, k, l)

    if out_warnings is not None:
        w = mc_variance_advisory(f, k, l, n_samples)
        if w is not None:
            out_warnings.append(w)

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
    max_tuples: int = 25_000_000,
    exact_auto_max_tuples: int = DEFAULT_EXACT_AUTO_MAX_TUPLES,
    negative_power_handling: str = "clip_nonneg",
    out_warnings: Optional[List[str]] = None,
) -> float:
    """
    ``‖f‖_{G(k,ℓ)} = (E[∏ f(x_i,y_j)])^{1/(kℓ)}``.

    For signed ``f`` (e.g. balanced ``1_A-α``), the multilinear average ``power`` can be
    negative. ``negative_power_handling`` controls the real ``(kℓ)``-th root step:

    - ``clip_nonneg`` (default): ``max(0, power)`` then root — nonnegative norm; matches
      nonnegative-indicator usage in the paper’s density-increment discussion when only
      magnitudes matter for clumping demos.
    - ``abs_then_root``: ``|power|^{1/(kℓ)}`` — magnitude of the multilinear average (Gowers
      box-type control on absolute value).
    - ``nan_if_negative``: return ``nan`` when ``power < 0`` (no ad-hoc clipping).
    """
    if negative_power_handling not in ("clip_nonneg", "abs_then_root", "nan_if_negative"):
        raise ValueError(
            "negative_power_handling must be 'clip_nonneg', 'abs_then_root', or 'nan_if_negative'"
        )
    power = grid_norm_power_kl(
        f,
        k,
        l,
        rng=rng,
        n_samples=n_samples,
        method=method,
        max_tuples=max_tuples,
        exact_auto_max_tuples=exact_auto_max_tuples,
        out_warnings=out_warnings,
    )
    if power < 0:
        if negative_power_handling == "nan_if_negative":
            return float("nan")
        if negative_power_handling == "abs_then_root":
            return abs(power) ** (1.0 / (k * l))
        power = 0.0
    return power ** (1.0 / (k * l))


def exact_grid_norm_tuple_count(n_rows: int, n_cols: int, k: int, l: int) -> int:
    """Number of (x₁,…,x_k, y₁,…,y_ℓ) tuples in the K_{k,ℓ} grid-norm average."""
    if k < 0 or l < 0:
        raise ValueError("k and l must be nonnegative")
    return (n_rows**k) * (n_cols**l)


def resolve_grid_norm_method(
    n: int,
    m: int,
    k: int,
    l: int,
    *,
    method: str,
    max_tuples: int,
    exact_auto_max_tuples: int,
) -> str:
    """
    Resolve ``method='auto'`` to ``'exact'`` or ``'mc'``.

    Exact is chosen iff ``n^k m^ℓ ≤ min(max_tuples, exact_auto_max_tuples)`` — the
    same tuple budget that ``grid_norm_exact`` would enumerate (cf. multilinear
    control, paper §5).
    """
    if method not in ("mc", "exact", "auto"):
        raise ValueError("method must be 'mc', 'exact', or 'auto'")
    if method != "auto":
        return method
    cnt = exact_grid_norm_tuple_count(n, m, k, l)
    cap = min(int(max_tuples), int(exact_auto_max_tuples))
    return "exact" if cnt <= cap else "mc"


def mc_variance_advisory(
    f: np.ndarray,
    k: int,
    l: int,
    n_samples: int,
) -> Optional[str]:
    """
    Emit a human-readable warning when Monte Carlo is likely very noisy in the
    sparse {0,1} regime (heuristic: few expected ``∏ 1`` products per sample).
    """
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2 or k < 1 or l < 1 or n_samples < 1:
        return None
    alpha_eff = float(np.mean(np.clip(f, 0.0, 1.0)))
    if alpha_eff <= 0.0:
        return (
            f"MC variance risk: α≈0 on support; products are almost always 0 for k={k}, l={l}. "
            "Use exact/auto on a small grid or increase n_samples."
        )
    exp_hit = alpha_eff ** (k * l)
    effective = n_samples * exp_hit
    if effective >= MIN_EFFECTIVE_MC_PRODUCTS:
        return None
    eff_s = f"{effective:.3g}" if effective < 0.01 else f"{effective:.1f}"
    return (
        f"MC variance risk: expected ≈{eff_s} all-one products per run (α≈{alpha_eff:.4g}, k={k}, l={l}, "
        f"n_samples={n_samples}). Prefer method='auto' or 'exact' when n^k m^ℓ is modest, "
        "or raise --samples (cf. paper §3.5 / relative sifting — sparse multilinear averages are noisy)."
    )


def load_grid_norm_pipe_v1(path: str | Path) -> np.ndarray:
    """
    Load a ``grid_norm_pipe_v1`` JSON export (matrix of reals / {0,1} indicators).

    Required keys: ``matrix`` (rectangular list-of-lists) or ``data`` with ``shape`` [n, m].
    Optional: ``format`` must be ``grid_norm_pipe_v1`` if present.
    """
    p = Path(path)
    raw: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    fmt = raw.get("format")
    if fmt is not None and fmt != PIPE_FORMAT:
        raise ValueError(f"unknown pipe format {fmt!r}; expected {PIPE_FORMAT!r}")
    mat = raw.get("matrix")
    if mat is None:
        mat = raw.get("data")
    if mat is None:
        raise ValueError("pipe JSON must contain 'matrix' or 'data'")
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim == 1:
        shape = raw.get("shape")
        if shape is None or len(shape) != 2:
            raise ValueError("1d 'data' requires integer 'shape': [n_rows, n_cols]")
        arr = arr.reshape(int(shape[0]), int(shape[1]))
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    return arr


def write_grid_norm_pipe_v1(
    path: str | Path,
    f: np.ndarray,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Write ``grid_norm_pipe_v1`` JSON for ``load_grid_norm_pipe_v1`` / external tools."""
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError("f must be 2D")
    payload = {
        "format": PIPE_FORMAT,
        "version": 1,
        "matrix": f.tolist(),
        "shape": [int(f.shape[0]), int(f.shape[1])],
        "meta": meta or {},
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    return grid_norm(f, k, l, method="exact", max_tuples=max_tuples)


def balanced_indicator(mask: np.ndarray) -> np.ndarray:
    """
    Balanced {0,1} indicator: f = 1_A - α with α = |A|/|Ω| (mean zero on the grid).
    Values in [-α, 1-α]; the multilinear average E[∏f] can be **negative**.
    For ``grid_norm(f,…)``, use ``negative_power_handling`` (``clip_nonneg``, ``abs_then_root``,
    ``nan_if_negative``) to control the real ``(kℓ)``-th root step; for nonnegative MC demos
    on ‖·‖, prefer ``balanced_nonneg_clip`` (§4 / density-increment style positivity in the paper).
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
    warns: List[str] = []

    rnd = random_binary_matrix(n, m, alpha, rng)
    clp = clumped_matrix(n, m, alpha, rng)

    # Paper uses nonnegative indicators on containers; ‖1_A‖_{G(k,ℓ)} detects clumping.
    # method="auto" uses exact enumeration on small tuple budgets (sparse-regime stability).
    nr1 = grid_norm(
        rnd, k, ell, rng=rng, n_samples=n_samples, method="auto", out_warnings=warns
    )
    nc1 = grid_norm(
        clp, k, ell, rng=rng, n_samples=n_samples, method="auto", out_warnings=warns
    )
    ks, es = min(k_sug, 3), min(ell_sug, 3)
    if ks != k or es != ell:
        nr_s = grid_norm(
            rnd, ks, es, rng=rng, n_samples=n_samples, method="auto", out_warnings=warns
        )
        nc_s = grid_norm(
            clp, ks, es, rng=rng, n_samples=n_samples, method="auto", out_warnings=warns
        )
    else:
        nr_s = nc_s = float("nan")
    # Balanced f = 1_A − α is signed; we report ‖·‖ on affine clip to [0,1] for stable MC.
    f_r = np.clip(balanced_nonneg_clip(balanced_indicator(rnd)), 0.0, 1.0)
    f_c = np.clip(balanced_nonneg_clip(balanced_indicator(clp)), 0.0, 1.0)
    nr = grid_norm(f_r, k, ell, rng=rng, n_samples=n_samples, method="auto", out_warnings=warns)
    nc = grid_norm(f_c, k, ell, rng=rng, n_samples=n_samples, method="auto", out_warnings=warns)

    inc_r = best_rectangle_lift(rnd)
    inc_c = best_rectangle_lift(clp)

    print("Gowers grid norm demo (arXiv:2504.07006-style K_{k,ℓ} averaging)")
    print(f"  grid {n}x{m}, target density α={alpha:.3f}")
    if warns:
        print("  Notes (auto MC / variance):")
        for w in dict.fromkeys(warns):
            print(f"    • {w}")
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


def main_cli(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description=(
            "Compute ‖f‖_{G(k,ℓ)} from a grid_norm_pipe_v1 JSON matrix "
            "(use Behrend --export-pipe or write_grid_norm_pipe_v1)."
        )
    )
    p.add_argument("--pipe", type=str, required=True, metavar="PATH", help="grid_norm_pipe_v1 JSON path")
    p.add_argument("--k", type=int, default=2, help="left parameter k in K_{k,ℓ}")
    p.add_argument("--l", type=int, default=2, help="right parameter ℓ in K_{k,ℓ}")
    p.add_argument("--samples", type=int, default=80_000, help="Monte Carlo draws when method is mc")
    p.add_argument(
        "--method",
        choices=("auto", "exact", "mc"),
        default="auto",
        help="auto: exact when n^k m^ℓ ≤ min(--max-tuples, --exact-auto-max); else MC",
    )
    p.add_argument(
        "--exact-auto-max",
        type=int,
        default=DEFAULT_EXACT_AUTO_MAX_TUPLES,
        metavar="N",
        help="tuple budget threshold for auto-selecting exact enumeration",
    )
    p.add_argument(
        "--max-tuples",
        type=int,
        default=25_000_000,
        metavar="N",
        help="hard refusal cap for exact enumeration (safety)",
    )
    p.add_argument(
        "--negative-power",
        choices=("clip_nonneg", "abs_then_root", "nan_if_negative"),
        default="clip_nonneg",
        help="how to take the (kℓ)-th root when E[∏f] < 0 (signed / balanced f)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sweep-k-max",
        type=int,
        default=0,
        metavar="K",
        help="if ≥2, also print k′=2..K with fixed --l (progressive k′ sweep)",
    )
    args = p.parse_args(argv)

    f = load_grid_norm_pipe_v1(args.pipe)
    n, m = f.shape
    rng = random.Random(args.seed)
    cnt = exact_grid_norm_tuple_count(n, m, args.k, args.l)
    resolved = resolve_grid_norm_method(
        n,
        m,
        args.k,
        args.l,
        method=args.method,
        max_tuples=args.max_tuples,
        exact_auto_max_tuples=args.exact_auto_max,
    )
    warns: List[str] = []
    v = grid_norm(
        f,
        args.k,
        args.l,
        rng=rng,
        n_samples=args.samples,
        method=args.method,
        max_tuples=args.max_tuples,
        exact_auto_max_tuples=args.exact_auto_max,
        negative_power_handling=args.negative_power,
        out_warnings=warns,
    )
    print(f"grid_norm_pipe: {args.pipe}  shape=({n},{m})  tuples=n^k m^l={cnt}")
    print(f"method={args.method!r} → resolved={resolved!r}  ‖f‖_G({args.k},{args.l}) = {v}")
    for w in warns:
        print(f"NOTE: {w}")
    if args.sweep_k_max >= 2:
        print(f"progressive k-sweep (ℓ={args.l}, same JSON):")
        for kk in range(2, args.sweep_k_max + 1):
            if kk == args.k:
                continue
            wlist: List[str] = []
            vv = grid_norm(
                f,
                kk,
                args.l,
                rng=rng,
                n_samples=args.samples,
                method=args.method,
                max_tuples=args.max_tuples,
                exact_auto_max_tuples=args.exact_auto_max,
                negative_power_handling=args.negative_power,
                out_warnings=wlist,
            )
            c = exact_grid_norm_tuple_count(n, m, kk, args.l)
            r = resolve_grid_norm_method(
                n, m, kk, args.l, method=args.method, max_tuples=args.max_tuples, exact_auto_max_tuples=args.exact_auto_max
            )
            print(f"  k′={kk}  tuples={c}  resolved={r!r}  ‖f‖_G({kk},{args.l})={vv}")
            for w in wlist:
                print(f"    NOTE: {w}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_cli()
    else:
        demo()
