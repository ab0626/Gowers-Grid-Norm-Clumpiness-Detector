"""Tests for grid_norm.py: exact norms, tuple counts, guards."""

import numpy as np
import pytest

from grid_norm import (
    exact_grid_norm_tuple_count,
    grid_norm,
    grid_norm_exact,
    grid_norm_power_kl,
    suggested_grid_size_from_density,
)


def test_exact_tuple_count():
    assert exact_grid_norm_tuple_count(3, 4, 2, 2) == 3**2 * 4**2


def test_constant_one_all_norms_one():
    f = np.ones((3, 5), dtype=np.float64)
    for k, l in [(1, 1), (2, 2), (2, 3)]:
        v = grid_norm_exact(f, k, l, max_tuples=10_000_000)
        assert abs(v - 1.0) < 1e-12, (k, l, v)


def test_all_zeros_norm_zero():
    f = np.zeros((4, 4), dtype=np.float64)
    v = grid_norm_exact(f, 2, 2, max_tuples=10_000_000)
    assert v == 0.0


def test_single_one_cell_box_norm():
    """One hot cell at (0,0) on 2x2 grid; exact G(2,2) is (1/16)^(1/4)."""
    f = np.zeros((2, 2), dtype=np.float64)
    f[0, 0] = 1.0
    p = grid_norm_power_kl(f, 2, 2, method="exact")
    assert abs(p - 1.0 / 16.0) < 1e-12
    v = grid_norm_exact(f, 2, 2, max_tuples=10_000_000)
    assert abs(v - (1.0 / 16.0) ** 0.25) < 1e-12


def test_grid_norm_exact_refuses_huge_enumeration():
    f = np.ones((50, 50), dtype=np.float64)
    with pytest.raises(ValueError, match="exact grid norm"):
        grid_norm_exact(f, 2, 6, max_tuples=1_000)  # 50^2 * 50^6 is enormous


def test_suggested_k_at_least_two():
    k, l = suggested_grid_size_from_density(0.5)
    assert k >= 2 and l >= 2


def test_mc_matches_exact_dense_small_grid():
    """Monte Carlo should approximate exact on dense {0,1} matrix."""
    rng = __import__("random").Random(42)
    f = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    ex = grid_norm_exact(f, 2, 2, max_tuples=10_000_000)
    mc = grid_norm(f, 2, 2, rng=rng, n_samples=200_000, method="mc")
    assert abs(mc - ex) < 0.03
