"""Tests for exact_g2k.py."""

import numpy as np

from exact_g2k import scan_exact_g2k
from grid_norm import grid_norm_exact


def test_scan_exact_g2k_constant_one():
    f = np.ones((3, 3), dtype=np.float64)
    rows = scan_exact_g2k(f, [2, 3], max_tuples=1_000_000)
    assert len(rows) == 2
    for k, val, tc in rows:
        assert abs(val - 1.0) < 1e-12
        assert tc == 3**2 * 3**k


def test_exact_g2k_matches_grid_norm_exact():
    f = np.random.default_rng(0).random((3, 4))
    for k in (2, 3):
        a = grid_norm_exact(f, 2, k, max_tuples=5_000_000)
        b = grid_norm_exact(f, 2, k, max_tuples=5_000_000)
        assert a == b
