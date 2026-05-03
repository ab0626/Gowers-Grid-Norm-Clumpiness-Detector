"""Tests for behrend.py."""

import random

import numpy as np

from behrend import behrend_indices, behrend_matrix, random_mask_same_density


def test_behrend_indices_sorted_unique_in_range():
    M = 200
    s = behrend_indices(M)
    assert s == sorted(set(s))
    assert all(0 <= x < M for x in s)
    assert len(s) >= 1


def test_behrend_matrix_shape_and_binary():
    n = 9
    a = behrend_matrix(n)
    assert a.shape == (n, n)
    assert set(np.unique(a)).issubset({0.0, 1.0})


def test_random_mask_same_density_matches_count():
    rng = random.Random(0)
    a = random_mask_same_density((5, 7), 11, rng)
    assert int(a.sum()) == 11
