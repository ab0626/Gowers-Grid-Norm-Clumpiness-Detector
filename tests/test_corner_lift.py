"""Tests for corner_lift.py."""

import numpy as np
import pytest

from corner_lift import (
    corner_free_lift_from_behrend,
    has_corner_in_grid,
    lift_mask_x_plus_2y,
)


def test_lift_line_no_corners_small_shell():
    """S = {0,2} has no 3-AP; lift on n=3 should be corner-free."""
    n = 3
    S = {0, 2}
    a = lift_mask_x_plus_2y(n, S)
    assert not has_corner_in_grid(a)


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 9, 10])
def test_corner_free_lift_from_behrend_has_no_corner(n: int):
    a = corner_free_lift_from_behrend(n)
    assert a.shape == (n, n)
    assert not has_corner_in_grid(a)


def test_corner_free_lift_binary():
    a = corner_free_lift_from_behrend(8)
    assert set(np.unique(a)).issubset({0.0, 1.0})
