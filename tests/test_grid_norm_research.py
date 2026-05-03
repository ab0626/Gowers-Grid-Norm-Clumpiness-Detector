"""Research-grade grid_norm: auto exact/MC, pipe JSON, variance notes, signed power."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from grid_norm import (
    DEFAULT_EXACT_AUTO_MAX_TUPLES,
    PIPE_FORMAT,
    exact_grid_norm_tuple_count,
    grid_norm,
    grid_norm_exact,
    load_grid_norm_pipe_v1,
    mc_variance_advisory,
    resolve_grid_norm_method,
    write_grid_norm_pipe_v1,
)


def test_resolve_auto_prefers_exact_on_tiny_budget():
    assert resolve_grid_norm_method(3, 3, 2, 2, method="auto", max_tuples=10**9, exact_auto_max_tuples=10**7) == "exact"
    assert exact_grid_norm_tuple_count(3, 3, 2, 2) == 3**2 * 3**2


def test_resolve_auto_mc_when_tuple_budget_huge():
    assert resolve_grid_norm_method(50, 50, 2, 6, method="auto", max_tuples=10**9, exact_auto_max_tuples=10**7) == "mc"


def test_auto_grid_norm_matches_exact_small_grid():
    rng = __import__("random").Random(0)
    f = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    ex = grid_norm_exact(f, 2, 2, max_tuples=10**7)
    au = grid_norm(
        f,
        2,
        2,
        rng=rng,
        n_samples=500,
        method="auto",
        max_tuples=10**7,
        exact_auto_max_tuples=DEFAULT_EXACT_AUTO_MAX_TUPLES,
    )
    assert abs(au - ex) < 1e-12


def test_pipe_json_roundtrip(tmp_path: Path):
    f = np.array([[0.0, 1.0], [1.0, 0.5]], dtype=np.float64)
    p = tmp_path / "t.json"
    write_grid_norm_pipe_v1(p, f, meta={"test": True})
    g = load_grid_norm_pipe_v1(p)
    assert np.allclose(f, g)
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["format"] == PIPE_FORMAT
    assert raw["meta"]["test"] is True


def test_mc_variance_advisory_when_sparse():
    f = np.zeros((18, 18), dtype=np.float64)
    f[0, 0] = 1.0
    msg = mc_variance_advisory(f, 2, 2, n_samples=500)
    assert msg is not None
    assert "MC variance risk" in msg


def test_negative_power_nan_for_signed_constant():
    f = np.full((1, 1), -1.0, dtype=np.float64)
    v = grid_norm(f, 1, 1, method="exact", negative_power_handling="nan_if_negative")
    assert np.isnan(v)
    v2 = grid_norm(f, 1, 1, method="exact", negative_power_handling="abs_then_root")
    assert abs(v2 - 1.0) < 1e-12


def test_cli_pipe_smoke(tmp_path: Path):
    f = np.ones((2, 2), dtype=np.float64)
    jp = tmp_path / "m.json"
    write_grid_norm_pipe_v1(jp, f)
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "grid_norm.py"), "--pipe", str(jp), "--k", "1", "--l", "1", "--method", "exact"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "resolved='exact'" in r.stdout or 'resolved="exact"' in r.stdout or "resolved=" in r.stdout
    assert "‖f‖_G(1,1)" in r.stdout or "G(1,1)" in r.stdout
