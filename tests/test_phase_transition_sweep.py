"""Smoke test for phase transition plot script (small grid, fast)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "phase_transition_sweep.py"


def test_phase_transition_script_runs(tmp_path: Path) -> None:
    out = tmp_path / "pt.png"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--n",
            "4",
            "--l",
            "2",
            "--k-min",
            "2",
            "--k-max",
            "3",
            "--seed",
            "0",
            "--samples",
            "5000",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.is_file() and out.stat().st_size > 2000
    assert "Wrote" in r.stdout
