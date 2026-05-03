"""Smoke test for scripts/export_mask_heatmaps.py."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_export_mask_heatmaps_script(tmp_path: Path) -> None:
    out = tmp_path / "out.png"
    script = ROOT / "scripts" / "export_mask_heatmaps.py"
    r = subprocess.run(
        [
            sys.executable,
            str(script),
            "--n",
            "6",
            "--scale",
            "8",
            "--seed",
            "1",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists()
    assert out.stat().st_size > 50
