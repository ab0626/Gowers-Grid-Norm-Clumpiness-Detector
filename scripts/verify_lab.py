# -*- coding: utf-8 -*-
"""
One-shot local verification: demos, README PNG figures, pytest.

From repo root:

    python scripts/verify_lab.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    py = sys.executable
    run([py, str(ROOT / "grid_norm.py")])
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        pipe_path = tmp.name
    try:
        run([py, str(ROOT / "behrend.py"), "--export-pipe", pipe_path, "-n", "5"])
        run(
            [
                py,
                str(ROOT / "grid_norm.py"),
                "--pipe",
                pipe_path,
                "--k",
                "2",
                "--l",
                "2",
                "--method",
                "auto",
            ]
        )
    finally:
        Path(pipe_path).unlink(missing_ok=True)
    run([py, str(ROOT / "behrend.py")])
    run([py, str(ROOT / "corner_lift.py")])
    run([py, str(ROOT / "exact_g2k.py"), "--n", "7", "--k-cap", "4"])
    run([py, str(ROOT / "scripts" / "gen_readme_pngs.py")])
    run(
        [
            py,
            str(ROOT / "scripts" / "export_mask_heatmaps.py"),
            "--n",
            "16",
            "--scale",
            "14",
            "--seed",
            "0",
            "--out",
            "docs/images/mask-lift-vs-random.png",
        ]
    )
    run([py, str(ROOT / "scripts" / "run_tests.py"), "-v"])
    print("verify_lab: OK")


if __name__ == "__main__":
    main()
