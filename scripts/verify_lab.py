# -*- coding: utf-8 -*-
"""
One-shot local verification: demos, PNG figure, pytest.

From repo root:

    python scripts/verify_lab.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    py = sys.executable
    run([py, str(ROOT / "grid_norm.py")])
    run([py, str(ROOT / "behrend.py")])
    run([py, str(ROOT / "corner_lift.py")])
    run([py, str(ROOT / "exact_g2k.py"), "--n", "7", "--k-cap", "4"])
    run(
        [
            py,
            str(ROOT / "scripts" / "export_mask_heatmaps.py"),
            "--n",
            "16",
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
