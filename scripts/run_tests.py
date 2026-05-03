"""
Run pytest with third-party setuptools plugins disabled.

Some Python installs autoload broken pytest11 entrypoints (e.g. web3), which
prevents pytest from starting. Setting PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 fixes
that for this repo's self-contained tests.

Usage (from repo root):

    python scripts/run_tests.py
    python scripts/run_tests.py -k grid_norm
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    import pytest

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    code = pytest.main(["tests", *sys.argv[1:]])
    raise SystemExit(code)


if __name__ == "__main__":
    main()
