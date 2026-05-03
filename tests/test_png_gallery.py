"""Committed README PNGs must exist (GitHub gallery)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "images"

EXPECTED = [
    "fig-corner-to-ap.png",
    "fig-k22-bipartite.png",
    "fig-k2k-bipartite.png",
    "fig-balanced-indicator.png",
    "fig-mc-vs-exact.png",
    "fig-density-increment.png",
    "mask-lift-vs-random.png",
    "phase-transition-behrend-vs-random.png",
]


def test_readme_png_gallery_files_exist() -> None:
    missing = []
    for name in EXPECTED:
        p = IMG / name
        if not p.is_file() or p.stat().st_size < 500:
            missing.append(name)
    assert not missing, f"Missing or tiny PNG gallery files: {missing}"
