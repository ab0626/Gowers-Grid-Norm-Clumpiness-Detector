"""All gallery SVGs must be well-formed XML (catches encoding / truncation bugs)."""

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG_DIR = ROOT / "docs" / "images"


def _svg_files() -> list[Path]:
    return sorted(SVG_DIR.glob("fig-*.svg"))


def test_each_figure_svg_parses() -> None:
    paths = _svg_files()
    assert len(paths) >= 6
    for p in paths:
        ET.parse(p)
        raw = p.read_bytes()
        assert raw.startswith(b"<?xml") or raw.startswith(b"<svg"), p.name
