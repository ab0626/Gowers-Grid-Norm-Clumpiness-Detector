"""Verify README-referenced image files exist on disk."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _image_paths_from_readme() -> list[str]:
    text = README.read_text(encoding="utf-8")
    # Markdown / HTML img src="..."
    paths = re.findall(r'(?:src|href)="(docs/images/[^"]+)"', text)
    # de-dup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def test_readme_image_paths_exist() -> None:
    assert README.is_file()
    missing: list[str] = []
    for rel in _image_paths_from_readme():
        path = ROOT / rel
        if not path.is_file():
            missing.append(rel)
    assert not missing, f"Missing README assets: {missing}"


def test_docs_images_manifest_exists() -> None:
    man = ROOT / "docs" / "images" / "README.md"
    assert man.is_file()
