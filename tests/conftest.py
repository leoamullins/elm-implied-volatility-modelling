"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Add the project's ``src`` directory to ``sys.path``.

    The project now follows a ``src/`` layout, so this helper makes sure
    imports such as ``import elm`` resolve without requiring an editable
    install during test runs.
    """

    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if src_path.exists():
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_src_on_path()
