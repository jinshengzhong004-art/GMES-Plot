from __future__ import annotations

import sys
from pathlib import Path


def asset_path(name: str) -> Path:
    """Return a bundled asset path in development and PyInstaller builds."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    candidate = root / "gmes_plot" / "assets" / name
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parent / "assets" / name

