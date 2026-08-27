from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from gmes_plot.domain.models import TerrainProfile
from gmes_plot.io.tabular import ParseOptions, load_dataset, preview_table


POSITION_HINTS = ("position", "distance", "station", "chainage", "x", "位置", "距离", "里程", "桩号")
ELEVATION_HINTS = ("elevation", "height", "altitude", "z", "高程", "标高", "海拔")


def suggest_terrain_fields(path: str | Path) -> tuple[str, str, ParseOptions]:
    preview = preview_table(path)
    names = preview.names
    numeric: list[str] = []
    for column_index, name in enumerate(names):
        values = [row[column_index] for row in preview.rows if len(row) > column_index]
        try:
            if values and all(np.isfinite(float(value)) for value in values):
                numeric.append(name)
        except ValueError:
            continue
    if len(numeric) < 2:
        raise ValueError("地形文件至少需要两列数值：位置和高程")

    def hinted(candidates: list[str], hints: tuple[str, ...]) -> str | None:
        return next((name for name in candidates if name.strip().lower() in hints), None)

    position = hinted(numeric, POSITION_HINTS) or numeric[0]
    remaining = [name for name in numeric if name != position]
    elevation = hinted(remaining, ELEVATION_HINTS) or remaining[0]
    options = ParseOptions(encoding=preview.encoding, delimiter=preview.delimiter, header=preview.has_header)
    return position, elevation, options


def load_terrain_profile(
    path: str | Path,
    position_field: str | None = None,
    elevation_field: str | None = None,
) -> TerrainProfile:
    file_path = Path(path)
    suggested_position, suggested_elevation, options = suggest_terrain_fields(file_path)
    position_field = position_field or suggested_position
    elevation_field = elevation_field or suggested_elevation
    dataset = load_dataset(file_path, {"position": position_field, "elevation": elevation_field}, options)
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return TerrainProfile(
        name=file_path.stem,
        position=dataset.role_values("position"),
        elevation=dataset.role_values("elevation"),
        source_path=str(file_path.resolve()),
        source_hash=digest,
    )

