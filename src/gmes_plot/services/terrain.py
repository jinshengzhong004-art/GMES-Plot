from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gmes_plot.domain.models import TerrainProfile


@dataclass(frozen=True, slots=True)
class TerrainCorrectedMesh:
    x: np.ndarray
    elevation: np.ndarray
    surface_elevation: np.ndarray
    source_columns: np.ndarray


def terrain_corrected_mesh(
    x_axis: np.ndarray,
    depth_axis: np.ndarray,
    terrain: TerrainProfile,
) -> TerrainCorrectedMesh:
    x_axis = np.asarray(x_axis, dtype=float)
    depth_axis = np.asarray(depth_axis, dtype=float)
    position, elevation = terrain.sorted_values()
    inside = (x_axis >= position.min()) & (x_axis <= position.max())
    columns = np.flatnonzero(inside)
    if len(columns) < 2:
        raise ValueError("地形覆盖范围内不足两个断面网格列")
    section_x = x_axis[columns]
    surface = np.interp(section_x, position, elevation)
    xx = np.broadcast_to(section_x[None, :], (len(depth_axis), len(section_x)))
    absolute_elevation = surface[None, :] - depth_axis[:, None]
    return TerrainCorrectedMesh(xx, absolute_elevation, surface, columns)

