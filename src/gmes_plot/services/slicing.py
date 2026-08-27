from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from gmes_plot.domain.models import GridResult


@dataclass(slots=True)
class VolumeSlice:
    name: str
    horizontal: np.ndarray
    vertical: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    values: np.ndarray
    horizontal_label: str
    vertical_label: str
    origin: tuple[float, float, float]
    normal: tuple[float, float, float]
    metadata: dict = field(default_factory=dict)

    @property
    def valid_fraction(self) -> float:
        return float(np.count_nonzero(np.isfinite(self.values)) / self.values.size)


def _interpolator(grid: GridResult) -> RegularGridInterpolator:
    if grid.spec.dimensions != 3:
        raise ValueError("三维剖面需要三维规则体")
    x_axis, y_axis, z_axis = grid.spec.axes()
    if grid.values.shape != (len(z_axis), len(y_axis), len(x_axis)):
        raise ValueError("三维网格数组形状与坐标轴不一致")
    return RegularGridInterpolator(
        (z_axis, y_axis, x_axis), np.asarray(grid.values, dtype=float),
        bounds_error=False, fill_value=np.nan,
    )


def _sample(grid: GridResult, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    points = np.column_stack((z.ravel(), y.ravel(), x.ravel()))
    return _interpolator(grid)(points).reshape(x.shape)


def orthogonal_slice(grid: GridResult, axis: str, coordinate: float) -> VolumeSlice:
    axis = axis.lower()
    x_axis, y_axis, z_axis = grid.spec.axes()
    xmin, xmax, ymin, ymax, zmin, zmax = grid.spec.bounds
    if axis == "x":
        if not xmin <= coordinate <= xmax:
            raise ValueError("X剖面位置超出三维网格范围")
        y, z = np.meshgrid(y_axis, z_axis, indexing="xy")
        x = np.full_like(y, coordinate)
        values = _sample(grid, x, y, z)
        return VolumeSlice(
            f"X={coordinate:g}", y_axis, z_axis, x, y, z, values,
            "Y", "Z", (coordinate, (ymin + ymax) / 2, (zmin + zmax) / 2), (1, 0, 0),
            {"type": "orthogonal", "axis": "x", "coordinate": coordinate},
        )
    if axis == "y":
        if not ymin <= coordinate <= ymax:
            raise ValueError("Y剖面位置超出三维网格范围")
        x, z = np.meshgrid(x_axis, z_axis, indexing="xy")
        y = np.full_like(x, coordinate)
        values = _sample(grid, x, y, z)
        return VolumeSlice(
            f"Y={coordinate:g}", x_axis, z_axis, x, y, z, values,
            "X", "Z", ((xmin + xmax) / 2, coordinate, (zmin + zmax) / 2), (0, 1, 0),
            {"type": "orthogonal", "axis": "y", "coordinate": coordinate},
        )
    if axis == "z":
        if not zmin <= coordinate <= zmax:
            raise ValueError("Z剖面位置超出三维网格范围")
        x, y = np.meshgrid(x_axis, y_axis, indexing="xy")
        z = np.full_like(x, coordinate)
        values = _sample(grid, x, y, z)
        return VolumeSlice(
            f"Z={coordinate:g}", x_axis, y_axis, x, y, z, values,
            "X", "Y", ((xmin + xmax) / 2, (ymin + ymax) / 2, coordinate), (0, 0, 1),
            {"type": "orthogonal", "axis": "z", "coordinate": coordinate},
        )
    raise ValueError("剖面轴必须是X、Y或Z")


def arbitrary_plane_slice(
    grid: GridResult,
    origin: tuple[float, float, float],
    normal: tuple[float, float, float],
    width: float,
    height: float,
    resolution: tuple[int, int] = (240, 180),
) -> VolumeSlice:
    if grid.spec.dimensions != 3:
        raise ValueError("任意平面剖面需要三维规则体")
    if width <= 0 or height <= 0:
        raise ValueError("剖面宽度和高度必须大于零")
    nu, nv = resolution
    if nu < 2 or nv < 2:
        raise ValueError("剖面分辨率每个方向至少为2")
    center = np.asarray(origin, dtype=float)
    n = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(n))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("剖面法向量不能为零")
    n /= norm

    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, n))) > 0.92:
        reference = np.array([0.0, 1.0, 0.0])
    u_direction = np.cross(reference, n)
    u_direction /= np.linalg.norm(u_direction)
    v_direction = np.cross(n, u_direction)
    v_direction /= np.linalg.norm(v_direction)

    horizontal = np.linspace(-width / 2, width / 2, nu)
    vertical = np.linspace(-height / 2, height / 2, nv)
    uu, vv = np.meshgrid(horizontal, vertical, indexing="xy")
    xyz = center[None, None, :] + uu[..., None] * u_direction + vv[..., None] * v_direction
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    values = _sample(grid, x, y, z)
    return VolumeSlice(
        "任意倾斜剖面", horizontal, vertical, x, y, z, values,
        "剖面U距离", "剖面V距离", tuple(float(item) for item in center), tuple(float(item) for item in n),
        {
            "type": "arbitrary", "origin": tuple(float(item) for item in center),
            "normal": tuple(float(item) for item in n), "width": width, "height": height,
            "resolution": (nu, nv),
        },
    )

