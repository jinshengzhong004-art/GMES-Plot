from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


class ThresholdMode(str, Enum):
    KEEP_ALL = "keep_all"
    KEEP_RANGE = "keep_range"
    HIDE_RANGE = "hide_range"
    ABOVE = "above"
    BELOW = "below"


@dataclass(frozen=True, slots=True)
class Threshold:
    mode: ThresholdMode = ThresholdMode.KEEP_ALL
    lower: float | None = None
    upper: float | None = None

    def mask(self, values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if self.mode is ThresholdMode.KEEP_ALL:
            return finite
        if self.mode is ThresholdMode.KEEP_RANGE:
            if self.lower is None or self.upper is None:
                raise ValueError("保留区间需要同时设置下限和上限")
            return finite & (values >= self.lower) & (values <= self.upper)
        if self.mode is ThresholdMode.HIDE_RANGE:
            if self.lower is None or self.upper is None:
                raise ValueError("隐藏区间需要同时设置下限和上限")
            return finite & ((values < self.lower) | (values > self.upper))
        if self.mode is ThresholdMode.ABOVE:
            if self.upper is None:
                raise ValueError("隐藏高值需要设置上限")
            return finite & (values <= self.upper)
        if self.mode is ThresholdMode.BELOW:
            if self.lower is None:
                raise ValueError("隐藏低值需要设置下限")
            return finite & (values >= self.lower)
        raise ValueError(f"未知阈值模式: {self.mode}")


@dataclass(frozen=True, slots=True)
class SpatialFilter:
    """Inclusive display-only XYZ region of interest.

    Spatial filtering is evaluated before the value threshold.  A missing
    range leaves that coordinate unconstrained.
    """

    x: tuple[float, float] | None = None
    y: tuple[float, float] | None = None
    z: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            limits = getattr(self, name)
            if limits is not None and (len(limits) != 2 or limits[0] > limits[1]):
                raise ValueError(f"{name.upper()}空间范围必须满足最小值≤最大值")

    def axis_mask(self, name: str, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        limits = getattr(self, name)
        finite = np.isfinite(data)
        return finite if limits is None else finite & (data >= limits[0]) & (data <= limits[1])

    @property
    def active(self) -> bool:
        return any(getattr(self, name) is not None for name in ("x", "y", "z"))


@dataclass(slots=True)
class Dataset:
    name: str
    columns: dict[str, np.ndarray]
    roles: dict[str, str]
    source_path: str | None = None
    source_hash: str | None = None
    units: dict[str, str] = field(default_factory=dict)
    crs: str | None = None
    parent_id: str | None = None
    derivation: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        lengths = {len(np.asarray(v)) for v in self.columns.values()}
        if len(lengths) > 1:
            raise ValueError("数据列长度不一致")
        frozen: dict[str, np.ndarray] = {}
        for key, values in self.columns.items():
            array = np.asarray(values).copy()
            array.setflags(write=False)
            frozen[key] = array
        self.columns = frozen

    @property
    def row_count(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def role_values(self, role: str) -> np.ndarray:
        try:
            return self.columns[self.roles[role]]
        except KeyError as exc:
            raise KeyError(f"数据集未映射字段角色: {role}") from exc

    def derive(self, name: str, row_mask: np.ndarray, operation: dict[str, Any]) -> "Dataset":
        mask = np.asarray(row_mask, dtype=bool)
        if mask.shape != (self.row_count,):
            raise ValueError("派生数据筛选掩膜长度不正确")
        return Dataset(
            name=name,
            columns={key: values[mask] for key, values in self.columns.items()},
            roles=dict(self.roles),
            source_path=self.source_path,
            source_hash=self.source_hash,
            units=dict(self.units),
            crs=self.crs,
            parent_id=self.id,
            derivation=operation,
        )


@dataclass(frozen=True, slots=True)
class GridSpec:
    bounds: tuple[float, ...]
    shape: tuple[int, ...]

    @property
    def dimensions(self) -> int:
        return len(self.shape)

    @property
    def cell_count(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def axes(self) -> tuple[np.ndarray, ...]:
        if self.dimensions == 2:
            xmin, xmax, ymin, ymax = self.bounds
            ny, nx = self.shape
            return np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        if self.dimensions == 3:
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            nz, ny, nx = self.shape
            return (
                np.linspace(xmin, xmax, nx),
                np.linspace(ymin, ymax, ny),
                np.linspace(zmin, zmax, nz),
            )
        raise ValueError("仅支持二维或三维网格")


@dataclass(slots=True)
class GridResult:
    name: str
    spec: GridSpec
    values: np.ndarray
    method: str
    source_dataset_id: str
    variance: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(slots=True)
class TerrainProfile:
    name: str
    position: np.ndarray
    elevation: np.ndarray
    source_path: str | None = None
    source_hash: str | None = None
    position_unit: str = "m"
    elevation_unit: str = "m"
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float).copy()
        elevation = np.asarray(self.elevation, dtype=float).copy()
        if position.ndim != 1 or elevation.ndim != 1 or len(position) != len(elevation):
            raise ValueError("地形位置与高程必须是一维等长数组")
        if len(position) < 2:
            raise ValueError("地形数据至少需要两个有效点")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(elevation)):
            raise ValueError("地形数据包含空值或非有限数值")
        position.setflags(write=False)
        elevation.setflags(write=False)
        self.position = position
        self.elevation = elevation

    @property
    def point_count(self) -> int:
        return len(self.position)

    def sorted_values(self) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.position, kind="stable")
        return self.position[order], self.elevation[order]


@dataclass(slots=True)
class Project:
    name: str = "未命名工程"
    id: str = field(default_factory=lambda: str(uuid4()))
    datasets: dict[str, Dataset] = field(default_factory=dict)
    grids: dict[str, GridResult] = field(default_factory=dict)
    terrains: dict[str, TerrainProfile] = field(default_factory=dict)
    pages: list[dict[str, Any]] = field(default_factory=list)
    styles: dict[str, Any] = field(default_factory=dict)
    colormaps: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=lambda: {"boreholes": [], "geology": []})
    standards_profile: dict[str, Any] = field(default_factory=dict)
    project_path: Path | None = None
    dirty: bool = False

    def add_dataset(self, dataset: Dataset) -> None:
        self.datasets[dataset.id] = dataset
        self.dirty = True

    def add_grid(self, grid: GridResult) -> None:
        self.grids[grid.id] = grid
        self.dirty = True

    def add_terrain(self, terrain: TerrainProfile) -> None:
        self.terrains[terrain.id] = terrain
        self.dirty = True

