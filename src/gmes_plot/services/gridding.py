from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
from scipy.interpolate import RBFInterpolator, griddata
from scipy.spatial import cKDTree

from gmes_plot.domain.models import Dataset, GridResult, GridSpec


@dataclass(frozen=True, slots=True)
class GridRecommendation:
    preview: GridSpec
    balanced: GridSpec
    fine: GridSpec
    nearest_distance_median: float
    notes: tuple[str, ...]


def _finite_points(dataset: Dataset, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    roles = ["x", "y"] + (["z"] if dimensions == 3 else [])
    coordinates = np.column_stack([np.asarray(dataset.role_values(role), dtype=float) for role in roles])
    values = np.asarray(dataset.role_values("value"), dtype=float)
    finite = np.isfinite(values) & np.all(np.isfinite(coordinates), axis=1)
    coordinates, values = coordinates[finite], values[finite]
    if len(values) < dimensions + 1:
        raise ValueError("有效数据点不足以进行网格化")
    return coordinates, values


def recommend_grid(dataset: Dataset, dimensions: int = 2) -> GridRecommendation:
    points, _ = _finite_points(dataset, dimensions)
    mins, maxs = points.min(axis=0), points.max(axis=0)
    spans = maxs - mins
    if np.any(spans <= 0):
        raise ValueError("坐标范围退化，无法建立完整网格")
    sample_count = min(len(points), 50_000)
    sample = points[np.linspace(0, len(points) - 1, sample_count, dtype=int)]
    distances, _ = cKDTree(sample).query(sample, k=2)
    nearest_median = float(np.median(distances[:, 1]))

    def shape_for_budget(budget: int) -> tuple[int, ...]:
        scale = (budget / prod(spans)) ** (1 / dimensions)
        axis_counts = np.maximum(16, np.rint(spans * scale).astype(int))
        if dimensions == 2:
            return int(axis_counts[1]), int(axis_counts[0])
        return int(axis_counts[2]), int(axis_counts[1]), int(axis_counts[0])

    bounds = tuple(value for pair in zip(mins, maxs) for value in pair)
    if dimensions == 2:
        budgets = (
            min(120_000, max(10_000, len(points) * 4)),
            min(500_000, max(40_000, len(points) * 12)),
            min(2_000_000, max(160_000, len(points) * 30)),
        )
    else:
        budgets = (
            min(500_000, max(50_000, len(points) * 40)),
            min(2_000_000, max(200_000, len(points) * 150)),
            min(8_000_000, max(1_000_000, len(points) * 500)),
        )
    specs = [GridSpec(bounds, shape_for_budget(item)) for item in budgets]
    notes = ["网格由空间范围、有效点数和最近邻尺度自动推荐；稀疏数据默认不再生成百万级预览网格。"]
    if specs[1].cell_count > len(points) * 200:
        notes.append("平衡网格相对测点较密，需结合交叉验证检查过度采样。")
    return GridRecommendation(specs[0], specs[1], specs[2], nearest_median, tuple(notes))


def estimate_memory(spec: GridSpec, output_arrays: int = 4, workspace_factor: float = 2.5) -> int:
    return int(spec.cell_count * 8 * output_arrays * workspace_factor)


def _idw(points: np.ndarray, values: np.ndarray, targets: np.ndarray, neighbors: int, power: float) -> np.ndarray:
    tree = cKDTree(points)
    k = min(max(1, neighbors), len(points))
    result = np.empty(len(targets), dtype=float)
    # Keep neighbor-distance and index work arrays bounded for multi-million
    # voxel jobs. Roughly target <= 64 MiB for both arrays per chunk.
    chunk_size = max(1_000, min(len(targets), int(64 * 1024**2 / max(16 * k, 1))))
    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        distances, indices = tree.query(targets[start:stop], k=k, workers=-1)
        if k == 1:
            distances, indices = distances[:, None], indices[:, None]
        exact = distances[:, 0] == 0
        safe = np.maximum(distances, np.finfo(float).eps)
        weights = 1.0 / safe**power
        chunk_result = np.sum(weights * values[indices], axis=1) / np.sum(weights, axis=1)
        chunk_result[exact] = values[indices[exact, 0]]
        result[start:stop] = chunk_result
    return result


def interpolate(
    dataset: Dataset,
    spec: GridSpec,
    method: str = "idw",
    neighbors: int = 24,
    power: float = 2.0,
) -> GridResult:
    points, values = _finite_points(dataset, spec.dimensions)
    axes = spec.axes()
    if spec.dimensions == 2:
        x_axis, y_axis = axes
        mesh = np.meshgrid(x_axis, y_axis, indexing="xy")
    else:
        x_axis, y_axis, z_axis = axes
        mesh = np.meshgrid(x_axis, y_axis, z_axis, indexing="xy")
        mesh = (mesh[0].transpose(2, 0, 1), mesh[1].transpose(2, 0, 1), mesh[2].transpose(2, 0, 1))
    targets = np.column_stack([item.ravel() for item in mesh])
    normalized = method.lower()
    if normalized == "idw":
        output = _idw(points, values, targets, neighbors, power)
    elif normalized in {"nearest", "linear", "cubic"}:
        if spec.dimensions == 3 and normalized == "cubic":
            raise ValueError("SciPy散点三维插值不支持cubic方法")
        output = griddata(points, values, targets, method=normalized, fill_value=np.nan)
    elif normalized == "rbf":
        model = RBFInterpolator(points, values, neighbors=min(neighbors, len(points)), smoothing=0.0)
        output = model(targets)
    elif normalized == "kriging":
        return ordinary_kriging(dataset, spec, neighbors=neighbors)
    else:
        raise ValueError(f"未知插值方法: {method}")
    return GridResult(
        name=f"{dataset.name}-{normalized}", spec=spec, values=output.reshape(spec.shape),
        method=normalized, source_dataset_id=dataset.id,
        metadata={"neighbors": neighbors, "power": power},
    )


def ordinary_kriging(dataset: Dataset, spec: GridSpec, neighbors: int = 24) -> GridResult:
    points, values = _finite_points(dataset, spec.dimensions)
    if spec.dimensions == 2 and len(points) > 10_000:
        raise ValueError(
            "当前MVP克里金求解器最多直接处理10,000个点。"
            "请先抽稀/分区，或等待局部克里金分块求解器接入。"
        )
    if spec.dimensions == 3 and (len(points) > 2_000 or spec.cell_count > 500_000):
        raise ValueError(
            "当前MVP三维克里金限制为2,000个输入点和500,000个体素；"
            "超出范围需要后续局部邻域分块求解器。"
        )
    try:
        if spec.dimensions == 2:
            from pykrige.ok import OrdinaryKriging

            x_axis, y_axis = spec.axes()
            model = OrdinaryKriging(
                points[:, 0], points[:, 1], values,
                variogram_model="spherical", verbose=False, enable_plotting=False,
            )
            estimate, variance = model.execute("grid", x_axis, y_axis, n_closest_points=min(neighbors, len(points)), backend="loop")
        else:
            from pykrige.ok3d import OrdinaryKriging3D

            x_axis, y_axis, z_axis = spec.axes()
            model = OrdinaryKriging3D(
                points[:, 0], points[:, 1], points[:, 2], values,
                variogram_model="spherical", verbose=False, enable_plotting=False,
            )
            estimate, variance = model.execute("grid", x_axis, y_axis, z_axis)
            estimate = np.asarray(estimate).transpose(2, 1, 0)
            variance = np.asarray(variance).transpose(2, 1, 0)
    except ImportError as exc:
        raise RuntimeError("普通克里金需要可选依赖 pykrige，请安装 gmes-plot[geostat]") from exc
    return GridResult(
        name=f"{dataset.name}-ordinary-kriging", spec=spec,
        values=np.asarray(estimate), variance=np.asarray(variance), method="ordinary_kriging",
        source_dataset_id=dataset.id,
        metadata={"variogram_model": "spherical", "neighbors": neighbors},
    )

