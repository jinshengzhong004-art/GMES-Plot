from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

from gmes_plot.domain.models import Dataset, GridResult


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    sample_count: int
    mean_error: float
    mae: float
    rmse: float
    r_squared: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def idw_cross_validation(dataset: Dataset, dimensions: int = 2, limit: int = 20_000, neighbors: int = 12) -> QualityMetrics:
    roles = ["x", "y"] + (["z"] if dimensions == 3 else [])
    points = np.column_stack([dataset.role_values(role).astype(float) for role in roles])
    values = dataset.role_values("value").astype(float)
    finite = np.isfinite(values) & np.all(np.isfinite(points), axis=1)
    points, values = points[finite], values[finite]
    if len(values) < 3:
        raise ValueError("交叉验证需要至少3个有效点")
    if len(values) > limit:
        indices = np.linspace(0, len(values) - 1, limit, dtype=int)
        validation_points, observed = points[indices], values[indices]
    else:
        validation_points, observed = points, values
    k = min(neighbors + 1, len(points))
    distances, indices = cKDTree(points).query(validation_points, k=k, workers=-1)
    distances, indices = distances[:, 1:], indices[:, 1:]
    weights = 1 / np.maximum(distances, np.finfo(float).eps) ** 2
    predicted = np.sum(weights * values[indices], axis=1) / np.sum(weights, axis=1)
    residual = predicted - observed
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return QualityMetrics(
        sample_count=len(observed), mean_error=float(residual.mean()),
        mae=float(np.mean(np.abs(residual))), rmse=float(np.sqrt(np.mean(residual**2))),
        r_squared=float(1 - ss_res / ss_tot) if ss_tot else float("nan"),
    )


def build_quality_report(dataset: Dataset, grid: GridResult, metrics: QualityMetrics | None = None) -> dict:
    values = dataset.role_values("value").astype(float)
    finite = values[np.isfinite(values)]
    report = {
        "dataset": {"id": dataset.id, "name": dataset.name, "rows": dataset.row_count, "valid_values": len(finite)},
        "value_summary": {
            "minimum": float(np.min(finite)), "maximum": float(np.max(finite)),
            "mean": float(np.mean(finite)), "standard_deviation": float(np.std(finite)),
        },
        "grid": {"shape": grid.spec.shape, "cells": grid.spec.cell_count, "method": grid.method, "bounds": grid.spec.bounds},
        "cross_validation": metrics.to_dict() if metrics else None,
        "warnings": [],
    }
    if grid.spec.cell_count > dataset.row_count * 200:
        report["warnings"].append({"level": "warning", "message": "网格相对测点很密，可能存在过度采样。"})
    if metrics and metrics.r_squared < 0.5:
        report["warnings"].append({"level": "high", "message": "交叉验证R²较低，结果不宜直接用于定量解释。"})
    return report

