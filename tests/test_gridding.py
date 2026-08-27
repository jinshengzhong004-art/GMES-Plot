import numpy as np

from gmes_plot.domain.models import Dataset, GridSpec
from gmes_plot.services.gridding import estimate_memory, interpolate, recommend_grid
from gmes_plot.services.quality import build_quality_report, idw_cross_validation


def make_plane(n: int = 9) -> Dataset:
    axis = np.linspace(0, 1, n)
    x, y = np.meshgrid(axis, axis)
    value = 2 * x + 3 * y + 5
    return Dataset(
        name="plane",
        columns={"X": x.ravel(), "Y": y.ravel(), "Value": value.ravel()},
        roles={"x": "X", "y": "Y", "value": "Value"},
    )


def test_grid_recommendation_and_idw():
    dataset = make_plane()
    recommendation = recommend_grid(dataset)
    assert recommendation.preview.cell_count > 0
    spec = GridSpec((0.0, 1.0, 0.0, 1.0), (31, 31))
    result = interpolate(dataset, spec, "idw", neighbors=8)
    assert result.values.shape == (31, 31)
    assert np.isfinite(result.values).all()
    center = result.values[15, 15]
    assert abs(center - 7.5) < 0.15


def test_linear_plane_and_quality_report():
    dataset = make_plane()
    spec = GridSpec((0.0, 1.0, 0.0, 1.0), (21, 21))
    result = interpolate(dataset, spec, "linear")
    metrics = idw_cross_validation(dataset, neighbors=8)
    report = build_quality_report(dataset, result, metrics)
    assert abs(result.values[10, 10] - 7.5) < 1e-9
    assert report["grid"]["cells"] == 441
    assert report["cross_validation"]["sample_count"] == dataset.row_count


def test_memory_estimate_increases_with_cells():
    small = estimate_memory(GridSpec((0, 1, 0, 1), (100, 100)))
    large = estimate_memory(GridSpec((0, 1, 0, 1), (200, 200)))
    assert large == small * 4


def test_sparse_3d_preview_no_longer_defaults_to_multi_million_voxels():
    rng = np.random.default_rng(7)
    dataset = Dataset(
        "sparse-volume",
        {"x": rng.random(600), "y": rng.random(600), "z": rng.random(600), "v": rng.normal(size=600)},
        {"x": "x", "y": "y", "z": "z", "value": "v"},
    )
    recommendation = recommend_grid(dataset, 3)
    assert recommendation.preview.cell_count <= 550_000
    assert recommendation.preview.cell_count < recommendation.balanced.cell_count < recommendation.fine.cell_count

