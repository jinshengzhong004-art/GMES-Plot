from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np

from gmes_plot.domain.models import Dataset, GridResult, GridSpec, Project, SpatialFilter, Threshold
from gmes_plot.services.research import layer_provenance, publication_checks, write_companion_report


def _objects():
    dataset = Dataset(
        "survey", {"X": np.array([0., 1.]), "Y": np.array([0., 1.]), "V": np.array([2., 3.])},
        {"x": "X", "y": "Y", "value": "V"}, source_hash="abc", units={"V": "mGal"}, crs="EPSG:32649",
    )
    grid = GridResult("grid", GridSpec((0., 1., 0., 1.), (2, 2)), np.ones((2, 2)), "idw", dataset.id, metadata={"power": 2})
    return dataset, grid


def test_provenance_and_companion_report(tmp_path: Path):
    dataset, grid = _objects()
    record = layer_provenance(
        dataset, grid, threshold=Threshold(), spatial_filter=SpatialFilter(x=(0., 1.)),
        cmap="viridis", style={}, view_kind="contour", physical_method="gravity",
    )
    assert record["source_data"]["field_mapping"]["value"] == "V"
    assert record["derived_grid"]["parameters"]["power"] == 2
    target = write_companion_report(tmp_path / "figure.png", record)
    assert target.name == "figure.reproducibility.md"
    assert "XYZ显示裁剪" in target.read_text(encoding="utf-8")


def test_publication_checker_flags_missing_quality_report():
    dataset, grid = _objects()
    record = layer_provenance(
        dataset, grid, threshold=Threshold(), spatial_filter=SpatialFilter(),
        cmap="viridis", style={}, view_kind="contour", physical_method="gravity",
    )
    figure = Figure(); axis = figure.add_subplot(111); axis.set(xlabel="X", ylabel="Y")
    findings = publication_checks(figure, Project(), record)
    assert any("质量报告" in item["message"] for item in findings)

