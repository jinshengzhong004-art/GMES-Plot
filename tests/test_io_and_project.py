from pathlib import Path

import numpy as np

from gmes_plot.domain.models import Project, TerrainProfile
from gmes_plot.io.project import load_project, save_project
from gmes_plot.io.tabular import ParseOptions, load_dataset, preview_table, suggest_roles
from gmes_plot.io.terrain import load_terrain_profile
from gmes_plot.services.gridding import interpolate, recommend_grid


def test_csv_import_and_project_roundtrip(tmp_path: Path):
    source = tmp_path / "sample.csv"
    source.write_text("X,Y,V\n0,0,1\n1,0,2\n0,1,3\n1,1,4\n", encoding="utf-8")
    preview = preview_table(source)
    assert preview.has_header
    assert preview.delimiter == ","
    dataset = load_dataset(source, {"x": "X", "y": "Y", "value": "V"})
    recommendation = recommend_grid(dataset)
    spec = recommendation.preview
    grid = interpolate(dataset, spec, "idw", neighbors=4)

    project = Project(name="roundtrip")
    project.add_dataset(dataset)
    project.add_grid(grid)
    project.add_terrain(TerrainProfile("terrain", np.array([0.0, 1.0]), np.array([100.0, 101.0])))
    project.constraints["boreholes"].append({"id": "ZK1", "reserved": True})
    project.standards_profile = {"style_library_version": "test"}
    path = tmp_path / "roundtrip.gpproj"
    save_project(project, path)
    restored = load_project(path)

    assert restored.name == "roundtrip"
    assert len(restored.datasets) == 1
    assert len(restored.grids) == 1
    assert len(restored.terrains) == 1
    assert restored.constraints["boreholes"][0]["id"] == "ZK1"
    assert restored.standards_profile["style_library_version"] == "test"
    restored_dataset = next(iter(restored.datasets.values()))
    restored_grid = next(iter(restored.grids.values()))
    np.testing.assert_allclose(restored_dataset.columns["V"], [1, 2, 3, 4])
    np.testing.assert_allclose(restored_grid.values, grid.values)
    restored_terrain = next(iter(restored.terrains.values()))
    np.testing.assert_allclose(restored_terrain.elevation, [100.0, 101.0])


def test_whitespace_dat_import(tmp_path: Path):
    source = tmp_path / "sample.dat"
    source.write_text("X Y Value\n0 0 10\n1 0 11\n", encoding="utf-8")
    dataset = load_dataset(source, {"x": "X", "y": "Y", "value": "Value"})
    assert dataset.row_count == 2
    assert dataset.role_values("value").tolist() == [10.0, 11.0]


def test_terrain_profile_auto_detection(tmp_path: Path):
    source = tmp_path / "terrain.txt"
    source.write_text("位置 高程\n0 102.5\n50 110.0\n100 107.5\n", encoding="utf-8")
    terrain = load_terrain_profile(source)
    assert terrain.point_count == 3
    assert terrain.position.tolist() == [0.0, 50.0, 100.0]
    assert terrain.elevation.tolist() == [102.5, 110.0, 107.5]


def test_role_suggestions_use_xyzv_and_xyv_column_conventions():
    assert suggest_roles(["Column_1", "Column_2", "Column_3"]) == {
        "x": "Column_1", "y": "Column_2", "value": "Column_3",
    }
    assert suggest_roles(["Column_1", "Column_2", "Column_3", "Column_4"]) == {
        "x": "Column_1", "y": "Column_2", "z": "Column_3", "value": "Column_4",
    }


def test_role_suggestions_recognize_geophysical_headers():
    assert suggest_roles(["Easting", "Northing", "Depth", "Resistivity"]) == {
        "x": "Easting", "y": "Northing", "z": "Depth", "value": "Resistivity",
    }

