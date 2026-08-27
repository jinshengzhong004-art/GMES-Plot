import numpy as np

from gmes_plot.domain.models import TerrainProfile
from gmes_plot.services.terrain import terrain_corrected_mesh


def test_terrain_corrected_mesh_follows_surface_and_depth():
    terrain = TerrainProfile("hill", np.array([0.0, 5.0, 10.0]), np.array([100.0, 110.0, 100.0]))
    mesh = terrain_corrected_mesh(np.linspace(-2, 12, 8), np.array([0.0, 10.0, 20.0]), terrain)
    assert mesh.x.shape == mesh.elevation.shape
    np.testing.assert_allclose(mesh.elevation[0], mesh.surface_elevation)
    np.testing.assert_allclose(mesh.elevation[1], mesh.surface_elevation - 10.0)
    np.testing.assert_allclose(mesh.elevation[2], mesh.surface_elevation - 20.0)
    assert mesh.x.min() >= 0.0 and mesh.x.max() <= 10.0

