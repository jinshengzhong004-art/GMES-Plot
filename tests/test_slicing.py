import numpy as np

from gmes_plot.domain.models import GridResult, GridSpec
from gmes_plot.services.slicing import arbitrary_plane_slice, orthogonal_slice


def analytic_grid() -> GridResult:
    spec = GridSpec((0.0, 4.0, 0.0, 3.0, 0.0, 2.0), (9, 10, 11))
    x, y, z = spec.axes()
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    values = xx + 2 * yy + 3 * zz
    return GridResult("analytic", spec, values, "analytic", "source")


def test_xyz_orthogonal_slices_match_analytic_field():
    grid = analytic_grid()
    for axis, coordinate in (("x", 2.0), ("y", 1.5), ("z", 1.0)):
        result = orthogonal_slice(grid, axis, coordinate)
        expected = result.x + 2 * result.y + 3 * result.z
        np.testing.assert_allclose(result.values, expected, atol=1e-10)
        assert result.valid_fraction == 1.0


def test_arbitrary_plane_matches_analytic_field_inside_volume():
    grid = analytic_grid()
    result = arbitrary_plane_slice(
        grid, origin=(2.0, 1.5, 1.0), normal=(1.0, 1.0, 0.5),
        width=2.0, height=1.2, resolution=(31, 21),
    )
    expected = result.x + 2 * result.y + 3 * result.z
    finite = np.isfinite(result.values)
    assert finite.mean() > 0.9
    np.testing.assert_allclose(result.values[finite], expected[finite], atol=1e-10)

