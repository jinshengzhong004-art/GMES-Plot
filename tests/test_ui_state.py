import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from gmes_plot.domain.models import Dataset, GridResult, GridSpec, SpatialFilter, Threshold, ThresholdMode
from gmes_plot.ui.main_window import MainWindow, PlotPage
from gmes_plot.ui.dialogs import VolumeSliceDialog
from gmes_plot.ui.geology_window import GeologySectionWindow


def _application():
    return QApplication.instance() or QApplication([])


def test_plot_page_has_threshold_before_first_plot():
    app = _application()
    page = PlotPage()
    assert page.threshold == Threshold()
    dataset = Dataset(
        name="sample", columns={"x": np.array([0., 1.]), "y": np.array([0., 1.]), "v": np.array([2., 3.])},
        roles={"x": "x", "y": "y", "value": "v"},
    )
    page.plot_points(dataset)
    assert page.view_kind == "points"
    page.close()
    assert app is not None


def test_threshold_is_applied_to_volume_values():
    app = _application()
    page = PlotPage()
    page.threshold = Threshold(ThresholdMode.ABOVE, upper=3.0)
    values = np.arange(8, dtype=float).reshape(2, 2, 2)
    grid = GridResult("volume", GridSpec((0, 1, 0, 1, 0, 1), (2, 2, 2)), values, "idw", "dataset")
    masked = page._masked_grid(grid)
    assert masked.count() == 4
    page.close()
    assert app is not None


def test_all_volume_render_modes_draw_without_changing_grid():
    app = _application()
    page = PlotPage()
    values = np.arange(27, dtype=float).reshape(3, 3, 3)
    original = values.copy()
    grid = GridResult("volume", GridSpec((0, 2, 0, 2, 0, 2), (3, 3, 3)), values, "idw", "dataset")
    dataset = Dataset(
        name="volume", columns={"x": np.array([0., 1.]), "y": np.array([0., 1.]), "z": np.array([0., 1.]), "v": np.array([1., 2.])},
        roles={"x": "x", "y": "y", "z": "z", "value": "v"},
    )
    for settings in (
        {"mode": "solid", "alpha": .6},
        {"mode": "transparent", "alpha": .2},
        {"mode": "isosurface", "alpha": .7, "isovalue": 13., "direction": "above"},
    ):
        page.plot_volume(dataset, grid, settings)
        assert page.view_kind == "volume"
    np.testing.assert_array_equal(grid.values, original)
    page.close()
    assert app is not None


def test_contour_can_show_decimated_grid_nodes_and_raw_points():
    app = _application()
    page = PlotPage(); page.show_grid_nodes = True; page.show_original_points = True
    dataset = Dataset(
        name="xyv", columns={"x": np.array([0., 1., 0., 1.]), "y": np.array([0., 0., 1., 1.]), "v": np.array([1., 2., 3., 4.])},
        roles={"x": "x", "y": "y", "value": "v"},
    )
    grid = GridResult("grid", GridSpec((0, 1, 0, 1), (8, 8)), np.arange(64, dtype=float).reshape(8, 8), "idw", dataset.id)
    page.plot_contour(dataset, grid)
    assert "网格节点 64/64" in page.coordinate_label.text()
    page.close(); assert app is not None


def test_ab_oblique_slice_uses_positive_down_depth():
    app = _application()
    grid = GridResult("volume", GridSpec((0, 10, 0, 10, 0, 10), (11, 11, 11)), np.arange(1331, dtype=float).reshape(11, 11, 11), "idw", "source")
    dialog = VolumeSliceDialog(grid)
    assert dialog.mode.currentData() == "ab"
    dialog.ax.setValue(0); dialog.ay.setValue(5); dialog.bx.setValue(10); dialog.by.setValue(5)
    dialog.ztop.setValue(0); dialog.zbottom.setValue(10); dialog.dip.setValue(90)
    result = dialog.build_slice()
    assert result.metadata["type"] == "arbitrary"
    assert result.valid_fraction > 0.8
    dialog.close(); assert app is not None


def test_geology_regular_shape_tools_create_expected_geometry():
    app = _application()
    dataset = Dataset("section", {"x": np.array([0., 1.]), "y": np.array([0., 1.]), "v": np.array([1., 2.])}, {"x": "x", "y": "y", "value": "v"})
    window = GeologySectionWindow(dataset, None, None, Threshold())
    window.shape_tool.setCurrentText("矩形")
    assert len(window._shape_points((0., 0.), (2., 1.))) == 4
    window.shape_tool.setCurrentText("椭圆")
    assert len(window._shape_points((0., 0.), (2., 1.))) == 72
    window.shape_tool.setCurrentText("圆形")
    points = np.asarray(window._shape_points((0., 0.), (2., 1.)))
    center = points.mean(axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    assert np.ptp(radii) < 1e-10
    window.close(); assert app is not None


def test_main_window_threshold_combo_converts_qt_string_to_enum():
    app = _application(); main = MainWindow()
    dataset = Dataset("values", {"x": np.arange(4.), "y": np.arange(4.), "v": np.arange(4.)}, {"x": "x", "y": "y", "value": "v"})
    main.project.add_dataset(dataset); main.active_dataset_id = dataset.id; main._refresh_project_view(); main.page.plot_points(dataset)
    main.threshold_mode.setCurrentIndex(main.threshold_mode.findData(ThresholdMode.ABOVE)); main.upper.setText("2")
    main._apply_threshold(silent=True)
    assert main.page.threshold == Threshold(ThresholdMode.ABOVE, upper=2.0)
    assert "Value 3 可见" in main.threshold_result_label.text()
    main._reset_threshold(); assert main.page.threshold == Threshold()
    main.project.dirty = False
    main.close(); assert app is not None


def test_custom_toolbar_wheel_zoom_changes_visible_range():
    app = _application(); page = PlotPage()
    dataset = Dataset("zoom", {"x": np.arange(10.), "y": np.arange(10.), "v": np.arange(10.)}, {"x": "x", "y": "y", "value": "v"})
    page.plot_points(dataset); axis = page.figure.axes[0]
    before = np.ptp(axis.get_xlim())
    page.toolbar._scroll(SimpleNamespace(inaxes=axis, xdata=5.0, ydata=5.0, button="up"))
    assert np.ptp(axis.get_xlim()) < before
    page.toolbar.back(); assert np.ptp(axis.get_xlim()) == pytest.approx(before)
    page.close(); assert app is not None


def test_geology_rectangle_mouse_drag_creates_layer_and_preserves_editability():
    app = _application()
    dataset = Dataset("section", {"x": np.array([0., 1.]), "y": np.array([0., 1.]), "v": np.array([1., 2.])}, {"x": "x", "y": "y", "value": "v"})
    window = GeologySectionWindow(dataset, None, None, Threshold())
    window._activate_tool("矩形"); axis = window.figure.axes[0]
    start_display = axis.transData.transform((0., 0.)); end_display = axis.transData.transform((2., 1.))
    window._press(SimpleNamespace(inaxes=axis, xdata=0., ydata=0., x=start_display[0], y=start_display[1], button=1))
    window._motion(SimpleNamespace(inaxes=axis, xdata=2., ydata=1., x=end_display[0], y=end_display[1], button=1))
    window._release(SimpleNamespace(inaxes=window.figure.axes[0], xdata=2., ydata=1., x=end_display[0], y=end_display[1], button=1))
    assert len(window.layers) == 1 and len(window.layers[0]["points"]) == 4
    assert window.mode == "select"
    window.close(); assert app is not None


def test_geology_plot_settings_control_titles_ranges_and_contour_color():
    app = _application()
    dataset = Dataset("section", {"x": np.array([0., 1.]), "y": np.array([0., 1.]), "v": np.array([1., 2.])}, {"x": "x", "y": "y", "value": "v"})
    grid = GridResult("grid", GridSpec((0, 10, 0, 5), (8, 9)), np.arange(72, dtype=float).reshape(8, 9), "idw", dataset.id)
    window = GeologySectionWindow(dataset, grid, None, Threshold())
    window.plot_settings.update(
        title="自定义图名", xlabel="测线距离 / m", ylabel="深度 / m",
        line_color="#ff0000", show_fill=False,
        xmin=1.0, xmax=9.0, ymin=0.5, ymax=4.5,
    )
    window._has_rendered = False; window.render()
    axis = window.figure.axes[0]
    assert axis.get_title() == "自定义图名"
    assert axis.get_xlabel() == "测线距离 / m" and axis.get_ylabel() == "深度 / m"
    assert axis.get_xlim() == pytest.approx((1.0, 9.0))
    assert axis.get_ylim() == pytest.approx((4.5, 0.5))
    np.testing.assert_allclose(axis.collections[0].get_edgecolors()[0], [1., 0., 0., 1.])
    assert window.layer_list.maximumHeight() <= 125
    assert window.draw_tool_combo.findData("圆形") >= 0 and window.draw_tool_combo.findData("移动") >= 0
    window.close(); assert app is not None


def test_geology_layers_and_plot_settings_are_persisted_together():
    app = _application(); main = MainWindow()
    dataset_id = "section-source"
    settings = {"title": "保存后的图名", "line_color": "#123456"}
    layers = [{"name": "L1", "points": [[0, 0], [1, 1]]}]
    main._store_geology_settings(dataset_id, settings)
    main._store_geology_layers(dataset_id, layers)
    record = next(page for page in main.project.pages if page.get("dataset_id") == dataset_id)
    assert record["settings"] == settings and record["layers"] == layers
    main.project.dirty = False; main.close(); assert app is not None


def test_xyz_spatial_filter_is_evaluated_before_value_threshold():
    app = _application(); page = PlotPage()
    dataset = Dataset(
        "xyzv", {"x": np.arange(5.), "y": np.arange(5.), "z": np.arange(5.), "v": np.arange(5.) * 10},
        {"x": "x", "y": "y", "z": "z", "value": "v"},
    )
    page.spatial_filter = SpatialFilter(x=(1., 3.), y=(0., 4.), z=(2., 4.))
    page.threshold = Threshold(ThresholdMode.ABOVE, upper=30.)
    spatial, value, visible = page.dataset_masks(dataset)
    assert spatial.tolist() == [False, False, True, True, False]
    assert value.tolist() == [True, True, True, True, False]
    assert visible.tolist() == [False, False, True, True, False]
    page.close(); assert app is not None


def test_grid_spatial_filter_uses_broadcast_mask_without_coordinate_cube():
    app = _application(); page = PlotPage()
    grid = GridResult("v", GridSpec((0, 3, 0, 3, 0, 3), (4, 4, 4)), np.ones((4, 4, 4)), "idw", "source")
    page.spatial_filter = SpatialFilter(x=(1., 2.), y=(0., 1.), z=(2., 3.))
    spatial = page.grid_spatial_mask(grid)
    assert spatial.shape == grid.values.shape and np.count_nonzero(spatial) == 8
    page.close(); assert app is not None


def test_data_zoom_is_clamped_and_toolbar_has_single_zoom_model():
    app = _application(); page = PlotPage()
    dataset = Dataset("zoom", {"x": np.arange(10.), "y": np.arange(10.), "v": np.arange(10.)}, {"x": "x", "y": "y", "value": "v"})
    page.plot_points(dataset); axis = page.figure.axes[0]; original = axis.get_xlim()
    for _ in range(20): page.toolbar.zoom(1.25)
    allowed_span = np.ptp(original) * 1.10
    assert np.ptp(axis.get_xlim()) <= allowed_span + 1e-9
    assert not hasattr(page.toolbar, "page_zoom")
    page.close(); assert app is not None


def test_colorbar_is_page_object_with_exact_geometry_and_annotations_survive_redraw():
    app = _application(); page = PlotPage()
    dataset = Dataset("plot", {"x": np.arange(4.), "y": np.arange(4.), "v": np.arange(4.)}, {"x": "x", "y": "y", "value": "v"})
    page.style.update(colorbar_x=.80, colorbar_y=.15, colorbar_width=.04, colorbar_height=.50)
    page.annotations.objects.append({"id": "note", "type": "text", "point": (.2, .8), "text": "科研注释", "color": "#202020", "size": 11})
    page.plot_points(dataset)
    assert page._colorbar_axes[-1].get_position().bounds == pytest.approx((.80, .15, .04, .50))
    assert "note" in page.annotations._artists
    page.redraw(); assert "note" in page.annotations._artists
    page.close(); assert app is not None


def test_canvas_recipe_restores_filter_style_and_page_annotations():
    app = _application(); main = MainWindow()
    dataset = Dataset("recipe", {"x": np.arange(4.), "y": np.arange(4.), "v": np.arange(4.)}, {"x": "x", "y": "y", "value": "v"})
    main.project.add_dataset(dataset); main.active_dataset_id = dataset.id
    main.page.spatial_filter = SpatialFilter(x=(1., 3.)); main.page.threshold = Threshold(ThresholdMode.ABOVE, upper=2.)
    main.page.style["colorbar_x"] = .77
    main.page.annotations.objects.append({"id": "saved-note", "type": "text", "point": (.1, .9), "text": "saved", "color": "#202020", "size": 11})
    main.page.plot_points(dataset); main._serialize_canvas_pages(); main._restore_canvas_pages()
    assert main.page.spatial_filter.x == (1., 3.) and main.page.threshold.upper == 2.
    assert main.page.style["colorbar_x"] == pytest.approx(.77)
    assert main.page.annotations.objects[0]["id"] == "saved-note"
    main.project.dirty = False; main.close(); assert app is not None


def test_default_left_drag_pans_without_pan_button_and_fit_restores_home():
    app = _application(); page = PlotPage()
    dataset = Dataset("nav", {"x": np.arange(10.), "y": np.arange(10.), "v": np.arange(10.)}, {"x": "x", "y": "y", "value": "v"})
    page.plot_points(dataset); axis = page.figure.axes[0]; home = (axis.get_xlim(), axis.get_ylim())
    assert "pan" not in page.toolbar._buttons
    page.toolbar.zoom(.5); before_pan = axis.get_xlim()
    page.toolbar._press(SimpleNamespace(button=1, dblclick=False, inaxes=axis, xdata=4., ydata=4.))
    page.toolbar._motion(SimpleNamespace(inaxes=axis, xdata=5., ydata=4.))
    page.toolbar._release(SimpleNamespace(inaxes=axis, xdata=5., ydata=4.))
    assert axis.get_xlim() != before_pan
    page.toolbar.fit(); assert axis.get_xlim() == pytest.approx(home[0]) and axis.get_ylim() == pytest.approx(home[1])
    page.close(); assert app is not None


def test_box_zoom_can_return_to_home_and_qt_wheel_reaches_canvas():
    app = _application(); page = PlotPage(); page.resize(900, 650); page.show(); app.processEvents()
    dataset = Dataset("nav", {"x": np.arange(10.), "y": np.arange(10.), "v": np.arange(10.)}, {"x": "x", "y": "y", "value": "v"})
    page.plot_points(dataset); app.processEvents(); axis = page.figure.axes[0]; home = axis.get_xlim()
    page.toolbar.set_mode("box")
    page.toolbar._press(SimpleNamespace(button=1, dblclick=False, inaxes=axis, xdata=2., ydata=2.))
    page.toolbar._release(SimpleNamespace(inaxes=axis, xdata=6., ydata=6.))
    assert np.ptp(axis.get_xlim()) < np.ptp(home)
    page.toolbar.home(); assert axis.get_xlim() == pytest.approx(home)
    center_display = axis.transData.transform((4.5, 4.5)); local = QPointF(center_display[0], page.canvas.height() - center_display[1])
    wheel = QWheelEvent(local, page.canvas.mapToGlobal(local.toPoint()), QPoint(), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
    before = np.ptp(axis.get_xlim()); QApplication.sendEvent(page.canvas, wheel); app.processEvents()
    assert np.ptp(axis.get_xlim()) < before
    page.close(); assert app is not None

