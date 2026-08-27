from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QFormLayout, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton, QSlider,
    QStatusBar, QTabWidget, QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from gmes_plot import __version__
from gmes_plot.domain.models import Dataset, GridResult, Project, SpatialFilter, TerrainProfile, Threshold, ThresholdMode
from gmes_plot.domain.standards import STANDARD_PROFILES, STRATIGRAPHY_PROFILE, STYLE_LIBRARY_VERSION
from gmes_plot.io.project import load_project, save_project
from gmes_plot.resources import asset_path
from gmes_plot.services.gridding import interpolate, recommend_grid
from gmes_plot.services.quality import build_quality_report, idw_cross_validation
from gmes_plot.services.research import layer_provenance, provenance_markdown, publication_checks, write_companion_report
from gmes_plot.services.slicing import VolumeSlice, orthogonal_slice
from gmes_plot.services.terrain import terrain_corrected_mesh
from gmes_plot.services.thresholds import METHOD_LABELS, recommend_anomaly_thresholds
from gmes_plot.ui.dialogs import DatasetPropertiesDialog, ImportDialog, InterpolationDialog, PlotPropertiesDialog, VolumeDisplayDialog, VolumeSliceDialog
from gmes_plot.ui.statistics_window import StatisticsWindow
from gmes_plot.ui.geology_window import GeologySectionWindow
from gmes_plot.ui.terrain_window import TerrainProfileWindow
from gmes_plot.ui.volume_window import AcceleratedSurfaceWindow, AcceleratedVolumeWindow, accelerated_available
from gmes_plot.ui.canvas_toolbar import ScientificCanvasToolbar
from gmes_plot.ui.canvas_annotations import CanvasAnnotationManager
from gmes_plot.visualization import configure_matplotlib

configure_matplotlib()


class TaskSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class FunctionTask(QRunnable):
    def __init__(self, function) -> None:
        super().__init__()
        self.function = function
        self.signals = TaskSignals()

    def run(self) -> None:
        try:
            self.signals.finished.emit(self.function())
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class PlotPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = ScientificCanvasToolbar(self.canvas, self.figure, self)
        self.toolbar.properties_requested.connect(self._request_properties)
        self.annotations = CanvasAnnotationManager(self.canvas, self.figure, self, self.toolbar)
        self.toolbar.annotation_requested.connect(self.annotations.set_tool)
        self.coordinate_label = QLabel("就绪")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        # The canvas is embedded directly.  A QScrollArea consumed wheel events
        # and mixed paper-size scrolling with scientific data navigation.
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.coordinate_label)
        self.title = ""
        self.xlabel = "X"
        self.ylabel = "Y"
        self.cmap = "viridis"
        self.style = {
            "contour_levels": 20, "font_size": 10, "line_width": 0.35,
            "point_size": 18.0, "show_grid": True, "aspect": "auto",
            "x_scale": "linear", "y_scale": "linear",
            "colorbar_orientation": "vertical", "background": "#ffffff",
            "colorbar_x": 0.91, "colorbar_y": 0.20,
            "colorbar_width": 0.025, "colorbar_height": 0.62,
        }
        self.threshold = Threshold()
        self.spatial_filter = SpatialFilter()
        self.show_original_points = True
        self.show_grid_nodes = False
        self.show_node_labels = False
        self.max_display_points = 50_000
        self.max_display_nodes = 30_000
        self.view_kind = "empty"
        self.last_dataset: Dataset | None = None
        self.last_grid: GridResult | None = None
        self.last_terrain: TerrainProfile | None = None
        self.last_volume_grid: GridResult | None = None
        self.last_volume_slice: VolumeSlice | None = None
        self.last_volume_render = {"mode": "solid", "alpha": 0.72, "isovalue": None, "direction": "above"}
        self._slice_axis = None
        self._colorbar_axes: list = []
        self._colorbar_drag = None
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_drag)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)

    def _request_properties(self) -> None:
        window = self.window()
        if hasattr(window, "_edit_plot_properties"):
            window._edit_plot_properties()

    def _draw_complete(self) -> None:
        self.figure.set_facecolor(self.style["background"])
        for axis in self.figure.axes:
            if hasattr(axis, "_colorbar"):
                continue
            axis.set_facecolor(self.style["background"])
            axis.title.set_fontsize(self.style["font_size"] + 2)
            axis.xaxis.label.set_fontsize(self.style["font_size"])
            axis.yaxis.label.set_fontsize(self.style["font_size"])
            axis.tick_params(labelsize=max(6, self.style["font_size"] - 1))
            if not hasattr(axis, "get_zlim3d"):
                try:
                    axis.set_xscale(self.style["x_scale"]); axis.set_yscale(self.style["y_scale"])
                    axis.set_aspect(self.style["aspect"])
                except ValueError:
                    axis.set_xscale("linear"); axis.set_yscale("linear")
                axis.grid(self.style["show_grid"], alpha=0.18)
        self.annotations.render()
        self.canvas.draw_idle()
        self.toolbar.capture_view()

    def _add_colorbar(self, artist, axis, **kwargs):
        orientation = self.style["colorbar_orientation"]
        width = max(.01, min(.8, self.style["colorbar_width"])); height = max(.03, min(.9, self.style["colorbar_height"]))
        rect = [
            max(0., min(1. - width, self.style["colorbar_x"])),
            max(0., min(1. - height, self.style["colorbar_y"])), width, height,
        ]
        cax = self.figure.add_axes(rect, label=f"gmes-colorbar-{len(self._colorbar_axes)}")
        axis.set_in_layout(False)
        for layout_only in ("pad", "fraction", "shrink", "aspect"):
            kwargs.pop(layout_only, None)
        colorbar = self.figure.colorbar(artist, cax=cax, orientation=orientation, **kwargs)
        colorbar.ax._gmes_colorbar = True; colorbar.ax.set_in_layout(False)
        self._colorbar_axes.append(colorbar.ax)
        return colorbar

    def _on_canvas_press(self, event) -> None:
        if event.button != 1:
            return
        if event.inaxes in self._colorbar_axes and event.x is not None and event.y is not None:
            bounds = event.inaxes.get_position().bounds
            self._colorbar_drag = (event.inaxes, float(event.x), float(event.y), bounds, event.key == "shift")

    def _on_canvas_drag(self, event) -> None:
        if self._colorbar_drag is None or event.x is None or event.y is None:
            return
        axis, x0, y0, bounds, resizing = self._colorbar_drag
        width_px, height_px = max(1, self.canvas.width()), max(1, self.canvas.height())
        dx, dy = (event.x - x0) / width_px, (event.y - y0) / height_px
        x, y, width, height = bounds
        if resizing:
            width = max(.01, min(.8, width + dx)); height = max(.03, min(.9, height + dy))
        else:
            x = max(0., min(1. - width, x + dx)); y = max(0., min(1. - height, y + dy))
        axis.set_position([x, y, width, height]); self.canvas.draw_idle()

    def _on_canvas_release(self, _event) -> None:
        if self._colorbar_drag is None:
            return
        axis = self._colorbar_drag[0]; x, y, width, height = axis.get_position().bounds
        self.style.update(colorbar_x=x, colorbar_y=y, colorbar_width=width, colorbar_height=height)
        self._colorbar_drag = None
        self.coordinate_label.setText("色标位置已更新；Shift+拖动可调整尺寸，普通拖动可移动")

    def _on_mouse_move(self, event) -> None:
        if self.last_volume_slice is None or event.inaxes is not self._slice_axis or event.xdata is None or event.ydata is None:
            return
        volume_slice = self.last_volume_slice
        column = int(np.argmin(np.abs(volume_slice.horizontal - event.xdata)))
        row = int(np.argmin(np.abs(volume_slice.vertical - event.ydata)))
        value = volume_slice.values[row, column]
        self.coordinate_label.setText(
            f"X={volume_slice.x[row, column]:.6g}  "
            f"Y={volume_slice.y[row, column]:.6g}  "
            f"Z={volume_slice.z[row, column]:.6g}  "
            f"Value={value:.6g}"
        )
    def dataset_masks(self, dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        spatial = np.ones(dataset.row_count, dtype=bool)
        for role in ("x", "y", "z"):
            if role in dataset.roles:
                spatial &= self.spatial_filter.axis_mask(role, dataset.role_values(role))
        value = self.threshold.mask(dataset.role_values("value"))
        return spatial, value, spatial & value

    def grid_spatial_mask(self, grid: GridResult, axes: tuple[np.ndarray, ...] | None = None) -> np.ndarray:
        axes = axes or grid.spec.axes()
        if grid.spec.dimensions == 2:
            x_axis, y_axis = axes
            return self.spatial_filter.axis_mask("y", y_axis)[:, None] & self.spatial_filter.axis_mask("x", x_axis)[None, :]
        x_axis, y_axis, z_axis = axes
        return (
            self.spatial_filter.axis_mask("z", z_axis)[:, None, None]
            & self.spatial_filter.axis_mask("y", y_axis)[None, :, None]
            & self.spatial_filter.axis_mask("x", x_axis)[None, None, :]
        )

    def grid_masks(self, grid: GridResult, axes: tuple[np.ndarray, ...] | None = None, values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = np.asarray(grid.values if values is None else values)
        spatial = self.grid_spatial_mask(grid, axes)
        value = self.threshold.mask(data)
        return spatial, value, spatial & value

    def _masked_grid(self, grid: GridResult) -> np.ma.MaskedArray:
        _spatial, _value, visible = self.grid_masks(grid)
        return np.ma.array(grid.values, mask=~visible)

    @staticmethod
    def _display_indices(count: int, maximum: int) -> np.ndarray:
        if count <= maximum:
            return np.arange(count)
        return np.linspace(0, count - 1, maximum, dtype=int)

    def plot_points(self, dataset: Dataset) -> None:
        self._colorbar_axes.clear(); self.figure.clear()
        ax = self.figure.add_subplot(111)
        x, y, values = dataset.role_values("x"), dataset.role_values("y"), dataset.role_values("value")
        _spatial, _value, visible = self.dataset_masks(dataset)
        if not np.any(visible):
            ax.text(0.5, 0.5, "当前显示阈值隐藏了全部数据", transform=ax.transAxes, ha="center", va="center")
            ax.set(title=self.title or dataset.name, xlabel=self.xlabel, ylabel=self.ylabel)
            self.last_dataset, self.last_grid = dataset, None
            self.view_kind = "points"
            self._draw_complete()
            return
        selected = np.flatnonzero(visible)
        shown = selected[self._display_indices(selected.size, self.max_display_points)]
        artist = ax.scatter(x[shown], y[shown], c=values[shown], cmap=self.cmap, s=self.style["point_size"], edgecolors="black", linewidths=0.15)
        self._add_colorbar(artist, ax, label=dataset.units.get(dataset.roles["value"], dataset.roles["value"]))
        ax.set(title=self.title or dataset.name, xlabel=self.xlabel, ylabel=self.ylabel)
        self.last_dataset, self.last_grid = dataset, None
        self.last_terrain = None
        self.last_volume_grid = None
        self.last_volume_slice = None
        self.view_kind = "points"
        self.coordinate_label.setText(f"原始点：显示 {shown.size:,} / 有效 {selected.size:,} / 全部 {values.size:,}；分析仍使用完整数据")
        self._draw_complete()

    def plot_contour(self, dataset: Dataset, grid: GridResult) -> None:
        self._colorbar_axes.clear(); self.figure.clear()
        ax = self.figure.add_subplot(111)
        x_axis, y_axis = grid.spec.axes()
        values = self._masked_grid(grid)
        if values.count() == 0:
            ax.text(0.5, 0.5, "当前显示阈值隐藏了全部网格", transform=ax.transAxes, ha="center", va="center")
            ax.set(title=self.title or grid.name, xlabel=self.xlabel, ylabel=self.ylabel)
            self.last_dataset, self.last_grid = dataset, grid
            self.view_kind = "contour"
            self._draw_complete()
            return
        artist = ax.contourf(x_axis, y_axis, values, levels=self.style["contour_levels"], cmap=self.cmap)
        ax.contour(x_axis, y_axis, values, levels=self.style["contour_levels"], colors="black", linewidths=self.style["line_width"], alpha=0.55)
        layer_parts = ["填色等值线", "等值线"]
        if self.show_original_points:
            raw_x, raw_y = dataset.role_values("x"), dataset.role_values("y")
            _raw_spatial, _raw_value, raw_visible = self.dataset_masks(dataset)
            candidates = np.flatnonzero(raw_visible)
            raw_indices = candidates[self._display_indices(candidates.size, self.max_display_points)]
            ax.scatter(raw_x[raw_indices], raw_y[raw_indices], s=7, c="#111111", alpha=0.42, marker="o", label="原始散点")
            layer_parts.append(f"原始点 {raw_indices.size:,}/{candidates.size:,} 可见")
        if self.show_grid_nodes:
            xx, yy = np.meshgrid(x_axis, y_axis)
            flat_values = np.asarray(grid.values).ravel()
            valid_nodes = np.flatnonzero(~np.ma.getmaskarray(values).ravel())
            node_indices = valid_nodes[self._display_indices(valid_nodes.size, self.max_display_nodes)]
            ax.scatter(xx.ravel()[node_indices], yy.ravel()[node_indices], s=9, facecolors="none", edgecolors="#f5f5f5", linewidths=0.45, alpha=0.85, marker="s", label="插值网格节点")
            layer_parts.append(f"网格节点 {node_indices.size:,}/{valid_nodes.size:,}")
            if self.show_node_labels and node_indices.size <= 1_500:
                for index in node_indices:
                    ax.annotate(f"{flat_values[index]:.3g}", (xx.ravel()[index], yy.ravel()[index]), fontsize=5, color="#222222", xytext=(2, 2), textcoords="offset points")
                layer_parts.append("节点值标签")
            elif self.show_node_labels:
                layer_parts.append("节点标签因超过1,500个自动隐藏")
        self._add_colorbar(artist, ax, label=dataset.units.get(dataset.roles["value"], dataset.roles["value"]))
        ax.set(title=self.title or grid.name, xlabel=self.xlabel, ylabel=self.ylabel)
        self.last_dataset, self.last_grid = dataset, grid
        self.last_terrain = None
        self.last_volume_grid = None
        self.last_volume_slice = None
        self.view_kind = "contour"
        self.coordinate_label.setText("；".join(layer_parts) + "；阈值仅影响显示层")
        if self.show_original_points or self.show_grid_nodes:
            ax.legend(loc="best", fontsize=8)
        self._draw_complete()

    def plot_surface(self, dataset: Dataset, grid: GridResult) -> None:
        if grid.spec.dimensions != 2:
            raise ValueError("当前曲面视图需要二维网格")
        self._colorbar_axes.clear(); self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        x_axis, y_axis = grid.spec.axes()
        x_indices = np.flatnonzero(self.spatial_filter.axis_mask("x", x_axis)); y_indices = np.flatnonzero(self.spatial_filter.axis_mask("y", y_axis))
        if not x_indices.size or not y_indices.size: raise ValueError("XY空间裁剪后没有可见曲面")
        sx, sy = max(1, int(np.ceil(x_indices.size / 80))), max(1, int(np.ceil(y_indices.size / 80)))
        x_indices, y_indices = x_indices[::sx], y_indices[::sy]
        xx, yy = np.meshgrid(x_axis[x_indices], y_axis[y_indices])
        sampled = np.asarray(grid.values[np.ix_(y_indices, x_indices)], dtype=float)
        masked = np.ma.array(sampled, mask=~self.threshold.mask(sampled))
        surface = ax.plot_surface(xx, yy, masked, cmap=self.cmap, linewidth=0, antialiased=False, shade=False)
        self._add_colorbar(surface, ax, shrink=0.65)
        ax.set(title=self.title or f"{grid.name} 三维曲面", xlabel=self.xlabel, ylabel=self.ylabel, zlabel="Value", xlim=(x_axis[x_indices].min(), x_axis[x_indices].max()), ylim=(y_axis[y_indices].min(), y_axis[y_indices].max()))
        self.last_dataset, self.last_grid = dataset, grid
        self.last_terrain = None
        self.last_volume_grid = None
        self.last_volume_slice = None
        self.view_kind = "surface"
        self._draw_complete()

    @staticmethod
    def _cell_edges(axis: np.ndarray) -> np.ndarray:
        if axis.size == 1:
            return np.array([axis[0] - 0.5, axis[0] + 0.5])
        middle = (axis[:-1] + axis[1:]) / 2
        return np.concatenate(([axis[0] - (axis[1] - axis[0]) / 2], middle, [axis[-1] + (axis[-1] - axis[-2]) / 2]))

    def plot_volume(self, dataset: Dataset, grid: GridResult, settings: dict | None = None) -> None:
        """Render an XYZV grid in solid, transparent, or isovalue-body mode."""
        if grid.spec.dimensions != 3:
            raise ValueError("三维数据体视图需要 XYZV 三维规则体")
        render = dict(self.last_volume_render)
        if settings:
            render.update(settings)
        mode = render.get("mode", "solid")
        alpha = float(render.get("alpha", 0.55))
        self._colorbar_axes.clear(); self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        x_axis, y_axis, z_axis = grid.spec.axes()
        nz, ny, nx = grid.values.shape
        sx, sy, sz = max(1, int(np.ceil(nx / 36))), max(1, int(np.ceil(ny / 36))), max(1, int(np.ceil(nz / 36)))
        sampled = np.asarray(grid.values[::sz, ::sy, ::sx], dtype=float)
        xa, ya, za = x_axis[::sx], y_axis[::sy], z_axis[::sz]
        spatial = (
            self.spatial_filter.axis_mask("z", za)[:, None, None]
            & self.spatial_filter.axis_mask("y", ya)[None, :, None]
            & self.spatial_filter.axis_mask("x", xa)[None, None, :]
        )
        visible = spatial & self.threshold.mask(sampled)
        finite = sampled[visible]
        if finite.size and np.any(visible):
            vmin, vmax = float(finite.min()), float(finite.max())
            if vmin == vmax:
                vmax = vmin + np.finfo(float).eps
            norm = matplotlib.colors.Normalize(vmin, vmax)
            colors_zyx = matplotlib.colormaps[self.cmap](norm(sampled))
            if mode == "transparent":
                zz, yy, xx = np.meshgrid(za, ya, xa, indexing="ij")
                point_mask = visible
                ax.scatter(xx[point_mask], yy[point_mask], zz[point_mask], c=sampled[point_mask], cmap=self.cmap, norm=norm, s=18, alpha=alpha, linewidths=0, depthshade=False)
                mode_label = "透明体（CPU试验）"
            else:
                body = visible.copy()
                if mode == "isosurface":
                    isovalue = float(render.get("isovalue") if render.get("isovalue") is not None else np.median(finite))
                    direction = render.get("direction", "above")
                    body &= sampled >= isovalue if direction == "above" else sampled <= isovalue
                    render["isovalue"] = isovalue
                    mode_label = f"等值面体（{'≥' if direction == 'above' else '≤'}{isovalue:.6g}）"
                else:
                    mode_label = "实心体素"
                colors_zyx[..., 3] = np.where(body, alpha, 0.0)
                filled = body.transpose(2, 1, 0)
                facecolors = colors_zyx.transpose(2, 1, 0, 3)
                xe, ye, ze = np.meshgrid(
                    self._cell_edges(xa), self._cell_edges(ya), self._cell_edges(za), indexing="ij"
                )
                if np.any(filled):
                    ax.voxels(xe, ye, ze, filled, facecolors=facecolors, edgecolor=(0, 0, 0, 0.04), shade=False)
                else:
                    ax.text2D(0.5, 0.5, "等值与显示阈值组合后没有可见体素", transform=ax.transAxes, ha="center")
            scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=self.cmap)
            self._add_colorbar(scalar, ax, shrink=0.65, label=dataset.roles["value"])
        else:
            ax.text2D(0.5, 0.5, "当前显示阈值隐藏了全部体素", transform=ax.transAxes, ha="center")
        ax.set(
            title=self.title or f"{grid.name} — XYZV三维数据体",
            xlabel="X", ylabel="Y", zlabel="Z",
            xlim=(x_axis.min(), x_axis.max()), ylim=(y_axis.min(), y_axis.max()), zlim=(z_axis.min(), z_axis.max()),
        )
        ax.invert_zaxis()
        self.coordinate_label.setText(
            f"{mode_label if finite.size and np.any(visible) else '无可见数据'}；原始 {nx}×{ny}×{nz}，显示 {len(xa)}×{len(ya)}×{len(za)}；深度Z向下为正；阈值仅作用于显示层"
        )
        self.last_dataset = dataset
        self.last_grid = None
        self.last_terrain = None
        self.last_volume_grid = grid
        self.last_volume_slice = None
        self.last_volume_render = render
        self._slice_axis = None
        self.view_kind = "volume"
        self._draw_complete()

    def plot_terrain_section(self, dataset: Dataset, grid: GridResult, terrain: TerrainProfile) -> None:
        if grid.spec.dimensions != 2:
            raise ValueError("带地形断面需要二维断面网格")
        self._colorbar_axes.clear(); self.figure.clear()
        ax = self.figure.add_subplot(111)
        x_axis, y_axis = grid.spec.axes()
        corrected = terrain_corrected_mesh(x_axis, y_axis, terrain)
        section_x = corrected.x[0]
        surface_elevation = corrected.surface_elevation
        values = self._masked_grid(grid)[:, corrected.source_columns]
        xx = corrected.x
        absolute_elevation = corrected.elevation

        artist = ax.contourf(xx, absolute_elevation, values, levels=self.style["contour_levels"], cmap=self.cmap)
        ax.contour(xx, absolute_elevation, values, levels=self.style["contour_levels"], colors="black", linewidths=self.style["line_width"], alpha=0.55)
        ax.plot(section_x, surface_elevation, color="#354d26", linewidth=2.2, label="地形线")
        relief_margin = max(float(np.ptp(surface_elevation)) * 0.12, 1.0)
        ax.fill_between(
            section_x, surface_elevation, surface_elevation + relief_margin,
            color="#9cc77b", alpha=0.75, linewidth=0,
        )
        ax.set(
            title=self.title or f"{grid.name} — 地形随动断面",
            xlabel=f"位置 ({terrain.position_unit})",
            ylabel=f"高程 ({terrain.elevation_unit})",
        )
        ax.set_xlim(section_x.min(), section_x.max())
        ax.set_ylim(float(np.nanmin(absolute_elevation)), float(np.nanmax(surface_elevation) + relief_margin))
        ax.grid(True, alpha=0.12)
        ax.legend(loc="upper right")
        self._add_colorbar(artist, ax, label=dataset.roles["value"])
        self.last_dataset, self.last_grid, self.last_terrain = dataset, grid, terrain
        self.last_volume_grid = None
        self.last_volume_slice = None
        self.view_kind = "terrain_section"
        self._draw_complete()

    def plot_volume_slice(self, dataset: Dataset, grid: GridResult, volume_slice: VolumeSlice) -> None:
        self._colorbar_axes.clear(); self.figure.clear()
        slice_ax = self.figure.add_subplot(121)
        space_ax = self.figure.add_subplot(122, projection="3d")
        spatial = (
            self.spatial_filter.axis_mask("x", volume_slice.x)
            & self.spatial_filter.axis_mask("y", volume_slice.y)
            & self.spatial_filter.axis_mask("z", volume_slice.z)
        )
        visible = spatial & self.threshold.mask(np.asarray(volume_slice.values))
        masked = np.ma.array(volume_slice.values, mask=~visible)
        finite = np.asarray(volume_slice.values)[visible]
        if finite.size == 0:
            raise ValueError("该剖面没有可显示数据：可能位于三维网格之外，或被当前显示阈值全部隐藏")
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            vmax = vmin + np.finfo(float).eps
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        color_map = matplotlib.colormaps[self.cmap]

        section_artist = slice_ax.contourf(
            volume_slice.horizontal, volume_slice.vertical, masked,
            levels=self.style["contour_levels"], cmap=color_map, norm=norm,
        )
        slice_ax.contour(
            volume_slice.horizontal, volume_slice.vertical, masked,
            levels=self.style["contour_levels"], colors="black", linewidths=self.style["line_width"], alpha=0.5,
        )
        slice_ax.set(
            title=volume_slice.name,
            xlabel=volume_slice.horizontal_label,
            ylabel=volume_slice.vertical_label,
        )
        if volume_slice.vertical_label == "Z":
            slice_ax.invert_yaxis()
        slice_ax.grid(True, alpha=0.16)
        self._add_colorbar(section_artist, slice_ax, label=dataset.roles["value"])

        facecolors = color_map(norm(np.asarray(volume_slice.values, dtype=float)))
        facecolors[~np.isfinite(volume_slice.values), 3] = 0.0
        row_step = max(1, volume_slice.values.shape[0] // 100)
        col_step = max(1, volume_slice.values.shape[1] // 100)
        space_ax.plot_surface(
            volume_slice.x, volume_slice.y, volume_slice.z,
            facecolors=facecolors, rstride=row_step, cstride=col_step,
            linewidth=0, antialiased=False, shade=False,
        )
        xmin, xmax, ymin, ymax, zmin, zmax = grid.spec.bounds
        corners = np.array([
            [x, y, z]
            for x in (xmin, xmax) for y in (ymin, ymax) for z in (zmin, zmax)
        ])
        space_ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], s=8, c="#555555")
        space_ax.set(
            title="剖面在三维体中的位置", xlabel="X", ylabel="Y", zlabel="Z",
            xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax),
        )
        space_ax.invert_zaxis()
        self.last_dataset = dataset
        self.last_grid = None
        self.last_terrain = None
        self.last_volume_grid = grid
        self.last_volume_slice = volume_slice
        self._slice_axis = slice_ax
        self.view_kind = "volume_slice"
        self.coordinate_label.setText(
            f"中心 XYZ={volume_slice.origin}  法向量={volume_slice.normal}  "
            f"有效覆盖={volume_slice.valid_fraction:.1%}"
        )
        self._draw_complete()

    def redraw(self) -> None:
        if self.last_dataset is None:
            return
        if self.view_kind == "volume" and self.last_volume_grid is not None:
            self.plot_volume(self.last_dataset, self.last_volume_grid, self.last_volume_render)
        elif self.view_kind == "volume_slice" and self.last_volume_grid is not None and self.last_volume_slice is not None:
            self.plot_volume_slice(self.last_dataset, self.last_volume_grid, self.last_volume_slice)
        elif self.view_kind == "terrain_section" and self.last_grid is not None and self.last_terrain is not None:
            self.plot_terrain_section(self.last_dataset, self.last_grid, self.last_terrain)
        elif self.view_kind == "surface" and self.last_grid is not None:
            self.plot_surface(self.last_dataset, self.last_grid)
        elif self.view_kind == "points" or self.last_grid is None:
            self.plot_points(self.last_dataset)
        else:
            self.plot_contour(self.last_dataset, self.last_grid)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.resize(1400, 900)
        self.project = Project()
        self._apply_default_standards_profile()
        self.active_dataset_id: str | None = None
        self.thread_pool = QThreadPool.globalInstance()
        self._workers: list[FunctionTask] = []
        self._terrain_windows: list[TerrainProfileWindow] = []
        self._auxiliary_windows: list[QWidget] = []
        self._threshold_range = (0.0, 1.0)
        self._threshold_timer = QTimer(self)
        self._threshold_timer.setSingleShot(True)
        self._threshold_timer.setInterval(150)
        self._threshold_timer.timeout.connect(lambda: self._apply_threshold(silent=True))
        self.setWindowIcon(QIcon(str(asset_path("gmes_plot_icon.png"))))
        self.setWindowTitle(f"重磁电震绘图 Pro 试验版 v{__version__} — 未命名工程")
        self._build_ui()
        self._build_actions()
        self._refresh_project_view()

    @property
    def page(self) -> PlotPage:
        return self.tabs.currentWidget()

    def _apply_default_standards_profile(self) -> None:
        if not self.project.standards_profile:
            self.project.standards_profile = {
                "style_library_version": STYLE_LIBRARY_VERSION,
                "standards": [dict(item) for item in STANDARD_PROFILES],
                "stratigraphy": dict(STRATIGRAPHY_PROFILE),
            }

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self._close_page)
        self.tabs.tabBarDoubleClicked.connect(self._rename_page)
        self.tabs.currentChanged.connect(self._page_changed)
        self.tabs.addTab(self._new_plot_page(), "绘图页 1")
        self.setCentralWidget(self.tabs)

        project_panel = QWidget()
        project_layout = QVBoxLayout(project_panel)
        project_layout.setContentsMargins(4, 4, 4, 4)
        self.project_list = QTreeWidget()
        self.project_list.setHeaderLabels(["工程对象", "状态"])
        self.project_list.setSelectionMode(QTreeWidget.SingleSelection)
        self.project_list.currentItemChanged.connect(self._selection_changed)
        self.project_list.itemDoubleClicked.connect(lambda _item, _column: self._edit_selected_object())
        self.project_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.project_list.customContextMenuRequested.connect(self._project_context_menu)
        project_layout.addWidget(self.project_list, 1)
        tree_buttons = QHBoxLayout()
        activate = QPushButton("设为当前"); activate.clicked.connect(self._activate_selected_object)
        edit = QPushButton("编辑…"); edit.clicked.connect(self._edit_selected_object)
        remove = QPushButton("移除…"); remove.clicked.connect(self._remove_selected_object)
        tree_buttons.addWidget(activate); tree_buttons.addWidget(edit); tree_buttons.addWidget(remove)
        project_layout.addLayout(tree_buttons)
        self.active_data_label = QLabel("当前数据：未选择")
        self.active_data_label.setWordWrap(True)
        project_layout.addWidget(self.active_data_label)
        self.project_dock = QDockWidget("工程 / 数据 / 图层", self)
        self.project_dock.setObjectName("projectDock")
        self.project_dock.setWidget(project_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.project_dock)

        properties = QWidget()
        form = QFormLayout(properties)
        self.threshold_mode = QComboBox()
        self.threshold_mode.addItem("全部显示", ThresholdMode.KEEP_ALL)
        self.threshold_mode.addItem("仅显示区间", ThresholdMode.KEEP_RANGE)
        self.threshold_mode.addItem("隐藏区间", ThresholdMode.HIDE_RANGE)
        self.threshold_mode.addItem("隐藏高于上限", ThresholdMode.ABOVE)
        self.threshold_mode.addItem("隐藏低于下限", ThresholdMode.BELOW)
        self.lower = QLineEdit()
        self.upper = QLineEdit()
        self.data_range_label = QLabel("当前值域：无数据")
        self.spatial_range_label = QLabel("空间范围：未启用（先XYZ，后Value）")
        self.spatial_checks: dict[str, QCheckBox] = {}
        self.spatial_min: dict[str, QLineEdit] = {}
        self.spatial_max: dict[str, QLineEdit] = {}
        self._spatial_rows: dict[str, QWidget] = {}
        for axis_name in ("x", "y", "z"):
            enabled = QCheckBox(f"{axis_name.upper()}裁剪")
            minimum, maximum = QLineEdit(), QLineEdit()
            minimum.setPlaceholderText("最小"); maximum.setPlaceholderText("最大")
            row = QWidget(); row_layout = QHBoxLayout(row); row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(enabled); row_layout.addWidget(minimum); row_layout.addWidget(QLabel("～")); row_layout.addWidget(maximum)
            enabled.toggled.connect(lambda checked, name=axis_name: self._spatial_toggled(name, checked))
            self.spatial_checks[axis_name], self.spatial_min[axis_name], self.spatial_max[axis_name] = enabled, minimum, maximum
            self._spatial_rows[axis_name] = row
        self.threshold_result_label = QLabel("可见/隐藏：—")
        self.lower_slider = QSlider(Qt.Horizontal); self.lower_slider.setRange(0, 1000)
        self.upper_slider = QSlider(Qt.Horizontal); self.upper_slider.setRange(0, 1000); self.upper_slider.setValue(1000)
        self.lower_slider.valueChanged.connect(lambda value: self._slider_changed("lower", value))
        self.upper_slider.valueChanged.connect(lambda value: self._slider_changed("upper", value))
        self.physical_method = QComboBox()
        for method, label in METHOD_LABELS.items():
            self.physical_method.addItem(label, method)
        self.anomaly_side = QComboBox(); self.anomaly_side.addItem("高值异常", "high"); self.anomaly_side.addItem("低值异常", "low")
        recommend_threshold = QPushButton("按物探方法推荐异常阈值")
        recommend_threshold.clicked.connect(self._recommend_anomaly_threshold)
        self.apply_threshold_button = QPushButton("应用显示阈值")
        self.apply_threshold_button.clicked.connect(self._apply_threshold)
        reset_threshold = QPushButton("重置为全部显示")
        reset_threshold.clicked.connect(self._reset_threshold)
        self.threshold_mode.currentIndexChanged.connect(self._threshold_mode_changed)
        properties_button = QPushButton("标题 / 坐标轴 / 色标…")
        properties_button.clicked.connect(self._edit_plot_properties)
        self.show_raw = QCheckBox("显示原始散点"); self.show_raw.setChecked(True)
        self.show_nodes = QCheckBox("显示插值网格节点")
        self.show_node_values = QCheckBox("显示节点数值（≤1500）")
        for control in (self.show_raw, self.show_nodes, self.show_node_values):
            control.toggled.connect(self._layer_display_changed)
        form.addRow(self.data_range_label)
        form.addRow(self.spatial_range_label)
        for axis_name in ("x", "y", "z"):
            form.addRow(self._spatial_rows[axis_name])
        form.addRow("阈值模式", self.threshold_mode)
        form.addRow("下限", self.lower)
        form.addRow("下限滑块", self.lower_slider)
        form.addRow("上限", self.upper)
        form.addRow("上限滑块", self.upper_slider)
        form.addRow("物探方法", self.physical_method)
        form.addRow("异常方向", self.anomaly_side)
        form.addRow(recommend_threshold)
        form.addRow(self.apply_threshold_button)
        form.addRow(reset_threshold)
        form.addRow(self.threshold_result_label)
        form.addRow(self.show_raw); form.addRow(self.show_nodes); form.addRow(self.show_node_values)
        form.addRow(properties_button)
        self.properties_dock = QDockWidget("对象属性", self)
        self.properties_dock.setObjectName("propertiesDock")
        self.properties_dock.setWidget(properties)
        self.addDockWidget(Qt.RightDockWidgetArea, self.properties_dock)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪")
        self._threshold_mode_changed()

    def _new_plot_page(self) -> PlotPage:
        page = PlotPage(); page.annotations.changed_callback = self._canvas_objects_changed
        return page

    def _canvas_objects_changed(self) -> None:
        self.project.dirty = True
        self._refresh_project_view()

    def _close_page(self, index: int) -> None:
        if self.tabs.count() <= 1:
            QMessageBox.information(self, "保留页面", "工程中至少保留一个绘图页面。")
            return
        widget = self.tabs.widget(index)
        self.tabs.removeTab(index)
        widget.deleteLater()

    def _rename_page(self, index: int) -> None:
        if index < 0:
            return
        name, ok = QInputDialog.getText(self, "重命名页面", "页面名称", text=self.tabs.tabText(index))
        if ok and name.strip():
            self.tabs.setTabText(index, name.strip())
            self.project.dirty = True

    def _threshold_mode_changed(self) -> None:
        mode = ThresholdMode(self.threshold_mode.currentData())
        lower_enabled = mode in (ThresholdMode.KEEP_RANGE, ThresholdMode.HIDE_RANGE, ThresholdMode.BELOW)
        upper_enabled = mode in (ThresholdMode.KEEP_RANGE, ThresholdMode.HIDE_RANGE, ThresholdMode.ABOVE)
        self.lower.setEnabled(lower_enabled); self.lower_slider.setEnabled(lower_enabled)
        self.upper.setEnabled(upper_enabled); self.upper_slider.setEnabled(upper_enabled)
        if mode is ThresholdMode.KEEP_ALL:
            self.lower.clear(); self.upper.clear()
            self.threshold_result_label.setText("可见/隐藏：全部有效值可见")

    def _page_changed(self, _index: int) -> None:
        if not hasattr(self, "threshold_mode") or self.tabs.currentWidget() is None:
            return
        threshold = self.page.threshold
        self.threshold_mode.setCurrentIndex(self.threshold_mode.findData(threshold.mode))
        self.lower.setText("" if threshold.lower is None else f"{threshold.lower:.12g}")
        self.upper.setText("" if threshold.upper is None else f"{threshold.upper:.12g}")
        self.show_raw.setChecked(self.page.show_original_points)
        self.show_nodes.setChecked(self.page.show_grid_nodes)
        self.show_node_values.setChecked(self.page.show_node_labels)
        self._sync_threshold_range()
        self._sync_spatial_controls()
        self._refresh_project_view()

    def _layer_display_changed(self) -> None:
        if self.tabs.currentWidget() is None:
            return
        self.page.show_original_points = self.show_raw.isChecked()
        self.page.show_grid_nodes = self.show_nodes.isChecked()
        self.page.show_node_labels = self.show_node_values.isChecked()
        if self.page.view_kind == "contour":
            self.page.redraw()

    def _available_axis_values(self, axis_name: str) -> np.ndarray | None:
        page = self.tabs.currentWidget()
        if isinstance(page, PlotPage):
            grid = page.last_volume_grid or page.last_grid
            if grid is not None:
                if axis_name == "z" and grid.spec.dimensions != 3:
                    return None
                index = {"x": 0, "y": 1, "z": 2}[axis_name]
                return grid.spec.axes()[index]
            if page.last_dataset is not None and axis_name in page.last_dataset.roles:
                return page.last_dataset.role_values(axis_name)
        dataset = self.project.datasets.get(self.active_dataset_id or "")
        return dataset.role_values(axis_name) if dataset is not None and axis_name in dataset.roles else None

    def _sync_spatial_controls(self) -> None:
        if self.tabs.currentWidget() is None:
            return
        spatial = self.page.spatial_filter
        for axis_name in ("x", "y", "z"):
            values = self._available_axis_values(axis_name)
            available = bool(values is not None and np.any(np.isfinite(values)))
            limits = getattr(spatial, axis_name)
            check = self.spatial_checks[axis_name]
            check.blockSignals(True); check.setEnabled(available); check.setChecked(limits is not None); check.blockSignals(False)
            self.spatial_min[axis_name].setEnabled(available and limits is not None)
            self.spatial_max[axis_name].setEnabled(available and limits is not None)
            if available:
                finite = np.asarray(values)[np.isfinite(values)]
                low, high = float(finite.min()), float(finite.max())
                self.spatial_min[axis_name].setPlaceholderText(f"{low:.6g}")
                self.spatial_max[axis_name].setPlaceholderText(f"{high:.6g}")
            if limits is None:
                self.spatial_min[axis_name].clear(); self.spatial_max[axis_name].clear()
            else:
                self.spatial_min[axis_name].setText(f"{limits[0]:.12g}"); self.spatial_max[axis_name].setText(f"{limits[1]:.12g}")

    def _spatial_toggled(self, axis_name: str, checked: bool) -> None:
        minimum, maximum = self.spatial_min[axis_name], self.spatial_max[axis_name]
        minimum.setEnabled(checked); maximum.setEnabled(checked)
        if checked and (not minimum.text().strip() or not maximum.text().strip()):
            values = self._available_axis_values(axis_name)
            if values is not None:
                finite = np.asarray(values)[np.isfinite(values)]
                if finite.size:
                    minimum.setText(f"{float(finite.min()):.12g}"); maximum.setText(f"{float(finite.max()):.12g}")

    def _sync_threshold_range(self) -> None:
        values = None
        page = self.tabs.currentWidget()
        if isinstance(page, PlotPage):
            if page.last_volume_grid is not None:
                values = page.last_volume_grid.values
            elif page.last_grid is not None:
                values = page.last_grid.values
            elif page.last_dataset is not None:
                values = page.last_dataset.role_values("value")
        if values is None and self.active_dataset_id in self.project.datasets:
            values = self.project.datasets[self.active_dataset_id].role_values("value")
        finite = np.asarray(values, dtype=float) if values is not None else np.array([])
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            self._threshold_range = (0.0, 1.0); self.data_range_label.setText("当前值域：无有效数据")
            return
        low, high = float(finite.min()), float(finite.max())
        if low == high:
            high = low + np.finfo(float).eps
        self._threshold_range = (low, high)
        self.data_range_label.setText(f"当前值域：{low:.6g} ～ {high:.6g}（{finite.size:,}个有效值）")
        self._set_slider_from_text()

    def _set_slider_from_text(self) -> None:
        low, high = self._threshold_range
        span = high - low
        for edit, slider, fallback in ((self.lower, self.lower_slider, 0), (self.upper, self.upper_slider, 1000)):
            try:
                position = round((float(edit.text()) - low) / span * 1000) if edit.text().strip() else fallback
            except ValueError:
                position = fallback
            slider.blockSignals(True); slider.setValue(max(0, min(1000, position))); slider.blockSignals(False)

    def _slider_changed(self, which: str, position: int) -> None:
        low, high = self._threshold_range
        value = low + (high - low) * position / 1000.0
        (self.lower if which == "lower" else self.upper).setText(f"{value:.12g}")
        if ThresholdMode(self.threshold_mode.currentData()) is not ThresholdMode.KEEP_ALL:
            self._threshold_timer.start()

    def _reset_threshold(self) -> None:
        self.threshold_mode.setCurrentIndex(self.threshold_mode.findData(ThresholdMode.KEEP_ALL))
        self.page.threshold = Threshold()
        self.page.spatial_filter = SpatialFilter()
        for axis_name in ("x", "y", "z"):
            self.spatial_checks[axis_name].setChecked(False)
            self.spatial_min[axis_name].clear(); self.spatial_max[axis_name].clear()
        self.lower_slider.setValue(0); self.upper_slider.setValue(1000)
        self.page.redraw()
        self._update_threshold_counts()

    def _action(self, text: str, slot, shortcut: str | None = None) -> QAction:
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        return action

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        for action in (
            self._action("新建工程", self.new_project, "Ctrl+N"),
            self._action("打开工程…", self.open_project, "Ctrl+O"),
            self._action("保存工程", self.save_project, "Ctrl+S"),
            self._action("工程另存为…", lambda: self.save_project(save_as=True), "Ctrl+Shift+S"),
            self._action("导入数据…", self.import_data, "Ctrl+Shift+I"),
            self._action("导出图片…", self.export_figure, "Ctrl+E"),
        ):
            file_menu.addAction(action)

        analysis = self.menuBar().addMenu("分析与绘图")
        analysis.addAction(self._action("绘制原始散点", self.plot_points))
        analysis.addAction(self._action("二维网格化 / 等值线…", self.grid_2d, "F5"))
        analysis.addAction(self._action("三维网格化计算…（只生成规则体）", self.grid_3d))
        analysis.addAction(self._action("XYZV三维数据体渲染…（显示已有规则体）", self.volume_3d))
        analysis.addAction(self._action("三维体任意 XYZ 剖面…", self.volume_slice))
        analysis.addAction(self._action("XYV三维曲面", self.surface_3d))
        analysis.addSeparator()
        analysis.addAction(self._action("一维与统计图…", self.open_statistics_window, "Ctrl+T"))
        analysis.addAction(self._action("地质剖面解释编辑器…", self.open_geology_window, "Ctrl+G"))
        analysis.addAction(self._action("地形数据窗口…", self.open_terrain_window))
        analysis.addAction(self._action("绘制带地形断面", self.terrain_section))
        analysis.addAction(self._action("生成插值质量报告", self.generate_quality_report))
        analysis.addAction(self._action("钻孔 / 已有地质约束（预留）…", self.show_constraint_reservation))

        research = self.menuBar().addMenu("科研复现")
        research.addAction(self._action("查看当前图层溯源…", self.show_current_provenance, "Ctrl+Shift+R"))
        research.addAction(self._action("科研排版检查…", self.show_publication_checks))
        research.addAction(self._action("导出图片与同名复现报告…", lambda: self.export_figure(force_report=True)))

        self.main_toolbar = QToolBar("常用工具")
        self.main_toolbar.setObjectName("mainToolbar")
        self.main_toolbar.setMovable(True)
        self.addToolBar(self.main_toolbar)
        for text, slot in (("导入", self.import_data), ("散点", self.plot_points), ("2D插值", self.grid_2d), ("3D网格化计算", self.grid_3d), ("3D体渲染", self.volume_3d), ("XYZ剖面", self.volume_slice), ("XYV曲面", self.surface_3d), ("统计图", self.open_statistics_window), ("地质解释", self.open_geology_window), ("地形", self.open_terrain_window), ("带地形断面", self.terrain_section), ("保存", self.save_project), ("导出", self.export_figure)):
            self.main_toolbar.addAction(self._action(text, slot))

        view_menu = self.menuBar().addMenu("视图")
        view_menu.addAction(self.main_toolbar.toggleViewAction())
        view_menu.addAction(self.project_dock.toggleViewAction())
        view_menu.addAction(self.properties_dock.toggleViewAction())
        status_action = QAction("状态栏", self); status_action.setCheckable(True); status_action.setChecked(True)
        status_action.toggled.connect(self.statusBar().setVisible); view_menu.addAction(status_action)

        help_menu = self.menuBar().addMenu("帮助")
        help_menu.addAction(self._action("制图标准与版本…", self.show_standards_info))

    def _selected_dataset(self) -> Dataset:
        if not self.active_dataset_id or self.active_dataset_id not in self.project.datasets:
            raise ValueError("请先在左侧选择数据集")
        return self.project.datasets[self.active_dataset_id]

    def _selection_changed(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None = None) -> None:
        if current is None:
            return
        kind, object_id = current.data(0, Qt.UserRole + 1), current.data(0, Qt.UserRole)
        if kind == "dataset" and object_id in self.project.datasets:
            self.active_dataset_id = object_id
            dataset = self.project.datasets[object_id]
            self.active_data_label.setText(f"当前数据：{dataset.name}\n{dataset.row_count:,}行｜字段 {', '.join(dataset.roles).upper()}")
            self._sync_threshold_range()
            self._sync_spatial_controls()
            self.statusBar().showMessage(f"当前绘图数据已切换为：{dataset.name}", 5000)
        elif kind == "grid" and object_id in self.project.grids:
            grid = self.project.grids[object_id]
            if grid.source_dataset_id in self.project.datasets:
                self.active_dataset_id = grid.source_dataset_id
                dataset = self.project.datasets[self.active_dataset_id]
                self.active_data_label.setText(f"当前结果：{grid.name}\n来源：{dataset.name}｜{grid.spec.cell_count:,}网格单元")
                if grid.spec.dimensions == 2:
                    self.page.plot_contour(dataset, grid)
                self._sync_threshold_range()
                self._sync_spatial_controls()
        elif kind == "canvas_object" and isinstance(self.page, PlotPage):
            self.page.annotations.selected_id = object_id; self.page.annotations.render()
            self.statusBar().showMessage("已选择页面绘图对象；可在画布拖动，Delete删除，双击编辑。", 6000)

    def _activate_selected_object(self) -> None:
        self._selection_changed(self.project_list.currentItem())

    def _project_context_menu(self, position) -> None:
        item = self.project_list.itemAt(position)
        if item is None or item.data(0, Qt.UserRole + 1) not in {"dataset", "grid", "terrain", "canvas_object"}:
            return
        self.project_list.setCurrentItem(item)
        menu = QMenu(self)
        menu.addAction("设为当前 / 显示", self._activate_selected_object)
        menu.addAction("编辑名称、字段映射或单位…", self._edit_selected_object)
        if item.data(0, Qt.UserRole + 1) in {"dataset", "grid"}:
            menu.addAction("查看科研溯源…", self.show_current_provenance)
        menu.addSeparator(); menu.addAction("从工程移除…", self._remove_selected_object)
        menu.exec(self.project_list.viewport().mapToGlobal(position))

    def _edit_selected_object(self) -> None:
        item = self.project_list.currentItem()
        if item is None:
            return
        kind, object_id = item.data(0, Qt.UserRole + 1), item.data(0, Qt.UserRole)
        if kind == "dataset" and object_id in self.project.datasets:
            dataset = self.project.datasets[object_id]
            dialog = DatasetPropertiesDialog(dataset, self)
            if not dialog.exec():
                return
            old_roles = dict(dataset.roles)
            dataset.name = dialog.result_name; dataset.roles = dialog.result_roles; dataset.units = dialog.result_units
            dependent = sum(grid.source_dataset_id == object_id for grid in self.project.grids.values())
            if old_roles != dataset.roles and dependent:
                QMessageBox.information(self, "字段映射已更新", f"字段映射已用于后续绘图和计算。已有 {dependent} 个派生网格不会自动重算。")
        elif kind == "grid" and object_id in self.project.grids:
            target = self.project.grids[object_id]
            name, ok = QInputDialog.getText(self, "重命名派生网格", "工程内名称", text=target.name)
            if not ok or not name.strip(): return
            target.name = name.strip()
        elif kind == "terrain" and object_id in self.project.terrains:
            target = self.project.terrains[object_id]
            name, ok = QInputDialog.getText(self, "重命名地形数据", "工程内名称", text=target.name)
            if not ok or not name.strip(): return
            target.name = name.strip()
        elif kind == "canvas_object" and isinstance(self.page, PlotPage):
            target = self.page.annotations._object(object_id)
            if target is None: return
            if target["type"] == "text":
                text, ok = QInputDialog.getMultiLineText(self, "编辑页面文字", "内容", target.get("text", ""))
                if not ok: return
                target["text"] = text
            else:
                width, ok = QInputDialog.getDouble(self, "页面对象属性", "线宽", target.get("width", 1.5), .1, 20., 1)
                if not ok: return
                target["width"] = width
            self.page.annotations.render(); self.page.annotations._changed()
        else:
            return
        self.project.dirty = True; self._refresh_project_view()
        if self.page.last_dataset is not None and self.page.last_dataset.id == object_id:
            self.page.redraw()

    def _remove_selected_object(self) -> None:
        item = self.project_list.currentItem()
        if item is None:
            return
        kind, object_id = item.data(0, Qt.UserRole + 1), item.data(0, Qt.UserRole)
        if kind not in {"dataset", "grid", "terrain", "canvas_object"}:
            QMessageBox.information(self, "请选择对象", "请选择一个数据集、派生网格或地形数据。")
            return
        if kind == "canvas_object":
            self.page.annotations.selected_id = object_id; self.page.annotations.delete_selected(); return
        if kind == "dataset":
            dataset = self.project.datasets[object_id]
            dependent_grids = [grid.id for grid in self.project.grids.values() if grid.source_dataset_id == object_id]
            box = QMessageBox(self)
            box.setWindowTitle("从工程移除数据")
            box.setIcon(QMessageBox.Warning)
            box.setText(f"将从工程移除“{dataset.name}”。磁盘上的原始源文件永远不会被删除。")
            box.setInformativeText(f"发现 {len(dependent_grids)} 个派生网格。请选择保留孤立结果，或级联移除依赖结果与关联页面。")
            keep = box.addButton("仅移除数据引用", QMessageBox.ActionRole)
            cascade = box.addButton("级联移除", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() not in (keep, cascade):
                return
            del self.project.datasets[object_id]
            if box.clickedButton() is cascade:
                for grid_id in dependent_grids:
                    self.project.grids.pop(grid_id, None)
                self.project.pages = [page for page in self.project.pages if page.get("dataset_id") != object_id and page.get("grid_id") not in dependent_grids]
            if self.active_dataset_id == object_id:
                self.active_dataset_id = next(iter(self.project.datasets), None)
        elif kind == "grid":
            grid = self.project.grids[object_id]
            if QMessageBox.question(self, "移除派生网格", f"从工程移除派生结果“{grid.name}”？原始数据不受影响。") != QMessageBox.Yes:
                return
            del self.project.grids[object_id]
            self.project.pages = [page for page in self.project.pages if page.get("grid_id") != object_id]
        else:
            terrain = self.project.terrains[object_id]
            if QMessageBox.question(self, "移除地形数据", f"从工程移除“{terrain.name}”？磁盘源文件不会删除。") != QMessageBox.Yes:
                return
            del self.project.terrains[object_id]
        self.project.dirty = True
        self._refresh_project_view()
        self.statusBar().showMessage("对象已从工程移除；源文件未删除。", 7000)

    def _refresh_project_view(self) -> None:
        selected_id = self.active_dataset_id
        self.project_list.blockSignals(True)
        self.project_list.clear()
        data_root = QTreeWidgetItem(["数据集", f"{len(self.project.datasets)}"])
        grid_root = QTreeWidgetItem(["派生网格", f"{len(self.project.grids)}"])
        terrain_root = QTreeWidgetItem(["地形", f"{len(self.project.terrains)}"])
        canvas_objects = self.page.annotations.objects if isinstance(self.tabs.currentWidget(), PlotPage) else []
        object_root = QTreeWidgetItem(["页面对象", f"{len(canvas_objects)}"])
        self.project_list.addTopLevelItems([data_root, grid_root, terrain_root, object_root])
        target_item = None
        for dataset in self.project.datasets.values():
            prefix = "🔒" if dataset.parent_id is None else "◇"
            item = QTreeWidgetItem([f"{prefix} {dataset.name}", f"{dataset.row_count:,}行"])
            item.setData(0, Qt.UserRole, dataset.id); item.setData(0, Qt.UserRole + 1, "dataset")
            data_root.addChild(item)
            if dataset.id == selected_id:
                target_item = item
        for grid in self.project.grids.values():
            orphan = grid.source_dataset_id not in self.project.datasets
            label = f"⚠ {grid.name}" if orphan else grid.name
            shape_text = "×".join(str(value) for value in reversed(grid.spec.shape))
            item = QTreeWidgetItem([label, f"{grid.spec.dimensions}D｜{shape_text}｜{grid.method}"])
            item.setData(0, Qt.UserRole, grid.id); item.setData(0, Qt.UserRole + 1, "grid")
            if orphan:
                item.setToolTip(0, "源数据引用已移除；该派生网格仍可保存和导出，但不能重新计算。")
            grid_root.addChild(item)
        for terrain in self.project.terrains.values():
            item = QTreeWidgetItem([f"▲ {terrain.name}", f"{terrain.point_count:,}点"])
            item.setData(0, Qt.UserRole, terrain.id); item.setData(0, Qt.UserRole + 1, "terrain")
            terrain_root.addChild(item)
        type_names = {"text": "文字", "arrow": "箭头", "line": "直线", "rectangle": "矩形", "ellipse": "椭圆"}
        for number, canvas_object in enumerate(canvas_objects, 1):
            summary = canvas_object.get("text", "").replace("\n", " ")[:18] if canvas_object.get("type") == "text" else f"对象 {number}"
            item = QTreeWidgetItem([f"{type_names.get(canvas_object.get('type'), '对象')}｜{summary}", "页面坐标"])
            item.setData(0, Qt.UserRole, canvas_object.get("id")); item.setData(0, Qt.UserRole + 1, "canvas_object")
            object_root.addChild(item)
        self.project_list.expandAll()
        self.project_list.blockSignals(False)
        if target_item is None and data_root.childCount():
            target_item = data_root.child(0)
        if target_item is not None:
            self.project_list.setCurrentItem(target_item)
            self._selection_changed(target_item)
        else:
            self.active_data_label.setText("当前数据：未选择")
        suffix = " *" if self.project.dirty else ""
        self.setWindowTitle(f"重磁电震绘图 Pro 试验版 v{__version__} — {self.project.name}{suffix}")

    def new_project(self) -> None:
        if self.project.dirty and QMessageBox.question(self, "未保存修改", "放弃当前未保存修改并新建工程？") != QMessageBox.Yes:
            return
        self.project = Project()
        self._apply_default_standards_profile()
        self.active_dataset_id = None
        self.tabs.clear(); self.tabs.addTab(self._new_plot_page(), "绘图页 1")
        self._refresh_project_view()

    def import_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入数据", "", "文本数据 (*.csv *.txt *.dat);;所有文件 (*)")
        if not path:
            return
        try:
            dialog = ImportDialog(path, self)
        except Exception as exc:
            QMessageBox.critical(self, "文件预览失败", str(exc)); return
        if dialog.exec() and dialog.result_dataset:
            self.project.add_dataset(dialog.result_dataset)
            self.active_dataset_id = dialog.result_dataset.id
            self._refresh_project_view()
            self.plot_points()

    def plot_points(self) -> None:
        try:
            self.page.plot_points(self._selected_dataset())
        except Exception as exc:
            QMessageBox.warning(self, "无法绘图", str(exc))

    def grid_2d(self) -> None:
        try:
            if self._workers:
                raise ValueError("已有计算任务正在运行，请等待其完成。")
            dataset = self._selected_dataset()
            recommendation = recommend_grid(dataset, 2)
            dialog = InterpolationDialog(recommendation, 2, self)
            if not dialog.exec() or dialog.result_spec is None:
                return
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("正在计算二维网格…")
            target_page = self.page
            task = FunctionTask(
                lambda: interpolate(dataset, dialog.result_spec, dialog.result_method, dialog.result_neighbors)
            )
            task.signals.finished.connect(
                lambda grid, worker=task: self._grid_finished(dataset, target_page, grid, worker)
            )
            task.signals.failed.connect(lambda message, worker=task: self._grid_failed(message, worker))
            self._workers.append(task)
            self.thread_pool.start(task)
        except Exception as exc:
            QMessageBox.critical(self, "插值失败", str(exc))
            QApplication.restoreOverrideCursor()

    def _grid_finished(self, dataset: Dataset, target_page: PlotPage, grid: GridResult, worker: FunctionTask) -> None:
        QApplication.restoreOverrideCursor()
        self.project.add_grid(grid)
        target_page.plot_contour(dataset, grid)
        self.statusBar().showMessage(f"完成：{grid.spec.cell_count:,} 个网格单元", 8000)
        self._refresh_project_view()
        if worker in self._workers:
            self._workers.remove(worker)

    def _grid_failed(self, message: str, worker: FunctionTask) -> None:
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(self, "插值失败", message)
        self.statusBar().showMessage("插值失败", 8000)
        if worker in self._workers:
            self._workers.remove(worker)

    def grid_3d(self) -> None:
        try:
            if self._workers:
                raise ValueError("已有计算任务正在运行，请等待其完成。")
            dataset = self._selected_dataset()
            if "z" not in dataset.roles:
                raise ValueError("三维网格化要求导入时映射 X、Y、Z 和 Value 字段")
            recommendation = recommend_grid(dataset, 3)
            dialog = InterpolationDialog(recommendation, 3, self)
            if not dialog.exec() or dialog.result_spec is None:
                return
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage("正在后台计算三维规则体…")
            task = FunctionTask(
                lambda: interpolate(dataset, dialog.result_spec, dialog.result_method, dialog.result_neighbors)
            )
            task.signals.finished.connect(
                lambda grid, worker=task: self._grid3d_finished(dataset, grid, worker)
            )
            task.signals.failed.connect(lambda message, worker=task: self._grid_failed(message, worker))
            self._workers.append(task)
            self.thread_pool.start(task)
        except Exception as exc:
            QMessageBox.critical(self, "三维网格化失败", str(exc))
            QApplication.restoreOverrideCursor()

    def _grid3d_finished(self, dataset: Dataset, grid: GridResult, worker: FunctionTask) -> None:
        QApplication.restoreOverrideCursor()
        self.project.add_grid(grid)
        self.statusBar().showMessage(
            f"三维网格化完成：生成 {grid.spec.cell_count:,} 个体素，尚未绘图。请点击“3D体渲染”或“XYZ剖面”。", 15000
        )
        self._refresh_project_view()
        QMessageBox.information(
            self, "三维网格化完成",
            f"已生成规则体“{grid.name}”\n形状：{grid.spec.shape}\n体素：{grid.spec.cell_count:,}\n\n"
            "“3D网格化计算”只生成可复用的数值规则体；“3D体渲染”才把选定规则体显示成三维图。",
        )
        if worker in self._workers:
            self._workers.remove(worker)

    def volume_3d(self) -> None:
        try:
            dataset = self._selected_dataset()
            candidates = [
                grid for grid in self.project.grids.values()
                if grid.spec.dimensions == 3 and grid.source_dataset_id == dataset.id
            ]
            if not candidates:
                raise ValueError("当前数据集还没有三维规则体，请先执行“三维网格化”")
            grid = candidates[-1]
            if len(candidates) > 1:
                labels = [f"{item.name}｜{'×'.join(str(v) for v in reversed(item.spec.shape))}｜{item.method}" for item in candidates]
                selected, ok = QInputDialog.getItem(self, "选择要渲染的三维规则体", "已有规则体", labels, len(labels) - 1, False)
                if not ok:
                    return
                grid = candidates[labels.index(selected)]
            dialog = VolumeDisplayDialog(grid, self)
            if not dialog.exec():
                return
            if accelerated_available():
                window = AcceleratedVolumeWindow(dataset, grid, self.page.threshold, self.page.spatial_filter, dialog.settings, self)
                window.show(); self._auxiliary_windows.append(window)
                self.statusBar().showMessage("已使用VTK/PyVista GPU三维渲染器。", 8000)
                return
            self.statusBar().showMessage("VTK/PyVista不可用：已自动使用抽稀CPU兼容预览。", 10000)
            source_page = self.page
            page = self._new_plot_page()
            page.cmap = source_page.cmap; page.style = dict(source_page.style)
            page.threshold = source_page.threshold; page.spatial_filter = source_page.spatial_filter
            page.threshold = source_page.threshold
            page.spatial_filter = source_page.spatial_filter
            self.tabs.addTab(page, "XYZV三维数据体")
            self.tabs.setCurrentWidget(page)
            page.plot_volume(dataset, grid, dialog.settings)
            self.statusBar().showMessage("未检测到VTK/PyVista，已使用内置CPU兼容渲染器。", 8000)
        except Exception as exc:
            QMessageBox.warning(self, "无法绘制三维数据体", str(exc))

    def volume_slice(self) -> None:
        try:
            dataset = self._selected_dataset()
            candidates = [
                grid for grid in self.project.grids.values()
                if grid.spec.dimensions == 3 and grid.source_dataset_id == dataset.id
            ]
            if not candidates:
                raise ValueError("当前数据集还没有三维规则体，请先执行三维网格化")
            grid = candidates[-1]
            if len(candidates) > 1:
                names = [item.name for item in candidates]
                selected, ok = QInputDialog.getItem(self, "选择三维规则体", "规则体", names, len(names) - 1, False)
                if not ok:
                    return
                grid = candidates[names.index(selected)]
            dialog = VolumeSliceDialog(grid, self)
            if not dialog.exec():
                return
            result = dialog.build_slice()
            source_page = self.page
            page = self._new_plot_page()
            page.cmap = source_page.cmap; page.style = dict(source_page.style)
            self.tabs.addTab(page, result.name)
            self.tabs.setCurrentWidget(page)
            page.plot_volume_slice(dataset, grid, result)
            self.project.pages.append({
                "type": "volume_slice", "grid_id": grid.id,
                "dataset_id": dataset.id, "definition": result.metadata,
            })
            self.project.dirty = True
            self.statusBar().showMessage(
                f"已生成三维剖面；有效覆盖 {result.valid_fraction:.1%}", 10000
            )
        except Exception as exc:
            QMessageBox.warning(self, "无法生成三维剖面", str(exc))

    def surface_3d(self) -> None:
        try:
            dataset = self._selected_dataset()
            source_page = self.page
            grid = source_page.last_grid
            if grid is None:
                raise ValueError("请先生成二维插值网格")
            if accelerated_available():
                window = AcceleratedSurfaceWindow(dataset, grid, source_page.threshold, source_page.spatial_filter, source_page.title, self)
                window.show(); self._auxiliary_windows.append(window)
                self.statusBar().showMessage("已使用VTK/PyVista GPU三维曲面渲染器。", 8000)
                return
            self.statusBar().showMessage("VTK/PyVista不可用：已自动使用低多边形CPU兼容曲面。", 10000)
            new_page = self._new_plot_page()
            new_page.cmap = source_page.cmap
            new_page.title = source_page.title
            new_page.xlabel = source_page.xlabel
            new_page.ylabel = source_page.ylabel
            new_page.style = dict(source_page.style)
            new_page.threshold = source_page.threshold
            new_page.spatial_filter = source_page.spatial_filter
            self.tabs.addTab(new_page, "三维曲面")
            self.tabs.setCurrentWidget(new_page)
            new_page.plot_surface(dataset, grid)
        except Exception as exc:
            QMessageBox.warning(self, "无法绘制三维曲面", str(exc))

    def open_terrain_window(self) -> None:
        window = TerrainProfileWindow(self)
        window.terrain_ready.connect(self._add_terrain)
        window.destroyed.connect(lambda: self._forget_terrain_window(window))
        self._terrain_windows.append(window)
        window.show()
        window.raise_()

    def open_statistics_window(self) -> None:
        try:
            window = StatisticsWindow(self._selected_dataset(), self)
            self._auxiliary_windows.append(window)
            window.destroyed.connect(lambda: self._forget_auxiliary_window(window))
            window.show()
            window.raise_()
        except Exception as exc:
            QMessageBox.warning(self, "无法打开统计图窗口", str(exc))

    def open_geology_window(self) -> None:
        try:
            dataset = self._selected_dataset()
            grid = self.page.last_grid
            terrain = self.page.last_terrain
            existing = next((
                page for page in reversed(self.project.pages)
                if page.get("type") == "geology_section" and page.get("dataset_id") == dataset.id
            ), {})
            window = GeologySectionWindow(
                dataset, grid, terrain, self.page.threshold, self,
                existing.get("layers", []), existing.get("settings", {}),
                spatial_filter=self.page.spatial_filter,
            )
            window.layers_changed.connect(lambda layers: self._store_geology_layers(dataset.id, layers))
            window.settings_changed.connect(lambda settings: self._store_geology_settings(dataset.id, settings))
            self._auxiliary_windows.append(window)
            window.destroyed.connect(lambda: self._forget_auxiliary_window(window))
            window.show()
            window.raise_()
        except Exception as exc:
            QMessageBox.warning(self, "无法打开地质剖面编辑器", str(exc))

    def _store_geology_layers(self, dataset_id: str, layers: list) -> None:
        existing = next((
            page for page in reversed(self.project.pages)
            if page.get("type") == "geology_section" and page.get("dataset_id") == dataset_id
        ), {})
        records = [page for page in self.project.pages if page.get("type") != "geology_section" or page.get("dataset_id") != dataset_id]
        records.append({
            "type": "geology_section", "dataset_id": dataset_id,
            "layers": layers, "settings": existing.get("settings", {}),
        })
        self.project.pages = records
        self.project.dirty = True

    def _store_geology_settings(self, dataset_id: str, settings: dict) -> None:
        existing = next((
            page for page in reversed(self.project.pages)
            if page.get("type") == "geology_section" and page.get("dataset_id") == dataset_id
        ), {})
        records = [page for page in self.project.pages if page.get("type") != "geology_section" or page.get("dataset_id") != dataset_id]
        records.append({
            "type": "geology_section", "dataset_id": dataset_id,
            "layers": existing.get("layers", []), "settings": settings,
        })
        self.project.pages = records
        self.project.dirty = True

    def _forget_auxiliary_window(self, window: QWidget) -> None:
        if window in self._auxiliary_windows:
            self._auxiliary_windows.remove(window)

    def _forget_terrain_window(self, window: TerrainProfileWindow) -> None:
        if window in self._terrain_windows:
            self._terrain_windows.remove(window)

    def _add_terrain(self, terrain: TerrainProfile) -> None:
        existing = next((item for item in self.project.terrains.values() if item.source_hash == terrain.source_hash), None)
        if existing:
            self.statusBar().showMessage("该地形文件已存在于工程中。", 8000)
            return
        self.project.add_terrain(terrain)
        self._refresh_project_view()
        self.statusBar().showMessage(f"已添加地形：{terrain.name}", 8000)

    def terrain_section(self) -> None:
        try:
            dataset = self._selected_dataset()
            grid = self.page.last_grid
            if grid is None:
                raise ValueError("请先在当前页面生成二维插值网格")
            if not self.project.terrains:
                raise ValueError("工程中没有地形数据，请先打开“地形数据窗口”导入位置—高程文件")
            terrains = list(self.project.terrains.values())
            terrain = terrains[0]
            if len(terrains) > 1:
                names = [item.name for item in terrains]
                selected, ok = QInputDialog.getItem(self, "选择地形", "地形剖面", names, 0, False)
                if not ok:
                    return
                terrain = terrains[names.index(selected)]
            x_axis, _ = grid.spec.axes()
            if terrain.position.max() < x_axis.min() or terrain.position.min() > x_axis.max():
                raise ValueError("地形位置范围与断面X范围没有重叠，请检查单位和起点")
            source_page = self.page
            page = self._new_plot_page()
            page.cmap = source_page.cmap
            page.title = source_page.title
            page.ylabel = source_page.ylabel
            page.threshold = source_page.threshold
            page.spatial_filter = source_page.spatial_filter
            page.style = dict(source_page.style)
            self.tabs.addTab(page, "带地形断面")
            self.tabs.setCurrentWidget(page)
            page.plot_terrain_section(dataset, grid, terrain)
        except Exception as exc:
            QMessageBox.warning(self, "无法绘制带地形断面", str(exc))

    def _apply_threshold(self, _checked: bool = False, silent: bool = False) -> None:
        try:
            mode = ThresholdMode(self.threshold_mode.currentData())
            lower_needed = mode in (ThresholdMode.KEEP_RANGE, ThresholdMode.HIDE_RANGE, ThresholdMode.BELOW)
            upper_needed = mode in (ThresholdMode.KEEP_RANGE, ThresholdMode.HIDE_RANGE, ThresholdMode.ABOVE)
            if lower_needed and not self.lower.text().strip():
                raise ValueError("当前模式需要设置下限")
            if upper_needed and not self.upper.text().strip():
                raise ValueError("当前模式需要设置上限")
            lower = float(self.lower.text()) if lower_needed else None
            upper = float(self.upper.text()) if upper_needed else None
            if mode in (ThresholdMode.KEEP_RANGE, ThresholdMode.HIDE_RANGE) and lower > upper:
                raise ValueError("下限不能大于上限")
            spatial_ranges: dict[str, tuple[float, float] | None] = {}
            for axis_name in ("x", "y", "z"):
                if not self.spatial_checks[axis_name].isChecked():
                    spatial_ranges[axis_name] = None; continue
                minimum_text, maximum_text = self.spatial_min[axis_name].text().strip(), self.spatial_max[axis_name].text().strip()
                if not minimum_text or not maximum_text:
                    raise ValueError(f"启用{axis_name.upper()}空间裁剪后必须同时填写最小值和最大值")
                minimum, maximum = float(minimum_text), float(maximum_text)
                if minimum > maximum:
                    raise ValueError(f"{axis_name.upper()}最小值不能大于最大值")
                spatial_ranges[axis_name] = (minimum, maximum)
            self.page.threshold = Threshold(mode, lower, upper)
            self.page.spatial_filter = SpatialFilter(**spatial_ranges)
            self.page.redraw()
            self._update_threshold_counts()
            self.project.dirty = True
            self.statusBar().showMessage("已按XYZ空间范围→Value阈值应用显示掩膜；原始数据与插值结果未被修改。", 8000)
        except Exception as exc:
            self.threshold_result_label.setText(f"阈值错误：{exc}")
            if not silent:
                QMessageBox.warning(self, "阈值无效", str(exc))

    def _update_threshold_counts(self) -> None:
        page = self.page
        if page.last_volume_grid is not None:
            spatial, _value, visible = page.grid_masks(page.last_volume_grid)
            layer = "三维体网格"
        elif page.last_grid is not None:
            spatial, _value, visible = page.grid_masks(page.last_grid)
            layer = "二维插值网格"
        elif page.last_dataset is not None:
            spatial, _value, visible = page.dataset_masks(page.last_dataset)
            layer = "原始点"
        else:
            self.threshold_result_label.setText("可见/隐藏：当前页面无数据")
            return
        visible_count = int(np.count_nonzero(visible))
        spatial_count = int(np.count_nonzero(spatial))
        total_count = int(np.size(spatial))
        self.threshold_result_label.setText(
            f"{layer}：总计 {total_count:,} → XYZ {spatial_count:,} → Value {visible_count:,} 可见"
        )
        active_axes = [name.upper() for name in ("x", "y", "z") if getattr(page.spatial_filter, name) is not None]
        self.spatial_range_label.setText("空间范围：" + ("、".join(active_axes) + " 已启用；先XYZ后Value" if active_axes else "未启用（先XYZ，后Value）"))

    def _recommend_anomaly_threshold(self) -> None:
        try:
            dataset = self._selected_dataset()
            recommendation = recommend_anomaly_thresholds(dataset.role_values("value"), self.physical_method.currentData())
            if self.anomaly_side.currentData() == "high":
                self.threshold_mode.setCurrentIndex(self.threshold_mode.findData(ThresholdMode.BELOW))
                self.lower.setText(f"{recommendation.high:.12g}")
                self.upper.clear()
                threshold_text = f"高值 ≥ {recommendation.high:.6g}"
            else:
                self.threshold_mode.setCurrentIndex(self.threshold_mode.findData(ThresholdMode.ABOVE))
                self.upper.setText(f"{recommendation.low:.12g}")
                self.lower.clear()
                threshold_text = f"低值 ≤ {recommendation.low:.6g}"
            self._set_slider_from_text()
            self.statusBar().showMessage(f"已推荐 {threshold_text}；{recommendation.transform}。点击“应用显示阈值”后生效。", 15000)
            QMessageBox.information(self, "物探异常阈值建议", f"{threshold_text}\n\n方法：{recommendation.transform}\n{recommendation.explanation}\n\n这是显示候选，不会修改或删除数据。")
        except Exception as exc:
            QMessageBox.warning(self, "无法推荐异常阈值", str(exc))

    def _edit_plot_properties(self) -> None:
        page = self.page
        dialog = PlotPropertiesDialog(page.title, page.xlabel, page.ylabel, page.cmap, page.style, self)
        if dialog.exec():
            page.title = dialog.title_edit.text()
            page.xlabel = dialog.xlabel_edit.text()
            page.ylabel = dialog.ylabel_edit.text()
            page.cmap = dialog.cmap.currentText()
            page.style.update(dialog.style)
            page.redraw()
            self.project.dirty = True

    def _serialize_canvas_pages(self) -> None:
        records = [record for record in self.project.pages if record.get("type") != "canvas_page"]
        for index in range(self.tabs.count()):
            page = self.tabs.widget(index)
            if not isinstance(page, PlotPage): continue
            grid = page.last_volume_grid or page.last_grid
            records.append({
                "type": "canvas_page", "name": self.tabs.tabText(index), "view_kind": page.view_kind,
                "dataset_id": page.last_dataset.id if page.last_dataset is not None else None,
                "grid_id": grid.id if grid is not None else None,
                "terrain_id": page.last_terrain.id if page.last_terrain is not None else None,
                "title": page.title, "xlabel": page.xlabel, "ylabel": page.ylabel, "cmap": page.cmap,
                "style": dict(page.style),
                "threshold": {"mode": page.threshold.mode.value, "lower": page.threshold.lower, "upper": page.threshold.upper},
                "spatial_filter": {name: getattr(page.spatial_filter, name) for name in ("x", "y", "z")},
                "annotations": deepcopy(page.annotations.objects), "volume_render": dict(page.last_volume_render),
            })
        self.project.pages = records

    def _restore_canvas_pages(self) -> None:
        records = [record for record in self.project.pages if record.get("type") == "canvas_page"]
        while self.tabs.count():
            widget = self.tabs.widget(0); self.tabs.removeTab(0); widget.deleteLater()
        if not records:
            self.tabs.addTab(self._new_plot_page(), "绘图页 1"); return
        for record in records:
            page = self._new_plot_page(); page.title = record.get("title", ""); page.xlabel = record.get("xlabel", "X"); page.ylabel = record.get("ylabel", "Y")
            page.cmap = record.get("cmap", "viridis"); page.style.update(record.get("style", {}))
            threshold = record.get("threshold", {}); page.threshold = Threshold(ThresholdMode(threshold.get("mode", "keep_all")), threshold.get("lower"), threshold.get("upper"))
            spatial = record.get("spatial_filter", {}); page.spatial_filter = SpatialFilter(**{name: tuple(spatial[name]) if spatial.get(name) is not None else None for name in ("x", "y", "z")})
            page.annotations.objects = deepcopy(record.get("annotations", [])); page.last_volume_render.update(record.get("volume_render", {}))
            dataset = self.project.datasets.get(record.get("dataset_id")); grid = self.project.grids.get(record.get("grid_id")); terrain = self.project.terrains.get(record.get("terrain_id"))
            try:
                kind = record.get("view_kind")
                if dataset is not None and kind == "points": page.plot_points(dataset)
                elif dataset is not None and grid is not None and kind == "contour": page.plot_contour(dataset, grid)
                elif dataset is not None and grid is not None and kind == "surface": page.plot_surface(dataset, grid)
                elif dataset is not None and grid is not None and kind == "volume": page.plot_volume(dataset, grid, page.last_volume_render)
                elif dataset is not None and grid is not None and terrain is not None and kind == "terrain_section": page.plot_terrain_section(dataset, grid, terrain)
                else:
                    page.figure.clear(); axis = page.figure.add_subplot(111); axis.text(.5, .5, "页面数据引用不可恢复，请重新选择数据绘图", ha="center", va="center", transform=axis.transAxes); page._draw_complete()
            except Exception as exc:
                page.figure.clear(); axis = page.figure.add_subplot(111); axis.text(.5, .5, f"页面恢复失败：{exc}", ha="center", va="center", transform=axis.transAxes); page._draw_complete()
            self.tabs.addTab(page, record.get("name", "绘图页"))

    def generate_quality_report(self) -> None:
        try:
            dataset = self._selected_dataset()
            grid = self.page.last_grid
            if grid is None:
                raise ValueError("请先完成二维插值")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            metrics = idw_cross_validation(dataset, 2)
            report = build_quality_report(dataset, grid, metrics)
            self.project.reports[grid.id] = report
            self.project.dirty = True
            text = json.dumps(report, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "插值质量报告", text[:6000])
            self._refresh_project_view()
        except Exception as exc:
            QMessageBox.critical(self, "报告生成失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def save_project(self, save_as: bool = False) -> None:
        path = self.project.project_path
        if save_as or path is None:
            selected, _ = QFileDialog.getSaveFileName(self, "保存GMES工程", "", "GMES工程 (*.gpproj)")
            if not selected:
                return
            path = Path(selected)
        try:
            self._serialize_canvas_pages()
            save_project(self.project, path)
            self._refresh_project_view()
            self.statusBar().showMessage(f"工程已保存：{path}", 8000)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def open_project(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "打开GMES工程", "", "GMES工程 (*.gpproj)")
        if not selected:
            return
        try:
            self.project = load_project(selected)
            self._apply_default_standards_profile()
            self._restore_canvas_pages()
            self._refresh_project_view()
            self.statusBar().showMessage(f"已打开：{selected}", 8000)
        except Exception as exc:
            QMessageBox.critical(self, "打开失败", str(exc))

    def show_constraint_reservation(self) -> None:
        QMessageBox.information(
            self, "钻孔 / 地质约束接口预留",
            "V1.1工程结构已预留 constraints.boreholes 与 constraints.geology。\n\n"
            "后续导入字段将包括：钻孔编号、孔口X/Y/Z、测深、层顶/层底、岩性、年代、倾角、来源文件与坐标系；"
            "已有地质界线将支持CSV、GIS和DXF/SVG交换。当前版本不会伪造导入能力。",
        )

    def show_standards_info(self) -> None:
        standards = "\n\n".join(
            f"{item['title']}\n状态：{item['status']}\n用途：{item['scope']}\n核验：{item['verified_on']}"
            for item in STANDARD_PROFILES
        )
        QMessageBox.information(
            self, "制图标准与版本",
            f"GMES样式库：{STYLE_LIBRARY_VERSION}\n地层年代：{STRATIGRAPHY_PROFILE['title']}\n\n{standards}\n\n"
            "说明：软件按这些标准组织图例和元数据，但用户自定义样式、比例尺及输出用途仍需在成果审查时复核；本软件不宣称经过标准符合性认证。",
        )

    def _current_provenance(self) -> dict:
        page = self.page
        grid = page.last_volume_grid or page.last_grid
        return layer_provenance(
            page.last_dataset, grid, threshold=page.threshold, spatial_filter=page.spatial_filter,
            cmap=page.cmap, style=page.style, view_kind=page.view_kind,
            physical_method=self.physical_method.currentData() or "generic",
        )

    def show_current_provenance(self) -> None:
        if self.page.last_dataset is None:
            QMessageBox.information(self, "科研溯源", "当前绘图页没有可追踪的数据图层。")
            return
        record = self._current_provenance()
        self.project.reports[f"lineage:{self.page.last_dataset.id}:{self.page.view_kind}"] = record
        self.project.dirty = True
        QMessageBox.information(self, "图—数据—参数溯源", provenance_markdown(record)[:7000])

    def show_publication_checks(self) -> list[dict[str, str]]:
        if self.page.last_dataset is None:
            findings = [{"level": "high", "message": "当前页面没有来源数据。"}]
        else:
            findings = publication_checks(self.page.figure, self.project, self._current_provenance())
        labels = {"pass": "通过", "info": "提示", "warning": "警告", "high": "高风险"}
        QMessageBox.information(
            self, "科研排版检查",
            "\n".join(f"[{labels.get(item['level'], item['level'])}] {item['message']}" for item in findings),
        )
        return findings

    def export_figure(self, force_report: bool = False) -> None:
        selected, selected_filter = QFileDialog.getSaveFileName(self, "导出图片", "figure.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif)")
        if not selected:
            return
        try:
            record = self._current_provenance() if self.page.last_dataset is not None else None
            if record is not None:
                findings = publication_checks(self.page.figure, self.project, record)
                high = [item for item in findings if item["level"] == "high"]
                if high and QMessageBox.question(
                    self, "导出前发现高风险", "\n".join(item["message"] for item in high) + "\n\n仍要继续导出吗？"
                ) != QMessageBox.Yes:
                    return
            suffix = Path(selected).suffix.lower()
            dpi = 600 if suffix in {".png", ".tif", ".tiff"} else 300
            self.page.figure.savefig(selected, dpi=dpi, bbox_inches="tight", transparent=False)
            report_path = None
            if record is not None and (force_report or QMessageBox.question(
                self, "科研复现报告", "是否同时生成同名的科研复现报告？"
            ) == QMessageBox.Yes):
                report_path = write_companion_report(selected, record)
            message = f"已导出：{selected}" + (f"；复现报告：{report_path.name}" if report_path else "")
            self.statusBar().showMessage(message, 10000)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def closeEvent(self, event) -> None:
        if self.project.dirty:
            answer = QMessageBox.question(self, "未保存修改", "工程尚未保存，仍要退出吗？", QMessageBox.Yes | QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore(); return
        event.accept()

