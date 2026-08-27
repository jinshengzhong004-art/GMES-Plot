from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QInputDialog, QLabel, QMainWindow, QVBoxLayout, QWidget

from gmes_plot.domain.models import Dataset, GridResult, SpatialFilter, Threshold


def accelerated_available() -> bool:
    try:
        import pyvista  # noqa: F401
        import pyvistaqt  # noqa: F401
    except ImportError:
        return False
    return True


class AcceleratedSurfaceWindow(QMainWindow):
    """VTK structured-surface viewer; keeps camera interaction off the CPU plot path."""

    def __init__(self, dataset: Dataset, grid: GridResult, threshold: Threshold, spatial_filter: SpatialFilter, title: str, parent=None) -> None:
        super().__init__(parent)
        import pyvista as pv
        from pyvistaqt import QtInteractor

        if grid.spec.dimensions != 2:
            raise ValueError("GPU曲面窗口需要二维XYV网格")
        self.scene_title = title or f"{grid.name} — 三维曲面"
        self.setWindowTitle(f"GPU三维曲面 — {grid.name}"); self.resize(1200, 850)
        container = QWidget(); layout = QVBoxLayout(container)
        self.plotter = QtInteractor(container); layout.addWidget(self.plotter.interactor, 1)
        layout.addWidget(QLabel("VTK/PyVista交互：左键旋转、中键平移、滚轮缩放；双击画布修改图名；色标可直接拖动缩放。"))
        self.setCentralWidget(container); self.plotter.interactor.installEventFilter(self)

        x_axis, y_axis = grid.spec.axes()
        xi = np.flatnonzero(spatial_filter.axis_mask("x", x_axis)); yi = np.flatnonzero(spatial_filter.axis_mask("y", y_axis))
        if not xi.size or not yi.size: raise ValueError("XY空间裁剪后没有可见曲面")
        xs, ys = slice(xi[0], xi[-1] + 1), slice(yi[0], yi[-1] + 1)
        values = np.asarray(grid.values[ys, xs], dtype=float).copy(); values[~threshold.mask(values)] = np.nan
        if not np.any(np.isfinite(values)): raise ValueError("Value阈值后没有可见曲面")
        xx, yy = np.meshgrid(x_axis[xs], y_axis[ys]); mesh = pv.StructuredGrid(xx, yy, values)
        mesh["Value"] = values.ravel(order="F")
        self.plotter.add_mesh(mesh, scalars="Value", cmap="viridis", smooth_shading=False, show_scalar_bar=False)
        self.scalar_bar = self.plotter.add_scalar_bar("Value", interactive=True, position_x=.86, position_y=.18, width=.07, height=.62, fmt="%.5g", outline=True)
        self.plotter.add_text(self.scene_title, position="upper_edge", font_size=14, name="scene-title")
        self.plotter.show_axes(); self.plotter.view_isometric(); self.plotter.reset_camera()
        interactor = getattr(getattr(self.plotter, "iren", None), "interactor", None)
        if interactor is not None: interactor.SetDesiredUpdateRate(30.0); interactor.SetStillUpdateRate(0.5)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.MouseButtonDblClick:
            title, ok = QInputDialog.getText(self, "修改三维图名", "图名", text=self.scene_title)
            if ok:
                self.scene_title = title.strip(); self.plotter.add_text(self.scene_title, position="upper_edge", font_size=14, name="scene-title"); self.plotter.render()
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:
        self.plotter.close(); super().closeEvent(event)


class AcceleratedVolumeWindow(QMainWindow):
    """Optional VTK/PyVista GPU volume renderer with a safe import boundary."""

    def __init__(self, dataset: Dataset, grid: GridResult, threshold: Threshold, spatial_filter: SpatialFilter, settings: dict, parent=None) -> None:
        super().__init__(parent)
        import pyvista as pv
        from pyvistaqt import QtInteractor

        self.setWindowTitle(f"GPU三维数据体 — {grid.name}")
        self.resize(1200, 850)
        container = QWidget(); layout = QVBoxLayout(container)
        self.plotter = QtInteractor(container); layout.addWidget(self.plotter.interactor, 1)
        layout.addWidget(QLabel("VTK/PyVista GPU交互：左键旋转，中键平移，滚轮缩放；深度Z向下为正；阈值仅作用于显示副本。"))
        self.setCentralWidget(container)

        self.scene_title = f"{grid.name} — XYZV三维数据体"
        self.plotter.interactor.installEventFilter(self)
        x_axis, y_axis, z_axis = grid.spec.axes()
        x_indices = np.flatnonzero(spatial_filter.axis_mask("x", x_axis))
        y_indices = np.flatnonzero(spatial_filter.axis_mask("y", y_axis))
        z_indices = np.flatnonzero(spatial_filter.axis_mask("z", z_axis))
        if not x_indices.size or not y_indices.size or not z_indices.size:
            raise ValueError("XYZ空间裁剪后没有可见体素")
        xs, ys, zs = slice(x_indices[0], x_indices[-1] + 1), slice(y_indices[0], y_indices[-1] + 1), slice(z_indices[0], z_indices[-1] + 1)
        x_axis, y_axis, z_axis = x_axis[xs], y_axis[ys], z_axis[zs]
        nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
        spacing = (
            float(x_axis[1] - x_axis[0]) if nx > 1 else 1.0,
            float(y_axis[1] - y_axis[0]) if ny > 1 else 1.0,
            float(z_axis[1] - z_axis[0]) if nz > 1 else 1.0,
        )
        image = pv.ImageData(dimensions=(nx, ny, nz), spacing=spacing, origin=(float(x_axis[0]), float(y_axis[0]), float(z_axis[0])))
        values = np.asarray(grid.values[zs, ys, xs], dtype=float).copy()
        values[~threshold.mask(values)] = np.nan
        image.point_data["Value"] = values.transpose(2, 1, 0).ravel(order="F")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("当前显示阈值隐藏了全部三维体素")

        mode = settings.get("mode", "solid")
        alpha = float(settings.get("alpha", 0.55))
        clim = (float(finite.min()), float(finite.max()))
        if mode == "isosurface":
            iso = float(settings.get("isovalue") if settings.get("isovalue") is not None else np.median(finite))
            surface = image.contour([iso], scalars="Value")
            self.plotter.add_mesh(surface, scalars="Value", cmap="viridis", opacity=alpha, clim=clim, show_scalar_bar=False)
        else:
            opacity = "linear" if mode == "transparent" else [0.0, 0.88, 1.0]
            try:
                actor = self.plotter.add_volume(image, scalars="Value", cmap="viridis", opacity=opacity, clim=clim, shade=True, mapper="gpu", show_scalar_bar=False)
            except Exception:
                actor = self.plotter.add_volume(image, scalars="Value", cmap="viridis", opacity=opacity, clim=clim, shade=True, mapper="smart", show_scalar_bar=False)
            mapper = getattr(actor, "mapper", None)
            for method in ("SetAutoAdjustSampleDistances", "SetInteractiveAdjustSampleDistances"):
                if mapper is not None and hasattr(mapper, method): getattr(mapper, method)(True)
        self.scalar_bar = self.plotter.add_scalar_bar(
            "Value", interactive=True, vertical=True, position_x=.86, position_y=.18,
            width=.07, height=.62, fmt="%.5g", outline=True,
        )
        self.plotter.add_mesh(image.outline(), color="#d7b65c", line_width=2)
        self.title_actor = self.plotter.add_text(self.scene_title, position="upper_edge", font_size=14, name="scene-title")
        self.plotter.show_axes(); self.plotter.view_isometric(); self.plotter.reset_camera()
        interactor = getattr(getattr(self.plotter, "iren", None), "interactor", None)
        if interactor is not None:
            interactor.SetDesiredUpdateRate(30.0); interactor.SetStillUpdateRate(0.5)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.MouseButtonDblClick:
            title, ok = QInputDialog.getText(self, "修改三维图名", "图名", text=self.scene_title)
            if ok:
                self.scene_title = title.strip()
                self.plotter.add_text(self.scene_title, position="upper_edge", font_size=14, name="scene-title")
                self.plotter.render()
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:
        self.plotter.close()
        super().closeEvent(event)

