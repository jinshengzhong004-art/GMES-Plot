from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from gmes_plot.domain.models import Dataset, GridSpec
from gmes_plot.io.tabular import ParseOptions, TablePreview, load_dataset, preview_table, suggest_roles
from gmes_plot.services.gridding import GridRecommendation, estimate_memory
from gmes_plot.services.slicing import VolumeSlice, arbitrary_plane_slice, orthogonal_slice


class ImportDialog(QDialog):
    def __init__(self, path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("导入数据与字段映射")
        self.resize(900, 600)
        self.path = path
        self.preview: TablePreview = preview_table(path)
        self.result_dataset: Dataset | None = None

        layout = QVBoxLayout(self)
        info = QLabel(f"文件：{Path(path).name}　编码：{self.preview.encoding}　分隔符：{repr(self.preview.delimiter) if self.preview.delimiter else '空白'}")
        layout.addWidget(info)
        if self.preview.warnings:
            warning = QLabel("；".join(self.preview.warnings))
            warning.setStyleSheet("color:#b06000")
            layout.addWidget(warning)

        self.table = QTableWidget(min(len(self.preview.rows), 50), len(self.preview.names))
        self.table.setHorizontalHeaderLabels(self.preview.names)
        for row_index, row in enumerate(self.preview.rows[:50]):
            for column_index, value in enumerate(row[:len(self.preview.names)]):
                self.table.setItem(row_index, column_index, QTableWidgetItem(value))
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table, 1)

        mapping = QGridLayout()
        self.role_boxes: dict[str, QComboBox] = {}
        choices = ["— 未指定 —", *self.preview.names]
        suggestions = suggest_roles(self.preview.names)
        for index, (role, label) in enumerate((("x", "X字段"), ("y", "Y字段"), ("z", "Z字段（可选）"), ("value", "Value字段"))):
            box = QComboBox()
            box.addItems(choices)
            suggested_name = suggestions.get(role)
            suggested = choices.index(suggested_name) if suggested_name in choices else 0
            box.setCurrentIndex(suggested)
            mapping.addWidget(QLabel(label), 0, index)
            mapping.addWidget(box, 1, index)
            self.role_boxes[role] = box
        layout.addLayout(mapping)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        roles = {role: box.currentText() for role, box in self.role_boxes.items() if box.currentIndex() > 0}
        if not {"x", "y", "value"}.issubset(roles):
            QMessageBox.warning(self, "字段映射不完整", "必须指定 X、Y 和 Value 字段。")
            return
        if len(set(roles.values())) != len(roles):
            QMessageBox.warning(self, "字段映射冲突", "同一字段不能同时承担多个角色。")
            return
        try:
            self.result_dataset = load_dataset(
                self.path, roles,
                ParseOptions(encoding=self.preview.encoding, delimiter=self.preview.delimiter, header=self.preview.has_header),
            )
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))
            return
        self.accept()


class InterpolationDialog(QDialog):
    def __init__(self, recommendation: GridRecommendation, dimensions: int = 2, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("二维网格化插值" if dimensions == 2 else "三维网格化插值")
        self.dimensions = dimensions
        self.recommendation = recommendation
        self.result_spec: GridSpec | None = None
        self.result_method = "idw"
        self.result_neighbors = 24

        layout = QFormLayout(self)
        self.preset = QComboBox()
        self.preset.addItems(["快速预览", "平衡（推荐）", "精细"])
        default_index = 1 if dimensions == 2 else 0
        self.preset.setCurrentIndex(default_index)
        self.preset.currentIndexChanged.connect(self._load_preset)
        layout.addRow("推荐档位", self.preset)

        self.method = QComboBox()
        self.method.addItems(["IDW", "Linear", "Nearest", "RBF", "Kriging"])
        layout.addRow("插值方法", self.method)

        self.nx, self.ny, self.nz = QSpinBox(), QSpinBox(), QSpinBox()
        for control in (self.nx, self.ny, self.nz):
            control.setRange(2, 4096)
            control.valueChanged.connect(self._update_budget)
        layout.addRow("NX", self.nx)
        layout.addRow("NY", self.ny)
        if dimensions == 3:
            layout.addRow("NZ", self.nz)

        self.neighbors = QSpinBox()
        self.neighbors.setRange(1, 256)
        self.neighbors.setValue(24)
        layout.addRow("局部邻点数", self.neighbors)
        self.memory = QLabel()
        layout.addRow("内存粗估", self.memory)
        self.note = QLabel("\n".join(recommendation.notes))
        self.note.setWordWrap(True)
        layout.addRow("推荐说明", self.note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self._load_preset(default_index)

    def _selected_spec(self) -> GridSpec:
        shape = (self.ny.value(), self.nx.value()) if self.dimensions == 2 else (self.nz.value(), self.ny.value(), self.nx.value())
        return GridSpec(self.recommendation.balanced.bounds, shape)

    def _load_preset(self, index: int) -> None:
        spec = (self.recommendation.preview, self.recommendation.balanced, self.recommendation.fine)[index]
        if self.dimensions == 2:
            ny, nx = spec.shape
            self.nx.setValue(nx); self.ny.setValue(ny)
        else:
            nz, ny, nx = spec.shape
            self.nx.setValue(nx); self.ny.setValue(ny); self.nz.setValue(nz)
        self._update_budget()

    def _update_budget(self) -> None:
        spec = self._selected_spec()
        estimate = estimate_memory(spec)
        self.memory.setText(f"{spec.cell_count:,} 单元；预计峰值约 {estimate / 1024**2:,.1f} MiB")
        self.memory.setStyleSheet("color:#b00020" if estimate > 4 * 1024**3 else "")

    def _accept(self) -> None:
        spec = self._selected_spec()
        if estimate_memory(spec) > 8 * 1024**3:
            QMessageBox.critical(self, "内存预算超限", "预计峰值超过8 GiB。请降低网格分辨率。")
            return
        self.result_spec = spec
        self.result_method = self.method.currentText().lower()
        self.result_neighbors = self.neighbors.value()
        self.accept()


class PlotPropertiesDialog(QDialog):
    def __init__(self, title: str, xlabel: str, ylabel: str, cmap: str, style: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("GMES图形参数")
        self.setMinimumWidth(440)
        style = style or {}
        form = QFormLayout(self)
        self.title_edit = QLineEdit(title)
        self.xlabel_edit = QLineEdit(xlabel)
        self.ylabel_edit = QLineEdit(ylabel)
        self.cmap = QComboBox()
        self.cmap.addItems(["viridis", "cividis", "plasma", "turbo", "terrain", "seismic", "coolwarm", "Spectral_r", "gray"])
        if cmap in [self.cmap.itemText(i) for i in range(self.cmap.count())]:
            self.cmap.setCurrentText(cmap)
        self.levels = QSpinBox(); self.levels.setRange(2, 256); self.levels.setValue(int(style.get("contour_levels", 20)))
        self.font_size = QSpinBox(); self.font_size.setRange(6, 48); self.font_size.setValue(int(style.get("font_size", 10)))
        self.line_width = QDoubleSpinBox(); self.line_width.setRange(0.05, 10); self.line_width.setValue(float(style.get("line_width", .35))); self.line_width.setSingleStep(.1)
        self.point_size = QDoubleSpinBox(); self.point_size.setRange(1, 200); self.point_size.setValue(float(style.get("point_size", 18)))
        self.grid = QCheckBox("显示坐标网格"); self.grid.setChecked(bool(style.get("show_grid", True)))
        self.aspect = QComboBox(); self.aspect.addItem("自动", "auto"); self.aspect.addItem("X/Y等比例", "equal"); self.aspect.setCurrentIndex(max(0, self.aspect.findData(style.get("aspect", "auto"))))
        self.x_scale = QComboBox(); self.x_scale.addItems(["linear", "log", "symlog"]); self.x_scale.setCurrentText(style.get("x_scale", "linear"))
        self.y_scale = QComboBox(); self.y_scale.addItems(["linear", "log", "symlog"]); self.y_scale.setCurrentText(style.get("y_scale", "linear"))
        self.colorbar_orientation = QComboBox(); self.colorbar_orientation.addItem("右侧竖直", "vertical"); self.colorbar_orientation.addItem("底部水平", "horizontal")
        self.colorbar_orientation.setCurrentIndex(max(0, self.colorbar_orientation.findData(style.get("colorbar_orientation", "vertical"))))
        self.colorbar_geometry: dict[str, QDoubleSpinBox] = {}
        for key, default in (("x", .91), ("y", .20), ("width", .025), ("height", .62)):
            control = QDoubleSpinBox(); control.setRange(0.0 if key in ("x", "y") else .01, 1.0)
            control.setDecimals(3); control.setSingleStep(.01); control.setValue(float(style.get(f"colorbar_{key}", default)))
            self.colorbar_geometry[key] = control
        self.background = style.get("background", "#ffffff")
        self.background_button = QPushButton("选择背景色"); self.background_button.setStyleSheet(f"background:{self.background}"); self.background_button.clicked.connect(self._choose_background)
        form.addRow("图名", self.title_edit)
        form.addRow("X轴标题", self.xlabel_edit)
        form.addRow("Y轴标题", self.ylabel_edit)
        form.addRow("色标", self.cmap)
        form.addRow("等值级数", self.levels); form.addRow("基础字号", self.font_size)
        form.addRow("等值线宽", self.line_width); form.addRow("散点大小", self.point_size)
        form.addRow(self.grid); form.addRow("坐标比例", self.aspect)
        form.addRow("X轴尺度", self.x_scale); form.addRow("Y轴尺度", self.y_scale)
        form.addRow("色标位置", self.colorbar_orientation); form.addRow(self.background_button)
        form.addRow("色标X（页面比例）", self.colorbar_geometry["x"]); form.addRow("色标Y（页面比例）", self.colorbar_geometry["y"])
        form.addRow("色标宽度", self.colorbar_geometry["width"]); form.addRow("色标高度", self.colorbar_geometry["height"])
        note = QLabel("滚轮、平移和框选缩放只改变数据视图。色标可在画布上拖动；Shift+拖动调整大小，也可在此精确输入页面比例坐标。")
        note.setWordWrap(True); note.setStyleSheet("color:#555"); form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _choose_background(self) -> None:
        color = QColorDialog.getColor(QColor(self.background), self, "选择画布背景色")
        if color.isValid():
            self.background = color.name(); self.background_button.setStyleSheet(f"background:{self.background}")

    @property
    def style(self) -> dict:
        return {
            "contour_levels": self.levels.value(), "font_size": self.font_size.value(),
            "line_width": self.line_width.value(), "point_size": self.point_size.value(),
            "show_grid": self.grid.isChecked(), "aspect": self.aspect.currentData(),
            "x_scale": self.x_scale.currentText(), "y_scale": self.y_scale.currentText(),
            "colorbar_orientation": self.colorbar_orientation.currentData(), "background": self.background,
            **{f"colorbar_{key}": control.value() for key, control in self.colorbar_geometry.items()},
        }


class DatasetPropertiesDialog(QDialog):
    """Edit project metadata and field bindings without mutating source columns."""

    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"数据属性与字段映射 — {dataset.name}")
        self.setMinimumWidth(460)
        form = QFormLayout(self)
        self.name_edit = QLineEdit(dataset.name); form.addRow("工程内名称", self.name_edit)
        columns = list(dataset.columns)
        self.roles: dict[str, QComboBox] = {}
        for role, label, optional in (("x", "X字段", False), ("y", "Y字段", False), ("z", "Z字段", True), ("value", "Value字段", False)):
            combo = QComboBox()
            if optional: combo.addItem("— 不使用 —", None)
            for column in columns: combo.addItem(column, column)
            if role in dataset.roles: combo.setCurrentIndex(combo.findData(dataset.roles[role]))
            self.roles[role] = combo; form.addRow(label, combo)
        self.units: dict[str, QLineEdit] = {}
        for role, label in (("x", "X单位"), ("y", "Y单位"), ("z", "Z单位"), ("value", "数值单位")):
            field = dataset.roles.get(role)
            edit = QLineEdit(dataset.units.get(field, "") if field else "")
            self.units[role] = edit; form.addRow(label, edit)
        note = QLabel("这里只修改工程内名称、字段角色和单位。原始列值保持只读；如需筛选或改值，应创建派生数据集。")
        note.setWordWrap(True); note.setStyleSheet("color:#555"); form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept); buttons.rejected.connect(self.reject); form.addRow(buttons)
        self.result_name = dataset.name; self.result_roles = dict(dataset.roles); self.result_units = dict(dataset.units)

    def _accept(self) -> None:
        name = self.name_edit.text().strip()
        roles = {role: combo.currentData() for role, combo in self.roles.items() if combo.currentData() is not None}
        if not name:
            QMessageBox.warning(self, "名称为空", "数据集名称不能为空。"); return
        if not {"x", "y", "value"}.issubset(roles):
            QMessageBox.warning(self, "字段不完整", "必须映射X、Y和Value字段。"); return
        if len(set(roles.values())) != len(roles):
            QMessageBox.warning(self, "字段重复", "同一列不能同时承担多个字段角色。"); return
        units = {}
        for role, column in roles.items():
            value = self.units[role].text().strip()
            if value: units[column] = value
        self.result_name, self.result_roles, self.result_units = name, roles, units
        self.accept()


class VolumeSliceDialog(QDialog):
    def __init__(self, grid, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("三维体任意 XYZ 剖面")
        self.grid = grid
        xmin, xmax, ymin, ymax, zmin, zmax = grid.spec.bounds
        self.bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
        self.setMinimumWidth(460)
        form = QFormLayout(self)

        self.mode = QComboBox()
        self.mode.addItem("俯视图 A—B 斜剖面（推荐）", "ab")
        self.mode.addItem("X = 常数", "x")
        self.mode.addItem("Y = 常数", "y")
        self.mode.addItem("Z = 常数", "z")
        self.mode.addItem("三点定义平面", "three_points")
        self.mode.addItem("中心 + 法向量（高级）", "arbitrary")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        form.addRow("剖面类型", self.mode)

        self.coordinate = QDoubleSpinBox()
        self.coordinate.setDecimals(6)
        form.addRow("轴向位置", self.coordinate)

        self.ax = QDoubleSpinBox(); self.ay = QDoubleSpinBox(); self.bx = QDoubleSpinBox(); self.by = QDoubleSpinBox()
        self.ztop = QDoubleSpinBox(); self.zbottom = QDoubleSpinBox(); self.dip = QDoubleSpinBox()
        for control in (self.ax, self.ay, self.bx, self.by, self.ztop, self.zbottom):
            control.setDecimals(6); control.setRange(-1e12, 1e12)
        self.ax.setValue(xmin); self.ay.setValue((ymin + ymax) / 2)
        self.bx.setValue(xmax); self.by.setValue((ymin + ymax) / 2)
        self.ztop.setValue(zmin); self.zbottom.setValue(zmax)
        self.dip.setRange(1, 90); self.dip.setValue(90); self.dip.setSuffix("°")
        form.addRow("A点 X / Y", self._pair(self.ax, self.ay)); form.addRow("B点 X / Y", self._pair(self.bx, self.by))
        form.addRow("顶部Z / 底部Z", self._pair(self.ztop, self.zbottom)); form.addRow("倾角（自水平面）", self.dip)

        self.point_edits = [QLineEdit(text) for text in (
            f"{xmin},{ymin},{zmin}", f"{xmax},{ymin},{zmax}", f"{xmin},{ymax},{zmax}"
        )]
        for index, edit in enumerate(self.point_edits, 1):
            edit.setPlaceholderText("X,Y,Z"); form.addRow(f"平面点 P{index}", edit)

        self.center_controls = [QDoubleSpinBox() for _ in range(3)]
        self.normal_controls = [QDoubleSpinBox() for _ in range(3)]
        for control in (*self.center_controls, *self.normal_controls):
            control.setDecimals(6)
            control.setRange(-1e12, 1e12)
        self.center_controls[0].setValue((xmin + xmax) / 2)
        self.center_controls[1].setValue((ymin + ymax) / 2)
        self.center_controls[2].setValue((zmin + zmax) / 2)
        self.normal_controls[0].setValue(1.0)
        self.normal_controls[1].setValue(0.0)
        self.normal_controls[2].setValue(0.0)
        form.addRow("中心 X", self.center_controls[0])
        form.addRow("中心 Y", self.center_controls[1])
        form.addRow("中心 Z", self.center_controls[2])
        form.addRow("法向量 NX", self.normal_controls[0])
        form.addRow("法向量 NY", self.normal_controls[1])
        form.addRow("法向量 NZ", self.normal_controls[2])

        diagonal_xy = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
        self.width = QDoubleSpinBox()
        self.height = QDoubleSpinBox()
        for control in (self.width, self.height):
            control.setDecimals(4)
            control.setRange(1e-9, 1e12)
        self.width.setValue(diagonal_xy)
        self.height.setValue(max(zmax - zmin, ymax - ymin, xmax - xmin))
        form.addRow("平面宽度", self.width)
        form.addRow("平面高度", self.height)

        self.nu, self.nv = QSpinBox(), QSpinBox()
        for control, value in ((self.nu, 240), (self.nv, 180)):
            control.setRange(2, 2000)
            control.setValue(value)
        form.addRow("横向采样数", self.nu)
        form.addRow("纵向采样数", self.nv)

        help_label = QLabel(
            "推荐方式：在XY俯视图确定A—B测线，再给顶部/底部深度与倾角；深度Z向下为正。"
            "三点法适合真正任意平面；中心+法向量保留给高级用户。平面超出三维体的部分显示为空。"
        )
        help_label.setWordWrap(True)
        form.addRow(help_label)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._mode_changed()

    @staticmethod
    def _pair(first: QWidget, second: QWidget) -> QWidget:
        widget = QWidget(); layout = QHBoxLayout(widget); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(first); layout.addWidget(second); return widget

    def _mode_changed(self) -> None:
        mode = self.mode.currentData()
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        ranges = {"x": (xmin, xmax), "y": (ymin, ymax), "z": (zmin, zmax)}
        is_orthogonal = mode in ranges
        self.coordinate.setEnabled(is_orthogonal)
        if is_orthogonal:
            low, high = ranges[mode]
            self.coordinate.setRange(low, high)
            self.coordinate.setValue((low + high) / 2)
        for control in (self.ax, self.ay, self.bx, self.by, self.ztop, self.zbottom, self.dip):
            control.setEnabled(mode == "ab")
        for control in self.point_edits:
            control.setEnabled(mode == "three_points")
        for control in (*self.center_controls, *self.normal_controls):
            control.setEnabled(mode == "arbitrary")
        for control in (self.width, self.height):
            control.setEnabled(mode in ("arbitrary", "three_points"))
        self.nu.setEnabled(not is_orthogonal); self.nv.setEnabled(not is_orthogonal)

    def build_slice(self) -> VolumeSlice:
        mode = self.mode.currentData()
        if mode in ("x", "y", "z"):
            return orthogonal_slice(self.grid, mode, self.coordinate.value())
        width, height = self.width.value(), self.height.value()
        if mode == "ab":
            a = np.array([self.ax.value(), self.ay.value(), self.ztop.value()], dtype=float)
            b = np.array([self.bx.value(), self.by.value(), self.ztop.value()], dtype=float)
            strike = b - a; strike[2] = 0.0
            line_length = float(np.linalg.norm(strike))
            if line_length <= 0:
                raise ValueError("A点和B点不能重合")
            if self.zbottom.value() <= self.ztop.value():
                raise ValueError("深度向下为正时，底部Z必须大于顶部Z")
            strike /= line_length
            horizontal_dip = np.array([-strike[1], strike[0], 0.0])
            dip_radians = np.deg2rad(self.dip.value())
            down_dip = horizontal_dip * np.cos(dip_radians) + np.array([0.0, 0.0, np.sin(dip_radians)])
            height = (self.zbottom.value() - self.ztop.value()) / max(np.sin(dip_radians), 1e-9)
            origin_vector = (a + b) / 2 + down_dip * height / 2
            origin = tuple(origin_vector)
            normal = tuple(np.cross(strike, down_dip))
            width = line_length
        elif mode == "three_points":
            points = []
            for edit in self.point_edits:
                try:
                    point = [float(value.strip()) for value in edit.text().split(",")]
                except ValueError as exc:
                    raise ValueError("三点坐标应使用 X,Y,Z 格式") from exc
                if len(point) != 3:
                    raise ValueError("每个平面点必须包含三个坐标：X,Y,Z")
                points.append(point)
            points_array = np.asarray(points, dtype=float)
            normal_array = np.cross(points_array[1] - points_array[0], points_array[2] - points_array[0])
            if np.linalg.norm(normal_array) <= 1e-12:
                raise ValueError("三个点不能共线")
            origin = tuple(points_array.mean(axis=0)); normal = tuple(normal_array)
        else:
            origin = tuple(control.value() for control in self.center_controls)
            normal = tuple(control.value() for control in self.normal_controls)
        return arbitrary_plane_slice(
            self.grid, origin, normal, width, height,
            (self.nu.value(), self.nv.value()),
        )


class VolumeDisplayDialog(QDialog):
    def __init__(self, grid, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("XYZV三维数据显示方式")
        form = QFormLayout(self)
        self.mode = QComboBox()
        self.mode.addItem("实心体素", "solid")
        self.mode.addItem("透明体（CPU试验）", "transparent")
        self.mode.addItem("等值面体", "isosurface")
        self.alpha = QDoubleSpinBox(); self.alpha.setRange(0.02, 1.0); self.alpha.setSingleStep(0.05); self.alpha.setValue(0.55)
        finite = np.asarray(grid.values, dtype=float)
        finite = finite[np.isfinite(finite)]
        self.isovalue = QDoubleSpinBox(); self.isovalue.setDecimals(8); self.isovalue.setRange(-1e100, 1e100)
        self.isovalue.setValue(float(np.median(finite)) if finite.size else 0.0)
        self.direction = QComboBox(); self.direction.addItem("显示 ≥ 等值", "above"); self.direction.addItem("显示 ≤ 等值", "below")
        self.note = QLabel("实心体素适合整体结构；透明体显示内部点；等值面体显示指定值一侧的外边界。显示阈值仍会叠加生效。")
        self.note.setWordWrap(True)
        form.addRow("显示模式", self.mode); form.addRow("透明度", self.alpha)
        form.addRow("等值", self.isovalue); form.addRow("等值方向", self.direction); form.addRow(self.note)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); form.addRow(buttons)
        self._mode_changed()

    def _mode_changed(self) -> None:
        enabled = self.mode.currentData() == "isosurface"
        self.isovalue.setEnabled(enabled); self.direction.setEnabled(enabled)

    @property
    def settings(self) -> dict:
        return {"mode": self.mode.currentData(), "alpha": self.alpha.value(), "isovalue": self.isovalue.value(), "direction": self.direction.currentData()}

