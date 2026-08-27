from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from gmes_plot.domain.models import TerrainProfile
from gmes_plot.io.tabular import TablePreview, preview_table
from gmes_plot.io.terrain import load_terrain_profile, suggest_terrain_fields
from gmes_plot.visualization import configure_matplotlib
from gmes_plot.ui.canvas_toolbar import ScientificCanvasToolbar
from gmes_plot.ui.canvas_annotations import CanvasAnnotationManager

configure_matplotlib()


class TerrainProfileWindow(QMainWindow):
    """Standalone position-elevation importer and terrain profile viewer."""

    terrain_ready = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("地形数据窗口 — 位置 / 高程")
        self.resize(1050, 720)
        self.file_path: str | None = None
        self.preview: TablePreview | None = None
        self.terrain: TerrainProfile | None = None
        self.plot_title = ""
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        top = QHBoxLayout()
        open_button = QPushButton("打开地形文件…")
        open_button.clicked.connect(self.open_file)
        self.path_label = QLabel("尚未选择文件")
        self.path_label.setWordWrap(True)
        top.addWidget(open_button)
        top.addWidget(self.path_label, 1)
        outer.addLayout(top)

        form = QFormLayout()
        self.position_field = QComboBox()
        self.elevation_field = QComboBox()
        self.position_unit = QLineEdit("m")
        self.elevation_unit = QLineEdit("m")
        load_button = QPushButton("识别并绘制地形")
        load_button.clicked.connect(self.load_and_plot)
        form.addRow("位置字段", self.position_field)
        form.addRow("高程字段", self.elevation_field)
        form.addRow("位置单位", self.position_unit)
        form.addRow("高程单位", self.elevation_unit)
        form.addRow(load_button)

        controls = QWidget()
        controls.setLayout(form)
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        left = QSplitter()
        left.setOrientation(Qt.Vertical)
        left.addWidget(controls)
        left.addWidget(self.table)
        left.setSizes([220, 420])

        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        figure_panel = QWidget()
        figure_layout = QVBoxLayout(figure_panel)
        figure_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_toolbar = ScientificCanvasToolbar(self.canvas, self.figure, self, allow_properties=True)
        self.canvas_toolbar.properties_requested.connect(self._edit_title)
        self.annotations = CanvasAnnotationManager(self.canvas, self.figure, self, self.canvas_toolbar)
        self.canvas_toolbar.annotation_requested.connect(self.annotations.set_tool)
        figure_layout.addWidget(self.canvas_toolbar)
        figure_layout.addWidget(self.canvas, 1)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(figure_panel)
        splitter.setSizes([320, 730])
        outer.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        self.summary = QLabel("地形尚未载入")
        add_button = QPushButton("添加到当前工程")
        add_button.clicked.connect(self.add_to_project)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        bottom.addWidget(self.summary, 1)
        bottom.addWidget(add_button)
        bottom.addWidget(close_button)
        outer.addLayout(bottom)

    def _edit_title(self) -> None:
        current = self.figure.axes[0].get_title() if self.figure.axes else self.plot_title
        title, ok = QInputDialog.getText(self, "修改图名", "图名", text=current)
        if ok:
            self.plot_title = title.strip()
            if self.figure.axes:
                self.figure.axes[0].set_title(self.plot_title); self.canvas.draw_idle()

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开位置—高程文件", "", "地形文本数据 (*.csv *.txt *.dat);;所有文件 (*)"
        )
        if not path:
            return
        try:
            self.preview = preview_table(path)
            suggested_position, suggested_elevation, _ = suggest_terrain_fields(path)
        except Exception as exc:
            QMessageBox.critical(self, "地形文件识别失败", str(exc))
            return
        self.file_path = path
        self.path_label.setText(str(Path(path).resolve()))
        self.position_field.clear()
        self.elevation_field.clear()
        self.position_field.addItems(self.preview.names)
        self.elevation_field.addItems(self.preview.names)
        self.position_field.setCurrentText(suggested_position)
        self.elevation_field.setCurrentText(suggested_elevation)
        self._show_preview()

    def _show_preview(self) -> None:
        assert self.preview is not None
        self.table.setRowCount(len(self.preview.rows))
        self.table.setColumnCount(len(self.preview.names))
        self.table.setHorizontalHeaderLabels(self.preview.names)
        for row_index, row in enumerate(self.preview.rows):
            for column_index, value in enumerate(row):
                if column_index < len(self.preview.names):
                    self.table.setItem(row_index, column_index, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()

    def load_and_plot(self) -> None:
        if not self.file_path:
            QMessageBox.information(self, "尚未选择文件", "请先打开一个地形数据文件。")
            return
        if self.position_field.currentText() == self.elevation_field.currentText():
            QMessageBox.warning(self, "字段冲突", "位置字段与高程字段不能相同。")
            return
        try:
            terrain = load_terrain_profile(
                self.file_path, self.position_field.currentText(), self.elevation_field.currentText()
            )
            terrain.position_unit = self.position_unit.text().strip() or "m"
            terrain.elevation_unit = self.elevation_unit.text().strip() or "m"
        except Exception as exc:
            QMessageBox.critical(self, "地形载入失败", str(exc))
            return
        self.terrain = terrain
        position, elevation = terrain.sorted_values()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        baseline = float(np.nanmin(elevation))
        margin = max(float(np.ptp(elevation)) * 0.15, 1.0)
        ax.fill_between(position, baseline - margin, elevation, color="#9cc77b", alpha=0.8)
        ax.plot(position, elevation, color="#4b6534", linewidth=1.8, marker="o", markersize=3)
        ax.set(
            title=self.plot_title or terrain.name,
            xlabel=f"位置 ({terrain.position_unit})",
            ylabel=f"高程 ({terrain.elevation_unit})",
        )
        ax.grid(True, alpha=0.22)
        self.annotations.render()
        self.canvas.draw_idle()
        self.canvas_toolbar.capture_view()
        self.summary.setText(
            f"{terrain.point_count:,} 点；位置 {position.min():g}—{position.max():g}；"
            f"高程 {elevation.min():g}—{elevation.max():g} {terrain.elevation_unit}"
        )

    def add_to_project(self) -> None:
        if self.terrain is None:
            QMessageBox.information(self, "尚无地形", "请先识别并绘制地形。")
            return
        self.terrain_ready.emit(self.terrain)
        QMessageBox.information(self, "已添加", "地形剖面已添加到当前工程，可用于带地形断面。")

