from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from gmes_plot.domain.models import Dataset
from gmes_plot.ui.canvas_toolbar import ScientificCanvasToolbar
from gmes_plot.ui.canvas_annotations import CanvasAnnotationManager


class StatisticsWindow(QDialog):
    """Interactive one-dimensional and statistical plotting workspace."""

    CHARTS = (
        "折线图", "阶梯图", "柱状图", "直方图", "散点图", "散点线性拟合",
        "箱线图", "饼图", "玫瑰图",
    )

    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(f"一维与统计图 — {dataset.name}")
        self.resize(1100, 760)
        self.dataset = dataset

        root = QVBoxLayout(self)
        controls = QWidget()
        form = QFormLayout(controls)
        self.chart = QComboBox(); self.chart.addItems(self.CHARTS)
        self.x_field = QComboBox(); self.x_field.addItems(dataset.columns)
        self.y_field = QComboBox(); self.y_field.addItems(dataset.columns)
        self.group_field = QComboBox(); self.group_field.addItems(["— 不分组 —", *dataset.columns])
        self.bins = QSpinBox(); self.bins.setRange(3, 180); self.bins.setValue(20)
        self.title_edit = QLineEdit()
        form.addRow("图表类型", self.chart)
        form.addRow("X / 分类 / 方位字段", self.x_field)
        form.addRow("Y / 数值字段", self.y_field)
        form.addRow("分组字段", self.group_field)
        form.addRow("直方/玫瑰分组数", self.bins)
        form.addRow("图名", self.title_edit)
        buttons = QHBoxLayout()
        draw = QPushButton("绘制 / 刷新"); draw.clicked.connect(self.draw)
        export = QPushButton("导出图片…"); export.clicked.connect(self.export)
        buttons.addWidget(draw); buttons.addWidget(export)
        form.addRow(buttons)
        root.addWidget(controls)

        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas_toolbar = ScientificCanvasToolbar(self.canvas, self.figure, self, allow_properties=True)
        self.canvas_toolbar.properties_requested.connect(self._edit_title)
        self.annotations = CanvasAnnotationManager(self.canvas, self.figure, self, self.canvas_toolbar)
        self.canvas_toolbar.annotation_requested.connect(self.annotations.set_tool)
        root.addWidget(self.canvas_toolbar)
        root.addWidget(self.canvas, 1)
        self.result_label = QLabel("选择图表类型和字段后绘制。")
        root.addWidget(self.result_label)

        roles = dataset.roles
        if roles.get("x") in dataset.columns:
            self.x_field.setCurrentText(roles["x"])
        if roles.get("value") in dataset.columns:
            self.y_field.setCurrentText(roles["value"])
        self.draw()

    def _edit_title(self) -> None:
        current = self.title_edit.text().strip() or (self.figure.axes[0].get_title() if self.figure.axes else "")
        title, ok = QInputDialog.getText(self, "修改图名", "图名", text=current)
        if ok:
            self.title_edit.setText(title.strip()); self.draw()

    def _numeric(self, field: str) -> np.ndarray:
        try:
            values = np.asarray(self.dataset.columns[field], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"字段“{field}”不是数值字段") from exc
        return values

    def _xy(self) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._numeric(self.x_field.currentText()), self._numeric(self.y_field.currentText())
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            raise ValueError("所选字段没有共同有效的数值记录")
        return x[valid], y[valid]

    def draw(self) -> None:
        try:
            self.figure.clear()
            chart = self.chart.currentText()
            polar = chart == "玫瑰图"
            ax = self.figure.add_subplot(111, projection="polar" if polar else None)
            y = self._numeric(self.y_field.currentText())
            y = y[np.isfinite(y)]
            if y.size == 0:
                raise ValueError("Y字段没有有效数值")
            result = f"有效样本数 n={y.size:,}"

            if chart in ("折线图", "阶梯图", "柱状图", "散点图", "散点线性拟合"):
                x, y = self._xy()
                order = np.argsort(x, kind="stable")
                if chart == "折线图":
                    ax.plot(x[order], y[order], marker="o", markersize=3)
                elif chart == "阶梯图":
                    ax.step(x[order], y[order], where="mid")
                elif chart == "柱状图":
                    ax.bar(x, y, width=(np.ptp(x) / max(len(x), 1) * 0.8) if np.ptp(x) else 0.8)
                else:
                    ax.scatter(x, y, s=22, alpha=0.75)
                    if chart == "散点线性拟合":
                        if len(x) < 2 or np.allclose(x, x[0]):
                            raise ValueError("线性拟合至少需要两个不同的X值")
                        slope, intercept = np.polyfit(x, y, 1)
                        predicted = slope * x + intercept
                        residual = y - predicted
                        ss_res = float(np.sum(residual ** 2))
                        ss_tot = float(np.sum((y - y.mean()) ** 2))
                        r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
                        xx = np.linspace(x.min(), x.max(), 200)
                        ax.plot(xx, slope * xx + intercept, color="#d62728", linewidth=2)
                        result = f"n={len(x):,}；y={slope:.6g}x+{intercept:.6g}；R²={r2:.6g}；RMSE={np.sqrt(ss_res / len(x)):.6g}"
                ax.set_xlabel(self.x_field.currentText()); ax.set_ylabel(self.y_field.currentText())
            elif chart == "直方图":
                ax.hist(y, bins=self.bins.value(), edgecolor="white", color="#4472c4")
                ax.set_xlabel(self.y_field.currentText()); ax.set_ylabel("频数")
                result += f"；均值={y.mean():.6g}；标准差={y.std(ddof=1) if y.size > 1 else 0:.6g}"
            elif chart == "箱线图":
                ax.boxplot(y, labels=[self.y_field.currentText()], showmeans=True)
            elif chart == "饼图":
                categories = np.asarray(self.dataset.columns[self.x_field.currentText()]).astype(str)
                raw_values = self._numeric(self.y_field.currentText())
                valid = np.isfinite(raw_values)
                labels, inverse = np.unique(categories[valid], return_inverse=True)
                totals = np.bincount(inverse, weights=raw_values[valid])
                if np.any(totals < 0) or totals.sum() <= 0:
                    raise ValueError("饼图聚合值必须非负且总和大于0")
                ax.pie(totals, labels=labels, autopct="%1.1f%%")
            elif chart == "玫瑰图":
                angles = np.deg2rad(np.mod(self._numeric(self.x_field.currentText()), 360.0))
                angles = angles[np.isfinite(angles)]
                counts, edges = np.histogram(angles, bins=self.bins.value(), range=(0, 2 * np.pi))
                widths = np.diff(edges)
                ax.bar(edges[:-1], counts, width=widths, align="edge", alpha=0.75, edgecolor="white")
                ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
                result = f"有效方位样本数 n={angles.size:,}"

            ax.set_title(self.title_edit.text().strip() or f"{self.dataset.name} — {chart}")
            if not polar and chart not in ("饼图", "箱线图"):
                ax.grid(True, alpha=0.2)
            self.result_label.setText(result)
            self.annotations.render()
            self.canvas.draw_idle()
            self.canvas_toolbar.capture_view()
        except Exception as exc:
            QMessageBox.warning(self, "统计图绘制失败", str(exc))

    def export(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(self, "导出统计图", "statistics.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if selected:
            self.figure.savefig(selected, dpi=600 if Path(selected).suffix.lower() == ".png" else 300, bbox_inches="tight")

