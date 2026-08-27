from __future__ import annotations

from copy import deepcopy

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from gmes_plot.domain.models import Dataset, GridResult, SpatialFilter, TerrainProfile, Threshold
from gmes_plot.domain.standards import STANDARD_PROFILES, STRATIGRAPHY_PROFILE, STRATIGRAPHY_TREE, STYLE_LIBRARY_VERSION
from gmes_plot.services.terrain import terrain_corrected_mesh
from gmes_plot.ui.canvas_toolbar import ScientificCanvasToolbar
from gmes_plot.ui.canvas_annotations import CanvasAnnotationManager


LITHOLOGY_STYLES: dict[str, tuple[str, str]] = {
    "第四纪松散堆积物": ("#ead9a1", ".."), "黏土": ("#c79b73", "--"),
    "砂": ("#f1d36b", "..."), "砂砾/砾石": ("#d6b46b", "oO"),
    "砂岩": ("#e7bd72", ".."), "泥岩/页岩": ("#9a8b73", "---"),
    "灰岩/白云岩": ("#9fc7cf", "//"), "煤层": ("#333333", "..."),
    "花岗岩": ("#e6a3a8", "++"), "闪长岩": ("#9eb1a4", "xx"),
    "安山岩": ("#9d8faa", "//"), "玄武岩": ("#555a63", "xx"),
    "凝灰岩": ("#c5b39a", "**"), "片麻岩": ("#c59cbd", "\\\\"),
    "片岩": ("#9bb7a8", "///"), "基岩（未细分）": ("#b6aaa0", "xx"),
    "自定义": ("#cccccc", ""),
}


class StratigraphyDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"选择地层年代 — {STRATIGRAPHY_PROFILE['id']}")
        self.resize(520, 600)
        layout = QVBoxLayout(self)
        note = QLabel(f"来源：{STRATIGRAPHY_PROFILE['title']}；颜色遵循ICS/CCGM年代色系。选择结果会记录数据版本，不把数值年龄固化为永久定义。")
        note.setWordWrap(True); layout.addWidget(note)
        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["年代地层单位", "颜色"]); layout.addWidget(self.tree, 1)

        def add(parent, records):
            for name, color, children in records:
                item = QTreeWidgetItem([name, color]); item.setData(0, Qt.UserRole, {"name": name, "color": color, "profile": STRATIGRAPHY_PROFILE["id"]})
                item.setBackground(1, QColor(color)); parent.addChild(item) if isinstance(parent, QTreeWidgetItem) else self.tree.addTopLevelItem(item)
                add(item, children)

        add(self.tree, STRATIGRAPHY_TREE); self.tree.expandAll()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.selection: dict | None = None

    def _accept(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            QMessageBox.warning(self, "未选择年代", "请选择一个年代地层单位。")
            return
        self.selection = dict(item.data(0, Qt.UserRole)); self.accept()


class GeologyPlotSettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("地质剖面图面设置")
        self.setMinimumWidth(470)
        self.settings = deepcopy(settings)
        form = QFormLayout(self)
        self.title = QLineEdit(settings["title"]); self.xlabel = QLineEdit(settings["xlabel"]); self.ylabel = QLineEdit(settings["ylabel"])
        form.addRow("图名", self.title); form.addRow("X轴标题", self.xlabel); form.addRow("Y轴标题", self.ylabel)
        self.cmap = QComboBox(); self.cmap.addItems(["gray", "viridis", "cividis", "turbo", "terrain", "Spectral_r", "coolwarm", "seismic"]); self.cmap.setCurrentText(settings["cmap"])
        self.levels = QSpinBox(); self.levels.setRange(2, 128); self.levels.setValue(settings["levels"])
        self.fill_alpha = QDoubleSpinBox(); self.fill_alpha.setRange(0, 1); self.fill_alpha.setSingleStep(.05); self.fill_alpha.setValue(settings["fill_alpha"])
        self.line_width = QDoubleSpinBox(); self.line_width.setRange(.05, 5); self.line_width.setSingleStep(.1); self.line_width.setValue(settings["line_width"])
        form.addRow("背景填色色带", self.cmap); form.addRow("等值级数", self.levels); form.addRow("背景透明度", self.fill_alpha); form.addRow("等值线宽", self.line_width)
        self.line_color = settings["line_color"]; self.terrain_color = settings["terrain_color"]
        self.line_color_button = QPushButton("选择等值线颜色"); self.line_color_button.setStyleSheet(f"background:{self.line_color}"); self.line_color_button.clicked.connect(lambda: self._choose_color("line"))
        self.terrain_color_button = QPushButton("选择地形线颜色"); self.terrain_color_button.setStyleSheet(f"background:{self.terrain_color}"); self.terrain_color_button.clicked.connect(lambda: self._choose_color("terrain"))
        form.addRow(self.line_color_button); form.addRow(self.terrain_color_button)
        self.show_fill = QCheckBox("显示背景填色"); self.show_fill.setChecked(settings["show_fill"])
        self.show_lines = QCheckBox("显示等值线"); self.show_lines.setChecked(settings["show_lines"])
        self.show_grid = QCheckBox("显示坐标网格"); self.show_grid.setChecked(settings["show_grid"])
        self.depth_down = QCheckBox("深度轴向下为正"); self.depth_down.setChecked(settings["depth_down"])
        form.addRow(self.show_fill); form.addRow(self.show_lines); form.addRow(self.show_grid); form.addRow(self.depth_down)
        self.xmin = QLineEdit("" if settings["xmin"] is None else f"{settings['xmin']:g}")
        self.xmax = QLineEdit("" if settings["xmax"] is None else f"{settings['xmax']:g}")
        self.ymin = QLineEdit("" if settings["ymin"] is None else f"{settings['ymin']:g}")
        self.ymax = QLineEdit("" if settings["ymax"] is None else f"{settings['ymax']:g}")
        self.vertical_exaggeration = QDoubleSpinBox(); self.vertical_exaggeration.setRange(0, 100); self.vertical_exaggeration.setSpecialValueText("自动"); self.vertical_exaggeration.setValue(settings["vertical_exaggeration"]); self.vertical_exaggeration.setSingleStep(.25)
        self.font_size = QSpinBox(); self.font_size.setRange(6, 36); self.font_size.setValue(settings["font_size"])
        form.addRow("X最小（空=自动）", self.xmin); form.addRow("X最大（空=自动）", self.xmax)
        form.addRow("Y最小（空=自动）", self.ymin); form.addRow("Y最大（空=自动）", self.ymax)
        form.addRow("垂向夸大", self.vertical_exaggeration); form.addRow("基础字号", self.font_size)
        reset = QPushButton("恢复自动范围"); reset.clicked.connect(lambda: [edit.clear() for edit in (self.xmin, self.xmax, self.ymin, self.ymax)]); form.addRow(reset)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); buttons.accepted.connect(self._accept); buttons.rejected.connect(self.reject); form.addRow(buttons)

    def _choose_color(self, target: str) -> None:
        current = self.line_color if target == "line" else self.terrain_color
        color = QColorDialog.getColor(QColor(current), self)
        if not color.isValid(): return
        if target == "line": self.line_color = color.name(); self.line_color_button.setStyleSheet(f"background:{self.line_color}")
        else: self.terrain_color = color.name(); self.terrain_color_button.setStyleSheet(f"background:{self.terrain_color}")

    @staticmethod
    def _optional_number(edit: QLineEdit) -> float | None:
        return float(edit.text()) if edit.text().strip() else None

    def _accept(self) -> None:
        try:
            xmin, xmax, ymin, ymax = (self._optional_number(edit) for edit in (self.xmin, self.xmax, self.ymin, self.ymax))
            if (xmin is None) != (xmax is None) or (ymin is None) != (ymax is None): raise ValueError("每个坐标轴必须同时填写最小值和最大值，或同时留空")
            if xmin is not None and xmin >= xmax: raise ValueError("X最小值必须小于X最大值")
            if ymin is not None and ymin >= ymax: raise ValueError("Y最小值必须小于Y最大值")
        except ValueError as exc:
            QMessageBox.warning(self, "坐标范围无效", str(exc)); return
        self.settings.update(
            title=self.title.text().strip(), xlabel=self.xlabel.text().strip() or "位置", ylabel=self.ylabel.text().strip() or "深度",
            cmap=self.cmap.currentText(), levels=self.levels.value(), fill_alpha=self.fill_alpha.value(), line_width=self.line_width.value(),
            line_color=self.line_color, terrain_color=self.terrain_color, show_fill=self.show_fill.isChecked(), show_lines=self.show_lines.isChecked(),
            show_grid=self.show_grid.isChecked(), depth_down=self.depth_down.isChecked(), xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            vertical_exaggeration=self.vertical_exaggeration.value(), font_size=self.font_size.value(),
        )
        self.accept()


class GeologySectionWindow(QDialog):
    """Mouse-driven geological section interpretation editor."""

    layers_changed = Signal(list)
    settings_changed = Signal(dict)

    def __init__(
        self, dataset: Dataset, grid: GridResult | None, terrain: TerrainProfile | None,
        threshold: Threshold, parent=None, initial_layers: list[dict] | None = None, initial_settings: dict | None = None,
        spatial_filter: SpatialFilter | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(f"地质剖面解释编辑器 — {dataset.name}")
        self.resize(1180, 760)
        self.setMinimumSize(760, 520)
        self.dataset, self.grid, self.terrain, self.threshold = dataset, grid, terrain, threshold
        self.spatial_filter = spatial_filter or SpatialFilter()
        self.layers: list[dict] = deepcopy(initial_layers or [])
        self.history: list[list[dict]] = []
        self.future: list[list[dict]] = []
        self.mode = "select"
        self.drawing: list[tuple[float, float]] = []
        self.drag_vertex: tuple[int, int] | None = None
        self.drag_layer: tuple[int, np.ndarray, tuple[float, float]] | None = None
        self.shape_start: tuple[float, float] | None = None
        self._copied_layer: dict | None = None
        self._has_rendered = False
        self._preview_artists: list = []
        self._pending_move: np.ndarray | None = None
        self.color = "#d94841"
        self.stratigraphy = {"name": "未指定", "color": "#cccccc", "profile": STRATIGRAPHY_PROFILE["id"]}
        self.plot_settings = {
            "title": "带地形物探背景与地质解释" if terrain is not None else "物探剖面与地质解释",
            "xlabel": "位置", "ylabel": "高程" if terrain is not None else "深度", "cmap": "gray",
            "levels": 20, "fill_alpha": .55, "line_width": .45, "line_color": "#606060", "terrain_color": "#354d26",
            "show_fill": True, "show_lines": True, "show_grid": True, "depth_down": terrain is None,
            "xmin": None, "xmax": None, "ymin": None, "ymax": None,
            "vertical_exaggeration": 0.0, "font_size": 10,
        }
        self.plot_settings.update(initial_settings or {})

        root = QVBoxLayout(self)
        splitter = QSplitter()
        root.addWidget(splitter, 1)
        side = QWidget(); form = QFormLayout(side); form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow); form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(side); scroll.setMinimumWidth(285); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        splitter.addWidget(scroll)
        self.layer_list = QListWidget(); self.layer_list.currentRowChanged.connect(self._load_selected)
        self.layer_list.itemChanged.connect(self._visibility_changed)
        self.layer_list.setMinimumHeight(82); self.layer_list.setMaximumHeight(125); self.layer_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form.addRow("解释图层", self.layer_list)
        self.kind = QComboBox(); self.kind.addItems(["地层/岩体", "分界线", "断层", "高值异常", "低值异常", "自由注释"])
        self.kind.currentTextChanged.connect(self._kind_changed)
        self.semantic = QComboBox(); self.semantic.setEditable(True)
        self.semantic.addItems(["高阻", "低阻", "高速", "低速", "高磁", "低磁", "高密度", "低密度", "自定义异常"])
        self.lithology = QComboBox(); self.lithology.addItems(LITHOLOGY_STYLES)
        self.lithology.currentTextChanged.connect(self._lithology_changed)
        self.name_edit = QLineEdit("新解释层")
        self.shape_tool = QComboBox(); self.shape_tool.addItems(["自由多边形/折线", "矩形", "椭圆", "圆形"])
        self.hatch = QComboBox(); self.hatch.setEditable(True); self.hatch.addItems(["", "..", "...", "//", "\\\\", "xx", "++", "--", "oO", "**"])
        self.color_button = QPushButton("选择填充/线颜色"); self.color_button.clicked.connect(self._choose_color)
        self.locked = QCheckBox("锁定，防止误编辑")
        self.snapping = QCheckBox("吸附到已有节点（12像素）"); self.snapping.setChecked(True)
        self.fault_type = QComboBox(); self.fault_type.addItems(["正断层", "逆断层"])
        self.fault_certainty = QComboBox(); self.fault_certainty.addItems(["实测", "推测"])
        self.dip_angle = QDoubleSpinBox(); self.dip_angle.setRange(0, 90); self.dip_angle.setDecimals(1); self.dip_angle.setValue(60)
        self.dip_direction = QComboBox(); self.dip_direction.addItems(["向右倾", "向左倾"])
        self.show_blocks = QCheckBox("显示上盘 / 下盘"); self.show_blocks.setChecked(True)
        self.age_button = QPushButton("选择地层年代…"); self.age_button.clicked.connect(self._choose_stratigraphy)
        self.age_label = QLabel("未指定")
        form.addRow("对象类型", self.kind); form.addRow("物性语义", self.semantic)
        form.addRow("岩性/地层", self.lithology); form.addRow("名称", self.name_edit)
        self.draw_tool_combo = QComboBox()
        for label, tool in (("选择 / 编辑节点", "选择"), ("自由多边形 / 折线", "自由多边形/折线"), ("矩形区域", "矩形"), ("椭圆区域", "椭圆"), ("圆形区域", "圆形"), ("整体移动选中对象", "移动")):
            self.draw_tool_combo.addItem(label, tool)
        self.draw_tool_combo.activated.connect(lambda _index: self._activate_tool(self.draw_tool_combo.currentData()))
        form.addRow("绘图工具", self.draw_tool_combo)
        form.addRow("填充图案", self.hatch); form.addRow(self.color_button); form.addRow(self.locked); form.addRow(self.snapping)
        form.addRow("断层类型", self.fault_type); form.addRow("可靠程度", self.fault_certainty)
        form.addRow("倾角", self.dip_angle); form.addRow("倾向", self.dip_direction); form.addRow(self.show_blocks)
        form.addRow(self.age_button); form.addRow("年代", self.age_label)

        finish = QPushButton("完成当前对象"); finish.clicked.connect(self.finish_draw)
        apply_style = QPushButton("应用属性"); apply_style.clicked.connect(self.apply_properties)
        plot_settings = QPushButton("图名 / 坐标轴 / 等值线设置…"); plot_settings.clicked.connect(self.edit_plot_settings)
        delete = QPushButton("删除选中图层"); delete.clicked.connect(self.delete_selected)
        clear_all = QPushButton("一键清除全部解释"); clear_all.clicked.connect(self.clear_all)
        up = QPushButton("上移"); up.clicked.connect(lambda: self.move_selected(-1))
        down = QPushButton("下移"); down.clicked.connect(lambda: self.move_selected(1))
        undo = QPushButton("撤销"); undo.clicked.connect(self.undo)
        redo = QPushButton("重做"); redo.clicked.connect(self.redo)
        def button_row(*buttons):
            widget = QWidget(); layout = QHBoxLayout(widget); layout.setContentsMargins(0, 0, 0, 0)
            for item in buttons: layout.addWidget(item)
            return widget
        form.addRow(plot_settings)
        form.addRow(button_row(finish, apply_style)); form.addRow(button_row(delete, clear_all))
        form.addRow(button_row(up, down)); form.addRow(button_row(undo, redo))
        delete.setShortcut("Delete"); undo.setShortcut("Ctrl+Z"); redo.setShortcut("Ctrl+Y")
        help_text = QLabel("自由工具：左键逐点、右键结束；矩形/椭圆/圆形：左键拖拽生成。选择模式可拖顶点，“整体拖动”可平移完整对象。Delete删除，Ctrl+C/Ctrl+V复制粘贴。清除和移动均可撤销。")
        help_text.setWordWrap(True); form.addRow(help_text)
        standards = QLabel(
            f"样式库 {STYLE_LIBRARY_VERSION}\n主依据：DZ/T 0069-2024、GB/T 958-2015；工程地质兼容GB/T 12328-1990（修订中）。"
        )
        standards.setWordWrap(True); standards.setStyleSheet("color:#555"); form.addRow(standards)

        canvas_widget = QWidget(); canvas_layout = QVBoxLayout(canvas_widget); canvas_layout.setContentsMargins(2, 2, 2, 2)
        splitter.addWidget(canvas_widget); splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 860]); splitter.setChildrenCollapsible(False)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas_toolbar = ScientificCanvasToolbar(self.canvas, self.figure, self, allow_properties=True, default_pan=False)
        self.canvas_toolbar.properties_requested.connect(self.edit_plot_settings)
        self.annotations = CanvasAnnotationManager(self.canvas, self.figure, self, self.canvas_toolbar)
        self.canvas_toolbar.annotation_requested.connect(self.annotations.set_tool)
        canvas_layout.addWidget(self.canvas_toolbar)
        canvas_layout.addWidget(self.canvas, 1)
        self.status = QLabel("就绪")
        self.status.setStyleSheet("padding:5px;background:#eaf2ff;border:1px solid #9db7df;font-weight:600")
        canvas_layout.addWidget(self.status)
        self.canvas.mpl_connect("button_press_event", self._press)
        self.canvas.mpl_connect("motion_notify_event", self._motion)
        self.canvas.mpl_connect("button_release_event", self._release)
        self._lithology_changed(self.lithology.currentText())
        self._kind_changed(self.kind.currentText())
        self._refresh_list()
        self.render()

    def _snapshot(self) -> None:
        self.history.append(deepcopy(self.layers)); self.future.clear()

    def _choose_color(self) -> None:
        selected = QColorDialog.getColor()
        if selected.isValid():
            self.color = selected.name(); self.color_button.setStyleSheet(f"background:{self.color}")

    def _choose_stratigraphy(self) -> None:
        dialog = StratigraphyDialog(self)
        if dialog.exec() and dialog.selection:
            self.stratigraphy = dialog.selection
            self.age_label.setText(self.stratigraphy["name"])
            if self.kind.currentText() == "地层/岩体":
                self.color = self.stratigraphy["color"]
                self.color_button.setStyleSheet(f"background:{self.color}")

    def _lithology_changed(self, name: str) -> None:
        color, hatch = LITHOLOGY_STYLES[name]
        self.color = color; self.hatch.setCurrentText(hatch)
        self.color_button.setStyleSheet(f"background:{color}")

    def _kind_changed(self, kind: str) -> None:
        is_fault = kind == "断层"
        for control in (self.fault_type, self.fault_certainty, self.dip_angle, self.dip_direction, self.show_blocks):
            control.setEnabled(is_fault)
        self.age_button.setEnabled(kind == "地层/岩体")

    @staticmethod
    def _is_polygon(kind: str) -> bool:
        return kind in ("地层/岩体", "高值异常", "低值异常")

    def _activate_tool(self, tool: str) -> None:
        index = self.draw_tool_combo.findData(tool)
        if index >= 0:
            self.draw_tool_combo.blockSignals(True); self.draw_tool_combo.setCurrentIndex(index); self.draw_tool_combo.blockSignals(False)
        self.canvas_toolbar.set_mode("none")
        self.drag_vertex = None; self.drag_layer = None; self.shape_start = None; self.drawing = []
        if tool == "选择":
            self.mode = "select"; self.status.setText("选择工具：单击对象或节点；拖动蓝色节点编辑几何")
            self.render(); return
        if tool == "移动":
            self.start_move_layer(); return
        self.shape_tool.setCurrentText(tool)
        self.start_draw()

    def edit_plot_settings(self) -> None:
        dialog = GeologyPlotSettingsDialog(self.plot_settings, self)
        if not dialog.exec(): return
        self.plot_settings = dialog.settings
        self._has_rendered = False
        self.render()
        self.settings_changed.emit(deepcopy(self.plot_settings))
        self.status.setText("图面参数已应用：图名、坐标轴、等值线与颜色设置已更新")

    def _snap_xy(self, event) -> tuple[float, float]:
        point = (float(event.xdata), float(event.ydata))
        if not self.snapping.isChecked() or not hasattr(event, "x") or not hasattr(event, "y"):
            return point
        best: tuple[float, float] | None = None; best_distance = 12.0
        for layer in self.layers:
            if not layer.get("visible", True): continue
            points = np.asarray(layer.get("points", []), dtype=float)
            if points.size == 0: continue
            display = event.inaxes.transData.transform(points)
            distances = np.hypot(display[:, 0] - event.x, display[:, 1] - event.y)
            index = int(np.argmin(distances))
            if distances[index] < best_distance:
                best_distance = float(distances[index]); best = (float(points[index, 0]), float(points[index, 1]))
        return best or point

    def _clear_preview(self) -> None:
        for artist in self._preview_artists:
            try: artist.remove()
            except (ValueError, AttributeError): pass
        self._preview_artists.clear()

    def _draw_preview(self, points: list | np.ndarray, polygon: bool) -> None:
        self._clear_preview()
        if not self.figure.axes or len(points) == 0: return
        array = np.asarray(points, dtype=float); axis = self.figure.axes[0]
        if polygon and len(array) >= 3:
            self._preview_artists.extend(axis.fill(array[:, 0], array[:, 1], facecolor=self.color, edgecolor="#0066cc", alpha=.30, linewidth=2, zorder=50))
        else:
            self._preview_artists.extend(axis.plot(array[:, 0], array[:, 1], "o--", color="#0066cc", linewidth=1.8, markersize=4, zorder=50))
        self.annotations.render()
        self.canvas.draw_idle()

    def start_draw(self) -> None:
        shape = self.shape_tool.currentText()
        if shape != "自由多边形/折线" and not self._is_polygon(self.kind.currentText()):
            QMessageBox.information(self, "区域工具", "矩形、椭圆和圆形只用于地层/岩体或异常区域；线对象请使用自由工具。")
            self.shape_tool.setCurrentIndex(0); shape = self.shape_tool.currentText()
        self.mode = "draw" if shape == "自由多边形/折线" else "shape"
        self.drawing = []; self.shape_start = None; self._clear_preview()
        self.status.setText("自由绘制：左键加点，右键完成，Backspace撤回节点，Esc取消" if self.mode == "draw" else f"{shape}工具已激活：在剖面画布内按住左键拖动，释放立即生成")

    def start_move_layer(self) -> None:
        row = self.layer_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "未选择对象", "请先选择要整体移动的解释对象。")
            return
        if self.layers[row].get("locked"):
            QMessageBox.information(self, "对象已锁定", "请先取消锁定再移动。")
            return
        self.mode = "move_layer"
        self.status.setText("整体移动：在画布中按住左键拖动选中对象，释放完成")

    def finish_draw(self) -> None:
        kind = self.kind.currentText()
        minimum = 3 if self._is_polygon(kind) else 2
        if len(self.drawing) < minimum:
            if self.drawing:
                QMessageBox.warning(self, "点数不足", f"{kind}至少需要{minimum}个点。")
            self.mode = "select"; self.drawing = []; self.render(); return
        points_array = np.asarray(self.drawing, dtype=float)
        if self._is_polygon(kind) and (np.ptp(points_array[:, 0]) <= 1e-12 or np.ptp(points_array[:, 1]) <= 1e-12):
            QMessageBox.warning(self, "区域无面积", "请拖出具有宽度和高度的区域。")
            self.mode = "select"; self.drawing = []; self.shape_start = None; self.render(); return
        self._snapshot()
        self.layers.append({
            "name": self.name_edit.text().strip() or kind, "kind": kind,
            "semantic": self.semantic.currentText(), "lithology": self.lithology.currentText(),
            "points": [[float(x), float(y)] for x, y in self.drawing],
            "color": self.color, "hatch": self.hatch.currentText(), "alpha": 0.55,
            "visible": True, "locked": self.locked.isChecked(),
            "stratigraphy": deepcopy(self.stratigraphy),
            "fault_type": self.fault_type.currentText(), "fault_certainty": self.fault_certainty.currentText(),
            "dip_angle": self.dip_angle.value(), "dip_direction": self.dip_direction.currentText(),
            "show_blocks": self.show_blocks.isChecked(),
            "standards": [item["id"] for item in STANDARD_PROFILES], "style_library_version": STYLE_LIBRARY_VERSION,
        })
        self.mode = "select"; self.drawing = []; self.shape_start = None
        self.draw_tool_combo.setCurrentIndex(self.draw_tool_combo.findData("选择"))
        self._refresh_list(); self.layer_list.setCurrentRow(len(self.layers) - 1)
        self._changed()

    def _press(self, event) -> None:
        if self.canvas_toolbar.mode != "none":
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if self.mode == "draw":
            if event.button == 3:
                self.finish_draw()
            elif event.button == 1:
                self.drawing.append(self._snap_xy(event)); self._draw_preview(self.drawing, self._is_polygon(self.kind.currentText()))
            return
        if self.mode == "shape" and event.button == 1:
            self.shape_start = self._snap_xy(event)
            self.drawing = [self.shape_start]
            return
        hit_row = self._hit_layer(event)
        if self.mode == "select" and hit_row >= 0 and hit_row != self.layer_list.currentRow():
            self.layer_list.setCurrentRow(hit_row)
        row = self.layer_list.currentRow()
        if event.button != 1 or row < 0 or self.layers[row].get("locked"):
            return
        if self.mode == "move_layer":
            self._snapshot()
            original = np.asarray(self.layers[row]["points"], dtype=float).copy()
            self.drag_layer = (row, original, (float(event.xdata), float(event.ydata)))
            return
        points = np.asarray(self.layers[row]["points"], dtype=float)
        display = event.inaxes.transData.transform(points)
        distance = np.hypot(display[:, 0] - event.x, display[:, 1] - event.y)
        vertex = int(np.argmin(distance))
        if distance[vertex] <= 12:
            self._snapshot(); self.drag_vertex = (row, vertex)

    def _hit_layer(self, event) -> int:
        from matplotlib.path import Path as MplPath
        for row in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[row]
            if not layer.get("visible", True):
                continue
            points = np.asarray(layer.get("points", []), dtype=float)
            if points.size == 0:
                continue
            if self._is_polygon(layer.get("kind", "")) and MplPath(points).contains_point((event.xdata, event.ydata)):
                return row
            display = event.inaxes.transData.transform(points)
            if np.min(np.hypot(display[:, 0] - event.x, display[:, 1] - event.y)) <= 10:
                return row
        return -1

    def _motion(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if self.mode == "shape" and self.shape_start is not None:
            self.drawing = self._shape_points(self.shape_start, self._snap_xy(event))
            self._draw_preview(self.drawing, True); return
        if self.drag_layer is not None:
            row, original, anchor = self.drag_layer
            delta = np.array([float(event.xdata) - anchor[0], float(event.ydata) - anchor[1]])
            self._pending_move = original + delta
            self._draw_preview(self._pending_move, self._is_polygon(self.layers[row]["kind"])); return
        if self.drag_vertex is not None:
            row, vertex = self.drag_vertex
            self.layers[row]["points"][vertex] = [float(event.xdata), float(event.ydata)]
            self.render()

    def _release(self, event) -> None:
        if self.mode == "shape" and self.shape_start is not None:
            self.finish_draw(); return
        if self.drag_layer is not None:
            row = self.drag_layer[0]
            if self._pending_move is not None:
                self.layers[row]["points"] = self._pending_move.tolist()
            self.drag_layer = None; self._pending_move = None; self._clear_preview(); self.mode = "select"; self._changed(); return
        if self.drag_vertex is not None:
            self.drag_vertex = None; self._changed()

    def _shape_points(self, start: tuple[float, float], end: tuple[float, float]) -> list[tuple[float, float]]:
        x0, y0 = start; x1, y1 = end
        shape = self.shape_tool.currentText()
        if shape == "矩形":
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if shape == "圆形":
            radius = max(abs(x1 - x0), abs(y1 - y0)) / 2
            rx = ry = radius
        else:
            rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
        return [(float(cx + rx * np.cos(a)), float(cy + ry * np.sin(a))) for a in angles]

    def _draw_background(self, ax) -> None:
        if self.grid is None or self.grid.spec.dimensions != 2:
            ax.set(xlabel=self.plot_settings["xlabel"], ylabel=self.plot_settings["ylabel"], title=self.plot_settings["title"] or "空白地质解释剖面")
            return
        x_axis, y_axis = self.grid.spec.axes()
        spatial = self.spatial_filter.axis_mask("y", y_axis)[:, None] & self.spatial_filter.axis_mask("x", x_axis)[None, :]
        masked = np.ma.array(self.grid.values, mask=~(spatial & self.threshold.mask(self.grid.values)))
        if masked.count() == 0:
            return
        if self.terrain is not None:
            corrected = terrain_corrected_mesh(x_axis, y_axis, self.terrain)
            values = masked[:, corrected.source_columns]
            if self.plot_settings["show_fill"]:
                ax.contourf(corrected.x, corrected.elevation, values, self.plot_settings["levels"], cmap=self.plot_settings["cmap"], alpha=self.plot_settings["fill_alpha"])
            if self.plot_settings["show_lines"]:
                ax.contour(corrected.x, corrected.elevation, values, self.plot_settings["levels"], colors=self.plot_settings["line_color"], linewidths=self.plot_settings["line_width"])
            ax.plot(corrected.x[0], corrected.surface_elevation, color=self.plot_settings["terrain_color"], linewidth=2)
        else:
            if self.plot_settings["show_fill"]:
                ax.contourf(x_axis, y_axis, masked, self.plot_settings["levels"], cmap=self.plot_settings["cmap"], alpha=self.plot_settings["fill_alpha"])
            if self.plot_settings["show_lines"]:
                ax.contour(x_axis, y_axis, masked, self.plot_settings["levels"], colors=self.plot_settings["line_color"], linewidths=self.plot_settings["line_width"])
        ax.set(xlabel=self.plot_settings["xlabel"], ylabel=self.plot_settings["ylabel"], title=self.plot_settings["title"])

    def render(self) -> None:
        previous_view = None
        if self._has_rendered and self.figure.axes:
            axis = self.figure.axes[0]
            previous_view = (axis.get_xlim(), axis.get_ylim())
        self._preview_artists.clear()
        if self.figure.axes:
            ax = self.figure.axes[0]; ax.clear()
        else:
            ax = self.figure.add_subplot(111)
        self._draw_background(ax)
        for index, layer in enumerate(self.layers):
            if not layer.get("visible", True):
                continue
            points = np.asarray(layer["points"], dtype=float)
            kind, color = layer["kind"], layer["color"]
            if self._is_polygon(kind):
                ax.fill(points[:, 0], points[:, 1], facecolor=color, edgecolor="black", alpha=layer.get("alpha", .55), hatch=layer.get("hatch", ""), linewidth=1.1, label=layer["name"])
            else:
                style = "--" if kind == "断层" and layer.get("fault_certainty") == "推测" else "-"
                width = 2.3 if kind == "断层" else 1.6
                ax.plot(points[:, 0], points[:, 1], style, color=color, linewidth=width, label=layer["name"])
                if kind == "断层":
                    midpoint = (points[0] + points[-1]) / 2
                    fault_type = layer.get("fault_type", "正断层")
                    dip = float(layer.get("dip_angle", 60.0))
                    direction = layer.get("dip_direction", "向右倾")
                    arrow = "↘" if direction == "向右倾" else "↙"
                    ax.annotate(f"{fault_type} {dip:g}° {arrow}", midpoint, xytext=(8, 8), textcoords="offset points", color=color, fontsize=8, fontweight="bold")
                    vector = points[-1] - points[0]
                    length = float(np.linalg.norm(vector))
                    if length > 0:
                        along = vector / length
                        side = np.array([-along[1], along[0]])
                        if direction == "向左倾":
                            side *= -1
                        motion = along if fault_type == "正断层" else -along
                        for side_sign, motion_sign in ((1.0, 1.0), (-1.0, -1.0)):
                            center = midpoint + side_sign * side * length * 0.035
                            delta = motion_sign * motion * length * 0.10
                            ax.annotate("", xy=center + delta / 2, xytext=center - delta / 2, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4})
                    if layer.get("show_blocks", True):
                        hanging_offset, foot_offset = ((12, -18), (-38, 12)) if direction == "向右倾" else ((-42, -18), (10, 12))
                        ax.annotate("上盘", midpoint, xytext=hanging_offset, textcoords="offset points", fontsize=8, color=color)
                        ax.annotate("下盘", midpoint, xytext=foot_offset, textcoords="offset points", fontsize=8, color=color)
            if index == self.layer_list.currentRow() and not layer.get("locked"):
                ax.plot(points[:, 0], points[:, 1], "o", color="#005bbb", markersize=4)
        if self.drawing:
            points = np.asarray(self.drawing)
            if self.mode == "shape" and len(points) >= 3:
                ax.fill(points[:, 0], points[:, 1], facecolor=self.color, edgecolor="#005bbb", alpha=0.35, linewidth=1.8)
            else:
                ax.plot(points[:, 0], points[:, 1], "o-", color=self.color, linewidth=1.5)
        if any(layer.get("visible", True) for layer in self.layers):
            ax.legend(loc="best", fontsize=8)
        size = self.plot_settings["font_size"]
        ax.title.set_fontsize(size + 2); ax.xaxis.label.set_fontsize(size); ax.yaxis.label.set_fontsize(size); ax.tick_params(labelsize=max(6, size - 1))
        ax.grid(self.plot_settings["show_grid"], alpha=0.12)
        if previous_view is not None:
            ax.set_xlim(previous_view[0]); ax.set_ylim(previous_view[1])
        else:
            if self.plot_settings["xmin"] is not None: ax.set_xlim(self.plot_settings["xmin"], self.plot_settings["xmax"])
            if self.plot_settings["ymin"] is not None: ax.set_ylim(self.plot_settings["ymin"], self.plot_settings["ymax"])
            if self.plot_settings["depth_down"] and self.terrain is None and not ax.yaxis_inverted(): ax.invert_yaxis()
        if self.plot_settings["vertical_exaggeration"] > 0:
            ax.set_aspect(self.plot_settings["vertical_exaggeration"])
        else:
            ax.set_aspect("auto")
        self.canvas.draw_idle()
        if not self._has_rendered:
            self.canvas_toolbar.capture_view(); self._has_rendered = True

    def _refresh_list(self) -> None:
        self.layer_list.blockSignals(True); self.layer_list.clear()
        for layer in self.layers:
            item = QListWidgetItem(f"{layer['kind']}｜{layer['name']}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if layer.get("visible", True) else Qt.Unchecked)
            self.layer_list.addItem(item)
        self.layer_list.blockSignals(False)

    def _load_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.layers):
            return
        layer = self.layers[row]
        self.kind.setCurrentText(layer["kind"]); self.semantic.setCurrentText(layer.get("semantic", ""))
        self.lithology.setCurrentText(layer.get("lithology", "自定义")); self.name_edit.setText(layer["name"])
        self.color = layer["color"]; self.hatch.setCurrentText(layer.get("hatch", "")); self.locked.setChecked(layer.get("locked", False))
        self.stratigraphy = deepcopy(layer.get("stratigraphy", self.stratigraphy)); self.age_label.setText(self.stratigraphy.get("name", "未指定"))
        self.fault_type.setCurrentText(layer.get("fault_type", "正断层")); self.fault_certainty.setCurrentText(layer.get("fault_certainty", "实测"))
        self.dip_angle.setValue(float(layer.get("dip_angle", 60.0))); self.dip_direction.setCurrentText(layer.get("dip_direction", "向右倾")); self.show_blocks.setChecked(layer.get("show_blocks", True))
        self.color_button.setStyleSheet(f"background:{self.color}"); self.render()

    def _visibility_changed(self, item: QListWidgetItem) -> None:
        row = self.layer_list.row(item)
        if 0 <= row < len(self.layers):
            self._snapshot(); self.layers[row]["visible"] = item.checkState() == Qt.Checked; self._changed()

    def apply_properties(self) -> None:
        row = self.layer_list.currentRow()
        if row < 0:
            return
        self._snapshot(); layer = self.layers[row]
        layer.update(
            name=self.name_edit.text().strip() or layer["kind"], kind=self.kind.currentText(), semantic=self.semantic.currentText(),
            lithology=self.lithology.currentText(), color=self.color, hatch=self.hatch.currentText(), locked=self.locked.isChecked(),
            stratigraphy=deepcopy(self.stratigraphy), fault_type=self.fault_type.currentText(), fault_certainty=self.fault_certainty.currentText(),
            dip_angle=self.dip_angle.value(), dip_direction=self.dip_direction.currentText(), show_blocks=self.show_blocks.isChecked(),
            standards=[item["id"] for item in STANDARD_PROFILES], style_library_version=STYLE_LIBRARY_VERSION,
        )
        self._refresh_list(); self.layer_list.setCurrentRow(row); self._changed()

    def delete_selected(self) -> None:
        row = self.layer_list.currentRow()
        if row >= 0:
            self._snapshot(); self.layers.pop(row); self._refresh_list(); self._changed()

    def clear_all(self) -> None:
        if not self.layers:
            return
        if QMessageBox.question(
            self, "清除全部解释",
            f"确定清除全部 {len(self.layers)} 个解释对象？此操作不会影响物探背景或原始数据，并可通过“撤销”恢复。",
        ) != QMessageBox.Yes:
            return
        self._snapshot(); self.layers.clear(); self.drawing = []; self.mode = "select"
        self._refresh_list(); self._changed()

    def duplicate_selected(self) -> None:
        row = self.layer_list.currentRow()
        if row < 0:
            return
        self._snapshot()
        copied = deepcopy(self.layers[row])
        copied["name"] = f"{copied['name']} 副本"
        points = np.asarray(copied["points"], dtype=float)
        if points.size:
            span = np.ptp(points, axis=0)
            offset = np.where(span > 0, span * 0.04, 1.0)
            copied["points"] = (points + offset).tolist()
        self.layers.append(copied); self._refresh_list(); self.layer_list.setCurrentRow(len(self.layers) - 1); self._changed()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.mode = "select"; self.drawing = []; self.shape_start = None; self.drag_layer = None; self.drag_vertex = None
            self.status.setText("已取消当前工具，返回选择模式"); self.render(); return
        if event.key() == Qt.Key_Backspace and self.mode == "draw" and self.drawing:
            self.drawing.pop(); self.status.setText(f"已撤回节点；当前 {len(self.drawing)} 个节点"); self.render(); return
        if event.key() == Qt.Key_Delete:
            self.delete_selected(); return
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_C:
            row = self.layer_list.currentRow()
            if row >= 0:
                self._copied_layer = deepcopy(self.layers[row]); self.status.setText("已复制选中解释对象")
            return
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_V and self._copied_layer is not None:
            self._snapshot(); copied = deepcopy(self._copied_layer); copied["name"] = f"{copied['name']} 副本"
            points = np.asarray(copied["points"], dtype=float); copied["points"] = (points + np.array([1.0, 1.0])).tolist()
            self.layers.append(copied); self._refresh_list(); self.layer_list.setCurrentRow(len(self.layers) - 1); self._changed(); return
        super().keyPressEvent(event)

    def move_selected(self, delta: int) -> None:
        row = self.layer_list.currentRow(); target = row + delta
        if row >= 0 and 0 <= target < len(self.layers):
            self._snapshot(); self.layers[row], self.layers[target] = self.layers[target], self.layers[row]
            self._refresh_list(); self.layer_list.setCurrentRow(target); self._changed()

    def undo(self) -> None:
        if self.history:
            self.future.append(deepcopy(self.layers)); self.layers = self.history.pop(); self._refresh_list(); self._changed(record=False)

    def redo(self) -> None:
        if self.future:
            self.history.append(deepcopy(self.layers)); self.layers = self.future.pop(); self._refresh_list(); self._changed(record=False)

    def _changed(self, record: bool = True) -> None:
        self.render(); self.layers_changed.emit(deepcopy(self.layers)); self.status.setText(f"解释图层 {len(self.layers)} 个；修改已记录")

