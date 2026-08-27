from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace

from PySide6.QtCore import QEvent, Signal
from PySide6.QtWidgets import QButtonGroup, QFileDialog, QHBoxLayout, QLabel, QMenu, QToolButton, QWidget


@dataclass
class _PanState:
    axis: object
    x: float
    y: float
    xlim: tuple[float, float]
    ylim: tuple[float, float]


class ScientificCanvasToolbar(QWidget):
    """Application-owned navigation bar shared by every scientific canvas."""

    properties_requested = Signal()
    annotation_requested = Signal(str)

    def __init__(self, canvas, figure, parent=None, *, allow_properties: bool = True, default_pan: bool = True) -> None:
        super().__init__(parent)
        self.canvas, self.figure = canvas, figure
        self.mode = "none"
        self.default_pan = default_pan
        self.navigation_suspended = False
        self._home: dict[int, tuple] = {}
        self._pan: _PanState | None = None
        self._box_start: tuple[object, float, float] | None = None
        self._history: list[list[tuple]] = []
        self._history_index = -1
        self._buttons: dict[str, QToolButton] = {}
        self._last_motion_draw = 0.0
        layout = QHBoxLayout(self); layout.setContentsMargins(4, 2, 4, 2); layout.setSpacing(3)
        for key, text, tip, callback in (
            ("home", "全图", "恢复绘制完成时的完整范围", self.home),
            ("back", "视图←", "返回上一个缩放/平移视图", self.back),
            ("forward", "视图→", "前往下一个视图", self.forward),
            ("in", "视图＋", "放大数据视图；默认不越过数据安全边界", lambda: self.zoom(0.8)),
            ("out", "视图－", "缩小数据视图；默认不越过数据安全边界", lambda: self.zoom(1.25)),
            ("box", "框选缩放", "拖出矩形精确放大", lambda: self.set_mode("box")),
            ("fit", "适应", "恢复可靠的数据完整范围", self.fit),
        ):
            button = QToolButton(); button.setText(text); button.setToolTip(tip); button.clicked.connect(callback)
            if key == "box":
                button.setCheckable(True)
            self._buttons[key] = button; layout.addWidget(button)
        self._mode_group = QButtonGroup(self); self._mode_group.setExclusive(False)
        self._mode_group.addButton(self._buttons["box"])
        if allow_properties:
            properties = QToolButton(); properties.setText("图形参数"); properties.setToolTip("标题、坐标轴、色标和样式")
            properties.clicked.connect(self.properties_requested); layout.addWidget(properties)
            annotations = QToolButton(); annotations.setText("添加对象"); annotations.setPopupMode(QToolButton.InstantPopup)
            menu = QMenu(annotations)
            for label, tool in (("选择/移动", "select"), ("文字", "text"), ("箭头", "arrow"), ("直线", "line"), ("矩形", "rectangle"), ("椭圆", "ellipse"), ("删除选中", "delete"), ("清除页面对象", "clear")):
                action = menu.addAction(label); action.triggered.connect(lambda _checked=False, value=tool: self.annotation_requested.emit(value))
            annotations.setMenu(menu); layout.addWidget(annotations)
        save = QToolButton(); save.setText("导出"); save.setToolTip("导出当前画布")
        save.clicked.connect(self.export); layout.addWidget(save)
        layout.addStretch(1)
        self.hint = QLabel("左键拖动平移｜滚轮缩放")
        self.hint.setStyleSheet("color:#555")
        layout.addWidget(self.hint)

        self.canvas.mpl_connect("scroll_event", self._scroll)
        self.canvas.mpl_connect("button_press_event", self._press)
        self.canvas.mpl_connect("motion_notify_event", self._motion)
        self.canvas.mpl_connect("button_release_event", self._release)
        self.canvas.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.canvas and event.type() == QEvent.Wheel and not self.navigation_suspended:
            position = event.position()
            x_pixel, y_pixel = float(position.x()), float(self.canvas.height() - position.y())
            axis = next((candidate for candidate in reversed(self.figure.axes) if candidate.bbox.contains(x_pixel, y_pixel)), None)
            if axis is None or getattr(axis, "_gmes_colorbar", False) or hasattr(axis, "get_zlim3d"):
                return False
            xdata, ydata = axis.transData.inverted().transform((x_pixel, y_pixel))
            self.zoom(0.82 if event.angleDelta().y() > 0 else 1.22, SimpleNamespace(inaxes=axis, xdata=xdata, ydata=ydata))
            event.accept(); return True
        return super().eventFilter(watched, event)

    def capture_view(self) -> None:
        self._home.clear()
        for axis in self.figure.axes:
            zlim = axis.get_zlim3d() if hasattr(axis, "get_zlim3d") else None
            self._home[id(axis)] = (axis, axis.get_xlim(), axis.get_ylim(), zlim)
        self._history = [self._current_view()]; self._history_index = 0

    def _bounded_limits(self, axis, limits: tuple[float, float], dimension: str) -> tuple[float, float]:
        home = self._home.get(id(axis))
        if home is None:
            return limits
        original = home[1] if dimension == "x" else home[2]
        bound_low, bound_high = sorted(original)
        span = max(bound_high - bound_low, 1e-12)
        allowed_low, allowed_high = bound_low - span * .05, bound_high + span * .05
        inverted = limits[0] > limits[1]
        low, high = sorted(limits)
        width = high - low
        allowed_width = allowed_high - allowed_low
        if width >= allowed_width:
            low, high = allowed_low, allowed_high
        else:
            if low < allowed_low: high += allowed_low - low; low = allowed_low
            if high > allowed_high: low -= high - allowed_high; high = allowed_high
        return (high, low) if inverted else (low, high)

    def _current_view(self) -> list[tuple]:
        return [(axis, axis.get_xlim(), axis.get_ylim(), axis.get_zlim3d() if hasattr(axis, "get_zlim3d") else None) for axis in self.figure.axes]

    def _push_view(self) -> None:
        self._history = self._history[:self._history_index + 1]
        self._history.append(self._current_view()); self._history_index = len(self._history) - 1

    def _apply_view(self, view: list[tuple]) -> None:
        for axis, xlim, ylim, zlim in view:
            if axis not in self.figure.axes: continue
            axis.set_xlim(xlim); axis.set_ylim(ylim)
            if zlim is not None: axis.set_zlim(zlim)
        self.canvas.draw_idle()

    def back(self) -> None:
        if self._history_index > 0:
            self._history_index -= 1; self._apply_view(self._history[self._history_index])

    def forward(self) -> None:
        if self._history_index + 1 < len(self._history):
            self._history_index += 1; self._apply_view(self._history[self._history_index])

    def home(self) -> None:
        for axis, xlim, ylim, zlim in self._home.values():
            if axis not in self.figure.axes:
                continue
            axis.set_xlim(xlim); axis.set_ylim(ylim)
            if zlim is not None:
                axis.set_zlim(zlim)
        self.canvas.draw_idle()
        self._push_view()

    def fit(self) -> None:
        self.home()
        self.hint.setText("已适应完整数据范围｜左键拖动平移｜滚轮缩放")

    def set_mode(self, mode: str) -> None:
        requested = "none" if self.mode == mode else mode
        self.mode = requested
        self._buttons["box"].setChecked(requested == "box")
        self.hint.setText({"box": "框选缩放：拖出矩形｜右键退出"}.get(requested, "左键拖动平移｜滚轮缩放"))

    @staticmethod
    def _scaled_limits(limits: tuple[float, float], factor: float, center: float | None = None) -> tuple[float, float]:
        left, right = limits
        pivot = (left + right) / 2 if center is None else center
        return pivot + (left - pivot) * factor, pivot + (right - pivot) * factor

    def zoom(self, factor: float, event=None) -> None:
        axes = [event.inaxes] if event is not None and event.inaxes is not None else list(self.figure.axes)
        for axis in axes:
            if axis is None or axis.get_label() == "<colorbar>" or hasattr(axis, "_colorbar"):
                continue
            if hasattr(axis, "get_zlim3d"):
                self.hint.setText("三维图使用相机滚轮缩放；不会改写XYZ数据边界")
                continue
            xcenter = event.xdata if event is not None and event.inaxes is axis else None
            ycenter = event.ydata if event is not None and event.inaxes is axis else None
            axis.set_xlim(self._bounded_limits(axis, self._scaled_limits(axis.get_xlim(), factor, xcenter), "x"))
            axis.set_ylim(self._bounded_limits(axis, self._scaled_limits(axis.get_ylim(), factor, ycenter), "y"))
        self.canvas.draw_idle()
        self._push_view()

    def _scroll(self, event) -> None:
        if event.inaxes is not None:
            self.zoom(0.82 if event.button == "up" else 1.22, event)

    def _press(self, event) -> None:
        if event.button == 1 and event.dblclick:
            for axis in self.figure.axes:
                texts = [axis.title, axis.xaxis.label, axis.yaxis.label]
                if hasattr(axis, "zaxis"): texts.append(axis.zaxis.label)
                if any(text.get_visible() and text.contains(event)[0] for text in texts):
                    self.properties_requested.emit(); return
        if self.navigation_suspended:
            return
        if event.button == 3 and self.mode == "box":
            self.set_mode("none"); return
        if event.button != 1 or event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if self.mode == "box" and not hasattr(event.inaxes, "get_zlim3d"):
            self._box_start = (event.inaxes, event.xdata, event.ydata)
        elif self.default_pan and not hasattr(event.inaxes, "get_zlim3d"):
            self._pan = _PanState(event.inaxes, event.xdata, event.ydata, event.inaxes.get_xlim(), event.inaxes.get_ylim())

    def _motion(self, event) -> None:
        if self._pan is None or event.inaxes is not self._pan.axis or event.xdata is None or event.ydata is None:
            return
        dx, dy = event.xdata - self._pan.x, event.ydata - self._pan.y
        self._pan.axis.set_xlim(self._bounded_limits(self._pan.axis, (self._pan.xlim[0] - dx, self._pan.xlim[1] - dx), "x"))
        self._pan.axis.set_ylim(self._bounded_limits(self._pan.axis, (self._pan.ylim[0] - dy, self._pan.ylim[1] - dy), "y"))
        now = perf_counter()
        if now - self._last_motion_draw >= 1 / 30:
            self.canvas.draw_idle(); self._last_motion_draw = now

    def _release(self, event) -> None:
        had_pan = self._pan is not None; self._pan = None
        if had_pan:
            self.canvas.draw_idle(); self._push_view()
        if self._box_start is None:
            return
        axis, x0, y0 = self._box_start; self._box_start = None
        if event.inaxes is axis and event.xdata is not None and event.ydata is not None and abs(event.xdata - x0) > 1e-12 and abs(event.ydata - y0) > 1e-12:
            axis.set_xlim(self._bounded_limits(axis, (min(x0, event.xdata), max(x0, event.xdata)), "x"))
            y_inverted = axis.yaxis_inverted()
            axis.set_ylim(self._bounded_limits(axis, (min(y0, event.ydata), max(y0, event.ydata)), "y"))
            if y_inverted: axis.invert_yaxis()
            self.canvas.draw_idle()
            self._push_view()
        self.set_mode("none")

    def export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出当前图", "figure.png", "PNG (*.png);;TIFF (*.tif);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.figure.savefig(path, dpi=600 if path.lower().endswith((".png", ".tif", ".tiff")) else 300, bbox_inches="tight")

