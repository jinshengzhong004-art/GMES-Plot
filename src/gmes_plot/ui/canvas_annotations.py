from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QInputDialog, QMessageBox


class CanvasAnnotationManager:
    """Small page-space object layer shared with scientific plot contents."""

    def __init__(self, canvas, figure, parent, toolbar) -> None:
        self.canvas, self.figure, self.parent, self.toolbar = canvas, figure, parent, toolbar
        self.objects: list[dict] = []
        self.mode = "select"
        self.selected_id: str | None = None
        self._artists: dict[str, object] = {}
        self._start: tuple[float, float] | None = None
        self._drag: tuple[str, tuple[float, float], dict] | None = None
        self._clipboard: dict | None = None
        self.changed_callback = None
        canvas.mpl_connect("button_press_event", self._press)
        canvas.mpl_connect("motion_notify_event", self._motion)
        canvas.mpl_connect("button_release_event", self._release)
        canvas.mpl_connect("key_press_event", self._key)

    def set_tool(self, tool: str) -> None:
        if tool == "delete":
            self.delete_selected(); return
        if tool == "clear":
            self.clear(); return
        self.mode = tool; self.toolbar.set_mode("none"); self.toolbar.navigation_suspended = tool != "select"
        self._notify({"select": "选择页面对象并拖动", "text": "单击放置文字", "arrow": "拖动绘制箭头", "line": "拖动绘制直线", "rectangle": "拖动绘制矩形", "ellipse": "拖动绘制椭圆"}.get(tool, tool))

    def _notify(self, text: str) -> None:
        label = getattr(self.parent, "coordinate_label", None)
        if label is not None: label.setText(text)

    def _page_point(self, event) -> tuple[float, float] | None:
        if event.x is None or event.y is None:
            return None
        return max(0., min(1., event.x / max(1, self.canvas.width()))), max(0., min(1., event.y / max(1, self.canvas.height())))

    def _object(self, object_id: str) -> dict | None:
        return next((item for item in self.objects if item["id"] == object_id), None)

    def _hit(self, event) -> str | None:
        for object_id, artist in reversed(list(self._artists.items())):
            try:
                if artist.contains(event)[0]: return object_id
            except (AttributeError, TypeError):
                continue
        return None

    def _press(self, event) -> None:
        if event.button != 1 or event.inaxes is not None and getattr(event.inaxes, "_gmes_colorbar", False):
            return
        point = self._page_point(event)
        if point is None: return
        hit = self._hit(event)
        if event.dblclick and hit:
            item = self._object(hit)
            if item and item["type"] == "text":
                text, ok = QInputDialog.getMultiLineText(self.parent, "编辑文字", "内容", item["text"])
                if ok: item["text"] = text; self.render(); self._changed()
            elif item:
                color = QColorDialog.getColor(QColor(item.get("color", "#202020")), self.parent, "对象颜色")
                if color.isValid(): item["color"] = color.name()
                width, ok = QInputDialog.getDouble(self.parent, "线宽", "线宽", item.get("width", 1.5), .1, 20., 1)
                if ok: item["width"] = width
                self.render(); self._changed()
            return
        if self.mode == "text":
            text, ok = QInputDialog.getMultiLineText(self.parent, "添加文字", "内容")
            if ok and text.strip():
                self.objects.append({"id": str(uuid4()), "type": "text", "point": point, "text": text.strip(), "color": "#202020", "size": 11})
                self.selected_id = self.objects[-1]["id"]; self.render(); self._changed()
            self.mode = "select"; self.toolbar.navigation_suspended = False
            return
        if self.mode == "select":
            self.selected_id = hit
            if hit:
                self.toolbar._pan = None
                self._drag = (hit, point, deepcopy(self._object(hit)))
            self.render(); return
        if self.mode in {"arrow", "line", "rectangle", "ellipse"}:
            self._start = point

    @staticmethod
    def _translate(item: dict, dx: float, dy: float) -> None:
        if item["type"] == "text":
            x, y = item["point"]; item["point"] = (x + dx, y + dy)
        else:
            x0, y0, x1, y1 = item["points"]; item["points"] = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)

    def _motion(self, event) -> None:
        point = self._page_point(event)
        if point is None or self._drag is None: return
        object_id, origin, original = self._drag
        item = self._object(object_id)
        if item is None: return
        item.clear(); item.update(deepcopy(original)); self._translate(item, point[0] - origin[0], point[1] - origin[1]); self.render()

    def _release(self, event) -> None:
        if self._drag is not None:
            self._drag = None; self._changed(); return
        if self._start is None: return
        end = self._page_point(event); start = self._start; self._start = None
        if end is None or abs(end[0] - start[0]) + abs(end[1] - start[1]) < .005: return
        self.objects.append({"id": str(uuid4()), "type": self.mode, "points": (*start, *end), "color": "#202020", "width": 1.5, "fill": "none"})
        self.selected_id = self.objects[-1]["id"]; self.mode = "select"; self.toolbar.navigation_suspended = False; self.render(); self._changed()

    def render(self) -> None:
        for artist in self._artists.values():
            try: artist.remove()
            except (ValueError, AttributeError): pass
        self._artists.clear()
        transform = self.figure.transFigure
        for item in self.objects:
            selected = item["id"] == self.selected_id
            color, width = item.get("color", "#202020"), item.get("width", 1.5) + (1.2 if selected else 0)
            if item["type"] == "text":
                artist = self.figure.text(*item["point"], item["text"], transform=transform, color=color, fontsize=item.get("size", 11), bbox={"edgecolor": "#1976d2" if selected else "none", "facecolor": "none", "pad": 2})
            else:
                x0, y0, x1, y1 = item["points"]
                if item["type"] == "line": artist = Line2D([x0, x1], [y0, y1], transform=transform, color=color, linewidth=width)
                elif item["type"] == "arrow": artist = FancyArrowPatch((x0, y0), (x1, y1), transform=transform, arrowstyle="-|>", mutation_scale=14, color=color, linewidth=width)
                elif item["type"] == "rectangle": artist = Rectangle((min(x0, x1), min(y0, y1)), abs(x1-x0), abs(y1-y0), transform=transform, facecolor="none", edgecolor=color, linewidth=width)
                else: artist = Ellipse(((x0+x1)/2, (y0+y1)/2), abs(x1-x0), abs(y1-y0), transform=transform, facecolor="none", edgecolor=color, linewidth=width)
                self.figure.add_artist(artist)
            artist.set_picker(6); self._artists[item["id"]] = artist
        self.canvas.draw()

    def delete_selected(self) -> None:
        if self.selected_id is None: return
        self.objects = [item for item in self.objects if item["id"] != self.selected_id]
        self.selected_id = None; self.render(); self._changed()

    def clear(self) -> None:
        if not self.objects: return
        if QMessageBox.question(self.parent, "清除页面对象", "确定清除本页全部文字和绘图对象？数据图层不会受影响。") != QMessageBox.Yes: return
        self.objects.clear(); self.selected_id = None; self.render(); self._changed()

    def _changed(self) -> None:
        if callable(self.changed_callback): self.changed_callback()

    def _key(self, event) -> None:
        key = (event.key or "").lower()
        if key in {"delete", "backspace"} and self.mode == "select": self.delete_selected()
        elif key == "escape": self.mode = "select"; self.toolbar.navigation_suspended = False; self._start = None; self._drag = None
        elif key == "ctrl+c" and self.selected_id:
            item = self._object(self.selected_id); self._clipboard = deepcopy(item) if item else None
        elif key == "ctrl+v" and self._clipboard:
            item = deepcopy(self._clipboard); item["id"] = str(uuid4()); self._translate(item, .02, -.02)
            self.objects.append(item); self.selected_id = item["id"]; self.render(); self._changed()

