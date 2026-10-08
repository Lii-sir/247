"""通用分割视图上的手选点扩展；不导入 point_matcher 的应用界面。"""

from math import hypot

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsItemGroup

from common.widgets.image_view import ImageView


class PointImageView(ImageView):
    point_picked = Signal(float, float)
    circle_picked = Signal(float, float, float)
    annotation_selected = Signal(int)
    annotation_changed = Signal(int, float, float, object)

    def __init__(self):
        super().__init__()
        self._press = None
        self._size = None
        self._markers = []
        self.picking_enabled = True
        self.mode = "point"
        self._points, self._radii = [], []
        self._selected = -1
        self._gesture = None
        self._draft = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, image, preserve_view=False):
        self.cancel_gesture()
        self._markers = []  # scene.clear 会销毁旧 Qt 对象。
        self._size = image.shape[1::-1] if image is not None else None
        super().set_image(image, preserve_view)

    def set_points(self, points, radii=None, selected=-1):
        self._points = list(points)
        self._radii = list(radii) if radii is not None else [None] * len(points)
        self._selected = selected
        for marker in self._markers:
            self.scene().removeItem(marker)
        self._markers.clear()
        for index, (x, y) in enumerate(points, start=1):
            group = QGraphicsItemGroup()
            self.scene().addItem(group)
            color = QColor("#45e5ff" if index - 1 == selected else "#ffe66d")
            pen = QPen(color, 2)
            group.addToGroup(self.scene().addEllipse(-6, -6, 12, 12, pen))
            group.addToGroup(self.scene().addLine(-10, 0, 10, 0, pen))
            group.addToGroup(self.scene().addLine(0, -10, 0, 10, pen))
            label = self.scene().addSimpleText(f"P{index}")
            label.setBrush(color)
            label.setPos(10, -24)
            group.addToGroup(label)
            group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            group.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            for child in group.childItems():
                child.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            group.setPos(x, y)
            group.setZValue(10)
            self._markers.append(group)
            radius = self._radii[index - 1]
            if radius is not None:
                # 圆边界使用原图尺寸随缩放变化，线宽固定屏幕像素；中心十字保持可见。
                circle_pen = QPen(color, 1.5)
                circle_pen.setCosmetic(True)
                circle = self.scene().addEllipse(x - radius, y - radius, 2 * radius, 2 * radius, circle_pen)
                circle.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                circle.setZValue(9)
                self._markers.append(circle)

    def set_mode(self, mode):
        self.cancel_gesture()
        self.mode = mode

    def cancel_gesture(self):
        self._gesture = None
        self._press = None
        if self._draft is not None:
            self.scene().removeItem(self._draft)
            self._draft = None

    def _inside(self, point):
        return self._size is not None and 0 <= point.x() <= self._size[0] - 1 and 0 <= point.y() <= self._size[1] - 1

    def _hit(self, point):
        tolerance = 10 / max(self.transform().m11(), 1e-9)
        # 优先中心，再找圆边缘；阈值为屏幕像素，不随图片缩放改变手感。
        for index in reversed(range(len(self._points))):
            x, y = self._points[index]
            if hypot(point.x() - x, point.y() - y) <= tolerance:
                return index, "center"
        for index in reversed(range(len(self._points))):
            radius = self._radii[index]
            x, y = self._points[index]
            if radius is not None and abs(hypot(point.x() - x, point.y() - y) - radius) <= tolerance:
                return index, "radius"
        return None

    def mousePressEvent(self, event):
        point = self.mapToScene(event.position().toPoint())
        if self.picking_enabled and event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            if self.mode == "circle" and self._inside(point):
                self._gesture = ("draw", point)
                pen = QPen(QColor("#45e5ff"), 1.5, Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                self._draft = self.scene().addEllipse(point.x(), point.y(), 0, 0, pen)
                event.accept()
                return
            hit = self._hit(point) if self.mode == "edit" else None
            if hit is not None:
                index, part = hit
                self._gesture = (part, index, point, self._points[index], self._radii[index])
                self.annotation_selected.emit(index)
                event.accept()
                return
        self._press = event.position().toPoint() if event.button() == Qt.MouseButton.LeftButton else None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._gesture is not None:
            self._move_gesture(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def _move_gesture(self, point):
        if not self.picking_enabled:
            self.cancel_gesture()
            return
        kind = self._gesture[0]
        if kind == "draw":
            center = self._gesture[1]
            radius = min(hypot(point.x() - center.x(), point.y() - center.y()), hypot(*self._size))
            self._draft.setRect(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius)
        else:
            _, index, start, center, radius = self._gesture
            x, y = center
            if kind == "center":
                x = min(max(x + point.x() - start.x(), 0), self._size[0] - 1)
                y = min(max(y + point.y() - start.y(), 0), self._size[1] - 1)
            else:
                radius = min(max(.001, hypot(point.x() - x, point.y() - y)), hypot(*self._size))
            self.annotation_changed.emit(index, x, y, radius)

    def mouseReleaseEvent(self, event):
        if self._gesture is not None and event.button() == Qt.MouseButton.LeftButton:
            gesture = self._gesture
            point = self.mapToScene(event.position().toPoint())
            self._move_gesture(point)
            if gesture[0] == "draw" and self.picking_enabled:
                center = gesture[1]
                radius = min(hypot(point.x() - center.x(), point.y() - center.y()), hypot(*self._size))
                self.cancel_gesture()
                if radius >= .5:
                    self.circle_picked.emit(center.x(), center.y(), radius)
            else:
                self.cancel_gesture()
            event.accept()
            return
        super().mouseReleaseEvent(event)
        position = event.position().toPoint()
        if (self.picking_enabled and self.mode == "point" and self._size and self._press is not None
                and event.button() == Qt.MouseButton.LeftButton
                and (position - self._press).manhattanLength() <= 4):
            point = self.mapToScene(position)
            width, height = self._size
            if 0 <= point.x() < width and 0 <= point.y() < height:
                self.point_picked.emit(min(point.x(), width - 1), min(point.y(), height - 1))
        self._press = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancel_gesture()
            event.accept()
            return
        if self.picking_enabled and self.mode == "edit" and 0 <= self._selected < len(self._points):
            step = .1 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 10 if event.modifiers() & Qt.KeyboardModifier.ControlModifier else 1
            x, y = self._points[self._selected]
            radius = self._radii[self._selected]
            direction = {Qt.Key.Key_Left: (-step, 0), Qt.Key.Key_Right: (step, 0),
                         Qt.Key.Key_Up: (0, -step), Qt.Key.Key_Down: (0, step)}
            if event.key() in direction:
                dx, dy = direction[event.key()]
                x = min(max(x + dx, 0), self._size[0] - 1)
                y = min(max(y + dy, 0), self._size[1] - 1)
            elif radius is not None and event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_Minus):
                radius = min(max(.001, radius + (-step if event.key() == Qt.Key.Key_Minus else step)), hypot(*self._size))
            else:
                return super().keyPressEvent(event)
            self.annotation_changed.emit(self._selected, x, y, radius)
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        if self._gesture is not None:
            event.accept()
            return
        super().wheelEvent(event)

