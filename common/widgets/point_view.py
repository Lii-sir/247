"""Shared image view with point picking and markers; no application imports."""

import cv2 as cv
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsItem, QGraphicsItemGroup, QGraphicsScene, QGraphicsView

class PointImageView(QGraphicsView):
    point_picked = Signal(float, float)
    position_changed = Signal(str)

    def __init__(self, allow_picking=False):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#121c2d"))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setMinimumSize(280, 220)
        self.allow_picking = allow_picking
        self.image_width = self.image_height = 0
        self._auto_fit = True
        self._press_position = None
        self._markers = []
        self._outline = None

    def set_image(self, image):
        self.scene().clear()
        self._markers = []
        self._outline = None
        self.image_width = self.image_height = 0
        if image is None:
            self.scene().setSceneRect(0, 0, 1, 1)
            self.viewport().update()
            return
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.scene().addPixmap(QPixmap.fromImage(qimage))
        self.image_width, self.image_height = width, height
        self.scene().setSceneRect(0, 0, width, height)
        self.fit_image()

    def set_markers(self, points, outline=None):
        for marker in self._markers:
            self.scene().removeItem(marker)
        self._markers = []
        if self._outline is not None:
            self.scene().removeItem(self._outline)
            self._outline = None
        if outline:
            pen = QPen(QColor("#39b8ff"), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._outline = self.scene().addPolygon(QPolygonF([QPointF(x, y) for x, y in outline]), pen)
        for index, (x, y) in enumerate(points, start=1):
            if not (0 <= x < self.image_width and 0 <= y < self.image_height):
                continue
            color = QColor("#ffe66d")
            pen = QPen(color, 2)
            group = QGraphicsItemGroup()
            self.scene().addItem(group)
            group.addToGroup(self.scene().addEllipse(-7, -7, 14, 14, pen))
            group.addToGroup(self.scene().addLine(-12, 0, 12, 0, pen))
            group.addToGroup(self.scene().addLine(0, -12, 0, 12, pen))
            label = self.scene().addSimpleText(f"P{index}")
            label.setBrush(color)
            font = label.font()
            font.setBold(True)
            font.setPointSize(11)
            label.setFont(font)
            label.setPos(12, -25)
            group.addToGroup(label)
            group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            group.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            for child in group.childItems():
                child.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            group.setPos(x, y)
            group.setZValue(10)
            self._markers.append(group)

    def fit_image(self):
        self._auto_fit = True
        if self.image_width:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_image()

    def wheelEvent(self, event):
        if not self.image_width:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        if 0.01 <= self.transform().m11() * factor <= 30:
            self._auto_fit = False
            self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        self._press_position = event.position().toPoint() if event.button() == Qt.MouseButton.LeftButton else None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        position = event.position().toPoint()
        if (self.allow_picking and event.button() == Qt.MouseButton.LeftButton
                and self._press_position is not None
                and (position - self._press_position).manhattanLength() <= 4):
            point = self.mapToScene(position)
            if 0 <= point.x() < self.image_width and 0 <= point.y() < self.image_height:
                self.point_picked.emit(min(point.x(), self.image_width - 1), min(point.y(), self.image_height - 1))
        self._press_position = None

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        point = self.mapToScene(event.position().toPoint())
        if 0 <= point.x() < self.image_width and 0 <= point.y() < self.image_height:
            self.position_changed.emit(f"x = {point.x():.1f}    y = {point.y():.1f}")
        else:
            self.position_changed.emit("")

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not self.image_width:
            painter.save()
            painter.resetTransform()
            painter.setPen(QColor("#7b8da9"))
            text = "请打开图片后选取标定点" if self.allow_picking else "请打开图片查看结果"
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, text)
            painter.restore()
