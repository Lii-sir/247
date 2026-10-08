"""共享的缩放/平移图片视图，不导入任何业务界面。"""

import cv2 as cv
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView


class ImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#17202c"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(240, 180)
        self._auto_fit = True

    def set_image(self, image, preserve_view=False):
        self.scene().clear()
        if image is None:
            self.scene().setSceneRect(0, 0, 1, 1)
            return
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.scene().addPixmap(QPixmap.fromImage(qimage))
        self.scene().setSceneRect(0, 0, width, height)
        if not preserve_view:
            self.fit_image()

    def fit_image(self):
        self._auto_fit = True
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        if 0.01 <= self.transform().m11() * factor <= 30:
            self._auto_fit = False
            self.scale(factor, factor)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_image()

