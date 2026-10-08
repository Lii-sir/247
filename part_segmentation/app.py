"""Qt 展示层：只负责交互与任务调度，推理在后台线程中运行。"""

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QDoubleSpinBox, QFileDialog,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QMainWindow,
    QMessageBox, QPushButton, QSlider, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from common.image_io import collect_images, read_image, write_image
from common.segmentation.inference import PartSegmenter
from common.segmentation.models import SegmentationSettings
from common.segmentation.visualization import class_color, render_comparison, render_overlay
from common.widgets.image_view import ImageView

IMAGE_FILTER = "图片 (*.bmp *.png *.jpg *.jpeg *.tif *.tiff *.webp)"


class SegmentationWorker(QThread):
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, weights, path, settings, engine=None, parent=None):
        super().__init__(parent)
        self.weights, self.path, self.settings = weights, path, settings
        self.engine = engine

    def run(self):
        try:
            if self.engine is None:
                self.engine = PartSegmenter(self.weights)
            self.result_ready.emit(self.engine.predict(self.path, self.settings))
        except Exception as exc:
            self.failed.emit(str(exc))


class SegmentationWindow(QMainWindow):
    def __init__(self, weights: Path, source: Path, settings: SegmentationSettings, recursive=False):
        super().__init__()
        self.settings = settings
        self.recursive = recursive
        self.paths = []
        self.result = None
        self.worker = None
        self.engine = None
        self._closing = False
        self.setWindowTitle("部件分割预览 · YOLO（独立于匹配找点）")
        self.resize(1380, 900)
        self._build_ui(weights)
        self.load_source(source)

    def _build_ui(self, weights):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("部件分割预览")
        title.setStyleSheet("font-size: 23px; font-weight: 600; padding: 6px;")
        layout.addWidget(title)

        weight_row = QHBoxLayout()
        weight_row.addWidget(QLabel("分割权重"))
        self.weight_edit = QLineEdit(str(Path(weights).resolve()))
        weight_row.addWidget(self.weight_edit, 1)
        self.weight_button = QPushButton("选择权重…")
        self.weight_button.clicked.connect(self.choose_weights)
        weight_row.addWidget(self.weight_button)
        self.image_button = QPushButton("打开图片…")
        self.image_button.clicked.connect(self.choose_image)
        weight_row.addWidget(self.image_button)
        self.folder_button = QPushButton("打开文件夹…")
        self.folder_button.clicked.connect(self.choose_folder)
        weight_row.addWidget(self.folder_button)
        layout.addLayout(weight_row)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("置信度"))
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0, 1)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(self.settings.confidence)
        controls.addWidget(self.confidence)
        controls.addWidget(QLabel("推理尺寸"))
        self.image_size = QSpinBox()
        self.image_size.setRange(32, max(4096, self.settings.image_size))
        self.image_size.setSingleStep(32)
        self.image_size.setValue(self.settings.image_size)
        controls.addWidget(self.image_size)
        controls.addWidget(QLabel(f"设备：{self.settings.device}   IoU：{self.settings.iou:g}"))
        self.run_button = QPushButton("开始分割当前图片")
        self.run_button.clicked.connect(self.start_prediction)
        controls.addWidget(self.run_button)
        controls.addStretch()
        fit = QPushButton("适应窗口")
        fit.clicked.connect(self.fit_views)
        controls.addWidget(fit)
        layout.addLayout(controls)

        splitter = QSplitter()
        self.image_list = QListWidget()
        self.image_list.setMinimumWidth(140)
        self.image_list.currentRowChanged.connect(self.select_image)
        splitter.addWidget(self.image_list)
        self.original_view, self.overlay_view = ImageView(), ImageView()
        for label, view in (("原图", self.original_view), ("分割叠加 · 同类同色", self.overlay_view)):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.addWidget(QLabel(label))
            panel_layout.addWidget(view, 1)
            splitter.addWidget(panel)
        splitter.setSizes([170, 560, 560])
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        layout.addWidget(splitter, 1)

        display = QHBoxLayout()
        self.alpha_label = QLabel("掩膜不透明度 45%")
        display.addWidget(self.alpha_label)
        self.alpha = QSlider(Qt.Orientation.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(45)
        self.alpha.setMaximumWidth(200)
        self.alpha.valueChanged.connect(self.redraw)
        display.addWidget(self.alpha)
        self.boxes = QCheckBox("显示检测框")
        self.labels = QCheckBox("显示类别 / 置信度")
        self.labels.setChecked(True)
        self.boxes.toggled.connect(self.redraw)
        self.labels.toggled.connect(self.redraw)
        display.addWidget(self.boxes)
        display.addWidget(self.labels)
        display.addStretch()
        self.save_button = QPushButton("保存当前效果…")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_result)
        display.addWidget(self.save_button)
        layout.addLayout(display)

        self.legend = QLabel("滚轮缩放 · 左键拖动平移 · 调整显示选项无需重新推理")
        layout.addWidget(self.legend)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["实例", "部件类别", "置信度", "掩膜面积（原图像素）"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(190)
        layout.addWidget(self.table)
        self.weight_edit.textChanged.connect(self.invalidate_result)
        self.confidence.valueChanged.connect(self.invalidate_result)
        self.image_size.valueChanged.connect(self.invalidate_result)
        self.statusBar().showMessage("选择图片后点击「开始分割当前图片」")

    def invalidate_result(self, *_):
        self.result = None
        self.overlay_view.set_image(None)
        self.table.setRowCount(0)
        self.save_button.setEnabled(False)
        self.legend.setText("滚轮缩放 · 左键拖动平移 · 调整显示选项无需重新推理")
        self.statusBar().showMessage("图片或参数已变更，请开始分割")

    def load_source(self, source):
        try:
            paths = collect_images(source, self.recursive)
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))
            return
        self.paths = paths
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for path in paths:
            self.image_list.addItem(path.name)
            self.image_list.item(self.image_list.count() - 1).setToolTip(str(path))
        self.image_list.blockSignals(False)
        self.image_list.setCurrentRow(0)

    def select_image(self, row):
        self.invalidate_result()
        self.original_view.set_image(None)
        self.run_button.setEnabled(False)
        if not 0 <= row < len(self.paths):
            return
        try:
            image = read_image(self.paths[row])
            self.original_view.set_image(image)
            self.run_button.setEnabled(True)
            self.statusBar().showMessage(f"{self.paths[row]} · {image.shape[1]} × {image.shape[0]}")
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))

    def choose_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择分割权重", self.weight_edit.text(), "PyTorch 权重 (*.pt)")
        if path:
            self.weight_edit.setText(path)

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开图片", str(self.paths[0].parent) if self.paths else "", IMAGE_FILTER)
        if path:
            self.load_source(path)

    def choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "打开图片文件夹")
        if path:
            self.load_source(path)

    def set_busy(self, busy):
        for widget in (self.weight_edit, self.weight_button, self.image_button, self.folder_button,
                       self.confidence, self.image_size, self.image_list, self.run_button):
            widget.setEnabled(not busy)
        self.run_button.setText("正在分割…" if busy else "开始分割当前图片")

    def start_prediction(self):
        row = self.image_list.currentRow()
        if self.worker is not None or not 0 <= row < len(self.paths):
            return
        try:
            settings = replace(self.settings, confidence=self.confidence.value(), image_size=self.image_size.value())
            weights = Path(self.weight_edit.text()).expanduser().resolve()
            if not weights.is_file():
                raise ValueError(f"权重文件不存在：{weights}")
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))
            return
        self.invalidate_result()
        self.set_busy(True)
        self.statusBar().showMessage("正在加载模型 / 分割；首次运行可能较慢…")
        engine = self.engine if self.engine is not None and self.engine.weights == weights else None
        self.worker = SegmentationWorker(weights, self.paths[row], settings, engine, self)
        self.worker.result_ready.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.prediction_finished)
        self.worker.start()

    def prediction_finished(self):
        self.engine = self.worker.engine
        self.worker.deleteLater()
        self.worker = None
        self.set_busy(False)
        if self._closing:
            self.close()

    def show_result(self, result):
        self.result = result
        self.original_view.set_image(result.image)
        self.redraw()
        self.overlay_view.fit_image()
        self.table.setRowCount(len(result.segments))
        counts = {}
        for row, segment in enumerate(result.segments):
            values = (f"#{row + 1}", segment.class_name, f"{segment.confidence:.3f}", str(segment.area))
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    b, g, r = class_color(segment.class_id)
                    item.setBackground(QColor(r, g, b, 90))
                self.table.setItem(row, column, item)
            counts[segment.class_name] = counts.get(segment.class_name, 0) + 1
        self.legend.setText("  |  ".join(f"{name}: {count}" for name, count in counts.items())
                            if counts else "未检测到部件，可尝试降低置信度阈值后重新分割")
        self.save_button.setEnabled(True)
        self.statusBar().showMessage(f"{result.image_path.name} · {len(result.segments)} 个部件 · "
                                    f"处理耗时 {result.elapsed_ms:.0f} ms（不含模型加载）")

    def display_options(self):
        return dict(alpha=self.alpha.value() / 100, show_boxes=self.boxes.isChecked(), show_labels=self.labels.isChecked())

    def redraw(self, *_):
        self.alpha_label.setText(f"掩膜不透明度 {self.alpha.value()}%")
        if self.result is not None:
            self.overlay_view.set_image(render_overlay(self.result, **self.display_options()), preserve_view=True)

    def fit_views(self):
        self.original_view.fit_image()
        self.overlay_view.fit_image()

    def save_result(self):
        if self.result is None:
            return
        default = Path(__file__).resolve().parent.parent / "outputs" / (self.result.image_path.stem + ".comparison.png")
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "保存效果（不覆盖原图）", str(default), "原图 + 分割对比 (*.png);;分割叠加图 (*.png)")
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".png")
        protected = [*self.paths, self.result.image_path, Path(self.weight_edit.text()).expanduser().resolve()]
        if target.resolve() in protected:
            self.show_error("不能覆盖输入图片或模型权重，请另选保存位置")
            return
        try:
            renderer = render_comparison if selected_filter.startswith("原图") else render_overlay
            write_image(target, renderer(self.result, **self.display_options()))
            self.statusBar().showMessage(f"已保存：{target}")
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))

    def show_error(self, message):
        self.statusBar().showMessage(f"失败：{message}")
        if not self._closing:
            QMessageBox.warning(self, "部件分割", message)

    def closeEvent(self, event):
        if self.worker is not None:
            self._closing = True
            self.statusBar().showMessage("等待当前推理完成后关闭，避免强制结束后台线程…")
            event.ignore()
        else:
            event.accept()


def run_app(weights, source, settings, recursive=False) -> int:
    # 初始路径错误直接反馈给 CLI，不在构造过程中弹出阻塞对话框。
    collect_images(source, recursive)
    app = QApplication.instance() or QApplication([])
    window = SegmentationWindow(weights, source, settings, recursive)
    window.show()
    return app.exec()
