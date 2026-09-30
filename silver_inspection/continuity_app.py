"""GUI for chip-ring silver continuity inspection."""

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPushButton, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from part_segmentation.image_io import collect_images, read_image, write_image
from part_segmentation.inference import PartSegmenter
from part_segmentation.models import SegmentationResult, SegmentationSettings
from part_segmentation.visualization import render_comparison as render_segmentation_comparison
from part_segmentation.visualization import render_overlay as render_segmentation_overlay
from point_matcher.app import IMAGE_FILTER, ImageView

from .continuity import ContinuitySettings
from .continuity_pipeline import ContinuityResult, evaluate_continuity
from .continuity_visualization import render_continuity_comparison, render_continuity_overlay


STATUS = {"ok": "检测合格", "disconnected": "银浆断连", "uncertain": "无法判定"}


class SegmentationWorker(QThread):
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, weights, path, segmentation_settings, engine=None, parent=None):
        super().__init__(parent)
        self.weights, self.path = weights, path
        self.segmentation_settings = segmentation_settings
        self.engine = engine

    def run(self):
        try:
            if self.engine is None:
                self.engine = PartSegmenter(self.weights)
            self.result_ready.emit(self.engine.predict(self.path, self.segmentation_settings))
        except Exception as exc:
            self.failed.emit(str(exc))


class ContinuityWindow(QMainWindow):
    def __init__(self, weights, source, settings, segmentation_settings, recursive=False):
        super().__init__()
        self.base_settings = settings
        self.base_segmentation_settings = segmentation_settings
        self.recursive = recursive
        self.paths, self.segmentation, self.result, self.worker, self.engine = [], None, None, None, None
        self._closing = False
        self.setWindowTitle("银浆检测 · 分割 / 断连 / 溢出")
        self.resize(1420, 900)
        self._build_ui(weights)
        self.load_source(source)

    def _build_ui(self, weights):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("银浆断连检测")
        title.setStyleSheet("font-size: 23px; font-weight: 600; padding: 6px;")
        layout.addWidget(title)
        info = QLabel("检测逻辑：取 chip 外接矩形向外扩展指定像素形成 360° 环带，按角度分段检查 silver；thin/bond 覆盖区域忽略。")
        info.setWordWrap(True)
        layout.addWidget(info)
        self.controls = []

        row = QHBoxLayout()
        row.addWidget(QLabel("权重"))
        self.weights = QLineEdit(str(Path(weights).resolve()))
        row.addWidget(self.weights, 1)
        self._button(row, "选择权重…", self.choose_weights)
        row.addWidget(QLabel("silver"))
        self.silver_class = QLineEdit(self.base_settings.silver_class)
        self.silver_class.setMaximumWidth(100)
        row.addWidget(self.silver_class)
        row.addWidget(QLabel("chip"))
        self.chip_class = QLineEdit(self.base_settings.chip_class)
        self.chip_class.setMaximumWidth(100)
        row.addWidget(self.chip_class)
        row.addWidget(QLabel("忽略类别"))
        self.occlusion_classes = QLineEdit(",".join(self.base_settings.occlusion_classes))
        self.occlusion_classes.setToolTip("例如 thin,bond；这些类别覆盖的环带像素不参与判定")
        self.occlusion_classes.setMaximumWidth(160)
        row.addWidget(self.occlusion_classes)
        layout.addLayout(row)

        row = QHBoxLayout()
        self._button(row, "待测图片…", self.choose_source)
        self._button(row, "图片文件夹…", self.choose_folder)
        self.outward = QSpinBox()
        self.outward.setRange(1, 1000)
        self.outward.setValue(self.base_settings.outward_length_px)
        self.sectors = QSpinBox()
        self.sectors.setRange(4, 720)
        self.sectors.setValue(self.base_settings.sector_count)
        self.min_silver = QSpinBox()
        self.min_silver.setRange(1, 1_000_000)
        self.min_silver.setValue(self.base_settings.min_sector_silver_px)
        self.min_coverage = QDoubleSpinBox()
        self.min_coverage.setRange(0, 1)
        self.min_coverage.setSingleStep(0.01)
        self.min_coverage.setValue(self.base_settings.min_sector_coverage)
        self.min_valid = QSpinBox()
        self.min_valid.setRange(1, 1_000_000)
        self.min_valid.setValue(self.base_settings.min_valid_sector_px)
        self.occlusion_dilation = QSpinBox()
        self.occlusion_dilation.setRange(0, 20)
        self.occlusion_dilation.setValue(self.base_settings.occlusion_dilation_px)
        for label, widget in (("外扩 px", self.outward), ("扇区", self.sectors),
                              ("每扇区最少 silver px", self.min_silver),
                              ("最少覆盖率", self.min_coverage), ("扇区最少有效 px", self.min_valid),
                              ("遮挡膨胀 px", self.occlusion_dilation)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        layout.addLayout(row)

        row = QHBoxLayout()
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0, 1)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(self.base_segmentation_settings.confidence)
        self.image_size = QSpinBox()
        self.image_size.setRange(32, 4096)
        self.image_size.setSingleStep(32)
        self.image_size.setValue(self.base_segmentation_settings.image_size)
        for label, widget in (("置信度", self.confidence), ("推理尺寸", self.image_size)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        self.device = QLineEdit(self.base_segmentation_settings.device)
        self.device.setMaximumWidth(90)
        row.addWidget(QLabel("设备"))
        row.addWidget(self.device)
        self.segment_button = self._button(row, "1. 开始分割", self.start_segmentation)
        self.continuity_button = self._button(row, "2. 断连检测", self.start_continuity)
        self.continuity_button.setEnabled(False)
        layout.addLayout(row)

        splitter = QSplitter()
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.select_image)
        splitter.addWidget(self.image_list)
        self.original_view, self.segmentation_view, self.result_view = ImageView(), ImageView(), ImageView()
        for label, view in (("原图", self.original_view), ("分割结果", self.segmentation_view),
                            ("检测结果", self.result_view)):
            panel = QWidget()
            column = QVBoxLayout(panel)
            column.addWidget(QLabel(label))
            column.addWidget(view, 1)
            splitter.addWidget(panel)
        splitter.setSizes([120, 520, 520, 520])
        layout.addWidget(splitter, 1)

        self.details = QLabel("第一步先点击“开始分割”查看 silver/chip/thin/bond 分割效果，再分别执行断连检测。")
        self.details.setWordWrap(True)
        self.details.setStyleSheet("padding: 8px; font-size: 14px;")
        layout.addWidget(self.details)
        row = QHBoxLayout()
        row.addWidget(QLabel("紫色：检查环带  ·  绿色：silver  ·  黄色：thin/bond 忽略  ·  红色：断连扇区"), 1)
        self.save_button = self._button(row, "保存对比图…", self.save_result)
        self.save_button.setEnabled(False)
        self._button(row, "适应窗口", self.fit_views)
        layout.addLayout(row)

        self.controls.extend((self.weights, self.silver_class, self.chip_class, self.occlusion_classes, self.device,
                              self.image_list, self.outward, self.sectors, self.min_silver,
                              self.min_coverage, self.min_valid, self.occlusion_dilation,
                              self.confidence, self.image_size))
        for widget in (self.weights, self.device):
            widget.textChanged.connect(self.invalidate_segmentation)
        for widget in (self.confidence, self.image_size):
            widget.valueChanged.connect(self.invalidate_segmentation)
        for widget in (self.silver_class, self.chip_class, self.occlusion_classes):
            widget.textChanged.connect(self.invalidate_continuity)
        for widget in (self.outward, self.sectors, self.min_silver, self.min_coverage, self.min_valid,
                       self.occlusion_dilation):
            widget.valueChanged.connect(self.invalidate_continuity)

    def _button(self, row, text, action):
        button = QPushButton(text)
        button.clicked.connect(action)
        row.addWidget(button)
        self.controls.append(button)
        return button

    def invalidate_segmentation(self, *_):
        self.result = None
        self.save_button.setEnabled(False)
        self.segmentation = None
        self.segmentation_view.set_image(None)
        self.result_view.set_image(None)
        self.details.setText("输入或参数已变更，请先重新分割。")
        self.continuity_button.setEnabled(False)

    def invalidate_continuity(self, *_):
        self.result = None
        self.result_view.set_image(None)
        self.save_button.setEnabled(self.segmentation is not None)
        if self.segmentation is not None:
            self.details.setText("断连参数已变更，请重新执行断连检测；当前分割结果仍可复用。")

    def load_source(self, source):
        if self.worker:
            return
        paths = collect_images(source, self.recursive)
        self.paths = paths
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for path in paths:
            self.image_list.addItem(path.name)
            self.image_list.item(self.image_list.count() - 1).setToolTip(str(path))
        self.image_list.blockSignals(False)
        self.image_list.setCurrentRow(0)

    def select_image(self, row):
        if self.worker:
            return
        self.invalidate_segmentation()
        if 0 <= row < len(self.paths):
            try:
                self.original_view.set_image(read_image(self.paths[row]))
                self.statusBar().showMessage(str(self.paths[row]))
            except (ValueError, OSError) as exc:
                self.show_error(str(exc))

    def choose_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择分割权重", self.weights.text(), "权重 (*.pt)")
        if path:
            self.weights.setText(path)

    def choose_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择待测图", "", IMAGE_FILTER)
        if path:
            self._try(lambda: self.load_source(path))

    def choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择待测图片文件夹")
        if path:
            self._try(lambda: self.load_source(path))

    def settings(self):
        classes = tuple(value.strip() for value in self.occlusion_classes.text().replace("，", ",").split(",") if value.strip())
        return ContinuitySettings(self.silver_class.text(), self.chip_class.text(), classes,
                                  self.outward.value(), self.sectors.value(), self.min_silver.value(),
                                  self.min_coverage.value(), self.min_valid.value(), self.occlusion_dilation.value())

    def _settings_for_run(self):
        settings = self.settings()
        segmentation = replace(self.base_segmentation_settings, confidence=self.confidence.value(),
                               image_size=self.image_size.value(), device=self.device.text().strip())
        weights = Path(self.weights.text()).expanduser().resolve()
        if not weights.is_file():
            raise ValueError(f"权重不存在：{weights}")
        return settings, segmentation, weights

    def start_segmentation(self):
        row = self.image_list.currentRow()
        if self.worker or not 0 <= row < len(self.paths):
            return
        try:
            _, segmentation, weights = self._settings_for_run()
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))
            return
        self.invalidate_segmentation()
        for widget in self.controls:
            widget.setEnabled(False)
        self.statusBar().showMessage(f"正在使用 {segmentation.device} 分割 chip、silver、thin/bond…")
        engine = self.engine if self.engine is not None and self.engine.weights == weights else None
        self.worker = SegmentationWorker(weights, self.paths[row], segmentation, engine, self)
        self.worker.result_ready.connect(self.show_segmentation)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.segmentation_finished)
        self.worker.start()

    def start_continuity(self):
        if self.worker:
            return
        if self.segmentation is None:
            self.show_error("请先点击“开始分割”，确认分割结果后再执行断连检测")
            return
        try:
            settings = self.settings()
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self.show_continuity(evaluate_continuity(self.segmentation, settings))

    def show_segmentation(self, result: SegmentationResult):
        self.segmentation = result
        self.segmentation_view.set_image(render_segmentation_overlay(result, alpha=0.45, show_boxes=True, show_labels=True))
        counts = {}
        for segment in result.segments:
            counts[segment.class_name] = counts.get(segment.class_name, 0) + 1
        self.details.setText("分割完成：" + ("、".join(f"{name} {count}" for name, count in counts.items())
                                      if counts else "未检测到实例") +
                            "。请先检查右侧分割效果，再点击“断连检测”。")
        self.save_button.setEnabled(True)
        self.statusBar().showMessage(f"{result.image_path.name} · 分割 {len(result.segments)} 个实例 · {result.elapsed_ms:.0f} ms")

    def segmentation_finished(self):
        self.engine = self.worker.engine
        self.worker.deleteLater()
        self.worker = None
        for widget in self.controls:
            widget.setEnabled(True)
        self.continuity_button.setEnabled(self.segmentation is not None)
        if self._closing:
            self.close()

    def show_continuity(self, result: ContinuityResult):
        self.result = result
        self.result_view.set_image(render_continuity_overlay(result))
        self.details.setText(f"{STATUS.get(result.status, result.status)} · {result.message}")
        if result.measurement:
            m = result.measurement
            self.details.setText(self.details.text() +
                                 f"\nchip {m.chip_area_px} px · ring {m.ring_area_px} px · "
                                 f"silver {m.silver_area_px} px · 遮挡忽略 {m.occluded_area_px} px · "
                                 f"有效扇区 {m.covered_sector_count}/{m.valid_sector_count}")
        self.save_button.setEnabled(True)
        self.statusBar().showMessage(f"{result.segmentation.image_path.name} · 断连检测完成")

    def save_result(self):
        if (self.result is None and self.segmentation is None) or self.worker:
            return
        default = "outputs/silver-continuity.png" if self.result is not None else "outputs/segmentation.png"
        path, _ = QFileDialog.getSaveFileName(self, "保存分割/检测结果", default, "PNG (*.png)")
        if not path:
            return
        target = Path(path if Path(path).suffix else path + ".png").resolve()
        protected = {Path(self.weights.text()).expanduser().resolve(), *self.paths}
        if target in protected:
            self.show_error("不能覆盖输入图片或权重")
            return
        renderer = render_continuity_comparison if self.result is not None else render_segmentation_comparison
        image = renderer(self.result) if self.result is not None else renderer(self.segmentation, show_boxes=True, show_labels=True)
        self._try(lambda: write_image(target, image))

    def fit_views(self):
        self.original_view.fit_image()
        self.segmentation_view.fit_image()
        self.result_view.fit_image()

    def _try(self, action):
        try:
            action()
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))

    def show_error(self, message):
        self.details.setText(f"无法判定：{message}")
        self.statusBar().showMessage(message)
        if not self._closing:
            QMessageBox.warning(self, "银浆断连检测", message)

    def closeEvent(self, event):
        if self.worker:
            self._closing = True
            self.statusBar().showMessage("等待当前推理完成后关闭…")
            event.ignore()
        else:
            event.accept()


def run_continuity_app(weights, source, settings, segmentation_settings, recursive=False):
    collect_images(source, recursive)
    app = QApplication.instance() or QApplication([])
    window = ContinuityWindow(weights, source, settings, segmentation_settings, recursive)
    window.show()
    return app.exec()
