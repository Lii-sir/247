"""Interactive boundary calibration and silver overflow inspection."""

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMainWindow, QMessageBox, QPushButton, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from part_segmentation.image_io import collect_images, read_image, write_image
from part_segmentation.inference import PartSegmenter
from point_matcher.app import IMAGE_FILTER, ImageView
from .calibration import Calibration, load_calibration, save_calibration
from .geometry import Boundary, OverflowSettings
from .pipeline import SilverInspector
from .visualization import render_comparison, render_overlay

STATUS = {"ok": "未发现溢出", "overflow": "溢出缺陷", "uncertain": "无法判定", "no_silver": "未检出银浆"}


class InspectionWorker(QThread):
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, weights, path, calibration, settings, segmentation_settings, engine=None, parent=None):
        super().__init__(parent)
        self.weights, self.path, self.calibration = weights, path, calibration
        self.settings, self.segmentation_settings, self.engine = settings, segmentation_settings, engine

    def run(self):
        try:
            if self.engine is None:
                self.engine = PartSegmenter(self.weights)
            inspector = SilverInspector(self.weights, self.calibration, self.settings,
                                        self.segmentation_settings, segmenter=self.engine)
            self.result_ready.emit(inspector.inspect(self.path))
        except Exception as exc:
            self.failed.emit(str(exc))


class InspectionWindow(QMainWindow):
    def __init__(self, weights, source, settings, segmentation_settings, calibration=None, recursive=False):
        super().__init__()
        self.segmentation_settings, self.recursive = segmentation_settings, recursive
        self.template_path, self.template_image, self.points = None, None, []
        self.paths, self.result, self.worker, self.engine = [], None, None, None
        self._closing = False
        self.setWindowTitle("银浆溢出检测 · 分割 / 匹配 / 边界判断")
        self.resize(1420, 900)
        self._build_ui(weights, settings)
        self.load_source(source)
        if calibration:
            self.set_calibration(calibration)

    def _build_ui(self, weights, settings):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("银浆溢出检测")
        title.setStyleSheet("font-size: 23px; font-weight: 600; padding: 6px;")
        layout.addWidget(title)
        help_text = QLabel("① 在模板上定义允许边界  →  ② 选择待测图片  →  ③ 分割、匹配找点、判断越界\n"
                           "闭合区域：按边缘顺序点击至少 3 点；单条直线：先点 2 个端点，再点允许的一侧。")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.controls = []

        row = QHBoxLayout()
        row.addWidget(QLabel("权重"))
        self.weights = QLineEdit(str(Path(weights).resolve()))
        row.addWidget(self.weights, 1)
        self._button(row, "选择权重…", self.choose_weights)
        row.addWidget(QLabel("银浆类别名"))
        self.silver_class = QLineEdit(settings.silver_class)
        self.silver_class.setMaximumWidth(140)
        self.silver_class.setToolTip("必须是权重中的真实银浆类别，不会把 bond 自动当作 silver")
        row.addWidget(self.silver_class)
        layout.addLayout(row)

        row = QHBoxLayout()
        self._button(row, "打开模板…", self.choose_template)
        self.mode = QComboBox()
        self.mode.addItem("闭合允许区域", "polygon")
        self.mode.addItem("直线 + 内侧点", "line")
        row.addWidget(self.mode)
        self._button(row, "撤销点", self.undo_point)
        self._button(row, "清空点", self.clear_points)
        self._button(row, "载入标定…", self.open_calibration)
        self._button(row, "保存标定…", self.save_boundary)
        self.point_count = QLabel("未设置模板")
        row.addWidget(self.point_count, 1)
        layout.addLayout(row)

        row = QHBoxLayout()
        self._button(row, "待测图片…", self.choose_source)
        self._button(row, "图片文件夹…", self.choose_folder)
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0, max(10000, settings.tolerance_px))
        self.tolerance.setValue(settings.tolerance_px)
        self.min_area = QSpinBox()
        self.min_area.setRange(1, max(100_000_000, settings.min_area_px))
        self.min_area.setValue(settings.min_area_px)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0, 1)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(self.segmentation_settings.confidence)
        self.image_size = QSpinBox()
        self.image_size.setRange(32, max(4096, self.segmentation_settings.image_size))
        self.image_size.setSingleStep(32)
        self.image_size.setValue(self.segmentation_settings.image_size)
        for label, widget in (("容差 px", self.tolerance), ("最小面积 px", self.min_area),
                              ("置信度", self.confidence), ("推理尺寸", self.image_size)):
            row.addWidget(QLabel(label))
            row.addWidget(widget)
        self.run_button = self._button(row, "开始检测当前图片", self.start_inspection)
        layout.addLayout(row)

        splitter = QSplitter()
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.select_image)
        splitter.addWidget(self.image_list)
        self.template_view, self.target_view = ImageView(allow_picking=True), ImageView()
        self.template_view.point_picked.connect(self.add_point)
        for label, view in (("模板 / 用户标定边界", self.template_view), ("待测图 / 越界结果", self.target_view)):
            panel = QWidget()
            column = QVBoxLayout(panel)
            column.addWidget(QLabel(label))
            column.addWidget(view, 1)
            splitter.addWidget(panel)
        splitter.setSizes([140, 600, 600])
        layout.addWidget(splitter, 1)
        self.details = QLabel("请打开模板并标定边界；不会自动把模板外框当作允许区域。")
        self.details.setWordWrap(True)
        self.details.setStyleSheet("padding: 8px; font-size: 14px;")
        layout.addWidget(self.details)
        row = QHBoxLayout()
        row.addWidget(QLabel("青色：边界  ·  绿色：银浆  ·  橙色：已被容差/面积过滤的越界  ·  红色：溢出缺陷"), 1)
        self.save_result_button = self._button(row, "保存对比图…", self.save_result)
        self.save_result_button.setEnabled(False)
        self._button(row, "适应窗口", self.fit_views)
        layout.addLayout(row)
        self.controls.extend((self.weights, self.silver_class, self.mode, self.tolerance, self.min_area,
                              self.confidence, self.image_size, self.image_list))
        for widget in (self.weights, self.silver_class):
            widget.textChanged.connect(self.invalidate)
        for widget in (self.tolerance, self.min_area, self.confidence, self.image_size):
            widget.valueChanged.connect(self.invalidate)
        self.mode.currentIndexChanged.connect(self.refresh_points)

    def _button(self, row, text, action):
        button = QPushButton(text)
        button.clicked.connect(action)
        row.addWidget(button)
        self.controls.append(button)
        return button

    def invalidate(self, *_):
        self.result = None
        self.save_result_button.setEnabled(False)
        self.target_view.set_image(None)
        self.details.setText("输入、标定或参数已变更，请重新检测。")

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
        self.invalidate()
        if 0 <= row < len(self.paths):
            try:
                self.target_view.set_image(read_image(self.paths[row]))
                self.statusBar().showMessage(str(self.paths[row]))
            except (ValueError, OSError) as exc:
                self.show_error(str(exc))

    def set_template(self, path):
        if self.worker:
            return
        image = read_image(path)
        self.template_path, self.template_image, self.points = Path(path).resolve(), image, []
        self.template_view.set_image(image)
        self.refresh_points()

    def set_calibration(self, calibration):
        if self.worker:
            return
        image = read_image(calibration.template_path)
        calibration.boundary.validate_image(image.shape)
        self.set_template(calibration.template_path)
        self.mode.setCurrentIndex(self.mode.findData(calibration.boundary.mode))
        self.points = list(calibration.boundary.points)
        self.refresh_points()

    def calibration(self):
        if self.template_image is None:
            raise ValueError("请先打开模板并选取边界点")
        boundary = Boundary(self.points, self.mode.currentData())
        boundary.validate_image(self.template_image.shape)
        return Calibration(self.template_path, boundary)

    def add_point(self, x, y):
        if self.worker or self.template_image is None:
            return
        if self.mode.currentData() == "line" and len(self.points) >= 3:
            self.statusBar().showMessage("直线模式只需 3 点，修改前请撤销或清空")
            return
        self.points.append((x, y))
        self.refresh_points()

    def undo_point(self):
        if self.points and not self.worker:
            self.points.pop()
            self.refresh_points()

    def clear_points(self):
        if not self.worker:
            self.points.clear()
            self.refresh_points()

    def refresh_points(self, *_):
        self.invalidate()
        outline = self.points if self.mode.currentData() == "polygon" else self.points[:2]
        self.template_view.set_markers(self.points, outline if len(outline) >= 2 else None)
        name = self.template_path.name if self.template_path else "未设置模板"
        self.point_count.setText(f"{name} · {len(self.points)} 点")

    def choose_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择参考模板", "", IMAGE_FILTER)
        if path:
            self._try(lambda: self.set_template(path))

    def choose_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择银浆分割权重", self.weights.text(), "权重 (*.pt)")
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

    def open_calibration(self):
        path, _ = QFileDialog.getOpenFileName(self, "载入边界标定", "", "JSON (*.json)")
        if path:
            self._try(lambda: self.set_calibration(load_calibration(path)))

    def save_boundary(self):
        try:
            calibration = self.calibration()
            path, _ = QFileDialog.getSaveFileName(self, "保存边界标定", "outputs/boundary.json", "JSON (*.json)")
            if path:
                save_calibration(path if Path(path).suffix else path + ".json", calibration,
                                 [*self.paths, Path(self.weights.text()).expanduser()])
                self.statusBar().showMessage(f"已保存标定：{path}")
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))

    def start_inspection(self):
        row = self.image_list.currentRow()
        if self.worker or not 0 <= row < len(self.paths):
            return
        try:
            calibration = self.calibration()
            settings = OverflowSettings(self.silver_class.text(), self.tolerance.value(), self.min_area.value())
            segmentation = replace(self.segmentation_settings, confidence=self.confidence.value(), image_size=self.image_size.value())
            weights = Path(self.weights.text()).expanduser().resolve()
            if not weights.is_file():
                raise ValueError(f"权重不存在：{weights}")
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))
            return
        self.invalidate()
        for widget in self.controls:
            widget.setEnabled(False)
        self.template_view.allow_picking = False
        self.statusBar().showMessage("正在分割、匹配和检查边界…")
        engine = self.engine if self.engine is not None and self.engine.weights == weights else None
        self.worker = InspectionWorker(weights, self.paths[row], calibration, settings, segmentation, engine, self)
        self.worker.result_ready.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.finished)
        self.worker.start()

    def show_result(self, result):
        self.result = result
        self.target_view.set_image(render_overlay(result))
        details = f"{STATUS[result.status]} · {result.message}"
        if result.measurement:
            m = result.measurement
            details += (f"\n银浆面积 {m.silver_area_px} px · 原始线外面积 {m.outside_area_px} px · "
                        f"有效溢出面积 {m.defect_area_px} px · 最大越界距离 {m.max_outside_distance_px:.2f} px")
        self.details.setText(details)
        self.statusBar().showMessage(f"{result.segmentation.image_path.name} · {result.elapsed_ms:.0f} ms（不含模型初始化）")

    def finished(self):
        self.engine = self.worker.engine
        self.worker.deleteLater()
        self.worker = None
        for widget in self.controls:
            widget.setEnabled(True)
        self.template_view.allow_picking = True
        self.save_result_button.setEnabled(self.result is not None)
        if self._closing:
            self.close()

    def save_result(self):
        if self.result is None or self.worker:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存原图和检测对比", "outputs/silver.comparison.png", "PNG (*.png)")
        if not path:
            return
        target = Path(path if Path(path).suffix else path + ".png").resolve()
        protected = {self.template_path, *self.paths, Path(self.weights.text()).expanduser().resolve()}
        if target in protected:
            self.show_error("不能覆盖输入图片、模板或权重")
            return
        self._try(lambda: write_image(target, render_comparison(self.result)))

    def fit_views(self):
        self.template_view.fit_image()
        self.target_view.fit_image()

    def _try(self, action):
        try:
            action()
        except (ValueError, OSError) as exc:
            self.show_error(str(exc))

    def show_error(self, message):
        self.details.setText(f"无法判定：{message}")
        self.statusBar().showMessage(message)
        if not self._closing:
            QMessageBox.warning(self, "银浆溢出检测", message)

    def closeEvent(self, event):
        if self.worker:
            self._closing = True
            self.statusBar().showMessage("等待当前推理完成后关闭…")
            event.ignore()
        else:
            event.accept()


def run_app(weights, source, settings, segmentation_settings, calibration=None, recursive=False):
    collect_images(source, recursive)
    if calibration:
        calibration.boundary.validate_image(read_image(calibration.template_path).shape)
    app = QApplication.instance() or QApplication([])
    window = InspectionWindow(weights, source, settings, segmentation_settings, calibration, recursive)
    window.show()
    return app.exec()
