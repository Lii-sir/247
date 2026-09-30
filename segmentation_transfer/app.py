"""双模板配对与映射预览。几何/推理/导出均委托独立模块。"""

from dataclasses import asdict
from math import hypot, isfinite
from pathlib import Path

import cv2 as cv
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QDoubleSpinBox, QFileDialog,
    QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QSlider, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from part_segmentation.image_io import read_image
from part_segmentation.models import SegmentationSettings
from part_segmentation.widgets import ImageView
from point_matcher.core import MatchSettings

from .calibration_io import load_calibration, save_calibration
from .export import export_result
from .geometry import fit_template_mapping
from .diagnostics import calibration_summary, transfer_summary
from .models import Calibration, MappingSettings
from .pipeline import LazyYoloSegmenter, TransferPipeline
from .visualization import render_views
from .widgets import PointImageView
from .annotation_editor import AnnotationEditor
from .alignment_widget import AlignmentPanel

IMAGE_FILTER = "图片 (*.bmp *.png *.jpg *.jpeg *.tif *.tiff *.webp)"
STATUS = {"ok": "已映射", "clipped": "边界裁切", "empty": "为空/越界"}


class TransferWorker(QThread):
    result_ready = Signal(object)
    progress = Signal(str)
    failed = Signal(str)

    def __init__(self, calibration, image_a, image_b, weights, segmentation, mapping, parent=None):
        super().__init__(parent)
        self.calibration, self.image_a, self.image_b = calibration, image_a, image_b
        self.weights, self.segmentation, self.mapping = weights, segmentation, mapping

    def run(self):
        try:
            self.progress.emit("校验模板标定并准备定位…")
            pipeline = TransferPipeline(self.calibration, mapping_settings=self.mapping)
            result = pipeline.run(self.image_a, self.image_b, LazyYoloSegmenter(self.weights),
                                  self.segmentation, self.progress.emit)
            self.result_ready.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class TransferWindow(QMainWindow):
    def __init__(self, weights, segmentation=None, mapping=None):
        super().__init__()
        self.paths = {}
        self.template_images = {}
        self.input_images = {}  # A1/B1 原图预览独立于可失效的分割/映射结果。
        self.points_a, self.points_b = [], []
        self.radii_a, self.radii_b = [], []
        self.pending_a = None
        self.pending_radius_a = None
        self.selected_annotation = None
        self.result = None
        self.worker = None
        self._closing = False
        self.result_metadata = {}
        self._segmentation = segmentation or SegmentationSettings()
        self._mapping = mapping or MappingSettings()
        self.setWindowTitle("跨模板分割映射 · A → B → A1 / B1")
        self.resize(1440, 960)
        self._build_ui(weights)

    def _build_ui(self, weights):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        heading = QLabel("跨模板分割映射  ·  A1 → A模板 → B模板 → B1")
        heading.setStyleSheet("font-size: 20px; font-weight: 600; padding: 4px;")
        layout.addWidget(heading)
        paths = QGridLayout()
        self.path_edits, self.input_buttons = {}, []
        for row, (key, title) in enumerate((("template_a", "A 模板"), ("template_b", "B 模板"),
                                           ("image_a", "图片 A1"), ("image_b", "图片 B1"))):
            paths.addWidget(QLabel(title), row, 0)
            entry = QLineEdit()
            entry.setReadOnly(True)
            entry.setPlaceholderText(f"请选择{title}")
            self.path_edits[key] = entry
            paths.addWidget(entry, row, 1)
            button = QPushButton("选择图片…")
            button.clicked.connect(lambda checked=False, key=key: self.choose_image(key))
            self.input_buttons.append(button)
            paths.addWidget(button, row, 2)
        paths.addWidget(QLabel("分割权重"), 4, 0)
        self.weights = QLineEdit(str(Path(weights).resolve()))
        self.weights.textChanged.connect(self.invalidate_result)
        paths.addWidget(self.weights, 4, 1)
        weight_button = QPushButton("选择权重…")
        weight_button.clicked.connect(self.choose_weights)
        self.input_buttons.append(weight_button)
        paths.addWidget(weight_button, 4, 2)
        layout.addLayout(paths)

        controls = QHBoxLayout()
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0, 1)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(self._segmentation.confidence)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(32, max(4096, self._segmentation.image_size))
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(self._segmentation.image_size)
        self.device = QLineEdit(self._segmentation.device)
        self.device.setMaximumWidth(70)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, max(1000, self._mapping.ransac_threshold))
        self.threshold.setValue(self._mapping.ransac_threshold)
        for title, widget in (("置信度", self.conf), ("推理尺寸", self.imgsz), ("设备", self.device),
                              ("人工点容差(px)", self.threshold)):
            controls.addWidget(QLabel(title))
            controls.addWidget(widget)
        for widget in (self.conf, self.imgsz, self.threshold):
            widget.valueChanged.connect(self.invalidate_result)
        self.device.textChanged.connect(self.invalidate_result)
        self.run_button = QPushButton("匹配 A1/B1 并映射分割")
        self.run_button.clicked.connect(self.start_transfer)
        controls.addWidget(self.run_button)
        controls.addStretch()
        self.export_button = QPushButton("导出结果…")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_current)
        controls.addWidget(self.export_button)
        layout.addLayout(controls)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        calibration_page = QWidget()
        cal_layout = QVBoxLayout(calibration_page)
        self.pair_hint = QLabel("先加载两张模板，再依次点击 A 模板的点和 B 模板的对应点；至少 4 对，建议 6–12 对。")
        self.pair_hint.setWordWrap(True)
        cal_layout.addWidget(self.pair_hint)
        self.annotation_editor = AnnotationEditor()
        self.annotation_editor.mode_changed.connect(self.set_annotation_mode)
        self.annotation_editor.selection_changed.connect(self.select_annotation)
        self.annotation_editor.values_changed.connect(self.update_selected_annotation)
        cal_layout.addWidget(self.annotation_editor)
        self.view_a, self.view_b = PointImageView(), PointImageView()
        self.view_a.point_picked.connect(self.pick_a)
        self.view_b.point_picked.connect(self.pick_b)
        self.view_a.circle_picked.connect(self.pick_circle_a)
        self.view_b.circle_picked.connect(self.pick_circle_b)
        self.view_a.annotation_selected.connect(lambda index: self.select_annotation(("a", index)))
        self.view_b.annotation_selected.connect(lambda index: self.select_annotation(("b", index)))
        self.view_a.annotation_changed.connect(lambda index, x, y, radius: self.update_annotation("a", index, x, y, radius))
        self.view_b.annotation_changed.connect(lambda index, x, y, radius: self.update_annotation("b", index, x, y, radius))
        cal_layout.addWidget(self.make_views(("A 模板 · 先选点", self.view_a), ("B 模板 · 再选对应点", self.view_b)), 1)
        actions = QHBoxLayout()
        self.calibration_buttons = []
        for title, callback in (("撤销上一对 / 待配点", self.undo_pair), ("清空配对", self.clear_pairs),
                                ("检查对应关系", self.check_calibration), ("保存标定…", self.save_pairs),
                                ("加载标定…", self.load_pairs)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            actions.addWidget(button)
            self.calibration_buttons.append(button)
        fit = QPushButton("适应窗口")
        fit.clicked.connect(lambda: (self.view_a.fit_image(), self.view_b.fit_image()))
        actions.addWidget(fit)
        cal_layout.addLayout(actions)
        self.pair_table = self.make_table(["点对", "类型", "A 模板 (x, y/R)", "B 模板 (x, y/R)", "B 模板误差(px)", "标定状态"])
        self.pair_table.setMaximumHeight(180)
        self.pair_table.cellClicked.connect(self.select_table_annotation)
        cal_layout.addWidget(self.pair_table)
        self.tabs.addTab(calibration_page, "1. 人工模板配对")

        result_page = QWidget()
        result_layout = QVBoxLayout(result_page)
        note = QLabel("B1 显示的是 A1 掩膜的几何映射，不是 B1 的 YOLO 检测；源置信度不代表映射精度。")
        note.setWordWrap(True)
        result_layout.addWidget(note)
        self.result_a, self.result_b = ImageView(), ImageView()
        result_layout.addWidget(self.make_views(("A1 · 原图 / 分割与定位点", self.result_a),
                                                 ("B1 · 原图 / 映射掩膜与定位点", self.result_b)), 1)
        display = QHBoxLayout()
        self.alpha = QSlider(Qt.Orientation.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(45)
        self.alpha.setMaximumWidth(180)
        display.addWidget(QLabel("掩膜不透明度"))
        display.addWidget(self.alpha)
        self.show_masks, self.show_points = QCheckBox("显示掩膜"), QCheckBox("显示定位点/模板边界")
        for control in (self.show_masks, self.show_points):
            control.setChecked(True)
            control.toggled.connect(self.redraw_result)
            display.addWidget(control)
        self.alpha.valueChanged.connect(self.redraw_result)
        fit_result = QPushButton("适应窗口")
        fit_result.clicked.connect(lambda: (self.result_a.fit_image(), self.result_b.fit_image()))
        display.addWidget(fit_result)
        display.addStretch()
        result_layout.addLayout(display)
        self.diagnostics = QLabel("尚未运行")
        self.diagnostics.setWordWrap(True)
        result_layout.addWidget(self.diagnostics)
        self.instance_table = self.make_table(["源实例", "类别", "A1 置信度", "A1 面积", "B1 面积", "源像素入视野比例", "状态"])
        self.instance_table.setMaximumHeight(170)
        result_layout.addWidget(self.instance_table)
        self.tabs.addTab(result_page, "2. 分割映射结果")
        self.alignment_panel = AlignmentPanel()
        self.tabs.addTab(self.alignment_panel, "3. 对齐检查 · B1 叠加 A1")
        self.statusBar().showMessage("两个模板中相同编号的点必须是人工确认的对应位置")

    @staticmethod
    def make_views(*pairs):
        splitter = QSplitter()
        for title, view in pairs:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QLabel(title))
            layout.addWidget(view, 1)
            splitter.addWidget(panel)
        splitter.setSizes([650, 650])
        return splitter

    @staticmethod
    def make_table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        return table

    def choose_image(self, key):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", IMAGE_FILTER)
        if path:
            try:
                self.set_image_path(key, Path(path))
                self.tabs.setCurrentIndex(0 if key.startswith("template") else 1)
            except (ValueError, OSError, cv.error) as exc:
                self.show_error(str(exc))

    def set_image_path(self, key, path):
        path = Path(path).resolve()
        image = read_image(path)  # 读取成功后才替换原状态。
        self.paths[key] = path
        self.path_edits[key].setText(str(path))
        if key.startswith("template"):
            self.template_images[key] = image
            (self.view_a if key == "template_a" else self.view_b).set_image(image)
            self.clear_pairs()  # 任意模板改变，成对清除，不留下错误的另一侧配对。
        else:
            self.input_images[key] = image
            self.invalidate_result()
            (self.result_a if key == "image_a" else self.result_b).set_image(image)

    def choose_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择分割权重", self.weights.text(), "权重 (*.pt)")
        if path:
            self.weights.setText(path)

    def pick_a(self, x, y):
        if self.worker is not None or len(self.template_images) != 2 or self.annotation_editor.mode.currentData() != "point":
            return
        self.pending_a = (x, y)
        self.pending_radius_a = None
        self.selected_annotation = ("a", len(self.points_a))
        self.invalidate_result()
        self.update_pairs()

    def pick_b(self, x, y):
        if self.worker is not None or self.annotation_editor.mode.currentData() != "point":
            return
        if self.pending_a is None:
            self.statusBar().showMessage("请先在 A 模板点击一个点，再点击 B 模板对应位置")
            return
        if self.pending_radius_a is not None:
            self.statusBar().showMessage("待配 A 标注是圆，请切回圆标注完成 B 圆，或撤销后重新选择")
            return
        self.points_a.append(self.pending_a)
        self.points_b.append((x, y))
        self.radii_a.append(None)
        self.radii_b.append(None)
        self.pending_a = None
        self.pending_radius_a = None
        self.selected_annotation = ("b", len(self.points_b) - 1)
        self.invalidate_result()
        self.update_pairs()

    def pick_circle_a(self, x, y, radius):
        if self.worker is not None or len(self.template_images) != 2 or self.annotation_editor.mode.currentData() != "circle":
            return
        self.pending_a = (x, y)
        self.pending_radius_a = radius
        self.selected_annotation = ("a", len(self.points_a))
        self.invalidate_result()
        self.update_pairs()

    def pick_circle_b(self, x, y, radius):
        if self.worker is not None or self.annotation_editor.mode.currentData() != "circle":
            return
        if self.pending_a is None or self.pending_radius_a is None:
            self.statusBar().showMessage("请先在 A 模板拖出一个圆，再在 B 模板拖出对应圆")
            return
        self.points_a.append(self.pending_a)
        self.points_b.append((x, y))
        self.radii_a.append(self.pending_radius_a)
        self.radii_b.append(radius)
        self.pending_a = None
        self.pending_radius_a = None
        self.selected_annotation = ("b", len(self.points_b) - 1)
        self.invalidate_result()
        self.update_pairs()

    def undo_pair(self):
        if self.worker is not None:
            return
        if self.pending_a is not None:
            self.pending_a = None
            self.pending_radius_a = None
        elif self.points_a:
            self.points_a.pop()
            self.points_b.pop()
            self.radii_a.pop()
            self.radii_b.pop()
        self.selected_annotation = None
        self.invalidate_result()
        self.update_pairs()

    def clear_pairs(self):
        self.view_a.cancel_gesture()
        self.view_b.cancel_gesture()
        self.points_a.clear()
        self.points_b.clear()
        self.radii_a.clear()
        self.radii_b.clear()
        self.pending_a = None
        self.pending_radius_a = None
        self.selected_annotation = None
        self.invalidate_result()
        self.update_pairs()

    def update_pairs(self, fit=None):
        points_a = self.points_a + ([self.pending_a] if self.pending_a is not None else [])
        radii_a = self.radii_a + ([self.pending_radius_a] if self.pending_a is not None else [])
        entries = [(f"A P{i + 1}" + ("（待配对）" if i == len(self.points_a) else ""), ("a", i))
                   for i in range(len(points_a))]
        entries += [(f"B P{i + 1}", ("b", i)) for i in range(len(self.points_b))]
        if self.selected_annotation not in [key for _, key in entries]:
            self.selected_annotation = None
        selected = self.selected_annotation
        self.annotation_editor.set_targets(entries, selected)
        self.view_a.set_points(points_a, radii_a, selected[1] if selected and selected[0] == "a" else -1)
        self.view_b.set_points(self.points_b, self.radii_b, selected[1] if selected and selected[0] == "b" else -1)
        if selected:
            side, index = selected
            points, radii = (points_a, radii_a) if side == "a" else (self.points_b, self.radii_b)
            self.annotation_editor.set_values(points[index], radii[index], self.template_images[f"template_{side}"].shape)
        else:
            self.annotation_editor.set_values(None)
        self.pair_table.setRowCount(len(self.points_a))
        for row, (a, b) in enumerate(zip(self.points_a, self.points_b, strict=True)):
            ra, rb = self.radii_a[row], self.radii_b[row]
            kind = "圆" if ra is not None or rb is not None else "点"
            a_text = f"({a[0]:.2f}, {a[1]:.2f}" + (f" / R={ra:.2f})" if ra is not None else ")")
            b_text = f"({b[0]:.2f}, {b[1]:.2f}" + (f" / R={rb:.2f})" if rb is not None else ")")
            values = (f"P{row + 1}", kind, a_text, b_text,
                      f"{fit.errors[row]:.3f}" if fit else "—",
                      ("内点" if fit.inliers[row] else "未参与拟合") if fit else "待检查")
            for column, value in enumerate(values):
                self.pair_table.setItem(row, column, QTableWidgetItem(value))
        pending = "请在 B 模板完成对应圆/点（再次点击 A 可修改待配点）" if self.pending_a is not None else "当前工具：点点击；圆拖出圆心和半径；微调可拖动或用键盘"
        self.pair_hint.setText(f"已完成 {len(self.points_a)} 对 · {pending} · 滚轮缩放 / 拖动平移 · 至少 4 对非共线点")

    def current_calibration(self):
        if not all(key in self.paths for key in ("template_a", "template_b")):
            raise ValueError("请先加载 A、B 两张模板")
        if self.pending_a is not None:
            raise ValueError("还有一个 A 模板点未配对，请在 B 模板选点或撤销")
        return Calibration(self.paths["template_a"], self.paths["template_b"], tuple(self.points_a), tuple(self.points_b),
                           tuple(self.radii_a), tuple(self.radii_b))

    def set_annotation_mode(self, mode):
        self.view_a.set_mode(mode)
        self.view_b.set_mode(mode)
        self.update_pairs()

    def select_annotation(self, selection):
        if self.worker is not None:
            return
        self.selected_annotation = selection
        self.update_pairs()

    def select_table_annotation(self, row, column):
        if self.worker is None:
            self.annotation_editor.mode.setCurrentIndex(2)
            self.select_annotation(("b" if column == 3 else "a", row))

    def update_selected_annotation(self, x, y, radius):
        selection = self.annotation_editor.target.currentData()
        if selection:
            self.update_annotation(selection[0], selection[1], x, y, radius)

    def update_annotation(self, side, index, x, y, radius):
        if self.worker is not None or side not in ("a", "b"):
            return
        points = self.points_a if side == "a" else self.points_b
        radii = self.radii_a if side == "a" else self.radii_b
        pending = side == "a" and self.pending_a is not None and index == len(points)
        if not pending and not 0 <= index < len(points):
            return
        if not all(isfinite(value) for value in (x, y)) or (radius is not None and not isfinite(radius)):
            return
        height, width = self.template_images[f"template_{side}"].shape[:2]
        point = (min(max(float(x), 0), width - 1), min(max(float(y), 0), height - 1))
        radius = min(float(radius), hypot(width, height)) if radius is not None and radius > 0 else None
        if pending:
            self.pending_a, self.pending_radius_a = point, radius
        else:
            points[index], radii[index] = point, radius
        self.selected_annotation = (side, index)
        self.invalidate_result()

    def mapping_settings(self):
        return MappingSettings(self.threshold.value(), self._mapping.min_inlier_ratio)

    def check_calibration(self):
        try:
            calibration = self.current_calibration()
            fit = fit_template_mapping(calibration.points_a, calibration.points_b,
                                       self.template_images["template_a"].shape, self.template_images["template_b"].shape,
                                       self.mapping_settings())
            self.update_pairs(fit)
            self.statusBar().showMessage(calibration_summary(fit) + "；低误差不等于未标注区域也准确")
        except (ValueError, cv.error) as exc:
            self.show_error(str(exc))

    def save_pairs(self):
        try:
            calibration = self.current_calibration()
            path, _ = QFileDialog.getSaveFileName(self, "保存人工配对标定", "calibration.json", "JSON (*.json)")
            if path:
                target = Path(path).with_suffix(".json")
                if target.resolve() in [*self.paths.values(), Path(self.weights.text()).resolve()]:
                    raise ValueError("不能覆盖输入图片或权重")
                save_calibration(target, calibration, self.mapping_settings())
                self.statusBar().showMessage(f"标定已保存：{target}")
        except (ValueError, OSError, cv.error) as exc:
            self.show_error(str(exc))

    def load_pairs(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载标定", "", "JSON (*.json)")
        if path:
            try:
                self.apply_calibration(load_calibration(Path(path)))
            except (ValueError, OSError, cv.error) as exc:
                self.show_error(str(exc))

    def apply_calibration(self, calibration):
        # 先验证两个文件，失败时不破坏现有标定。
        a, b = read_image(calibration.template_a), read_image(calibration.template_b)
        self.paths.update(template_a=calibration.template_a, template_b=calibration.template_b)
        self.template_images.update(template_a=a, template_b=b)
        for key in ("template_a", "template_b"):
            self.path_edits[key].setText(str(self.paths[key]))
        self.view_a.set_image(a)
        self.view_b.set_image(b)
        self.points_a, self.points_b = list(calibration.points_a), list(calibration.points_b)
        self.radii_a = list(calibration.radii_a or (None,) * len(self.points_a))
        self.radii_b = list(calibration.radii_b or (None,) * len(self.points_b))
        self.pending_a = None
        self.pending_radius_a = None
        self.selected_annotation = None
        self.invalidate_result()
        self.update_pairs()
        self.tabs.setCurrentIndex(0)

    def invalidate_result(self, *_):
        had_result = self.result is not None
        self.result = None
        self.result_metadata = {}
        if hasattr(self, "result_a"):
            # 清除旧叠加结果时恢复两侧原图；尚未推理时不重建图片、不改变缩放。
            # 新选图片由 set_image_path 单独更新，另一侧始终保持可见。
            if had_result:
                self.result_a.set_image(self.input_images.get("image_a"), preserve_view=True)
                self.result_b.set_image(self.input_images.get("image_b"), preserve_view=True)
            self.instance_table.setRowCount(0)
            self.diagnostics.setText("当前显示已选原图；输入/参数已改变，请重新运行")
            self.export_button.setEnabled(False)
            self.alignment_panel.clear()
            self.update_pairs()  # 清除旧误差/内点标记。

    def set_busy(self, busy):
        for widget in [*self.input_buttons, *self.calibration_buttons, self.weights,
                       self.conf, self.imgsz, self.device, self.threshold, self.run_button]:
            widget.setEnabled(not busy)
        self.view_a.picking_enabled = self.view_b.picking_enabled = not busy
        self.annotation_editor.setEnabled(not busy)
        if busy:
            self.view_a.cancel_gesture()
            self.view_b.cancel_gesture()
        self.export_button.setEnabled(not busy and self.result is not None)

    def start_transfer(self):
        if self.worker is not None:
            return
        try:
            calibration = self.current_calibration()
            if not all(key in self.paths for key in ("image_a", "image_b")):
                raise ValueError("请选择 A1 和 B1 两张实际图片")
            segmentation = SegmentationSettings(self.conf.value(), self._segmentation.iou,
                                                self.imgsz.value(), self.device.text())
            mapping = self.mapping_settings()
            weights = Path(self.weights.text()).expanduser().resolve()
            if not weights.is_file():
                raise ValueError("权重文件不存在")
            # 快速校验选点，不在主线程执行 SIFT 或 YOLO。
            fit_template_mapping(calibration.points_a, calibration.points_b,
                                 self.template_images["template_a"].shape, self.template_images["template_b"].shape, mapping)
        except (ValueError, OSError, cv.error) as exc:
            self.show_error(str(exc))
            return
        self.invalidate_result()
        self.result_metadata = {"weights": str(weights), "segmentation": asdict(segmentation),
                                "mapping": asdict(mapping), "matching": asdict(MatchSettings())}
        self.set_busy(True)
        self.worker = TransferWorker(calibration, self.paths["image_a"], self.paths["image_b"], weights, segmentation, mapping, self)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.result_ready.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def worker_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.set_busy(False)
        if self._closing:
            self.close()

    def show_result(self, result):
        self.result = result
        self.alignment_panel.set_result(result)
        self.update_pairs(result.fit)
        self.redraw_result()
        self.result_a.fit_image()
        self.result_b.fit_image()
        self.instance_table.setRowCount(len(result.instances))
        for row, item in enumerate(result.instances):
            values = (f"#{item.source_id}", item.class_name, f"{item.source_confidence:.3f}",
                      str(item.source_area), str(item.area), f"{item.source_coverage:.1%}", STATUS[item.status])
            for col, value in enumerate(values):
                self.instance_table.setItem(row, col, QTableWidgetItem(value))
        diagnostics = transfer_summary(result)
        if result.warnings:
            diagnostics += "\n" + "\n".join(result.warnings)
        self.diagnostics.setText(diagnostics)
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage("映射完成；第 3 页“对齐检查”可将 B1 原图叠加到 A1 上，调节透明度核对边缘")
        self.export_button.setEnabled(self.worker is None)

    def redraw_result(self, *_):
        if self.result is not None:
            a, b = render_views(self.result, self.alpha.value() / 100, self.show_masks.isChecked(), self.show_points.isChecked())
            self.result_a.set_image(a, preserve_view=True)
            self.result_b.set_image(b, preserve_view=True)

    def export_current(self):
        if self.result is None:
            return
        parent = QFileDialog.getExistingDirectory(self, "选择结果的父目录（自动新建 transfer_001 等子目录）")
        if not parent:
            return
        number = 1
        while (Path(parent) / f"transfer_{number:03d}").exists():
            number += 1
        try:
            output = export_result(Path(parent) / f"transfer_{number:03d}", self.result, self.result_metadata)
            self.statusBar().showMessage(f"已导出：{output}")
        except (ValueError, OSError, cv.error) as exc:
            self.show_error(str(exc))

    def show_error(self, message):
        self.statusBar().showMessage(message)
        if not self._closing:
            QMessageBox.warning(self, "跨模板分割映射", message)

    def closeEvent(self, event):
        if self.worker is not None:
            self._closing = True
            self.statusBar().showMessage("等待当前匹配/推理完成后安全关闭…")
            event.ignore()
        else:
            event.accept()


def run_app(weights, calibration=None, image_a=None, image_b=None, segmentation=None, mapping=None):
    app = QApplication.instance() or QApplication([])
    window = TransferWindow(weights, segmentation, mapping)
    if calibration is not None:
        window.apply_calibration(calibration)
    for key, path in (("image_a", image_a), ("image_b", image_b)):
        if path is not None:
            window.set_image_path(key, path)
    window.show()
    return app.exec()
