"""Desktop UI: select template points, then locate them in an image or folder."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QAbstractItemView,
)

# Retain direct script launch as well as python -m point_matcher.app.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.image_io import IMAGE_FILTER, collect_images, read_image, write_image
from common.matching import MatchResult, MatchSettings, TemplateMatcher, annotate_image
from common.widgets.point_view import PointImageView as ImageView
from point_matcher.export import export_csv, export_json

STATUS_TEXT = {"ok": "成功", "partial": "部分点越界", "outside": "全部点越界", "error": "未找到"}


class MatchWorker(QThread):
    result_ready = Signal(int, object)
    progress = Signal(int, int)
    failed = Signal(str)

    def __init__(self, template, points, paths, settings, parent=None):
        super().__init__(parent)
        self.template = template
        self.points = list(points)
        self.paths = list(paths)
        self.settings = settings

    def run(self):
        try:
            matcher = TemplateMatcher(self.template, self.points, self.settings)
            for index, path in enumerate(self.paths):
                if self.isInterruptionRequested():
                    break
                result = matcher.match_path(path)
                self.result_ready.emit(index, result)
                self.progress.emit(index + 1, len(self.paths))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("模板点定位 · Point Matcher")
        self.resize(1380, 930)
        self.setMinimumSize(1000, 740)
        self.template_image = None
        self.template_path = ""
        self.points = []
        self.target_paths = []
        self.results = {}
        self.worker = None
        self._closing = False
        self._run_error = ""
        self._build_ui()
        self._update_actions()

    @property
    def busy(self):
        # Remain busy until the GUI handles finished, not just until run() returns.
        return self.worker is not None

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(22, 18, 22, 14)
        layout.setSpacing(12)
        self.setCentralWidget(central)
        title = QLabel("模板点定位")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("01  打开模板并选点    →    02  选择目标图片或文件夹    →    03  找到对应位置")
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)

        input_panel = QFrame()
        input_panel.setObjectName("panel")
        inputs = QGridLayout(input_panel)
        self.template_entry = QLineEdit()
        self.template_entry.setReadOnly(True)
        self.template_entry.setPlaceholderText("模板路径 · 左侧图片上可选择任意多个点")
        self.template_button = QPushButton("打开模板图")
        self.template_button.clicked.connect(self.choose_template)
        inputs.addWidget(QLabel("模板"), 0, 0)
        inputs.addWidget(self.template_entry, 0, 1, 1, 2)
        inputs.addWidget(self.template_button, 0, 3)
        self.target_entry = QLineEdit()
        self.target_entry.setPlaceholderText("选择目标，或粘贴图片 / 文件夹路径")
        self.target_entry.textChanged.connect(self._target_changed)
        self.image_button = QPushButton("选择目标图")
        self.image_button.clicked.connect(self.choose_target_image)
        self.folder_button = QPushButton("选择文件夹")
        self.folder_button.clicked.connect(self.choose_target_folder)
        inputs.addWidget(QLabel("目标"), 1, 0)
        inputs.addWidget(self.target_entry, 1, 1)
        inputs.addWidget(self.image_button, 1, 2)
        inputs.addWidget(self.folder_button, 1, 3)
        inputs.setColumnStretch(1, 1)
        layout.addWidget(input_panel)

        action_bar = QHBoxLayout()
        self.undo_button = QPushButton("撤销上一点")
        self.undo_button.clicked.connect(self.undo_point)
        self.clear_button = QPushButton("清空选点")
        self.clear_button.clicked.connect(self.clear_points)
        self.point_count = QLabel("已选 0 个点")
        self.recursive_check = QCheckBox("包含子文件夹")
        self.recursive_check.toggled.connect(self._target_changed)
        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(0.4, 0.9)
        self.ratio_spin.setSingleStep(0.05)
        self.ratio_spin.setValue(0.7)
        self.ratio_spin.setToolTip("越小越严格；默认 0.70。修改后需重新找点。")
        self.ratio_spin.valueChanged.connect(self._settings_changed)
        self.start_button = QPushButton("开始找点")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start_matching)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_matching)
        for widget in (self.undo_button, self.clear_button, self.point_count):
            action_bar.addWidget(widget)
        action_bar.addStretch()
        action_bar.addWidget(self.recursive_check)
        action_bar.addSpacing(14)
        action_bar.addWidget(QLabel("匹配比值"))
        action_bar.addWidget(self.ratio_spin)
        action_bar.addWidget(self.start_button)
        action_bar.addWidget(self.stop_button)
        layout.addLayout(action_bar)

        self.template_view = ImageView(True)
        self.target_view = ImageView(False)
        self.template_view.point_picked.connect(self.add_point)
        views = QSplitter(Qt.Orientation.Horizontal)
        views.addWidget(self._image_panel("模板 / 点击选点", self.template_view))
        views.addWidget(self._image_panel("目标 / 定位结果", self.target_view))
        views.setSizes([650, 650])

        self.image_table = self._table(["目标图片", "状态", "内点 / 匹配", "说明"])
        self.image_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.image_table.itemSelectionChanged.connect(self.show_selected_target)
        self.image_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.image_table.setColumnWidth(0, 160)
        self.image_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.point_table = self._table(["点", "模板 X", "模板 Y", "目标 X", "目标 Y", "状态"])
        self.point_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tables = QSplitter(Qt.Orientation.Horizontal)
        tables.addWidget(self.image_table)
        tables.addWidget(self.point_table)
        tables.setSizes([650, 650])
        vertical = QSplitter(Qt.Orientation.Vertical)
        vertical.addWidget(views)
        vertical.addWidget(tables)
        vertical.setSizes([470, 170])
        layout.addWidget(vertical, 1)

        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("等待开始")
        self.export_button = QPushButton("导出坐标 CSV / JSON")
        self.export_button.clicked.connect(self.export_results)
        self.save_image_button = QPushButton("保存当前标注图")
        self.save_image_button.clicked.connect(self.save_annotated_image)
        bottom.addWidget(self.progress, 1)
        bottom.addWidget(self.export_button)
        bottom.addWidget(self.save_image_button)
        layout.addLayout(bottom)
        hint = QLabel("左键点击选点 / 拖动平移 · 滚轮缩放 · 坐标为原图像素，左上角 (0, 0) · 适用于同一平面目标")
        hint.setObjectName("muted")
        layout.addWidget(hint)
        self.statusBar().showMessage("请先打开模板图片")
        self._config_widgets = [self.template_button, self.image_button, self.folder_button,
                                self.target_entry, self.recursive_check, self.ratio_spin]

    @staticmethod
    def _table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().hide()
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        return table

    @staticmethod
    def _image_panel(title, view):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        row = QHBoxLayout()
        heading = QLabel(title)
        heading.setObjectName("heading")
        coordinates = QLabel("")
        coordinates.setObjectName("muted")
        view.position_changed.connect(coordinates.setText)
        fit_button = QPushButton("适应窗口")
        fit_button.clicked.connect(view.fit_image)
        row.addWidget(heading)
        row.addStretch()
        row.addWidget(coordinates)
        row.addWidget(fit_button)
        layout.addLayout(row)
        layout.addWidget(view)
        return panel

    def choose_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模板图片", "", IMAGE_FILTER)
        if path:
            self.load_template(path)

    def load_template(self, path):
        if self.busy:
            return
        try:
            image = read_image(path)
        except Exception as exc:
            QMessageBox.warning(self, "模板读取失败", str(exc))
            return
        self.template_image = image
        self.template_path = str(Path(path).resolve())
        self.template_entry.setText(self.template_path)
        self.points = []
        self.template_view.set_image(image)
        self._points_changed()
        self.statusBar().showMessage(f"模板 {image.shape[1]} × {image.shape[0]} · 请在左侧图片上点击选点")

    def choose_target_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择目标图片", "", IMAGE_FILTER)
        if path:
            self.target_entry.setText(path)
            self.load_targets()

    def choose_target_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择目标文件夹")
        if path:
            self.target_entry.setText(path)
            self.load_targets()

    def _target_changed(self, *_):
        if self.busy:
            return
        self.target_paths = []
        self.results.clear()
        self.image_table.setRowCount(0)
        self.target_view.set_image(None)
        self.progress.setValue(0)
        self.progress.setFormat("等待开始")
        self.refresh_point_table()
        self._update_actions()

    def _settings_changed(self, *_):
        if not self.busy:
            self._invalidate_results()

    def load_targets(self):
        try:
            raw_path = self.target_entry.text().strip().strip('"')
            if not raw_path:
                raise ValueError("请先选择目标图片或文件夹")
            paths = collect_images(raw_path, self.recursive_check.isChecked())
            if not paths:
                raise ValueError("文件夹中没有支持的图片。图片在子目录中时请勾选「包含子文件夹」。")
        except Exception as exc:
            QMessageBox.warning(self, "目标读取失败", str(exc))
            return False
        self.target_paths = paths
        self.results.clear()
        self.image_table.blockSignals(True)
        self.image_table.setRowCount(len(paths))
        base = Path(raw_path)
        for index, path in enumerate(paths):
            name = str(path.relative_to(base.resolve())) if base.is_dir() else path.name
            self._set_image_row(index, [name, "待处理", "—", ""])
            self.image_table.item(index, 0).setToolTip(str(path))
        self.image_table.blockSignals(False)
        self.image_table.selectRow(0)
        self.show_selected_target()
        self.statusBar().showMessage(f"已加载 {len(paths)} 张目标图片")
        self._update_actions()
        return True

    def _set_image_row(self, row, texts):
        for column, text in enumerate(texts):
            item = self.image_table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self.image_table.setItem(row, column, item)
            item.setText(str(text))

    def add_point(self, x, y):
        if not self.busy and self.template_image is not None:
            self.points.append((x, y))
            self._points_changed()

    def undo_point(self):
        if self.points and not self.busy:
            self.points.pop()
            self._points_changed()

    def clear_points(self):
        if not self.busy:
            self.points.clear()
            self._points_changed()

    def _points_changed(self):
        self.point_count.setText(f"已选 {len(self.points)} 个点")
        self.template_view.set_markers(self.points)
        self._invalidate_results()

    def _invalidate_results(self):
        self.results.clear()
        self.target_view.set_markers([])
        for row in range(self.image_table.rowCount()):
            name = self.image_table.item(row, 0).text()
            self._set_image_row(row, [name, "待处理", "—", ""])
        self.progress.setValue(0)
        self.progress.setFormat("等待开始")
        self.refresh_point_table()
        self._update_actions()

    def _update_actions(self):
        active = not self.busy
        self.start_button.setEnabled(active and self.template_image is not None and bool(self.points) and bool(self.target_entry.text().strip()))
        self.undo_button.setEnabled(active and bool(self.points))
        self.clear_button.setEnabled(active and bool(self.points))
        self.stop_button.setEnabled(not active and not self.worker.isInterruptionRequested())
        self.export_button.setEnabled(active and bool(self.results))
        selected = self.results.get(self.image_table.currentRow())
        self.save_image_button.setEnabled(active and selected is not None and bool(selected.points))
        self.template_view.allow_picking = active
        for widget in self._config_widgets:
            widget.setEnabled(active)

    def start_matching(self):
        if self.busy:
            return
        if self.template_image is None or not self.points:
            QMessageBox.information(self, "还未选点", "请打开模板图并选择至少一个点。")
            return
        if not self.load_targets():
            return
        self._run_error = ""
        self.progress.setRange(0, len(self.target_paths))
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m 张")
        self.worker = MatchWorker(self.template_image, self.points, self.target_paths,
                                  MatchSettings(ratio_threshold=self.ratio_spin.value()), self)
        self.worker.result_ready.connect(self._on_result)
        self.worker.progress.connect(lambda current, total: self.progress.setValue(current))
        self.worker.failed.connect(self._on_failure)
        self.worker.finished.connect(self._on_finished)
        self._update_actions()
        self.statusBar().showMessage("正在提取模板特征并找点；可切换查看图片，停止将在当前图片处理后生效…")
        self.worker.start()

    def _on_result(self, index, result):
        self.results[index] = result
        name = self.image_table.item(index, 0).text()
        self._set_image_row(index, [name, STATUS_TEXT[result.status],
                                   f"{result.inliers} / {result.good_matches}", result.message])
        color = "#047857" if result.status == "ok" else "#b45309" if result.points else "#b91c1c"
        self.image_table.item(index, 1).setForeground(QColor(color))
        if self.image_table.currentRow() == index:
            self.show_selected_target()

    def _on_failure(self, message):
        self._run_error = message

    def _on_finished(self):
        worker = self.worker
        cancelled = worker.isInterruptionRequested()
        self.worker = None
        worker.deleteLater()
        for index in range(len(self.target_paths)):
            if index not in self.results:
                self.image_table.item(index, 1).setText("未处理")
                self.image_table.item(index, 3).setText(self._run_error or "用户已停止")
        successful = sum(result.status == "ok" for result in self.results.values())
        uncertain = sum(result.status in {"partial", "outside"} for result in self.results.values())
        failed = sum(result.status == "error" for result in self.results.values())
        prefix = "已停止" if cancelled else "处理失败" if self._run_error else "已完成"
        self.statusBar().showMessage(f"{prefix} · 已处理 {len(self.results)}/{len(self.target_paths)} 张 · 成功 {successful} · 越界 {uncertain} · 未找到 {failed}")
        self._update_actions()
        if self._closing:
            self.close()
        elif self._run_error:
            QMessageBox.warning(self, "找点未完成", self._run_error)

    def stop_matching(self):
        if self.worker:
            self.worker.requestInterruption()
            self.stop_button.setEnabled(False)
            self.statusBar().showMessage("正在停止：等待当前图片处理结束，已完成的结果将保留。")

    def show_selected_target(self):
        row = self.image_table.currentRow()
        if not 0 <= row < len(self.target_paths):
            return
        try:
            self.target_view.set_image(read_image(self.target_paths[row]))
            result = self.results.get(row)
            if result:
                self.target_view.set_markers([(point.target_x, point.target_y) for point in result.points], result.template_outline)
        except Exception as exc:
            self.target_view.set_image(None)
            self.statusBar().showMessage(str(exc))
        self.refresh_point_table()
        self._update_actions()

    def refresh_point_table(self):
        result = self.results.get(self.image_table.currentRow())
        self.point_table.setRowCount(len(self.points))
        for index, (x, y) in enumerate(self.points):
            mapped = result.points[index] if result and index < len(result.points) else None
            values = [f"P{index + 1}", f"{x:.2f}", f"{y:.2f}",
                      f"{mapped.target_x:.2f}" if mapped else "—",
                      f"{mapped.target_y:.2f}" if mapped else "—",
                      ("图内" if mapped.inside_image else "图外") if mapped else "未找到" if result else "待定位"]
            for column, value in enumerate(values):
                self.point_table.setItem(index, column, QTableWidgetItem(value))

    def export_results(self):
        if self.busy or not self.results:
            return
        path, selected_filter = QFileDialog.getSaveFileName(self, "导出已处理图片的定位结果", "point_results.csv", "CSV 坐标表 (*.csv);;JSON 完整结果 (*.json)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".json" if selected_filter.startswith("JSON") else ".csv"
        if Path(path).suffix.lower() not in {".csv", ".json"}:
            QMessageBox.warning(self, "导出格式错误", "请使用 .csv 或 .json 扩展名，不能将坐标结果写入图片文件。")
            return
        try:
            results = [self.results[index] for index in sorted(self.results)]
            if Path(path).suffix.lower() == ".json":
                export_json(path, self.template_path, self.points, results)
            else:
                export_csv(path, results, self.points)
            self.statusBar().showMessage(f"已导出 {len(results)} 张图片的结果：{path}")
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def save_annotated_image(self):
        result = self.results.get(self.image_table.currentRow())
        if self.busy or not result or not result.points:
            return
        suggested = Path(result.image_path).stem + "_points.png"
        path, _ = QFileDialog.getSaveFileName(self, "保存当前标注图（不覆盖原图）", suggested, "PNG 图片 (*.png)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".png"
        protected_paths = {Path(self.template_path).resolve(), *(p.resolve() for p in self.target_paths)}
        if Path(path).resolve() in protected_paths:
            QMessageBox.warning(self, "请另存为", "不能覆盖模板或目标原图，请使用新的文件名。")
            return
        try:
            write_image(path, annotate_image(read_image(result.image_path), result))
            self.statusBar().showMessage(f"已保存标注图：{path}")
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))

    def closeEvent(self, event):
        if self.busy:
            if self._closing or QMessageBox.question(self, "正在找点", "停止处理并关闭窗口？当前图片处理完成后将退出。") == QMessageBox.StandardButton.Yes:
                self._closing = True
                self.stop_matching()
            event.ignore()
        else:
            event.accept()


STYLE = """
QMainWindow, QWidget { font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif; font-size: 12px; color: #24344d; }
QMainWindow { background: #eef2f7; }
QLabel#title { font-size: 25px; font-weight: 700; color: #132743; }
QLabel#heading { font-size: 13px; font-weight: 700; }
QLabel#muted { color: #697b94; }
QFrame#panel { background: white; border: 1px solid #dce4ef; border-radius: 8px; }
QPushButton { padding: 7px 13px; background: white; border: 1px solid #cbd6e5; border-radius: 5px; }
QPushButton:hover { background: #eff6ff; border-color: #6b9cf5; }
QPushButton:pressed { background: #dceaff; }
QPushButton:disabled { color: #a1adbd; background: #f3f5f8; border-color: #e0e6ee; }
QPushButton#primary { background: #2463eb; color: white; font-weight: 700; border-color: #2463eb; padding: 8px 23px; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { background: #adc3ed; border-color: #adc3ed; }
QLineEdit, QDoubleSpinBox { background: white; border: 1px solid #cbd6e5; padding: 6px; border-radius: 4px; }
QLineEdit:read-only { color: #697b94; background: #f7f9fc; }
QTableWidget { background: white; alternate-background-color: #f5f8fc; gridline-color: #e5ebf3; border: 1px solid #dce4ef; selection-background-color: #dbeafe; selection-color: #163f83; }
QHeaderView::section { background: #e9eff7; padding: 7px; font-weight: 600; border: none; border-right: 1px solid #dae3ef; }
QProgressBar { border: 1px solid #d5e0ed; border-radius: 5px; background: white; text-align: center; min-height: 25px; }
QProgressBar::chunk { background: #bdd4ff; border-radius: 4px; }
QStatusBar { background: #e5edf7; color: #49617e; }
QSplitter::handle { background: transparent; width: 8px; height: 8px; }
"""


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
