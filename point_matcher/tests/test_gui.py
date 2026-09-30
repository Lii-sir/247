from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2 as cv
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from point_matcher.app import MainWindow, STYLE
from point_matcher.core import MatchResult, read_image, write_image
from point_matcher.tests.test_core import textured_image


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # Windows' offscreen plugin has no system font database by default.
        if not QFontDatabase.families() and os.name == "nt":
            font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
            for font_name in ("msyh.ttc", "msyhbd.ttc", "segoeui.ttf"):
                QFontDatabase.addApplicationFont(str(font_dir / font_name))
        cls.app.setStyle("Fusion")
        cls.app.setStyleSheet(STYLE)
        cv.setNumThreads(2)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="匹配界面_")
        self.root = Path(self.directory.name)
        self.template = textured_image()
        write_image(self.root / "模板.png", self.template)
        self.targets = self.root / "目标"
        self.targets.mkdir()
        matrix = np.array([[1, 0.05, 60], [0.02, 1, 30], [0.00005, 0, 1]], dtype=float)
        self.target = cv.warpPerspective(self.template, matrix, (850, 600))
        write_image(self.targets / "01_正常.png", self.target)
        (self.targets / "02_损坏.jpg").write_bytes(b"broken")
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        if self.window.worker:
            self.window.worker.requestInterruption()
            self.wait_finished()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.directory.cleanup()

    def wait_finished(self):
        deadline = time.monotonic() + 30
        while self.window.busy and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(20)
        self.assertFalse(self.window.busy, "Worker did not finish")

    def prepare(self):
        self.window.load_template(self.root / "模板.png")
        self.app.processEvents()
        for x, y in [(100, 100), (300, 210), (500, 330)]:
            position = self.window.template_view.mapFromScene(QPointF(x, y))
            QTest.mouseClick(self.window.template_view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.window.target_entry.setText(str(self.targets))
        self.app.processEvents()

    def test_complete_folder_workflow_and_invalidation(self):
        self.assertFalse(self.window.start_button.isEnabled())
        self.prepare()
        self.assertEqual(len(self.window.points), 3)
        self.assertAlmostEqual(self.window.points[0][0], 100, delta=2)
        self.assertTrue(self.window.start_button.isEnabled())
        QTest.mouseClick(self.window.start_button, Qt.MouseButton.LeftButton)
        self.assertFalse(self.window.template_button.isEnabled())
        self.wait_finished()
        self.assertEqual(len(self.window.results), 2)
        self.assertEqual(self.window.results[0].status, "ok")
        self.assertEqual(self.window.results[1].status, "error")
        self.assertTrue(self.window.export_button.isEnabled())
        self.assertNotEqual(self.window.point_table.item(0, 3).text(), "—")
        self.assertEqual(len(self.window.target_view._markers), 3)
        self.window.image_table.selectRow(1)
        self.assertEqual(self.window.point_table.item(0, 3).text(), "—")
        self.assertFalse(self.window.save_image_button.isEnabled())
        self.window.image_table.selectRow(0)
        QTest.mouseClick(self.window.undo_button, Qt.MouseButton.LeftButton)
        self.assertEqual(len(self.window.points), 2)
        self.assertFalse(self.window.results)
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertEqual(len(self.window.target_view._markers), 0)

    def test_single_image_and_export_actions(self):
        self.prepare()
        self.window.target_entry.setText(str(self.targets / "01_正常.png"))
        self.window.start_matching()
        self.wait_finished()
        self.assertEqual(len(self.window.results), 1)
        with patch("point_matcher.app.QFileDialog.getSaveFileName", return_value=(str(self.root / "坐标.csv"), "CSV 坐标表 (*.csv)")):
            self.window.export_results()
        self.assertTrue((self.root / "坐标.csv").is_file())
        with patch("point_matcher.app.QFileDialog.getSaveFileName", return_value=(str(self.root / "坐标.json"), "JSON 完整结果 (*.json)")):
            self.window.export_results()
        self.assertTrue((self.root / "坐标.json").is_file())
        with patch("point_matcher.app.QFileDialog.getSaveFileName", return_value=(str(self.targets / "01_正常.png"), "CSV 坐标表 (*.csv)")), patch("point_matcher.app.QMessageBox.warning") as warning:
            self.window.export_results()
            warning.assert_called_once()
        with patch("point_matcher.app.QFileDialog.getSaveFileName", return_value=(str(self.root / "标注.png"), "PNG 图片 (*.png)")):
            self.window.save_annotated_image()
        self.assertTrue((self.root / "标注.png").is_file())
        with patch("point_matcher.app.QFileDialog.getSaveFileName", return_value=(str(self.targets / "01_正常.png"), "PNG 图片 (*.png)")), patch("point_matcher.app.QMessageBox.warning") as warning:
            self.window.save_annotated_image()
            warning.assert_called_once()
        np.testing.assert_array_equal(read_image(self.targets / "01_正常.png"), self.target)

    def test_zoomed_selection_uses_original_coordinates(self):
        self.window.load_template(self.root / "模板.png")
        self.app.processEvents()
        self.window.template_view.scale(2, 2)
        self.window.template_view.centerOn(300, 210)
        position = self.window.template_view.mapFromScene(QPointF(300, 210))
        QTest.mouseClick(self.window.template_view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.assertAlmostEqual(self.window.points[0][0], 300, delta=1)
        self.assertAlmostEqual(self.window.points[0][1], 210, delta=1)

    def test_clear_and_new_template_reset_results(self):
        self.prepare()
        QTest.mouseClick(self.window.clear_button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.points, [])
        self.assertFalse(self.window.start_button.isEnabled())
        self.window.add_point(40, 60)
        self.window.load_template(self.root / "模板.png")
        self.assertEqual(self.window.points, [])
        self.assertEqual(self.window.point_table.rowCount(), 0)

    def test_dragging_does_not_add_a_point(self):
        self.window.load_template(self.root / "模板.png")
        self.app.processEvents()
        view = self.window.template_view
        start = view.mapFromScene(QPointF(250, 200))
        end = view.mapFromScene(QPointF(330, 200))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
        self.assertEqual(self.window.points, [])

    def test_empty_folder_does_not_start_worker(self):
        self.prepare()
        empty = self.root / "empty"
        empty.mkdir()
        self.window.target_entry.setText(str(empty))
        with patch("point_matcher.app.QMessageBox.warning") as warning:
            self.window.start_matching()
            warning.assert_called_once()
        self.assertFalse(self.window.busy)
        self.assertFalse(self.window.results)

    def test_cancellation_preserves_completed_results(self):
        self.prepare()
        for i in range(3, 12):
            write_image(self.targets / f"{i:02d}.png", self.target)

        def slow_match(matcher, path):
            time.sleep(0.08)
            return MatchResult(str(path), "error", "test failure")

        with patch("point_matcher.app.TemplateMatcher.match_path", slow_match):
            self.window.start_matching()
            deadline = time.monotonic() + 15
            while not self.window.results and self.window.busy and time.monotonic() < deadline:
                self.app.processEvents()
                QTest.qWait(10)
            self.window.stop_matching()
            self.wait_finished()
        self.assertGreater(len(self.window.results), 0)
        self.assertLess(len(self.window.results), len(self.window.target_paths))
        self.assertTrue(self.window.export_button.isEnabled())

    def test_bad_template_reports_error_and_unlocks_controls(self):
        blank_path = self.root / "空白.png"
        write_image(blank_path, np.full_like(self.template, 255))
        self.window.load_template(blank_path)
        self.window.add_point(100, 100)
        self.window.target_entry.setText(str(self.targets))
        with patch("point_matcher.app.QMessageBox.warning") as warning:
            self.window.start_matching()
            self.wait_finished()
            warning.assert_called_once()
        self.assertTrue(self.window.template_button.isEnabled())
        self.assertFalse(self.window.results)


if __name__ == "__main__":
    unittest.main()
