"""选点配对、配置恢复、后台流程及旧结果失效的离屏 Qt 回归。"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem

from part_segmentation.image_io import write_image
from point_matcher.core import MappedPoint, MatchResult
from segmentation_transfer.app import TransferWindow
from segmentation_transfer.calibration_io import load_calibration
from segmentation_transfer.models import Calibration, HomographyFit, TransferResult
from segmentation_transfer.pipeline import transfer_instances
from segmentation_transfer.registration import TemplateRegistration
from test_transfer import source_result


class TransferGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if not QFontDatabase.families() and os.name == "nt":
            for name in ("msyh.ttc", "segoeui.ttf"):
                QFontDatabase.addApplicationFont(str(Path("C:/Windows/Fonts") / name))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.weights = self.root / "best.pt"
        self.weights.touch()
        self.window = TransferWindow(self.weights)
        for key in ("template_a", "template_b", "image_a", "image_b"):
            path = self.root / f"{key}.png"
            write_image(path, source_result().image)
            self.window.set_image_path(key, path)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.wait_finished()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def wait_finished(self):
        deadline = time.monotonic() + 10
        while self.window.worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(10)
        self.assertIsNone(self.window.worker)

    def prepare_pairs(self):
        for x, y in ((10, 10), (140, 10), (140, 90), (10, 90)):
            self.window.pick_a(x, y)
            self.window.pick_b(x, y)

    def fake_result(self):
        calibration = self.window.current_calibration()
        points = [MappedPoint(i + 1, *p, *p, True) for i, p in enumerate(calibration.points_a)]
        match = MatchResult("", "ok", "ok", points, 20, 20, 1, 0, np.eye(3).tolist(), [], (150, 100))
        source = source_result(self.window.paths["image_a"])
        return TransferResult(calibration, HomographyFit(np.eye(3), np.ones(4, dtype=bool), np.zeros(4)),
                              match, match, np.eye(3), np.zeros(4), source, self.window.paths["image_b"],
                              source.image, transfer_instances(source, np.eye(3), source.image.shape), ())

    def assert_preview(self, view, expected):
        items = [item for item in view.scene().items() if isinstance(item, QGraphicsPixmapItem)]
        self.assertEqual(len(items), 1, "已选择的图片应一直保留在预览中")
        image = items[0].pixmap().toImage()
        self.assertEqual((image.height(), image.width()), expected.shape[:2])
        for x, y in ((0, 0), (35, 25), (70, 50)):
            color = image.pixelColor(x, y)
            self.assertEqual((color.blue(), color.green(), color.red()), tuple(expected[y, x]))

    def test_both_input_previews_in_either_selection_order(self):
        a = np.full((100, 150, 3), (20, 80, 160), dtype=np.uint8)
        b = np.full((120, 210, 3), (170, 90, 30), dtype=np.uint8)
        write_image(self.root / "new_a.png", a)
        write_image(self.root / "new_b.png", b)
        for order in (("image_a", "image_b"), ("image_b", "image_a")):
            with self.subTest(order=order):
                for key in order:
                    path = self.root / ("new_a.png" if key == "image_a" else "new_b.png")
                    with patch("segmentation_transfer.app.QFileDialog.getOpenFileName", return_value=(str(path), "图片")):
                        self.window.choose_image(key)
                self.assertEqual(self.window.tabs.currentIndex(), 1)
                self.assert_preview(self.window.result_a, a)
                self.assert_preview(self.window.result_b, b)

    def test_replacing_one_input_clears_overlay_but_retains_other_original(self):
        self.prepare_pairs()
        self.window.show_result(self.fake_result())
        replacement = np.full((130, 190, 3), (35, 75, 180), dtype=np.uint8)
        path = self.root / "replacement.png"
        write_image(path, replacement)
        self.window.set_image_path("image_a", path)
        self.assertIsNone(self.window.result)
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertEqual(self.window.instance_table.rowCount(), 0)
        self.assert_preview(self.window.result_a, replacement)
        self.assert_preview(self.window.result_b, source_result().image)

    def test_parameter_changes_restore_originals_not_blank_views(self):
        self.prepare_pairs()
        result = self.fake_result()
        for update in (lambda: self.window.conf.setValue(.55),
                       lambda: self.window.weights.setText(str(self.root / "other.pt")),
                       lambda: self.window.pick_a(20, 20)):
            self.window.show_result(result)
            update()
            self.assertIsNone(self.window.result)
            self.assert_preview(self.window.result_a, source_result().image)
            self.assert_preview(self.window.result_b, source_result().image)

    def test_corrupt_replacement_retains_paths_previews_and_result(self):
        self.prepare_pairs()
        result = self.fake_result()
        self.window.show_result(result)
        old_paths = dict(self.window.paths)
        old_a = self.window.result_a.scene().items()[0].pixmap().toImage()
        old_b = self.window.result_b.scene().items()[0].pixmap().toImage()
        bad = self.root / "bad.png"
        bad.write_bytes(b"not an image")
        with patch("segmentation_transfer.app.QFileDialog.getOpenFileName", return_value=(str(bad), "图片")), \
                patch("segmentation_transfer.app.QMessageBox.warning") as warning:
            self.window.choose_image("image_b")
            warning.assert_called_once()
        self.assertEqual(self.window.paths, old_paths)
        self.assertIs(self.window.result, result)
        self.assertEqual(self.window.result_a.scene().items()[0].pixmap().toImage(), old_a)
        self.assertEqual(self.window.result_b.scene().items()[0].pixmap().toImage(), old_b)

    def test_pair_order_pending_and_undo(self):
        self.window.pick_b(10, 10)
        self.assertEqual(self.window.points_b, [])
        self.window.pick_a(10, 10)
        self.window.pick_a(20, 20)
        self.assertEqual(self.window.pending_a, (20, 20))
        with self.assertRaisesRegex(ValueError, "未配对"):
            self.window.current_calibration()
        self.window.pick_b(30, 30)
        self.assertEqual(self.window.points_a, [(20, 20)])
        self.assertEqual(self.window.points_b, [(30, 30)])
        self.window.pick_a(40, 40)
        self.window.undo_pair()
        self.assertEqual(len(self.window.points_a), 1)
        self.window.undo_pair()
        self.assertEqual(self.window.points_a, [])
        self.assertEqual(self.window.points_b, [])

    def drag_in_image(self, view, start, end):
        start_pixel = view.mapFromScene(QPointF(*start))
        end_pixel = view.mapFromScene(QPointF(*end))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start_pixel)
        QTest.mouseMove(view.viewport(), end_pixel)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end_pixel)
        self.app.processEvents()

    def test_alignment_tab_opacity_cache_and_invalidation(self):
        self.prepare_pairs()
        result = self.fake_result()
        b = np.full_like(result.target_image, 200)
        result = replace(result, target_image=b)
        self.window.show_result(result)
        panel = self.window.alignment_panel
        self.window.tabs.setCurrentIndex(2)
        self.app.processEvents()
        self.assertTrue(panel.controls.isEnabled())
        self.assert_preview(panel.view, np.full_like(b, 164))
        panel.view.scale(2, 2)
        transform = panel.view.transform()
        # 拖透明度不重新计算投影，更不能重新推理。
        with patch("segmentation_transfer.alignment_widget.prepare_alignment") as prepare:
            panel.presets[0].click()
            self.assert_preview(panel.view, result.source.image)
            panel.presets[2].click()
            self.assert_preview(panel.view, b)
            prepare.assert_not_called()
        self.assertEqual(panel.view.transform(), transform)
        self.assertIs(self.window.result, result)
        self.window.conf.setValue(.55)
        self.assertIsNone(panel.preview)
        self.assertEqual(panel.view.scene().items(), [])
        self.assertFalse(panel.controls.isEnabled())

    def test_identity_diagnostics_do_not_present_zero_inliers_as_failure(self):
        self.prepare_pairs()
        result = self.fake_result()
        registration = TemplateRegistration(result.source.image, result.calibration.points_a)
        match = registration.match(result.source.image)
        self.window.show_result(replace(result, match_a=match, match_b=match))
        text = self.window.diagnostics.text()
        self.assertIn("同图基准", text)
        self.assertIn("全部点误差", text)
        self.assertEqual(text.count("未运行 SIFT"), 2)
        self.assertNotIn("匹配内点 0", text)

    def test_alignment_not_affected_by_segmentation_display_options(self):
        self.prepare_pairs()
        result = self.fake_result()
        self.window.show_result(result)
        panel = self.window.alignment_panel
        original_preview = panel.preview
        self.window.alpha.setValue(90)
        self.window.show_points.setChecked(False)
        self.window.show_masks.setChecked(False)
        self.assertIs(panel.preview, original_preview)
        self.assert_preview(panel.view, result.source.image)
        self.window.set_image_path("image_b", self.window.paths["image_b"])
        self.assertIsNone(panel.preview)

    def test_draw_circle_at_zoom_and_refine_pending_before_pairing(self):
        self.window.tabs.setCurrentIndex(0)
        self.app.processEvents()
        self.window.annotation_editor.mode.setCurrentIndex(1)
        view = self.window.view_a
        view.scale(1.5, 1.5)
        view.centerOn(75, 50)
        self.drag_in_image(view, (75, 50), (95, 50))
        self.assertAlmostEqual(self.window.pending_a[0], 75, delta=.6)
        self.assertAlmostEqual(self.window.pending_a[1], 50, delta=.6)
        self.assertAlmostEqual(self.window.pending_radius_a, 20, delta=.6)
        self.assertEqual(self.window.points_a, [])
        self.assertEqual(self.window.annotation_editor.target.currentData(), ("a", 0))
        self.window.annotation_editor.x.setValue(74.125)
        self.window.annotation_editor.radius.setValue(21.375)
        self.assertAlmostEqual(self.window.pending_a[0], 74.125)
        self.assertAlmostEqual(self.window.pending_radius_a, 21.375)
        self.drag_in_image(self.window.view_b, (70, 50), (95, 50))
        self.assertIsNone(self.window.pending_a)
        self.assertEqual(len(self.window.points_a), 1)
        self.assertAlmostEqual(self.window.points_a[0][0], 74.125)
        self.assertAlmostEqual(self.window.radii_a[0], 21.375)
        self.assertAlmostEqual(self.window.radii_b[0], 25, delta=.6)
        calibration = self.window.current_calibration()
        self.assertEqual(len(calibration.points_a), 1, "一个圆只能提供一个对应圆心，不能冒充四个对应点")

    def test_circle_center_rim_drag_and_keyboard_fine_adjustment(self):
        self.window.tabs.setCurrentIndex(0)
        self.app.processEvents()
        self.window.annotation_editor.mode.setCurrentIndex(1)
        self.window.pick_circle_a(70, 50, 20)
        self.window.pick_circle_b(70, 50, 22)
        self.window.annotation_editor.mode.setCurrentIndex(2)
        view = self.window.view_a
        self.drag_in_image(view, (70, 50), (75, 55))
        x, y = self.window.points_a[0]
        self.assertAlmostEqual(x, 75, delta=.6)
        self.assertAlmostEqual(y, 55, delta=.6)
        self.assertAlmostEqual(self.window.radii_a[0], 20)
        self.assertEqual(self.window.points_b[0], (70, 50))
        self.drag_in_image(view, (x + 20, y), (x + 25, y))
        radius = self.window.radii_a[0]
        self.assertAlmostEqual(radius, 25, delta=.6)
        self.assertAlmostEqual(self.window.annotation_editor.radius.value(), radius, delta=.001)
        before = self.window.points_a[0]
        QTest.keyClick(view, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        QTest.keyClick(view, Qt.Key.Key_Down)
        QTest.keyClick(view, Qt.Key.Key_Equal, Qt.KeyboardModifier.ShiftModifier)
        self.assertAlmostEqual(self.window.points_a[0][0], before[0] + .1)
        self.assertAlmostEqual(self.window.points_a[0][1], before[1] + 1)
        self.assertAlmostEqual(self.window.radii_a[0], radius + .1)
        QTest.keyClick(view, Qt.Key.Key_Left, Qt.KeyboardModifier.ControlModifier)
        self.assertAlmostEqual(self.window.points_a[0][0], before[0] - 9.9)
        self.assertEqual(len(self.window.points_a), 1)

    def test_circle_cancel_mode_switch_undo_and_clear(self):
        self.window.annotation_editor.mode.setCurrentIndex(1)
        view = self.window.view_a
        start = view.mapFromScene(QPointF(70, 50))
        end = view.mapFromScene(QPointF(95, 50))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), end)
        QTest.keyClick(view, Qt.Key.Key_Escape)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
        self.assertIsNone(self.window.pending_a)
        self.assertIsNone(view._draft)
        self.window.pick_circle_a(70, 50, 20)
        self.window.annotation_editor.mode.setCurrentIndex(0)
        self.window.pick_b(70, 50)  # 防止待配圆被静默转成普通点。
        self.assertEqual(self.window.points_a, [])
        self.assertEqual(self.window.pending_radius_a, 20)
        self.window.undo_pair()
        self.assertIsNone(self.window.pending_radius_a)
        self.window.annotation_editor.mode.setCurrentIndex(1)
        self.window.pick_circle_a(70, 50, 20)
        self.window.pick_circle_b(70, 50, 21)
        self.window.undo_pair()
        self.assertEqual(self.window.radii_a, [])
        self.assertEqual(self.window.radii_b, [])
        self.window.pick_circle_a(70, 50, 20)
        self.window.clear_pairs()
        self.assertEqual(view._points, [])
        self.assertIsNone(self.window.selected_annotation)

    def test_numeric_edit_invalidates_results_and_preserves_original_previews(self):
        self.prepare_pairs()
        self.window.show_result(self.fake_result())
        self.window.select_table_annotation(0, 2)
        self.window.annotation_editor.x.setValue(11.125)
        self.window.annotation_editor.radius.setValue(5.375)
        self.assertEqual(self.window.points_a[0], (11.125, 10))
        self.assertEqual(self.window.radii_a[0], 5.375)
        self.assertEqual(self.window.points_b[0], (10, 10))
        self.assertIsNone(self.window.result)
        self.assertFalse(self.window.export_button.isEnabled())
        self.assert_preview(self.window.result_a, source_result().image)
        self.assert_preview(self.window.result_b, source_result().image)
        self.window.select_table_annotation(0, 3)
        self.window.annotation_editor.y.setValue(12.25)
        self.assertEqual(self.window.points_b[0], (10, 12.25))

    def test_circle_metadata_round_trip_and_radius_does_not_change_mapping_points(self):
        self.prepare_pairs()
        self.window.select_annotation(("a", 0))
        self.window.annotation_editor.radius.setValue(6.25)
        self.window.select_annotation(("b", 0))
        self.window.annotation_editor.radius.setValue(8.5)
        path = self.root / "circles.json"
        with patch("segmentation_transfer.app.QFileDialog.getSaveFileName", return_value=(str(path), "JSON")):
            self.window.save_pairs()
        calibration = load_calibration(path)
        self.window.clear_pairs()
        self.window.apply_calibration(calibration)
        self.assertEqual(self.window.radii_a, [6.25, None, None, None])
        self.assertEqual(self.window.radii_b, [8.5, None, None, None])
        self.assertEqual(self.window.points_a[0], (10, 10))
        self.window.set_image_path("template_b", self.window.paths["template_b"])
        self.assertEqual(self.window.radii_a, [])
        self.assertEqual(self.window.radii_b, [])

    def test_micro_adjustment_clamped_to_image_and_locked_when_busy(self):
        self.prepare_pairs()
        self.window.select_annotation(("a", 0))
        self.window.update_annotation("a", 0, -100, 300, 20)
        self.assertEqual(self.window.points_a[0], (0, 99))
        self.window.worker = object()
        try:
            self.window.set_busy(True)
            self.assertFalse(self.window.annotation_editor.isEnabled())
            self.window.update_annotation("a", 0, 20, 20, 10)
            self.assertEqual(self.window.points_a[0], (0, 99))
            self.window.select_annotation(("b", 0))
            self.assertEqual(self.window.selected_annotation, ("a", 0))
        finally:
            self.window.worker = None
            self.window.set_busy(False)

    def test_zoom_and_drag_use_original_coordinates(self):
        view = self.window.view_a
        view.scale(2, 2)
        view.centerOn(75, 50)
        position = view.mapFromScene(QPointF(75, 50))
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.assertAlmostEqual(self.window.pending_a[0], 75, delta=1)
        self.assertAlmostEqual(self.window.pending_a[1], 50, delta=1)
        self.window.undo_pair()
        start = view.mapFromScene(QPointF(60, 50))
        end = view.mapFromScene(QPointF(90, 50))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
        self.assertIsNone(self.window.pending_a)

    def test_save_load_and_replacing_template_clears_both_sides(self):
        self.prepare_pairs()
        path = self.root / "calibration.json"
        with patch("segmentation_transfer.app.QFileDialog.getSaveFileName", return_value=(str(path), "JSON")):
            self.window.save_pairs()
        self.assertTrue(path.is_file())
        calibration = load_calibration(path)
        self.window.set_image_path("template_a", self.window.paths["template_a"])
        self.assertEqual(self.window.points_a, [])
        self.assertEqual(self.window.points_b, [])
        self.window.apply_calibration(calibration)
        self.assertEqual(len(self.window.points_a), 4)
        self.assertEqual(self.window.pair_table.rowCount(), 4)

    def test_threaded_pipeline_and_result_invalidation(self):
        self.prepare_pairs()
        result = self.fake_result()
        with patch("segmentation_transfer.app.TransferPipeline") as pipeline:
            pipeline.return_value.run.return_value = result
            self.window.start_transfer()
            self.assertFalse(self.window.view_a.picking_enabled)
            self.assertFalse(self.window.run_button.isEnabled())
            self.assert_preview(self.window.result_a, source_result().image)
            self.assert_preview(self.window.result_b, source_result().image)
            self.window.pick_a(80, 80)
            self.assertIsNone(self.window.pending_a)
            self.wait_finished()
        self.assertTrue(self.window.export_button.isEnabled())
        self.assertEqual(self.window.instance_table.rowCount(), 1)
        self.window.alpha.setValue(60)
        self.assertIsNotNone(self.window.result)
        self.window.conf.setValue(.6)
        self.assertIsNone(self.window.result)
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertEqual(self.window.instance_table.rowCount(), 0)

    def test_worker_failure_and_pending_close(self):
        self.prepare_pairs()
        with patch("segmentation_transfer.app.TransferPipeline", side_effect=ValueError("failed")), \
                patch("segmentation_transfer.app.QMessageBox.warning") as warning:
            self.window.start_transfer()
            self.wait_finished()
            warning.assert_called_once()
        self.assertTrue(self.window.run_button.isEnabled())
        self.assert_preview(self.window.result_a, source_result().image)
        self.assert_preview(self.window.result_b, source_result().image)
        result = self.fake_result()

        def slow(*args):
            time.sleep(.1)
            return result

        with patch("segmentation_transfer.app.TransferPipeline") as pipeline:
            pipeline.return_value.run.side_effect = slow
            self.window.start_transfer()
            self.window.close()
            self.assertTrue(self.window._closing)
            self.wait_finished()
        self.assertFalse(self.window.isVisible())


if __name__ == "__main__":
    unittest.main()
