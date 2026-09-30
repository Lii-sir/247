"""Offscreen interaction tests; inference stays mocked and deterministic."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from part_segmentation.image_io import write_image
from part_segmentation.models import SegmentationSettings
from silver_inspection.app import InspectionWindow
from silver_inspection.geometry import OverflowSettings
from silver_inspection.pipeline import evaluate
from test_silver_inspection import fixture


class SilverGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path, self.weights = self.root / "target.png", self.root / "model.pt"
        self.weights.write_bytes(b"mock weights")
        self.segmentation, self.match, self.boundary = fixture()
        write_image(self.path, self.segmentation.image)
        self.window = InspectionWindow(self.weights, self.path, OverflowSettings(), SegmentationSettings())
        self.window.set_template(self.path)
        for x, y in self.boundary.points:
            self.window.add_point(x, y)

    def tearDown(self):
        self.wait_for_worker()
        self.window.close()
        self.app.processEvents()
        self.tmp.cleanup()

    def wait_for_worker(self):
        deadline = time.monotonic() + 10
        while self.window.worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertIsNone(self.window.worker)

    def test_calibration_modes_and_invalidation(self):
        self.assertEqual(self.window.calibration().boundary, self.boundary)
        result = evaluate(self.segmentation, self.match, self.boundary, OverflowSettings())
        self.window.show_result(result)
        self.window.tolerance.setValue(2)
        self.assertIsNone(self.window.result)
        self.assertFalse(self.window.save_result_button.isEnabled())
        self.window.clear_points()
        self.window.mode.setCurrentIndex(1)
        for x, y in ((20, 10), (20, 30), (10, 20), (5, 5)):
            self.window.add_point(x, y)
        self.assertEqual(len(self.window.points), 3)
        self.assertEqual(self.window.calibration().boundary.mode, "line")
        self.window.undo_point()
        self.assertEqual(len(self.window.points), 2)

    def test_worker_reuses_model_and_locks_boundary(self):
        result = evaluate(self.segmentation, self.match, self.boundary, OverflowSettings())
        self.window.engine = SimpleNamespace(weights=self.weights.resolve(), class_names=("silver",))
        with patch("silver_inspection.app.SilverInspector") as factory:
            factory.return_value.inspect.return_value = result
            self.window.start_inspection()
            self.assertFalse(self.window.template_view.allow_picking)
            self.assertFalse(self.window.run_button.isEnabled())
            self.window.add_point(5, 5)
            self.assertEqual(len(self.window.points), 4)
            self.wait_for_worker()
            self.assertIs(factory.call_args.kwargs["segmenter"], self.window.engine)
        self.assertEqual(self.window.result.status, "overflow")
        self.assertTrue(self.window.save_result_button.isEnabled())
        self.assertTrue(self.window.template_view.allow_picking)

    def test_missing_class_failure_recovers_controls(self):
        self.window.engine = SimpleNamespace(weights=self.weights.resolve(), class_names=("bond", "wire"))
        with patch("silver_inspection.app.QMessageBox.warning") as warning:
            self.window.start_inspection()
            self.wait_for_worker()
            self.assertIn("bond, wire", warning.call_args.args[2])
        self.assertIsNone(self.window.result)
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertFalse(self.window.save_result_button.isEnabled())

    def test_saves_overlay_but_never_overwrites_input(self):
        self.window.show_result(evaluate(self.segmentation, self.match, self.boundary, OverflowSettings()))
        original = self.path.read_bytes()
        with patch("silver_inspection.app.QFileDialog.getSaveFileName", return_value=(str(self.path), "PNG")), \
             patch("silver_inspection.app.QMessageBox.warning") as warning:
            self.window.save_result()
            warning.assert_called_once()
        self.assertEqual(original, self.path.read_bytes())
        output = self.root / "comparison.png"
        with patch("silver_inspection.app.QFileDialog.getSaveFileName", return_value=(str(output), "PNG")):
            self.window.save_result()
        self.assertTrue(output.is_file())

    def test_close_waits_for_worker_and_finishes_cleanly(self):
        result = evaluate(self.segmentation, self.match, self.boundary, OverflowSettings())
        self.window.engine = SimpleNamespace(weights=self.weights.resolve(), class_names=("silver",))
        def inspect(path):
            time.sleep(.1)
            return result
        with patch("silver_inspection.app.SilverInspector") as factory:
            factory.return_value.inspect.side_effect = inspect
            self.window.show()
            self.window.start_inspection()
            self.window.close()
            self.assertTrue(self.window._closing)
            self.wait_for_worker()
        self.assertFalse(self.window.isVisible())


if __name__ == "__main__":
    unittest.main()
