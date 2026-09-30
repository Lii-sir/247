"""Qt 离屏回归：真实线程、模拟推理，不打开桌面窗口。"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from part_segmentation.app import SegmentationWindow
from part_segmentation.image_io import write_image
from part_segmentation.models import SegmentationSettings
from test_segmentation import sample_result


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if os.name == "nt" and not QFontDatabase.families():
            fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
            for name in ("msyh.ttc", "segoeui.ttf"):
                QFontDatabase.addApplicationFont(str(fonts / name))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "图片.png"
        self.weights = self.root / "best.pt"
        self.weights.touch()
        write_image(self.path, sample_result().image)
        self.window = SegmentationWindow(self.weights, self.path, SegmentationSettings())
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

    def test_threaded_prediction_reuse_and_invalidation(self):
        with patch("part_segmentation.app.PartSegmenter") as engine:
            engine.return_value.weights = self.weights.resolve()
            engine.return_value.predict.return_value = sample_result(self.path)
            self.window.start_prediction()
            self.assertFalse(self.window.image_list.isEnabled())
            self.wait_finished()
            self.assertEqual(self.window.table.rowCount(), 1)
            self.assertTrue(self.window.save_button.isEnabled())
            self.window.alpha.setValue(70)
            self.assertIsNotNone(self.window.result)
            self.window.start_prediction()
            self.wait_finished()
            self.assertEqual(engine.call_count, 1)
            self.window.confidence.setValue(0.5)
            self.assertIsNone(self.window.result)
            self.assertEqual(self.window.table.rowCount(), 0)
            self.assertFalse(self.window.save_button.isEnabled())

    def test_worker_error_is_recoverable(self):
        with patch("part_segmentation.app.PartSegmenter", side_effect=ValueError("bad weights")), \
                patch("part_segmentation.app.QMessageBox.warning") as warning:
            self.window.start_prediction()
            self.wait_finished()
            warning.assert_called_once()
            self.assertTrue(self.window.run_button.isEnabled())
            self.assertFalse(self.window.save_button.isEnabled())

    def test_save_and_protect_original(self):
        self.window.show_result(sample_result(self.path))
        with patch("part_segmentation.app.QFileDialog.getSaveFileName", return_value=(str(self.path), "原图")), \
                patch("part_segmentation.app.QMessageBox.warning") as warning:
            self.window.save_result()
            warning.assert_called_once()
        output = self.root / "output" / "对比.png"
        with patch("part_segmentation.app.QFileDialog.getSaveFileName", return_value=(str(output), "原图")):
            self.window.save_result()
        self.assertTrue(output.is_file())

    def test_close_waits_for_worker(self):
        def slow_predict(*_):
            time.sleep(0.1)
            return sample_result(self.path)

        with patch("part_segmentation.app.PartSegmenter") as engine:
            engine.return_value.predict.side_effect = slow_predict
            self.window.start_prediction()
            self.window.close()
            self.assertTrue(self.window._closing)
            self.wait_finished()
            self.assertFalse(self.window.isVisible())


if __name__ == "__main__":
    unittest.main()
