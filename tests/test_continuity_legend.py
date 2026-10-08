"""Continuity legend and configured occlusion regression tests (no inference)."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from part_segmentation.models import SegmentationSettings
from silver_inspection.continuity import ContinuitySettings, measure_continuity
from silver_inspection.continuity_app import ContinuityWindow
from test_silver_continuity import make_segments


class ContinuityLegendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_legend_matches_settings_on_start_and_edit(self):
        with patch.object(ContinuityWindow, "load_source"):
            window = ContinuityWindow("model.pt", "unused", ContinuitySettings(
                occlusion_classes=("thin", "bond", "wire")), SegmentationSettings())
        try:
            for text, expected in (("thin,bond,wire", ("thin", "bond", "wire")),
                                   (" wire， thin , ", ("wire", "thin")),
                                   ("", ())):
                with self.subTest(text=text):
                    window.occlusion_classes.setText(text)
                    self.assertEqual(window.settings().occlusion_classes, expected)
                    label = "/".join(expected) + " 忽略" if expected else "未配置忽略类别"
                    self.assertIn("黄色：" + label, window.legend.text())
        finally:
            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_wire_is_ignored_only_when_configured(self):
        segments, settings = make_segments(occluded_sector=4)
        segments = (*segments[:2], replace(segments[2], class_name="wire"))
        without_wire = measure_continuity(segments, (140, 140, 3), settings)
        with_wire = measure_continuity(segments, (140, 140, 3), replace(
            settings, occlusion_classes=("thin", "bond", "wire")))
        self.assertTrue(without_wire.is_disconnected)
        self.assertFalse(with_wire.is_disconnected)
        self.assertGreater(with_wire.occluded_area_px, 0)


if __name__ == "__main__":
    unittest.main()
