import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generate_report", ROOT / "generate_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "2d"
        self.images.mkdir()
        for name in ("10.png", "2.png", "1.png"):
            Image.new("RGB", (100, 80), (40, 50, 70)).save(self.images / name)
        values = np.arange(100 * 80, dtype=np.uint16).reshape(80, 100) + 32768
        self.depth = self.root / "depth.png"
        Image.fromarray(values).save(self.depth)
        self.response = {"ok": True, "board_id": "board-1", "results_board": {
            "is_ok": False, "registration_ok": True,
            "pieces": [{"piece_id": 1, "is_ok": False, "image_2d_size": [100, 80],
                        "depth_start_row": 10, "depth_end_row": 30}]
        }}

    def tearDown(self):
        self.temp.cleanup()

    def test_natural_sort(self):
        self.assertEqual([p.name for p in report.image_files(self.images)], ["1.png", "2.png", "10.png"])

    def test_full_width_depth_piece(self):
        result = report.build_payload(self.response, base_dir=self.root, images_2d=self.images, depth_path=self.depth)
        piece = result["pieces"][0]
        self.assertEqual((piece["preview_3d"]["width"], piece["preview_3d"]["height"]), (100, 20))
        self.assertEqual(piece["depth_origin_row"], 10)
        self.assertTrue(piece["preview_2d"]["url"].startswith("data:image/jpeg;base64,"))

    def test_join_stream_diagnostics(self):
        reply = {"ok": True, "board_id": "board-1", "piece_id": 1, "results_2d": {
            "is_ok": False, "silver_overflow_2d": {"is_ok": False, "status": "overflow"}
        }}
        result = report.build_payload([reply, self.response], base_dir=self.root)
        self.assertFalse(result["pieces"][0]["results_2d"]["silver_overflow_2d"]["is_ok"])
        self.assertFalse(result["pieces"][0]["is_ok"])

    def test_missing_is_not_ok(self):
        result = report.build_payload({"ok": False, "error": "missing_2d_pieces"}, base_dir=self.root)
        self.assertEqual(result["pieces"], [])
        self.assertNotIn("is_ok", result["response"])

    def test_no_algorithm_import(self):
        self.assertFalse(any(name.startswith("algorithm.") for name in sys.modules))

    def test_mismatched_image_disables_overlay(self):
        self.response["results_board"]["pieces"][0]["image_2d_size"] = [200, 100]
        result = report.build_payload(self.response, base_dir=self.root, images_2d=self.images)
        self.assertFalse(result["pieces"][0]["overlay_2d_valid"])

    def test_json_escape(self):
        encoded = report.safe_script_json({"name": "</script><img onerror='alert(1)'>"})
        self.assertNotIn("</script>", encoded)
        self.assertEqual(json.loads(encoded)["name"], "</script><img onerror='alert(1)'>")

    def test_calibration_preview_preserves_original_coordinates_and_digest(self):
        document = {"response": {"ok": True, "results_2d": {"is_ok": None}},
                    "calibration": {"enabled": True, "template_size": [5320, 3032],
                                    "template_points": [[1234, 789], [1800, 789], [1800, 1600]],
                                    "template_sha256": "a" * 64}}
        payload = report.build_payload(document, base_dir=self.root)
        self.assertEqual(payload["calibration"], document["calibration"])
        html = report.render_report(document, self.root / "calibration.html", base_dir=self.root).read_text(encoding="utf-8")
        self.assertIn('id="calibration-export"', html)
        self.assertIn('id="show-silver-boundary"', html)

    def test_silver_boundary_and_registration_survive_report_join(self):
        diagnostic = {"status": "overflow", "is_ok": False,
                      "boundary": {"mode": "polygon", "coordinate_system": "image",
                                   "points": [[15, 10], [80, 10], [80, 60], [15, 60]]},
                      "registration": {"method": "sift_ransac", "matches": 42, "inliers": 40,
                                       "homography_template_to_image": [[1, 0, 7], [0, 1, 8], [0, 0, 1]]},
                      "measurement": {"defect_contours": [[[81, 25], [85, 25], [85, 30]]]}}
        reply = {"ok": True, "board_id": "board-1", "piece_id": 1,
                 "results_2d": {"is_ok": False, "silver_overflow_2d": diagnostic}}
        payload = report.build_payload([reply, self.response], base_dir=self.root)
        self.assertEqual(payload["pieces"][0]["results_2d"]["silver_overflow_2d"], diagnostic)

    def test_report_is_standalone_and_no_overwrite(self):
        target = self.root / "report.html"
        report.render_report(self.response, target, base_dir=self.root)
        html = target.read_text(encoding="utf-8")
        self.assertNotIn("__REPORT_JSON__", html)
        self.assertNotIn("__VIEWER_JS__", html)
        self.assertNotIn("__VIEWER_CSS__", html)
        self.assertIn('id="canvas2d"', html)
        self.assertIn('id="canvas3d"', html)
        with self.assertRaises(FileExistsError):
            report.render_report(self.response, target, base_dir=self.root)

    def test_invalid_depth_range_is_not_silently_cropped(self):
        self.response["results_board"]["pieces"][0]["depth_end_row"] = 999
        payload = report.build_payload(self.response, base_dir=self.root, depth_path=self.depth)
        self.assertNotIn("preview_3d", payload["pieces"][0])
        self.assertTrue(payload["warnings"])

    def test_reject_multiple_boards(self):
        reply = {"board_id": "board-2", "piece_id": 1, "results_2d": {"is_ok": True}}
        with self.assertRaises(ValueError):
            report.build_payload([reply, self.response], base_dir=self.root)

    def test_standalone_3d_points_survive(self):
        response = {"ok": True, "stage": "3d", "results_3d": {"is_ok": True, "pieces": [
            {"piece_id": 1, "is_ok": True, "points": [{"x": 15, "y": 8, "height_diff_mm": 0.14}]}
        ]}}
        result = report.build_payload(response, base_dir=self.root, depth_path=self.depth)
        self.assertEqual(result["pieces"][0]["points"][0]["height_diff_mm"], 0.14)
        self.assertIn("preview_3d", result["pieces"][0])


if __name__ == "__main__":
    unittest.main()
