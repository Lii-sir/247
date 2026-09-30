"""不依赖真实权重的分割单元测试。真实推理通过 CLI 单独验收。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from part_segmentation.export import export_batch
from part_segmentation.image_io import collect_images, read_image, write_image
from part_segmentation.inference import PartSegmenter
from part_segmentation.models import Segment, SegmentationResult, SegmentationSettings
from part_segmentation.visualization import class_color, render_comparison, render_overlay


def sample_result(path=Path("sample.png")):
    image = np.full((120, 160, 3), 100, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[20:100, 20:140] = True
    mask[45:75, 60:100] = False  # 孔洞不应被填充。
    segment = Segment(0, "bond", 0.9, (20, 20, 140, 100), mask)
    return SegmentationResult(path, image, (segment,), 10)


def tensor(array):
    value = Mock()
    value.cpu.return_value.numpy.return_value = np.asarray(array)
    return value


class SegmentationTests(unittest.TestCase):
    def test_settings_validation(self):
        for values in ({"confidence": -0.1}, {"confidence": float("nan")}, {"iou": 0},
                       {"image_size": 33}, {"image_size": 0}, {"device": " "}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                SegmentationSettings(**values)

    def test_unicode_io_and_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = sample_result().image
            path = root / "中文图片.PNG"
            write_image(path, image)
            write_image(root / "子目录" / "第二张.bmp", image)
            np.testing.assert_array_equal(read_image(path), image)
            self.assertEqual(collect_images(root), [path])
            self.assertEqual(len(collect_images(root, recursive=True)), 2)
            self.assertEqual(collect_images(path), [path])

    def test_invalid_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                collect_images(root)
            with self.assertRaises(ValueError):
                collect_images(root / "missing")
            path = root / "broken.bmp"
            path.write_bytes(b"broken")
            with self.assertRaises(ValueError):
                read_image(path)
            with self.assertRaises(ValueError):
                write_image(root / "invalid.pt", sample_result().image)

    def test_render_preserves_original_holes_and_background(self):
        result = sample_result()
        original = result.image.copy()
        overlay = render_overlay(result, alpha=1, show_labels=False)
        np.testing.assert_array_equal(result.image, original)
        np.testing.assert_array_equal(overlay[30, 30], class_color(0))
        np.testing.assert_array_equal(overlay[60, 80], original[60, 80])
        np.testing.assert_array_equal(overlay[0, 0], original[0, 0])
        self.assertEqual(result.segments[0].area, 80 * 120 - 30 * 40)

    def test_no_detections_and_comparison(self):
        sample = sample_result()
        empty = SegmentationResult(sample.image_path, sample.image, (), 0)
        np.testing.assert_array_equal(render_overlay(empty), sample.image)
        comparison = render_comparison(empty)
        self.assertEqual(comparison.shape[1], sample.image.shape[1] * 2)
        np.testing.assert_array_equal(comparison[-120:, :160], sample.image)

    def test_invalid_overlay(self):
        with self.assertRaises(ValueError):
            render_overlay(sample_result(), alpha=2)

    def test_inference_adapter(self):
        sample = sample_result()
        engine = PartSegmenter.__new__(PartSegmenter)
        # 一个轻量替身，模拟 Ultralytics 张量接口，不加载 Torch。
        class Boxes:
            xyxy = tensor([[20, 20, 140, 100]])
            cls = tensor([0])
            conf = tensor([0.9])

            def __len__(self):
                return 1

        prediction = SimpleNamespace(boxes=Boxes(), masks=SimpleNamespace(data=tensor([sample.segments[0].mask])), names={0: "bond"})
        engine._model = Mock()
        engine._model.predict.return_value = [prediction]
        with patch("part_segmentation.inference.read_image", return_value=sample.image):
            result = engine.predict("test.png", SegmentationSettings())
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].mask.dtype, np.bool_)
        self.assertEqual(result.segments[0].area, sample.segments[0].area)
        self.assertTrue(engine._model.predict.call_args.kwargs["retina_masks"])
        self.assertFalse(engine._model.predict.call_args.kwargs["save"])
        prediction.masks = None
        with patch("part_segmentation.inference.read_image", return_value=sample.image), self.assertRaises(ValueError):
            engine.predict("test.png", SegmentationSettings())

    def test_inference_empty_result(self):
        sample = sample_result()
        engine = PartSegmenter.__new__(PartSegmenter)
        engine._model = Mock()
        engine._model.predict.return_value = [SimpleNamespace(boxes=[], masks=None)]
        with patch("part_segmentation.inference.read_image", return_value=sample.image):
            result = engine.predict("test.png", SegmentationSettings())
        self.assertEqual(result.segments, ())

    def test_missing_weights(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            PartSegmenter(Path(tmp) / "missing.pt")

    def test_export_continues_on_error_and_avoids_filename_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "input", root / "output"
            for name in ("same.png", "same.bmp", "nested/same.png"):
                write_image(source / name, sample_result().image)
            (source / "broken.png").write_bytes(b"broken")

            def predict(path, settings):
                if path.name == "broken.png":
                    raise ValueError("broken image")
                return sample_result(path)

            with patch("part_segmentation.export.PartSegmenter") as engine:
                engine.return_value.predict.side_effect = predict
                status = export_batch(root / "best.pt", source, output, SegmentationSettings(), recursive=True)
            self.assertEqual(status, 1)
            self.assertTrue((output / "same.png.overlay.png").is_file())
            self.assertTrue((output / "same.bmp.overlay.png").is_file())
            self.assertTrue((output / "nested/same.png.overlay.png").is_file())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["images"]), 4)
            self.assertEqual(sum(r["status"] == "error" for r in summary["images"]), 1)
            with self.assertRaises(ValueError):
                export_batch(root / "best.pt", source, source / "output", SegmentationSettings())


if __name__ == "__main__":
    unittest.main()
