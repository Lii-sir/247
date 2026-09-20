"""验证 VisA 辅助数据整理脚本不混入异常或测试图片。"""

from __future__ import annotations

import csv
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from download_visa_aux import main


class VisaAuxTests(unittest.TestCase):
    def test_official_category_first_layout(self) -> None:
        for prefix in ("", "VisA_20220922/"):
            for split_mode in ("official", "without_category", "absent"):
                with self.subTest(prefix=prefix, split_mode=split_mode), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    archive = root / "visa.tar"
                    output = root / "visa_aux"
                    files = {
                        "candle/Data/Images/Normal/train.JPG": b"candle-train",
                        "candle/Data/Images/Normal/test.JPG": b"candle-test",
                        "candle/Data/Images/Anomaly/defect.JPG": b"candle-defect",
                        "candle/Data/Masks/Normal/mask.png": b"not-an-image-sample",
                        "capsules/Data/Images/Normal/train.JPG": b"capsules-train",
                    }
                    if split_mode != "absent":
                        rows = [
                            ["candle", "train", "normal", "candle/Data/Images/Normal/train.JPG", ""],
                            ["candle", "test", "normal", "candle/Data/Images/Normal/test.JPG", ""],
                            ["candle", "train", "anomaly", "candle/Data/Images/Anomaly/defect.JPG", ""],
                            ["candle", "train", "normal", "candle/Data/Masks/Normal/mask.png", ""],
                            ["capsules", "train", "normal", "capsules/Data/Images/Normal/train.JPG", ""],
                        ]
                        table = io.StringIO(newline="")
                        writer = csv.writer(table)
                        header = ["object", "split", "label", "image", "mask"]
                        start = 1 if split_mode == "without_category" else 0
                        writer.writerow(header[start:])
                        writer.writerows(row[start:] for row in rows)
                        files["split_csv/1cls.csv"] = table.getvalue().encode("utf-8")
                    with tarfile.open(archive, "w") as handle:
                        for name, content in files.items():
                            member = tarfile.TarInfo(prefix + name)
                            member.size = len(content)
                            handle.addfile(member, io.BytesIO(content))

                    self.assertEqual(main(["--archive", str(archive), "--output", str(output)]), 0)

                    expected = {
                        "candle/train.JPG": b"candle-train",
                        "capsules/train.JPG": b"capsules-train",
                    }
                    if split_mode == "absent":
                        expected["candle/test.JPG"] = b"candle-test"
                    actual = {
                        path.relative_to(output).as_posix(): path.read_bytes()
                        for path in output.rglob("*")
                        if path.is_file() and path.name != "manifest.json"
                    }
                    self.assertEqual(actual, expected)
                    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                    self.assertEqual(manifest["total_images"], len(expected))
                    self.assertEqual(set(manifest["classes"]), {"candle", "capsules"})

    def _make_archive(self, root: Path) -> Path:
        source = root / "VisA_20220922"
        files = {
            "Data/Images/candle/Normal/train.JPG": b"candle-train",
            "Data/Images/candle/Normal/test.JPG": b"candle-test",
            "Data/Images/candle/Anomaly/defect.JPG": b"candle-defect",
            "Data/Images/capsules/Normal/train.JPG": b"capsules-train",
        }
        for relative, content in files.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        split = source / "split_csv" / "1cls.csv"
        split.parent.mkdir(parents=True, exist_ok=True)
        with split.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["object", "split", "label", "image_path", "mask_path"])
            writer.writerow(["candle", "train", "normal", "Data/Images/candle/Normal/train.JPG", ""])
            writer.writerow(["candle", "test", "normal", "Data/Images/candle/Normal/test.JPG", ""])
            writer.writerow(["candle", "test", "anomaly", "Data/Images/candle/Anomaly/defect.JPG", ""])
            writer.writerow(["capsules", "train", "normal", "Data/Images/capsules/Normal/train.JPG", ""])

        archive = root / "VisA_20220922.tar"
        with tarfile.open(archive, "w") as handle:
            for path in source.rglob("*"):
                handle.add(path, arcname=path.relative_to(root))
        return archive

    def test_selects_train_normal_images_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self._make_archive(root)
            output = root / "visa_aux"

            self.assertEqual(main(["--archive", str(archive), "--output", str(output)]), 0)

            self.assertEqual((output / "candle" / "train.JPG").read_bytes(), b"candle-train")
            self.assertEqual((output / "capsules" / "train.JPG").read_bytes(), b"capsules-train")
            self.assertFalse((output / "candle" / "test.JPG").exists())
            self.assertFalse((output / "candle" / "defect.JPG").exists())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["selection"]["mode"], "split_csv_train_normal")
            self.assertEqual(manifest["total_images"], 2)

    def test_exclude_and_max_per_class_are_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self._make_archive(root)
            output = root / "visa_aux"

            self.assertEqual(
                main([
                    "--archive", str(archive),
                    "--output", str(output),
                    "--exclude-class", "capsules",
                    "--max-per-class", "1",
                ]),
                0,
            )
            self.assertTrue((output / "candle" / "train.JPG").exists())
            self.assertFalse((output / "capsules").exists())


if __name__ == "__main__":
    unittest.main()
