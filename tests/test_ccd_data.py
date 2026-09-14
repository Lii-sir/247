"""使用临时合成图片验证数据快照，不读取或修改真实数据集。"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from ccd_data import list_categories, prepare_manifest, resolve_data_root


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.camera = self.root / "CCD1"
        for index in range(10):
            self.make_image(f"train/good/{index:02}.png", index)

    def make_image(self, relative: str, value: int, old: bool = True) -> Path:
        """用不同像素值创建测试图片。"""
        path = self.camera / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), (value, value, value)).save(path)
        if old:
            old_time = time.time() - 600
            os.utime(path, (old_time, old_time))
        return path

    def manifest(self, **kwargs):
        return prepare_manifest(self.root, "CCD1", **kwargs)

    def test_split_is_reproducible_and_has_no_leak(self) -> None:
        self.make_image("test/good/normal.png", 50)
        self.make_image("test/scratch/nested/abnormal.png", 80)
        snapshot = self.manifest()
        second = self.manifest()
        self.assertEqual(snapshot["train"], second["train"])
        self.assertEqual(snapshot["val"], second["val"])
        self.assertEqual((len(snapshot["train"]), len(snapshot["val"])), (8, 2))
        split_hashes = [{record["sha256"] for record in snapshot[split]} for split in ("train", "val", "test")]
        self.assertFalse(split_hashes[0] & split_hashes[1])
        self.assertFalse(split_hashes[0] & split_hashes[2])
        self.assertFalse(split_hashes[1] & split_hashes[2])
        self.assertEqual({record["label"] for record in snapshot["train"] + snapshot["val"]}, {0})
        self.assertEqual(snapshot["summary"]["test_good"], 1)
        self.assertEqual(snapshot["summary"]["test_anomaly"], 1)
        for record in snapshot["train"] + snapshot["val"] + snapshot["test"]:
            self.assertTrue(Path(record["path"]).is_absolute())
            self.assertEqual((record["width"], record["height"]), (16, 12))
            self.assertGreater(record["size_bytes"], 0)
            self.assertGreater(record["mtime_ns"], 0)
            self.assertEqual(len(record["sha256"]), 64)
            self.assertNotIn("mask", record)

    def test_corrupt_recent_and_temporary_files_are_skipped(self) -> None:
        corrupt = self.camera / "train/good/corrupt.png"
        corrupt.write_bytes(b"broken image")
        old_time = time.time() - 600
        os.utime(corrupt, (old_time, old_time))
        self.make_image("train/good/recent.png", 90, old=False)
        (self.camera / "train/good/downloading.bmp.part").write_bytes(b"unfinished")
        summary = self.manifest()["summary"]
        self.assertEqual(summary["skipped_by_reason"]["unreadable_image"], 1)
        self.assertEqual(summary["skipped_by_reason"]["too_recent"], 1)
        self.assertEqual(summary["skipped_by_reason"]["temporary_file"], 1)
        self.assertEqual(summary["train"] + summary["val"], 10)

    def test_truncated_pixels_are_skipped(self) -> None:
        path = self.make_image("train/good/truncated.bmp", 100)
        path.write_bytes(path.read_bytes()[:-100])
        old_time = time.time() - 600
        os.utime(path, (old_time, old_time))
        snapshot = self.manifest()
        self.assertEqual(snapshot["summary"]["skipped_by_reason"]["unreadable_image"], 1)

    def test_empty_test_is_allowed(self) -> None:
        snapshot = self.manifest()
        self.assertEqual(snapshot["test"], [])
        self.assertEqual(snapshot["summary"]["test"], 0)
        self.assertEqual(snapshot["summary"]["test_good"], 0)
        self.assertEqual(snapshot["summary"]["test_anomaly"], 0)

    def test_scan_does_not_modify_original_files(self) -> None:
        files = list((self.camera / "train/good").glob("*.png"))
        before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
        self.manifest()
        after = {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
        self.assertEqual(before, after)

    def test_file_changed_during_scan_is_skipped(self) -> None:
        changing_path = self.camera / "train/good/00.png"
        original_open = Image.open
        changed = False

        def change_mtime_on_open(path, *args, **kwargs):
            nonlocal changed
            if Path(path) == changing_path and not changed:
                changed = True
                stat = changing_path.stat()
                os.utime(changing_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            return original_open(path, *args, **kwargs)

        with patch("ccd_data.Image.open", side_effect=change_mtime_on_open):
            snapshot = self.manifest()
        self.assertEqual(snapshot["summary"]["skipped_by_reason"]["changed_during_scan"], 1)
        self.assertEqual(snapshot["summary"]["train"] + snapshot["summary"]["val"], 9)

    def test_missing_explicit_root_does_not_fall_back(self) -> None:
        with patch("ccd_data.DEFAULT_DATA_ROOTS", (self.root,)):
            self.assertEqual(resolve_data_root(), self.root.resolve())
            with self.assertRaises(FileNotFoundError):
                resolve_data_root(self.root / "missing")
        with self.assertRaises(FileNotFoundError):
            prepare_manifest(self.root, "CCD_missing")

    def test_category_listing_and_path_validation(self) -> None:
        (self.root / "CCD2" / "test").mkdir(parents=True)
        (self.root / "downloading").mkdir()
        self.assertEqual(list_categories(self.root), ["CCD1", "CCD2"])
        for category in ("../CCD1", "..", "", "a/b", "a\\b"):
            with self.subTest(category=category), self.assertRaises(ValueError):
                prepare_manifest(self.root, category)

    def test_duplicate_content_is_kept(self) -> None:
        train_directory = self.camera / "train/good"
        shutil.copy2(train_directory / "00.png", train_directory / "00(1).png")
        test_good = self.camera / "test/good"
        test_good.mkdir(parents=True)
        shutil.copy2(train_directory / "01.png", test_good / "overlap.png")
        self.make_image("test/defect/a.png", 70)
        shutil.copy2(self.camera / "test/defect/a.png", self.camera / "test/defect/a(1).png")
        snapshot = self.manifest()
        self.assertEqual(snapshot["summary"]["skipped_by_reason"], {})
        self.assertEqual(snapshot["summary"]["train"] + snapshot["summary"]["val"], 11)
        self.assertEqual(snapshot["summary"]["test"], 3)
        report = snapshot["duplicate_report"]
        self.assertEqual(report["group_count"], 3)
        self.assertEqual(report["file_count"], 6)
        self.assertEqual(report["extra_copy_count"], 3)
        self.assertEqual(report["cross_train_test_group_count"], 1)
        self.assertEqual(report["label_conflict_group_count"], 0)
        train_hashes = {record["sha256"] for record in snapshot["train"] + snapshot["val"]}
        self.assertTrue(train_hashes & {record["sha256"] for record in snapshot["test"]})

    def test_conflicting_labels_are_kept(self) -> None:
        defect = self.camera / "test/defect"
        defect.mkdir(parents=True)
        shutil.copy2(self.camera / "train/good/00.png", defect / "conflict.png")
        snapshot = self.manifest()
        self.assertEqual(snapshot["summary"]["skipped_by_reason"], {})
        self.assertEqual(snapshot["duplicate_report"]["group_count"], 1)
        self.assertEqual(snapshot["duplicate_report"]["cross_train_test_group_count"], 1)
        self.assertEqual(snapshot["duplicate_report"]["label_conflict_group_count"], 1)
        test_hashes = {record["sha256"] for record in snapshot["test"]}
        self.assertIn(next(iter(test_hashes)), {record["sha256"] for record in snapshot["train"] + snapshot["val"]})

    def test_conflicting_labels_inside_test_are_kept(self) -> None:
        normal = self.make_image("test/good/normal.png", 120)
        defect = self.camera / "test/defect"
        defect.mkdir(parents=True)
        shutil.copy2(normal, defect / "conflict.png")
        snapshot = self.manifest()
        self.assertEqual(snapshot["summary"]["skipped_by_reason"], {})
        self.assertEqual(snapshot["summary"]["test"], 2)
        self.assertEqual(snapshot["duplicate_report"]["group_count"], 1)
        self.assertEqual(snapshot["duplicate_report"]["cross_train_test_group_count"], 0)
        self.assertEqual(snapshot["duplicate_report"]["label_conflict_group_count"], 1)

    def test_minimum_split_sizes_and_invalid_configuration(self) -> None:
        for ratio in (0.01, 0.99):
            snapshot = self.manifest(val_ratio=ratio)
            self.assertGreaterEqual(len(snapshot["train"]), 2)
            self.assertGreaterEqual(len(snapshot["val"]), 2)
        for ratio in (0, 1, float("nan")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.manifest(val_ratio=ratio)
        for age in (-1, float("inf")):
            with self.subTest(age=age), self.assertRaises(ValueError):
                self.manifest(min_age_seconds=age)
        # 删除仅发生在本测试独占的临时目录中。
        for path in sorted((self.camera / "train/good").glob("*.png"))[3:]:
            path.unlink()
        with self.assertRaisesRegex(ValueError, "至少需要 4 张"):
            self.manifest()


if __name__ == "__main__":
    unittest.main()
