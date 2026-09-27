"""Branch diagnostics must reproduce the model's calibrated fusion."""

import unittest
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from export_branch_maps import extract_maps, collect_images
from self_efficientad.torch_model import EfficientAdModel


class BranchMapTests(unittest.TestCase):
    def test_folder_scan_recurses_and_excludes_previous_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            output = root / "results"
            output.mkdir()
            for name in ("a.PNG", "a.bmp", "nested/a.PNG", "results/st_raw.png", "note.txt"):
                (root / name).touch()
            found = collect_images(root, output)
            self.assertEqual({p.relative_to(root).as_posix() for p in found},
                             {"a.PNG", "a.bmp", "nested/a.PNG"})
            with self.assertRaisesRegex(ValueError, "output"):
                collect_images(root, root)
            with self.assertRaisesRegex(ValueError, "output"):
                collect_images(root, root.parent)
            (root / "empty").mkdir()
            with self.assertRaisesRegex(ValueError, "No supported"):
                collect_images(root / "empty", output)

    def test_export_matches_model_inference_and_keeps_raw_values(self):
        torch.manual_seed(12)
        model = EfficientAdModel(backbone="resnet50_layer3").eval()
        # Deterministic small STAE error guarantees negative calibrated values;
        # random untrained ResNet activations need not fall below qa_ae.
        with torch.no_grad():
            model.student.head.weight.zero_()
            model.student.head.bias.fill_(0.25)
            model.ae.decoder[-1].weight.zero_()
            model.ae.decoder[-1].bias.fill_(0.1)
        model.mean_std["mean"].data.fill_(0.2)
        model.mean_std["std"].data.fill_(0.7)
        for name, value in {"qa_st": 1., "qb_st": 3., "qa_ae": 2., "qb_ae": 5.}.items():
            model.quantiles[name].data.fill_(value)
        x = torch.rand(1, 3, 512, 512)
        result = extract_maps(model, x)
        with torch.inference_mode():
            expected = model(x).anomaly_map[0, 0].numpy()
            student, distance = model.compute_student_teacher_distance(x)
            ae = model.autoencoder_features(x, student.shape[-2:])
            teacher = model.teacher(x)
            if model.is_set(model.mean_std):
                teacher = (teacher - model.mean_std["mean"]) / model.mean_std["std"]
        np.testing.assert_allclose(result["fused_calibrated"], expected, atol=1e-7)
        np.testing.assert_allclose(result["st_native"], distance.mean(1)[0].numpy(), atol=1e-7)
        np.testing.assert_allclose(result["stae_native"],
                                   (ae - student[:, 1024:]).square().mean(1)[0].numpy(), atol=1e-7)
        np.testing.assert_allclose(result["teacher_ae_native"],
                                   (teacher - ae).square().mean(1)[0].numpy(), atol=1e-7)
        self.assertEqual(result["st_raw"].shape, (512, 512))
        self.assertEqual(result["st_native"].shape, (32, 32))
        self.assertTrue((result["st_raw"] >= 0).all())
        self.assertTrue((result["stae_raw"] >= 0).all())
        self.assertTrue((result["teacher_ae_raw"] >= 0).all())
        self.assertTrue((result["stae_calibrated"] < 0).any())

    def test_uncalibrated_checkpoint_keeps_only_raw_maps(self):
        model = EfficientAdModel(backbone="pdn_small").eval()
        x = torch.rand(1, 3, 256, 256)
        result = extract_maps(model, x)
        with torch.inference_mode():
            st, stae = model.get_maps(x, normalize=False)
        np.testing.assert_allclose(result["st_raw"], st[0, 0].numpy(), atol=1e-7)
        np.testing.assert_allclose(result["stae_raw"], stae[0, 0].numpy(), atol=1e-7)
        self.assertNotIn("fused_calibrated", result)

    def test_cli_saves_images_arrays_and_metadata_offline(self):
        import efficientad_ccd as cli
        cli.load_runtime()
        config = dict(imagenette_dir="unused", model_size="small", lr=1e-4,
                      weight_decay=1e-5, device="cpu", backbone="resnet18_layer2",
                      image_size=256)
        model = cli.new_model(config).eval()
        for name, value in {"qa_st": 1., "qb_st": 3., "qa_ae": 2., "qb_ae": 5.}.items():
            model.model.quantiles[name].data.fill_(value)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            cli.save_checkpoint(checkpoint, model, config, {"category": "synthetic"}, 0)
            image_path = root / "input.png"
            Image.fromarray(np.random.default_rng(3).integers(0, 256, (300, 400, 3), dtype=np.uint8)).save(image_path)
            script = Path(__file__).resolve().parents[1] / "export_branch_maps.py"
            subprocess.run([sys.executable, str(script), "--checkpoint", str(checkpoint),
                            "--image", str(image_path), "--device", "cpu", "--output-dir", str(root / "out")],
                           check=True, capture_output=True, text=True, timeout=90)
            output, = (root / "out").iterdir()
            for name in ("st_raw.png", "stae_raw.png", "teacher_ae_raw.png",
                         "branches_raw.png", "branches_calibrated.png"):
                with Image.open(output / name) as rendered:
                    self.assertGreater(rendered.width, 100)
                    self.assertGreater(rendered.height, 100)
                    rendered.verify()
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["mask_applied"])
            self.assertEqual(metadata["backbone"], "resnet18_layer2")
            with np.load(output / "maps.npz") as maps:
                self.assertEqual(maps["st_native"].shape, (32, 32))
                self.assertIn("teacher_ae_native", maps.files)
                self.assertIn("teacher_ae_raw", maps.files)
                np.testing.assert_allclose(maps["fused_calibrated"],
                                           0.5 * (maps["st_calibrated"] + maps["stae_calibrated"]), atol=1e-7)

            inputs = root / "inputs"
            (inputs / "nested").mkdir(parents=True)
            for relative in ("same.png", "same.bmp", "nested/same.png"):
                with Image.open(image_path) as image:
                    image.save(inputs / relative)
            batch_output = inputs / "results"
            batch_output.mkdir()
            (batch_output / "previous.png").write_bytes(image_path.read_bytes())
            subprocess.run([sys.executable, str(script), "--checkpoint", str(checkpoint),
                            "--image-dir", str(inputs), "--device", "cpu", "--output-dir", str(batch_output),
                            "--vmax", "0.2"],
                           check=True, capture_output=True, text=True, timeout=90)
            run, = (path for path in batch_output.iterdir() if path.is_dir())
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["image_count"], 3)
            for relative in ("same.png", "same.bmp", "nested/same.png"):
                exported = run / relative
                self.assertTrue((exported / "maps.npz").is_file())
                self.assertTrue((exported / "branches_raw.png").is_file())
                meta = json.loads((exported / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["image"], str((inputs / relative).resolve()))
                self.assertEqual(meta["display_scales"]["raw"]["vmax"], 0.2)


if __name__ == "__main__":
    unittest.main()
