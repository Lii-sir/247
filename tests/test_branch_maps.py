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

from export_branch_maps import extract_maps
from self_efficientad.torch_model import EfficientAdModel


class BranchMapTests(unittest.TestCase):
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
        np.testing.assert_allclose(result["fused_calibrated"], expected, atol=1e-7)
        np.testing.assert_allclose(result["st_native"], distance.mean(1)[0].numpy(), atol=1e-7)
        np.testing.assert_allclose(result["stae_native"],
                                   (ae - student[:, 1024:]).square().mean(1)[0].numpy(), atol=1e-7)
        self.assertEqual(result["st_raw"].shape, (512, 512))
        self.assertEqual(result["st_native"].shape, (32, 32))
        self.assertTrue((result["st_raw"] >= 0).all())
        self.assertTrue((result["stae_raw"] >= 0).all())
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
            for name in ("st_raw.png", "stae_raw.png", "branches_raw.png", "branches_calibrated.png"):
                with Image.open(output / name) as rendered:
                    self.assertGreater(rendered.width, 100)
                    self.assertGreater(rendered.height, 100)
                    rendered.verify()
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["mask_applied"])
            self.assertEqual(metadata["backbone"], "resnet18_layer2")
            with np.load(output / "maps.npz") as maps:
                self.assertEqual(maps["st_native"].shape, (32, 32))
                np.testing.assert_allclose(maps["fused_calibrated"],
                                           0.5 * (maps["st_calibrated"] + maps["stae_calibrated"]), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
