"""Export unpooled EfficientAD branch errors from a CLI checkpoint and one image.

No mask is applied: inspecting ignored regions is intentional. Raw errors still
use the checkpoint's channel-standardized teacher. Calibrated errors additionally
use its stored anomaly-map quantiles. Neither PNG colors nor these branch maps
are independently calibrated defect probabilities or segmentation thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor


@torch.inference_mode()
def extract_maps(model, image: torch.Tensor) -> dict[str, np.ndarray]:
    """Return native-grid errors, input-sized raw errors and calibrated branches."""
    if model.training:
        raise ValueError("Call model.eval() before exporting branch maps.")
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Expected one image with shape [1, 3, H, W].")
    student, distance = model.compute_student_teacher_distance(image)
    ae = model.autoencoder_features(image, student.shape[-2:])
    st = distance.mean(dim=1, keepdim=True)
    stae = (ae - student[:, model.teacher_out_channels:]).square().mean(dim=1, keepdim=True)
    tensors = {"st_native": st, "stae_native": stae}
    # Match EfficientAdModel.compute_maps, including the PDN border convention.
    if model.pad_maps:
        st, stae = (F.pad(value, (4, 4, 4, 4)) for value in (st, stae))
    st, stae = (F.interpolate(value, size=image.shape[-2:], mode="bilinear", align_corners=False)
                for value in (st, stae))
    tensors.update(st_raw=st, stae_raw=stae)
    if model.is_set(model.quantiles):
        q = model.quantiles
        if not all(torch.isfinite(v).all().item() for v in q.values()):
            raise ValueError("Checkpoint map quantiles contain NaN/Inf.")
        if q["qb_st"] <= q["qa_st"] or q["qb_ae"] <= q["qa_ae"]:
            raise ValueError("Checkpoint map quantiles have a non-positive normalization range.")
        st_cal = 0.1 * (st - q["qa_st"]) / (q["qb_st"] - q["qa_st"])
        stae_cal = 0.1 * (stae - q["qa_ae"]) / (q["qb_ae"] - q["qa_ae"])
        tensors.update(st_calibrated=st_cal, stae_calibrated=stae_cal,
                       fused_calibrated=0.5 * (st_cal + stae_cal))
    result = {key: value[0, 0].detach().cpu().numpy() for key, value in tensors.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Branch errors contain NaN/Inf; inspect checkpoint normalization statistics.")
    return result


def save_figures(image: Image.Image, maps: dict[str, np.ndarray], output: Path,
                 vmax: float | None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = {}
    groups = [("raw", ["st_raw", "stae_raw"])]
    if "fused_calibrated" in maps:
        groups.append(("calibrated", ["st_calibrated", "stae_calibrated", "fused_calibrated"]))
    for group, keys in groups:
        # Same scale for every panel within a group, no independent branch scaling.
        # Negative calibrated values remain in NPZ; PNG displays them at zero.
        upper = vmax if vmax is not None else max(float(maps[key].max()) for key in keys)
        upper = max(upper, 1e-12)
        scales[group] = {"vmin": 0, "vmax": upper}
        fig, axes = plt.subplots(2, len(keys) + 1, figsize=(5 * (len(keys) + 1), 8),
                                 squeeze=False, constrained_layout=True)
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original image")
        axes[1, 0].text(0, 0.8, "No mask, pooling or boxes applied.\nShared scale within this figure.\n"
                        "Negative calibrated values display at 0.\nExact signed values are in maps.npz.",
                        transform=axes[1, 0].transAxes, va="top", fontsize=10)
        for column, key in enumerate(keys, 1):
            values = maps[key]
            # Match visualization to the EXIF-corrected original aspect ratio.
            visible = np.asarray(Image.fromarray(values).resize(image.size, Image.Resampling.BILINEAR))
            axes[0, column].imshow(visible, cmap="turbo", vmin=0, vmax=upper)
            axes[0, column].set_title(f"{key}\nmin={values.min():.5g}, max={values.max():.5g}")
            axes[1, column].imshow(image)
            colored = axes[1, column].imshow(visible, cmap="turbo", vmin=0, vmax=upper, alpha=0.45)
            axes[1, column].set_title(f"{key} overlay")
            fig.colorbar(colored, ax=list(axes[:, column]), shrink=0.75)
            plt.imsave(output / f"{key}.png", visible, cmap="turbo", vmin=0, vmax=upper)
        for axis in axes.flat:
            axis.axis("off")
        fig.savefig(output / f"branches_{group}.png", dpi=140)
        plt.close(fig)
    return scales


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="CLI model.pt or checkpoints/last.pt")
    parser.add_argument("--image", type=Path, required=True, help="Original source image, not a heatmap montage")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or visible GPU index (e.g. 0)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs/branch_maps")
    parser.add_argument("--vmax", type=float, help="Optional fixed positive PNG upper limit; does not alter NPZ")
    args = parser.parse_args(argv)
    if args.vmax is not None and (not np.isfinite(args.vmax) or args.vmax <= 0):
        parser.error("--vmax must be finite and positive")

    import efficientad_ccd as cli
    cli.load_runtime()
    saved = cli.read_checkpoint(args.checkpoint)
    config = dict(saved["config"])
    config["device"] = cli.choose_device(args.device)
    config.setdefault("model_size", "small")
    model = cli.new_model(config).eval()
    model.model.load_state_dict(saved["model_state"], strict=True)
    with Image.open(args.image) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    size = int(config["image_size"])
    resized = original.resize((size, size), Image.Resampling.BILINEAR)
    tensor = to_tensor(resized).unsqueeze(0).to(config["device"])
    maps = extract_maps(model.model, tensor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="branches_", dir=args.output_dir.resolve()))
    np.savez_compressed(output / "maps.npz", **maps)
    # Rendering can be bounded independently of inference resolution.
    original.thumbnail((800, 800), Image.Resampling.LANCZOS)
    scales = save_figures(original, maps, output, args.vmax)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "image": str(args.image.resolve()),
        "backbone": model.model.backbone, "input_size": size,
        "mask_applied": False, "pooling_applied": False,
        "teacher_channel_standardization_applied": bool(model.model.is_set(model.model.mean_std)),
        "display_scales": scales,
        "definitions": {
            "st": "mean_C((standardized_teacher - student[:C])**2)",
            "stae": "mean_C((autoencoder - student[C:])**2); NOT teacher-autoencoder error",
            "native": "Original feature grid before PDN border padding and image-size interpolation",
            "raw": "Input-sized error before anomaly-map quantile normalization",
            "calibrated": "Stored checkpoint quantile normalization, signed and unclipped in NPZ",
            "fused_calibrated": "0.5 * st_calibrated + 0.5 * stae_calibrated",
        },
        "arrays": {key: {"shape": list(value.shape), "min": float(value.min()),
                         "max": float(value.max()), "mean": float(value.mean())}
                   for key, value in maps.items()},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                     allow_nan=False), encoding="utf-8")
    print(f"Branch diagnostics: {output}")
    return output


if __name__ == "__main__":
    main()
