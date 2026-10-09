"""加载本项目已校准的 model.pt，对单图或文件夹进行 EfficientAD 推理。"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

from . import cli
from .mask import category_config, default_mask_record, load_configs, read_rgb


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _finite_threshold(value: float | str) -> float:
    """CLI 与 API 共用校验；分数不是概率，因此不限制阈值范围。"""
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold 必须是有限数值（不能为 NaN/Inf）。") from error
    if not math.isfinite(threshold):
        raise ValueError("threshold 必须是有限数值（不能为 NaN/Inf）。")
    return threshold


def collect_images(root: Path, output_root: Path) -> list[Path]:
    """递归获取文件清单，排除输出树、符号链接和 Windows junction。"""
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    if not root.is_dir():
        raise ValueError(f"图片目录不存在：{root}")
    if root.is_relative_to(output_root):
        raise ValueError("输出目录不能等于或包含输入目录。")
    images = []
    for directory, folders, files in os.walk(root):
        current = Path(directory)
        folders[:] = sorted(
            name for name in folders
            if not (current / name).is_symlink() and not (current / name).is_junction()
            and not (current / name).resolve().is_relative_to(output_root)
        )
        images.extend(
            current / name for name in sorted(files)
            if (current / name).suffix.lower() in IMAGE_EXTENSIONS
            and (current / name).is_file() and not (current / name).is_symlink()
        )
    if not images:
        raise ValueError(f"目录内没有支持的图片：{root}")
    return sorted(images)


class EfficientAdPredictor:
    """模型只加载一次；predict() 逐图复用，不访问训练集或下载教师权重。"""

    def __init__(self, checkpoint: Path | str, device: str = "auto", *,
                 mask: Path | str | None = None, circle_config: Path | str | None = None,
                 threshold: float | None = None, score_mode: str = cli.SCORE_MODE_CHECKPOINT,
                 score_pool_kernel: int | None = None, score_topk_ratio: float | None = None) -> None:
        """仅覆盖本实例的 score/阈值；默认沿用 checkpoint，不修改权重或重新校准。"""
        score_mode = cli.normalize_score_mode(score_mode, allow_checkpoint=True)
        if threshold is not None:
            threshold = _finite_threshold(threshold)
        if mask is not None and circle_config is not None:
            raise ValueError("mask 与 circle_config 不能同时指定。")
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"找不到权重：{self.checkpoint}")
        cli.load_runtime()
        # 复用旧 checkpoint 的版本/输出边界/Teacher 激活兼容逻辑，严格加载 state_dict。
        self.model, self.config, saved = cli.restore_for_inference(SimpleNamespace(
            checkpoint=self.checkpoint, device=device, num_workers=0,
        ))
        self.config = dict(self.config)
        self.category = saved.get("manifest", {}).get("category", "unknown")
        self.calibration = dict(saved["calibration"])
        self.score_mode = cli.calibration_score_mode(self.calibration)
        self.threshold = float(self.calibration["threshold"])
        display_max = float(self.calibration["display_max"])
        if not math.isfinite(self.threshold) or not math.isfinite(display_max) or display_max <= 0:
            raise ValueError("checkpoint 的 threshold 必须有限，display_max 必须有限且大于 0。")
        self._score_settings(score_mode, threshold, score_pool_kernel, score_topk_ratio)
        if threshold is not None:
            # 定位与可视化同样读取 calibration，必须与整图判定使用同一阈值。
            self.threshold = threshold
            self.calibration["threshold"] = threshold
        size = self.config.get("image_size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("checkpoint 的 image_size 必须为正整数。")
        cli.check_statistics(self.model.model.mean_std)
        cli.check_statistics(self.model.model.quantiles, quantiles=True)
        self.mask_params, self.mask_base_dir, self.mask_source = self._mask_settings(mask, circle_config)

    def _score_settings(self, requested: str, threshold: float | None,
                        pool_kernel: int | None, topk_ratio: float | None) -> None:
        """选择离线 score 定义，禁止给新公式静默套用旧阈值或伪造归一化。"""
        has_pool_overrides = pool_kernel is not None or topk_ratio is not None
        if has_pool_overrides and requested != cli.SCORE_MODE_POOL_TOPK:
            raise ValueError("score-pool-kernel / score-topk-ratio 仅可与 --score-mode pool+top 一起使用。")
        if requested == cli.SCORE_MODE_CHECKPOINT:
            return
        saved_mode = self.score_mode
        saved_method = self.calibration.get("score_method") or {}
        changed = requested != saved_mode
        if requested == cli.SCORE_MODE_TOP:
            method = cli.build_score_method([], self.config, requested)
        elif requested == cli.SCORE_MODE_POOL_TOPK:
            method = cli.build_score_method([], self.config, requested)
            if saved_mode == requested:
                method.update(saved_method)
            original_kernel, original_ratio = method["pool_kernel"], method["topk_ratio"]
            if pool_kernel is not None:
                method["pool_kernel"] = pool_kernel
            if topk_ratio is not None:
                method["topk_ratio"] = float(topk_ratio)
            kernel, ratio = method["pool_kernel"], method["topk_ratio"]
            if isinstance(kernel, bool) or not isinstance(kernel, int) or kernel < 1 or kernel % 2 == 0:
                raise ValueError("score-pool-kernel 必须为正奇数。")
            if not math.isfinite(ratio) or not 0 < ratio <= 1:
                raise ValueError("score-topk-ratio 必须是 (0, 1] 内的有限数值。")
            changed = changed or kernel != original_kernel or ratio != original_ratio
        else:
            if saved_mode != cli.SCORE_MODE_MULTISCALE:
                raise ValueError(
                    "checkpoint 未保存 multiscale_pool 所需的正常集归一化参数；"
                    "请先用 evaluate --score-mode multiscale_pool 生成匹配的 model.pt。"
                )
            method = dict(saved_method)
            self._check_multiscale_method(method)
        if changed and threshold is None:
            raise ValueError(
                "修改 score 计算方式或池化参数时必须同时指定 --threshold（API: threshold）；"
                "不同公式不能直接复用 checkpoint 的旧阈值。"
            )
        self.score_mode = requested
        method["mode"] = requested
        self.config["score_mode"] = requested
        self.calibration.update(score_mode=requested, score_method=method)
        if changed:
            self.calibration["inference_score_override"] = {
                "checkpoint_score_mode": saved_mode, "recalibrated": False,
            }

    @staticmethod
    def _check_multiscale_method(method: dict) -> None:
        """显式选择多尺度时提前检查保存的基准；不读取任何验证/待测数据。"""
        try:
            kernels = method["pool_kernels"]
            ratio = method["topk_ratio"]
            if not kernels or not math.isfinite(ratio) or not 0 < ratio <= 1:
                raise ValueError
            for kernel in kernels:
                if isinstance(kernel, bool) or not isinstance(kernel, int) or kernel < 1 or kernel % 2 == 0:
                    raise ValueError
                reference = method["normalization"][str(kernel)]
                if (not math.isfinite(reference["median"])
                        or not math.isfinite(reference["denominator"])
                        or reference["denominator"] <= 0):
                    raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("checkpoint 的 multiscale_pool 归一化参数缺失或无效；请重新校准。") from error

    def _mask_settings(self, mask, circle_config) -> tuple[dict | None, Path | None, str]:
        """默认复现训练 mask；迁移机器时可显式重定位，不静默取消忽略区域。"""
        if mask is not None:
            path = Path(mask).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"找不到 mask：{path}")
            settings = ({"default_mask": str(path)}, path.parent, "explicit_mask")
        elif circle_config is not None:
            path = Path(circle_config).resolve()
            settings = (category_config(load_configs(path), self.category), path.parent, "explicit_config")
        else:
            configured_path = self.config.get("circle_config")
            params = self.config.get("circle_params")
            if configured_path:
                path = Path(configured_path)
                params = params or category_config(load_configs(path), self.category)
                return params, path.parent, "checkpoint"
            if params:
                # 本项目训练入口通常同时保存 circle_config 和 circle_params。
                return params, self.checkpoint.parent, "checkpoint"
            return None, None, "none"
        warnings.warn(
            "显式覆盖了 checkpoint 的 mask 设置。只有与训练 mask 等效时才可复用原阈值；"
            "改变有效检测区域应重新校准。", RuntimeWarning, stacklevel=3,
        )
        return settings

    def _prepare_batch(self, image_path: Path):
        stat = image_path.stat()
        record = {"path": str(image_path), "size_bytes": stat.st_size,
                  "mtime_ns": stat.st_mtime_ns, "label": 0}
        rgb = read_rgb(image_path)
        if self.mask_params is not None:
            try:
                record["circle"] = default_mask_record(rgb, self.mask_params, base_dir=self.mask_base_dir)
            except (ValueError, FileNotFoundError, OSError) as error:
                raise ValueError(
                    f"无法使用训练/指定 mask：{error}。请用 --mask 或 --circle-config 指定本机路径；"
                    "不会自动改为无 mask 推理。"
                ) from error
        # 与原 predict 命令完全一致：EXIF -> RGB -> BILINEAR 方形缩放 -> [0,1]。
        batch = cli.collate_batch([cli.SnapshotDataset([record], self.config["image_size"])[0]])
        if batch.ignore_mask.all().item():
            raise ValueError("mask 遮住了全部像素，没有有效检测区域。")
        return batch, (int(rgb.shape[1]), int(rgb.shape[0]))

    def predict(self, image: Path | str, output_dir: Path | str | None = None, *,
                save_heatmaps: bool = True, save_maps: bool = True) -> dict:
        """返回 JSON 兼容的分数、OK/NG 与定位信息；可选保存图像和数组。"""
        image_path = Path(image).resolve()
        batch, original_size = self._prepare_batch(image_path)
        device = self.config["device"]
        with cli.torch.inference_mode():
            prediction = self.model.model(batch.image.to(device))
            scores = cli.prediction_scores(prediction, batch, self.config, self.calibration)
            if not all(math.isfinite(value) for value in scores.values()):
                raise ValueError(f"预测分数包含 NaN/Inf：{image_path}")
            localization = cli.prediction_localization(
                prediction, batch, self.config, self.calibration, scores,
            )
        anomaly_map = prediction.anomaly_map[0, 0].detach().cpu().numpy()
        ignore_mask = batch.ignore_mask[0, 0].cpu().numpy()
        map_h, map_w = anomaly_map.shape
        width, height = original_size
        boxes_original = [{
            "x0": max(0, math.floor(box["x0"] * width / map_w)),
            "y0": max(0, math.floor(box["y0"] * height / map_h)),
            "x1": min(width, math.ceil(box["x1"] * width / map_w)),
            "y1": min(height, math.ceil(box["y1"] * height / map_h)),
            "source_scales": box["source_scales"],
        } for box in localization["boxes"]]
        result = {
            "image": str(image_path), "checkpoint": str(self.checkpoint),
            "category": self.category, "backbone": self.config["backbone"],
            "score_mode": self.score_mode, **scores, "threshold": self.threshold,
            "score_method": self.calibration.get("score_method") or {"name": cli.TOP_SCORE_METHOD},
            "prediction": "NG" if scores["score"] > self.threshold else "OK",
            "decision_rule": "score > threshold", "mask_source": self.mask_source,
            "image_size_original": {"width": width, "height": height},
            "anomaly_map_shape": [map_h, map_w], "boxes_original": boxes_original,
            "localization": cli.localization_summary(localization),
        }
        if output_dir is not None:
            output = Path(output_dir).resolve()
            output.mkdir(parents=True, exist_ok=True)
            # NPY 保留有符号浮点异常值；PNG 二值图只是定位启发式，不是标注 mask。
            if save_maps:
                cli.np.save(output / "anomaly_map.npy", anomaly_map.astype(cli.np.float32))
                cli.Image.fromarray(ignore_mask.astype("uint8") * 255).save(output / "ignore_mask.png")
                binary = cli.Image.fromarray(localization["binary_map"].astype("uint8") * 255)
                binary.resize(original_size, cli.Image.Resampling.NEAREST).save(output / "anomaly_mask.png")
            if save_heatmaps:
                from .report import save_heatmap
                save_heatmap(
                    image_path, anomaly_map, output / "prediction.png",
                    display_max=self.calibration["display_max"], score=scores["score"],
                    threshold=self.threshold, ignore_mask=ignore_mask, localization=localization,
                    multiscale_output_path=output / "prediction_scales.png",
                )
            result["output_dir"] = str(output)
            cli.write_json(output / "prediction.json", result)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="本项目训练/重新校准生成的 model.pt")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="单张原始图片")
    inputs.add_argument("--image-dir", type=Path, help="递归推理该目录的图片（不要求 train/test 结构）")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda 或单个可见 GPU 编号，如 0")
    parser.add_argument(
        "--threshold", type=_finite_threshold, default=None,
        help="本次推理使用的有限数值阈值；默认沿用 checkpoint，不修改权重；score > threshold 为 NG",
    )
    parser.add_argument(
        "--score-mode", type=cli.parse_evaluation_score_mode, default=cli.SCORE_MODE_CHECKPOINT,
        help="checkpoint=沿用模型（默认）；top=最大单像素；pool+top=池化 Top-K；multiscale_pool=保存的多尺度归一化",
    )
    parser.add_argument("--score-pool-kernel", type=int, help="pool+top 的池化核，正奇数；默认沿用模型设置或 21")
    parser.add_argument("--score-topk-ratio", type=float, help="pool+top 的 Top-K 比例，(0, 1]；默认沿用模型设置或 0.001")
    parser.add_argument("--output-dir", type=Path, default=cli.PROJECT_DIR / "outputs" / "inference")
    masks = parser.add_mutually_exclusive_group()
    masks.add_argument("--mask", type=Path, help="显式指定 ignore mask：非零忽略，0 保留；默认沿用训练设置")
    masks.add_argument("--circle-config", type=Path, help="本机 mask 配置 JSON，按 checkpoint 类别读取")
    parser.add_argument("--no-heatmaps", action="store_true", help="不生成可视化 PNG")
    parser.add_argument("--no-maps", action="store_true", help="不保存异常数组及二值 mask")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    if args.image_dir is not None:
        images = collect_images(args.image_dir, args.output_dir)
        input_root = args.image_dir.resolve()
    else:
        if not args.image.is_file():
            raise FileNotFoundError(f"图片不存在：{args.image}")
        images, input_root = [args.image.resolve()], None
    predictor = EfficientAdPredictor(
        args.checkpoint, args.device, mask=args.mask, circle_config=args.circle_config,
        threshold=args.threshold,
        score_mode=args.score_mode, score_pool_kernel=args.score_pool_kernel,
        score_topk_ratio=args.score_topk_ratio,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="inference_", dir=args.output_dir.resolve()))
    rows = []
    for index, image in enumerate(images, 1):
        # 保留扩展名作为目录名，避免 same.png / same.bmp 或跨目录同名覆盖。
        relative = image.relative_to(input_root) if input_root is not None else Path(image.name)
        result = predictor.predict(image, output / "images" / relative,
                                   save_heatmaps=not args.no_heatmaps, save_maps=not args.no_maps)
        rows.append({key: result[key] for key in (
            "image", "prediction", "score", "score_topk", "score_max", "threshold", "score_mode", "output_dir",
        )} | {"box_count": len(result["boxes_original"])})
        print(f"[{index}/{len(images)}] {result['prediction']} score={result['score']:.6g} | {image}", flush=True)
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cli.write_json(output / "summary.json", {
        "checkpoint": str(predictor.checkpoint), "category": predictor.category,
        "backbone": predictor.config["backbone"], "device": predictor.config["device"],
        "model_image_size": predictor.config["image_size"], "score_mode": predictor.score_mode,
        "threshold": predictor.threshold, "decision_rule": "score > threshold",
        "score_method": predictor.calibration.get("score_method") or {"name": cli.TOP_SCORE_METHOD},
        "mask_source": predictor.mask_source, "mask_params": predictor.mask_params,
        "input_directory": str(input_root) if input_root is not None else None,
        "image_count": len(rows), "ok_count": sum(row["prediction"] == "OK" for row in rows),
        "ng_count": sum(row["prediction"] == "NG" for row in rows),
        "save_heatmaps": not args.no_heatmaps, "save_maps": not args.no_maps,
        "calibration": predictor.calibration,
    })
    print(f"推理结果：{output}")
    return output


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
