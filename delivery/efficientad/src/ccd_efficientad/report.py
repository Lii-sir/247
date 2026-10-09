"""生成 CCD 异常检测的图像级评估报告和统一色阶的异常热图。"""

from __future__ import annotations

import csv
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, roc_auc_score


def _finite_number(value: Any, name: str) -> float:
    """检查数值，避免无效分数悄悄进入评估结果。"""
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是有限数值，实际为 {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 不能是 NaN 或 Inf")
    return result


def _validated_rows(rows: list[dict]) -> list[dict]:
    """规范化评估字段，同时保留调用方提供的图片路径。"""
    if not rows:
        raise ValueError("测试结果为空：至少需要一张已完成预测的图片才能生成报告。")
    validated = []
    for index, row in enumerate(rows):
        label = row.get("label")
        if not isinstance(label, Real) or not math.isfinite(float(label)) or label not in (0, 1):
            raise ValueError(f"第 {index + 1} 条结果的 label 必须为 0 或 1，实际为 {label!r}")
        score = _finite_number(row.get("score"), f"第 {index + 1} 条结果的 score")
        score_topk = _finite_number(row.get("score_topk", score), f"第 {index + 1} 条结果的 score_topk")
        score_max = _finite_number(row.get("score_max", score), f"第 {index + 1} 条结果的 score_max")
        defect_type = row.get("defect_type", "unspecified")
        if not isinstance(defect_type, str):
            raise ValueError(f"第 {index + 1} 条结果的 defect_type 必须是字符串")
        validated.append({
            **row,
            "label": int(label),
            "score": score,
            "score_topk": score_topk,
            "score_max": score_max,
            "defect_type": defect_type or "unspecified",
        })
    return validated


def _ratio(numerator: int, denominator: int) -> float | None:
    """分母为零时返回空值，不把无定义指标误报为零。"""
    return numerator / denominator if denominator else None


def compute_metrics(rows: list[dict], threshold: float) -> dict:
    """计算图像级指标；异常判定统一使用 score > threshold。"""
    threshold = _finite_number(threshold, "threshold")
    rows = _validated_rows(rows)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    predictions = scores > threshold
    tn = int(np.sum((labels == 0) & ~predictions))
    fp = int(np.sum((labels == 0) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    tp = int(np.sum((labels == 1) & predictions))
    normal, anomaly = tn + fp, fn + tp
    notes = ["仅计算图像级指标；数据未提供像素标注，因此不计算像素级指标。"]
    if normal and anomaly:
        roc_auc = float(roc_auc_score(labels, scores))
    else:
        roc_auc = None
        notes.append("测试集只包含一个类别，ROC AUC 无定义，记录为 null。")
    if anomaly:
        average_precision = float(average_precision_score(labels, scores))
    else:
        average_precision = None
        notes.append("测试集没有异常图片，Average Precision 无定义，记录为 null。")
    notes.append("所有分母为零的指标记录为 null；等于阈值的分数判为正常。")

    by_defect: dict[str, dict] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        if row["label"] == 0:
            continue
        group = by_defect.setdefault(row["defect_type"], {"support": 0, "tp": 0, "fn": 0})
        group["support"] += 1
        group["tp" if prediction else "fn"] += 1
    for group in by_defect.values():
        group["recall"] = _ratio(group["tp"], group["support"])

    return {
        "evaluation_level": "image",
        "threshold": threshold,
        "decision_rule": "score > threshold",
        "counts": {"total": len(rows), "normal": normal, "anomaly": anomaly},
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": _ratio(tn + tp, len(rows)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_negative_rate": _ratio(fn, fn + tp),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "defect_type_recall": dict(sorted(by_defect.items())),
        "notes": notes,
    }


def _pyplot():
    """延迟载入无界面的绘图后端，适配远程或后台训练。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _json_default(value: Any):
    """允许校准信息包含路径和 NumPy 标量。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法序列化到 JSON 的类型：{type(value).__name__}")


def write_report(
    output_dir: Path,
    rows: list[dict],
    threshold: float,
    calibration: dict,
    *,
    speed: dict | None = None,
) -> dict:
    """写入指标、逐图预测和分数分布图，并返回保存的指标字典。"""
    rows = _validated_rows(rows)
    metrics = compute_metrics(rows, threshold)
    metrics["calibration"] = calibration
    if speed is not None:
        metrics["inference_speed"] = speed
    # 先检查 JSON，避免无效校准值导致产生部分报告。
    payload = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(payload + "\n", encoding="utf-8")
    # UTF-8 BOM 便于 Windows Excel 直接识别中文路径。
    with (output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        scale_fields = sorted({
            key
            for row in rows
            for key in row
            if key.startswith("score_kernel_") or key.startswith("score_normalized_kernel_")
        })
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "path", "label", "score", "score_topk", "score_max",
                *scale_fields, "pred_label", "defect_type", "correct",
            ],
        )
        writer.writeheader()
        for row in rows:
            prediction = int(row["score"] > threshold)
            writer.writerow({
                "path": str(row.get("path", "")),
                "label": row["label"],
                "score": row["score"],
                "score_topk": row["score_topk"],
                "score_max": row["score_max"],
                **{field: row.get(field, "") for field in scale_fields},
                "pred_label": prediction,
                "defect_type": row["defect_type"],
                "correct": int(prediction == row["label"]),
            })

    plt = _pyplot()
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    # 两类使用同一组分箱，保证分布图可以直接比较。
    bins = np.histogram_bin_edges(scores, bins=min(50, max(5, int(math.sqrt(len(rows))))))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    try:
        for label, title, color in [(0, "Normal", "#2980b9"), (1, "Anomaly", "#e67e22")]:
            values = scores[labels == label]
            if len(values):
                ax.hist(values, bins=bins, alpha=0.6, label=f"{title} (n={len(values)})", color=color)
        ax.axvline(threshold, color="#c0392b", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.5g}")
        ax.set(xlabel="Image anomaly score", ylabel="Image count", title="Test score distribution (image-level)")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "score_distribution.png")
    finally:
        plt.close(fig)
    return metrics


def _resized_map(values: np.ndarray, size: tuple[int, int], *, binary: bool = False) -> np.ndarray:
    """将模型空间响应映射回展示尺寸；二值图必须使用最近邻插值。"""
    mode = Image.Resampling.NEAREST if binary else Image.Resampling.BILINEAR
    if binary:
        source = Image.fromarray(np.asarray(values, dtype=np.uint8) * 255)
        return np.asarray(source.resize(size, mode)) > 0
    return np.asarray(Image.fromarray(np.asarray(values, dtype=np.float32)).resize(size, mode))


def _draw_boxes(axis, boxes: list[dict], map_shape: tuple[int, int], display_size: tuple[int, int]) -> None:
    """把异常图坐标的半开区间外接框映射到原图展示坐标。"""
    from matplotlib.patches import Rectangle

    map_height, map_width = map_shape
    display_width, display_height = display_size
    scale_x, scale_y = display_width / map_width, display_height / map_height
    for index, box in enumerate(boxes, start=1):
        x = box["x0"] * scale_x
        y = box["y0"] * scale_y
        width = (box["x1"] - box["x0"]) * scale_x
        height = (box["y1"] - box["y0"]) * scale_y
        axis.add_patch(Rectangle(
            (x, y), width, height, fill=False, edgecolor="#ff2020", linewidth=2.2
        ))
        axis.text(
            x, max(0, y - 3), str(index), color="white", fontsize=8,
            bbox={"facecolor": "#d00000", "edgecolor": "none", "pad": 1.5},
        )


def _save_multiscale_diagnostic(
    original: Image.Image,
    localization: dict,
    output_path: Path,
    *,
    raw_display_max: float,
) -> None:
    """保存逐尺度池化图与 Overlay；默认 3 个尺度时为 2×3。"""
    scales = localization.get("scales", [])
    if localization.get("score_mode") != "multiscale_pool" or not scales:
        return
    plt = _pyplot()
    column_count = len(scales)
    image_ratio = original.height / original.width
    fig_height = min(14.0, max(6.0, 5.0 * image_ratio * 2 + 1.7))
    fig, axes = plt.subplots(
        2, column_count, figsize=(5 * column_count, fig_height), dpi=100, squeeze=False
    )
    colored = None
    try:
        for column, scale in enumerate(scales):
            pooled = np.clip(scale["map"], 0, raw_display_max)
            shown = _resized_map(pooled, original.size)
            title = (
                f"kernel={scale['kernel']} | raw_topk={scale['raw_score']:.5g}\n"
                f"normalized={scale['normalized_score']:.5g} | active={scale['active']}"
            )
            colored = axes[0, column].imshow(
                shown, cmap="turbo", vmin=0, vmax=raw_display_max
            )
            axes[0, column].set_title(f"P{scale['kernel']} pooled map\n{title}", fontsize=9)
            axes[1, column].imshow(original)
            axes[1, column].imshow(
                shown, cmap="turbo", vmin=0, vmax=raw_display_max, alpha=0.45
            )
            _draw_boxes(axes[1, column], scale["boxes"], scale["map"].shape, original.size)
            axes[1, column].set_title(
                f"Scale overlay | boxes={len(scale['boxes'])}", fontsize=9
            )
            axes[0, column].set_axis_off()
            axes[1, column].set_axis_off()
        fig.suptitle("Multiscale pooling diagnostics (shared raw anomaly scale)", fontsize=13)
        # 为水平色条及其标签预留独立空间，避免保存长宽比较大的 CCD 图时被裁切。
        fig.subplots_adjust(
            left=0.015, right=0.985, bottom=0.17, top=0.88,
            wspace=0.05, hspace=0.18,
        )
        if colored is not None:
            color_axis = fig.add_axes((0.35, 0.075, 0.30, 0.022))
            fig.colorbar(
                colored, cax=color_axis, orientation="horizontal",
                label="Raw pooled anomaly value (fixed validation scale)",
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    finally:
        plt.close(fig)


def save_heatmap(
    image_path: Path,
    anomaly_map: np.ndarray,
    output_path: Path,
    *,
    display_max: float,
    score: float,
    threshold: float,
    ignore_mask: np.ndarray | None = None,
    localization: dict | None = None,
    multiscale_output_path: Path | None = None,
) -> None:
    """保存 2×3 主可视化，并可选保存多尺度逐尺度诊断图。"""
    display_max = _finite_number(display_max, "display_max")
    score = _finite_number(score, "score")
    threshold = _finite_number(threshold, "threshold")
    if display_max <= 0:
        raise ValueError("display_max 必须大于 0，且应由正常验证集统一估计。")
    raw_values = np.asarray(anomaly_map)
    if raw_values.ndim > 2:
        raw_values = np.squeeze(raw_values)
    if raw_values.ndim != 2 or not raw_values.size:
        raise ValueError("anomaly_map 必须是非空二维数组。")
    if (not np.issubdtype(raw_values.dtype, np.number)
            or np.iscomplexobj(raw_values) or not np.isfinite(raw_values).all()):
        raise ValueError("anomaly_map 必须包含有限实数，不能有 NaN 或 Inf。")
    raw_values = raw_values.astype(np.float32, copy=True)
    if ignore_mask is not None:
        ignore_mask = np.asarray(ignore_mask).squeeze().astype(bool)
        if ignore_mask.shape != raw_values.shape:
            raise ValueError("ignore_mask 必须与 anomaly_map 具有相同的二维形状。")
        raw_values[ignore_mask] = 0
    if localization is None:
        localization = {
            "score_mode": "top",
            "response_map": raw_values,
            "response_threshold": threshold,
            "response_title": "Raw response M",
            "response_display_max": display_max,
            "binary_map": raw_values > threshold if score > threshold else np.zeros_like(raw_values, dtype=bool),
            "boxes": [],
            "scales": [],
        }
    response = np.asarray(localization["response_map"], dtype=np.float32)
    binary = np.asarray(localization["binary_map"], dtype=bool)
    if response.shape != raw_values.shape or binary.shape != raw_values.shape:
        raise ValueError("定位响应图、二值图必须与 anomaly_map 形状一致。")
    if not np.isfinite(response).all():
        raise ValueError("定位响应图不能包含 NaN 或 Inf。")
    response_display_max = _finite_number(
        localization["response_display_max"], "response_display_max"
    )
    if response_display_max <= 0:
        raise ValueError("response_display_max 必须大于 0。")
    with Image.open(image_path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    # 限制输出尺寸，同时保留原图长宽比，避免展示超大 CCD 图时占用过多内存。
    original.thumbnail((460, 900), Image.Resampling.LANCZOS)
    raw_heatmap = _resized_map(np.clip(raw_values, 0, display_max), original.size)
    response_heatmap = _resized_map(
        np.clip(response, 0, response_display_max), original.size
    )
    binary_heatmap = _resized_map(binary, original.size, binary=True)
    height = min(18.0, max(7.0, 10.0 * original.height / original.width + 2.0))
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(15, height), dpi=100, squeeze=False)
    try:
        axes[0, 0].imshow(original)
        axes[0, 0].set_title("Original")
        raw_colored = axes[0, 1].imshow(
            raw_heatmap, cmap="turbo", vmin=0, vmax=display_max
        )
        axes[0, 1].set_title("Raw anomaly heatmap M")
        axes[0, 2].imshow(original)
        axes[0, 2].imshow(
            raw_heatmap, cmap="turbo", vmin=0, vmax=display_max, alpha=0.45
        )
        axes[0, 2].set_title("Raw anomaly overlay")

        response_colored = axes[1, 0].imshow(
            response_heatmap, cmap="turbo", vmin=0, vmax=response_display_max
        )
        if localization.get("score_mode") == "pool_topk" and localization.get("scales"):
            scale = localization["scales"][0]
            response_detail = (
                f"pooled_max={scale['pooled_max']:.5g} | topk_mean={scale['raw_score']:.5g}"
            )
        elif localization.get("score_mode") == "multiscale_pool":
            active = [str(scale["kernel"]) for scale in localization.get("scales", []) if scale["active"]]
            response_detail = f"active kernels={','.join(active) if active else 'none'}"
        else:
            response_detail = f"max={float(response.max()):.5g}"
        axes[1, 0].set_title(f"{localization['response_title']}\n{response_detail}", fontsize=10)
        axes[1, 1].imshow(binary_heatmap, cmap="gray", vmin=0, vmax=1)
        axes[1, 1].set_title(
            f"Threshold mask | response > {localization['response_threshold']:.5g}"
        )
        axes[1, 2].imshow(original)
        axes[1, 2].imshow(
            response_heatmap, cmap="turbo", vmin=0,
            vmax=response_display_max, alpha=0.45,
        )
        _draw_boxes(
            axes[1, 2], localization.get("boxes", []), response.shape, original.size
        )
        axes[1, 2].set_title(
            f"Localization overlay | boxes={len(localization.get('boxes', []))}"
        )
        for axis in axes.flat:
            axis.set_axis_off()
        prediction = "Anomaly" if score > threshold else "Normal"
        fig.suptitle(
            f"{prediction} | mode={localization.get('score_mode', 'unknown')} | "
            f"score={score:.5g} | threshold={threshold:.5g}", fontsize=13,
        )
        fig.subplots_adjust(
            left=0.015, right=0.985, bottom=0.17, top=0.90,
            wspace=0.06, hspace=0.20,
        )
        raw_color_axis = fig.add_axes((0.12, 0.075, 0.30, 0.018))
        fig.colorbar(
            raw_colored, cax=raw_color_axis, orientation="horizontal",
            label="Raw anomaly value (fixed validation scale)",
        )
        response_color_axis = fig.add_axes((0.58, 0.075, 0.30, 0.018))
        fig.colorbar(
            response_colored, cax=response_color_axis, orientation="horizontal",
            label="Localization response (fixed scale)",
        )
        fig.text(
            0.5, 0.015,
            "Model localization heuristic; boxes are not pixel-ground-truth annotations.",
            ha="center", fontsize=8,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    finally:
        plt.close(fig)
    if multiscale_output_path is not None:
        _save_multiscale_diagnostic(
            original, localization, multiscale_output_path,
            raw_display_max=display_max,
        )
