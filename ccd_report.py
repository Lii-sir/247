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
        writer = csv.DictWriter(
            stream,
            fieldnames=["path", "label", "score", "score_topk", "score_max", "pred_label", "defect_type", "correct"],
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


def save_heatmap(
    image_path: Path,
    anomaly_map: np.ndarray,
    output_path: Path,
    *,
    display_max: float,
    score: float,
    threshold: float,
    ignore_mask: np.ndarray | None = None,
) -> None:
    """保存原图、热图和叠加图；色阶上限由调用方用正常验证集统一估计。"""
    display_max = _finite_number(display_max, "display_max")
    score = _finite_number(score, "score")
    threshold = _finite_number(threshold, "threshold")
    if display_max <= 0:
        raise ValueError("display_max 必须大于 0，且应由正常验证集统一估计。")
    values = np.asarray(anomaly_map)
    if values.ndim > 2:
        values = np.squeeze(values)
    if values.ndim != 2 or not values.size:
        raise ValueError("anomaly_map 必须是非空二维数组。")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError("anomaly_map 必须包含有限实数，不能有 NaN 或 Inf。")
    if ignore_mask is not None:
        ignore_mask = np.asarray(ignore_mask).squeeze().astype(bool)
        if ignore_mask.shape != values.shape:
            raise ValueError("ignore_mask 必须与 anomaly_map 具有相同的二维形状。")
        values = values.copy()
        values[ignore_mask] = 0
    with Image.open(image_path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    # 限制输出尺寸，同时保留原图长宽比，避免展示超大 CCD 图时占用过多内存。
    original.thumbnail((460, 900), Image.Resampling.LANCZOS)
    values = np.clip(values, 0, display_max).astype(np.float32)
    heatmap = np.asarray(Image.fromarray(values).resize(original.size, Image.Resampling.BILINEAR))
    height = min(11.0, max(3.4, 5.0 * original.height / original.width + 1.4))
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(15, height), dpi=100)
    try:
        axes[0].imshow(original)
        axes[0].set_title("Original")
        colored = axes[1].imshow(heatmap, cmap="turbo", vmin=0, vmax=display_max)
        axes[1].set_title("Anomaly heatmap")
        axes[2].imshow(original)
        axes[2].imshow(heatmap, cmap="turbo", vmin=0, vmax=display_max, alpha=0.45)
        axes[2].set_title("Overlay")
        for axis in axes:
            axis.set_axis_off()
        prediction = "Anomaly" if score > threshold else "Normal"
        fig.suptitle(f"{prediction} | score={score:.5g} | threshold={threshold:.5g}", fontsize=13)
        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.18, top=0.85, wspace=0.06)
        color_axis = fig.add_axes((0.35, 0.105, 0.30, 0.025))
        fig.colorbar(colored, cax=color_axis, orientation="horizontal", label="Anomaly value (fixed validation scale)")
        fig.text(0.5, 0.02, "Model heatmap, not pixel annotations. Values clipped to the shared display range.", ha="center", fontsize=9)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    finally:
        plt.close(fig)
