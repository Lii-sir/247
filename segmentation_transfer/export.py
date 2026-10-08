"""结果导出：独立实例二值掩膜 + 映射矩阵/对应点/匹配诊断 + 预览图。"""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common.image_io import write_image

from .calibration_io import calibration_document
from .models import TransferResult
from .visualization import render_comparison, render_views


def export_result(output: Path, result: TransferResult, metadata: dict | None = None) -> Path:
    output = Path(output).expanduser().resolve()
    # 每次导出要求新目录，不覆盖输入/旧实验；失败时不留下看似完整的结果。
    if output.exists():
        raise ValueError("输出目录已存在，请选择一个新的结果目录")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".transfer-", dir=output.parent) as temp:
        root = Path(temp)
        a, b = render_views(result)
        write_image(root / "a1.overlay.png", a)
        write_image(root / "b1.transferred.png", b)
        write_image(root / "comparison.jpg", render_comparison(result))
        instances = []
        for instance in result.instances:
            mask_path = f"masks/instance_{instance.source_id:03d}.png"
            write_image(root / mask_path, instance.mask.astype(np.uint8) * 255)
            instances.append({"source_id": instance.source_id, "class_id": instance.class_id,
                              "class_name": instance.class_name, "source_confidence": instance.source_confidence,
                              "status": instance.status, "source_area_pixels": instance.source_area,
                              "mapped_area_pixels": instance.area, "source_coverage": instance.source_coverage,
                              "box_xyxy_exclusive": instance.box, "mask": mask_path})
        document = {
            "schema_version": 1, "calibration": calibration_document(result.calibration),
            "metadata": metadata or {}, "a1": str(result.source.image_path), "b1": str(result.target_path),
            "target_size_wh": list(result.target_image.shape[1::-1]),
            "matrix_convention": "column vectors; A1->B1 = B->B1 @ A->B @ inverse(A->A1)",
            "h_template_a_to_b": result.fit.matrix.tolist(),
            "h_a1_to_b1": result.matrix_a1_to_b1.tolist(),
            "calibration_inliers": result.fit.inliers.tolist(),
            "calibration_errors_b_pixels": result.fit.errors.tolist(),
            "point_chain_errors_b1_pixels": result.point_errors_b1.tolist(),
            "point_chain_error_note": "Consistency only, not independent B1 accuracy measurement",
            "match_a": asdict(result.match_a), "match_b": asdict(result.match_b),
            "instances": instances, "warnings": list(result.warnings),
            "confidence_note": "Confidence is from A1 YOLO only; B1 is a geometric transfer, not a detection",
        }
        (root / "mapping.json").write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        # 同盘原子重命名；目录一旦出现即代表所有产物已写完。
        root.rename(output)
    return output

