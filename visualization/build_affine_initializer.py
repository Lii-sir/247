"""Fit a guarded coarse-relative affine prior; validate matrices without image pixels."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from algorithm.vision3d.alignment.registration import (_affine_parameters_from_matrix, _affine_regions,
    _calibration_seed, _full_affine)


def affine_matrix(raw):
    if (not isinstance(raw, list) or len(raw) not in (2, 3)
            or any(not isinstance(row, list) or len(row) != 3 for row in raw)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for row in raw for v in row)):
        raise ValueError("Expected a numeric 2x3 or 3x3 affine matrix")
    matrix = np.asarray(raw if len(raw) == 3 else [*raw, [0., 0., 1.]], dtype=float)
    if not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
        raise ValueError("Nonfinite or singular affine matrix")
    matrix = matrix / matrix[2, 2]
    if not np.allclose(matrix[2], [0., 0., 1.], rtol=0, atol=1e-12) or np.linalg.det(matrix[:2, :2]) <= 0:
        raise ValueError("Expected an orientation-preserving affine matrix")
    return matrix


def context_key(context):
    for key in ("image_size", "depth_size"):
        size = context.get(key)
        if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError("Missing or invalid image/depth dimensions")
    if (context.get("source_space") != "independent_2d_image"
            or context.get("target_space") != "depth_piece_local"
            or not isinstance(context.get("preprocessing"), dict)):
        raise ValueError("Initializer requires independent 2D to depth-piece coordinates")
    return json.dumps([context["source_space"], context["target_space"], context["image_size"],
                       context["depth_size"][0], context["preprocessing"]], sort_keys=True)


def load_rows(document, csv_path=None):
    if (not isinstance(document, dict) or document.get("format") != "board_piece_registration_v1"
            or document.get("complete") is not True):
        raise ValueError("Expected a complete board_piece_registration_v1 report")
    rows, keys = [], set()
    for board in document["boards"]:
        board_id = str(board["board_id"])
        if not board_id.isascii() or not board_id.isdigit():
            raise ValueError("Board IDs must be nonnegative integers")
        board_id = str(int(board_id))
        contexts = {p["piece_id"]: p["context"] for p in board["preparation"]["pieces"]}
        for piece in board["pieces"]:
            if type(piece["piece_id"]) is not int or piece["piece_id"] < 1:
                raise ValueError("Piece IDs must be positive integers")
            key = (board_id, piece["piece_id"])
            if key in keys:
                raise ValueError(f"Duplicate board/piece: {key}")
            keys.add(key)
            if piece["registration"].get("is_valid") is not True:
                raise ValueError(f"Invalid registration cannot train the prior: {key}")
            context = copy.deepcopy(contexts[piece["piece_id"]])
            context_key(context)
            if piece["depth_size"] != context["depth_size"]:
                raise ValueError(f"Depth dimensions disagree: {key}")
            final = affine_matrix(piece["registration"]["homography_2d_to_3d"])
            coarse = affine_matrix(piece["coarse_registration"]["homography_2d_to_3d"])
            rows.append(dict(board_id=key[0], piece_id=key[1], context=context,
                             final=final, coarse=coarse, correction=final @ np.linalg.inv(coarse)))
    if not rows or len({row["board_id"] for row in rows}) < 2:
        raise ValueError("At least two boards are required for held-out-board validation")
    if len({context_key(row["context"]) for row in rows}) != 1:
        raise ValueError("Mixed coordinate/Pitch contexts; fit separate profiles")
    if csv_path is not None:
        with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        by_key = {(str(int(r["board_id"])), int(r["piece_id"])): r for r in csv_rows}
        if len(by_key) != len(csv_rows) or set(by_key) != keys:
            raise ValueError("CSV and JSON board/piece sets disagree or contain duplicates")
        for row in rows:
            record = by_key[(row["board_id"], row["piece_id"])]
            for column, matrix in (("matrix_2d_to_3d", row["final"]),
                                   ("coarse_matrix_2d_to_3d", row["coarse"])):
                if not np.allclose(affine_matrix(json.loads(record[column])), matrix, rtol=0, atol=1e-12):
                    raise ValueError(f"CSV/JSON matrix disagreement: {row['board_id']}/{row['piece_id']}")
    return rows


def fit_profile(rows, provenance=None):
    matrix = affine_matrix(np.mean([row["correction"] for row in rows], axis=0).tolist())
    return dict(format="affine_initializer_profile", version=1, mode="coarse_relative",
        matrix=matrix.tolist(),
        context=copy.deepcopy(rows[0]["context"]),
        source=dict(board_ids=sorted({row["board_id"] for row in rows}), piece_count=len(rows),
                    method="mean(final_matrix @ inverse(current_coarse_matrix))",
                    usage="Initialization only; never a final alignment matrix",
                    validation="Matrix prediction, not image fitting or height accuracy",
                    **(provenance or {})))


def seed_matrix(row, profile, values):
    scale = values.get("preview_scale", .5)
    p0 = _affine_parameters_from_matrix(row["coarse"][:2], scale)
    delta, shear = values.get("scale_delta", .02), values.get("shear_limit", .02)
    tx, ty = values.get("translation_delta", [35., 25.])
    radius = np.array([delta, delta, shear, shear, tx, ty])
    lower, upper = p0-radius, p0+radius
    lower[:2] = np.maximum(lower[:2], .05)
    try:
        parameters = _calibration_seed(profile, row["context"], p0, scale, lower, upper)
    except ValueError as error:
        return row["coarse"], str(error)
    return np.vstack([_full_affine(parameters, scale), [0., 0., 1.]]), None


def cross_validate(rows, values):
    records, folds = [], []
    for held_out in sorted({row["board_id"] for row in rows}):
        training = [row for row in rows if row["board_id"] != held_out]
        testing = [row for row in rows if row["board_id"] == held_out]
        prior = fit_profile(training)
        legacy = copy.deepcopy(prior)
        legacy.update(format="registration_lab_matrix", matrix=np.eye(3).tolist())
        linear = np.mean([row["final"][:2, :2] for row in training], axis=0)
        for i in range(2):
            legacy["matrix"][i][:2] = linear[i].tolist()
        folds.append(dict(held_out_board=held_out, training_boards=prior["source"]["board_ids"],
                          training_piece_count=len(training), testing_piece_count=len(testing),
                          profile=prior))
        for row in testing:
            width, height = row["context"]["depth_size"]
            points = []
            for _, (x0, y0, x1, y1) in _affine_regions(values.get("rois"), width, height,
                                                      values.get("selected_regions")):
                points.extend([[x, y, 1.] for y in np.linspace(y0, y1-1, 5)
                               for x in np.linspace(x0, x1-1, 5)])
            if not points:
                raise ValueError("No scoring regions for matrix validation")
            target = np.asarray(points)
            source = target @ np.linalg.inv(row["final"]).T
            for name, profile in (("coarse", None), ("linear_only", legacy), ("coarse_relative", prior)):
                matrix, reason = (row["coarse"], None) if profile is None else seed_matrix(row, profile, values)
                distance = np.linalg.norm(source @ matrix[:2].T-target[:, :2], axis=1)
                records.append(dict(method=name, held_out_board=held_out, piece_id=row["piece_id"],
                    point_count=len(points), squared_error=float(np.sum(distance**2)),
                    rms_depth_px=float(np.sqrt(np.mean(distance**2))), max_depth_px=float(distance.max()),
                    fallback_reason=reason))
    summary = {}
    for method in ("coarse", "linear_only", "coarse_relative"):
        subset = [row for row in records if row["method"] == method]
        summary[method] = dict(piece_count=len(subset),
            rms_depth_px=float(np.sqrt(sum(r["squared_error"] for r in subset)/sum(r["point_count"] for r in subset))),
            p95_piece_rms_depth_px=float(np.percentile([r["rms_depth_px"] for r in subset], 95)),
            max_depth_px=max(r["max_depth_px"] for r in subset),
            fallback_count=sum(r["fallback_reason"] is not None for r in subset))
    return dict(method="Leave one board out; compare predicted matrices to stored fitted matrices, not ground truth",
                pixel_domain="Depth-piece scoring regions; 5x5 sampled points per region", folds=folds,
                summary=summary, pieces=records, image_refitting_performed=False, speedup_verified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--matrices-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "visualization/output/affine-initializer")
    args = parser.parse_args(argv)
    document = json.loads(args.results.read_text(encoding="utf-8-sig"))
    rows = load_rows(document, args.matrices_csv)
    provenance = dict(results_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest())
    if args.matrices_csv:
        provenance["matrices_csv_sha256"] = hashlib.sha256(args.matrices_csv.read_bytes()).hexdigest()
    profile = fit_profile(rows, provenance)
    validation = cross_validate(rows, document["affine_parameters"])
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in (("initializer-profile.json", profile), ("calibration-block.json", dict(enabled=False, profile=profile)),
                        ("cross-validation.json", validation)):
        (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    fold_folder = folder / "folds"
    fold_folder.mkdir(exist_ok=True)
    for fold in validation["folds"]:
        (fold_folder / f"holdout-{fold['held_out_board']}.json").write_text(
            json.dumps(fold["profile"], ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (folder / "cross-validation.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(validation["pieces"][0]))
        writer.writeheader(); writer.writerows(validation["pieces"])
    lines = ["# 四板仿射初始化模型", "", "模型：`M_start = R @ M_coarse_current`，R 为训练片 `M_final @ inverse(M_coarse)` 的均值。",
        "每片使用自身的粗配准，随后仍运行原有六参数精修。上下文、边界和完整评分检查均保留。",
        "", "| 初始化方式 | 留出板矩阵预测 RMS / 深度像素 | 单点最大差异 | 回退数 |", "|---|---:|---:|---:|"]
    for name, row in validation["summary"].items():
        lines.append(f"| {name} | {row['rms_depth_px']:.6f} | {row['max_depth_px']:.6f} | {row['fallback_count']} |")
    lines += ["", "这里的差异以已保存的拟合矩阵为参考，不是真实几何精度或测高精度；未使用原图重新评分，也未验证提速。",
        "每轮只使用另外三板（一般情况下为 N-1 板）拟合，留出整板验证；最终文件再使用全部有效片拟合。",
        "", "Pitch/尺寸上下文：", "```json", json.dumps(profile["context"], ensure_ascii=False, indent=2), "```",
        "", "`initializer-profile.json` 只用于 dynamic_affine_3d.parameters.calibration.profile，不能当最终配准矩阵复用。",
        "`calibration-block.json` 默认关闭，便于在原图所在机器完成 current/calibration 对照后启用。生成器不修改生产配置。",
        "可将 profile 嵌入上述 calibration.profile，并设置 calibration.enabled=true。上下文不匹配、越界或起点评分不改善时自动回退粗配准。",
        "原图复测时使用 folds/holdout-板号.json，该文件仅由另外三板训练，避免训练板参与自身测试。",
        "在项目根目录，例如对留出的第1板运行：", "```powershell",
        '.\\.venv\\Scripts\\python.exe -B visualization/benchmark_affine.py --images-dir demodatasets/20261106-2/2D --depth demodatasets/20261106-2/3D/1.png --board-id 1 --workers 1 --opencv-threads 1 --variants current calibration --profile "'+str(fold_folder / 'holdout-1.json')+'" --repeats 3 --output-dir visualization/output/affine-initializer-image-check',
        "```", "将板号、深度图文件名及 holdout 文件一起改为2/3/4，可完成另外三轮。跨电脑复制时相应调整 --profile 路径。", ""]
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(output=str(folder), profile=profile, validation_summary=validation["summary"]), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
