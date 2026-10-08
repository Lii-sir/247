"""Point matching coordinate exports; not part of shared registration logic."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from common.matching import MappedPoint, MatchResult


def export_csv(path: str | Path, results: Sequence[MatchResult], template_points: Sequence) -> None:
    """One row per image/point, retaining failure rows instead of silently dropping them."""
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "status", "message", "point_id", "template_x", "template_y", "target_x", "target_y", "inside_image", "good_matches", "inliers", "inlier_ratio", "median_error_px"])
        for result in results:
            points = result.points or [MappedPoint(i + 1, x, y, 0, 0, False) for i, (x, y) in enumerate(template_points)]
            for point in points:
                writer.writerow([
                    result.image_path, result.status, result.message, point.point_id,
                    round(point.template_x, 4), round(point.template_y, 4),
                    round(point.target_x, 4) if result.points else "",
                    round(point.target_y, 4) if result.points else "",
                    point.inside_image if result.points else "", result.good_matches,
                    result.inliers, round(result.inlier_ratio, 4),
                    round(result.median_error, 4) if result.median_error is not None else "",
                ])


def export_json(path: str | Path, template_path: str, template_points: Sequence, results: Sequence[MatchResult]) -> None:
    data = {
        "schema_version": 1,
        "coordinate_system": "original image pixels, origin top-left, x right, y down",
        "template_path": template_path,
        "template_points": [list(point) for point in template_points],
        "results": [asdict(result) for result in results],
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

