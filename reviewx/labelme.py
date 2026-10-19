from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .geometry import polygon_to_cropped_mask
from .types import Instance, Issue


def find_pairs(dataset_dir: Path) -> tuple[list[tuple[Path, Path]], list[Issue]]:
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    pairs: list[tuple[Path, Path]] = []
    dataset_issues: list[Issue] = []
    for annotation_path in sorted(dataset_dir.rglob("*.json")):
        image_path = next(
            (
                annotation_path.with_suffix(ext)
                for ext in image_extensions
                if annotation_path.with_suffix(ext).is_file()
            ),
            None,
        )
        if image_path is None:
            dataset_issues.append(
                Issue(
                    issue_type="MISSING_IMAGE",
                    severity="FAIL",
                    reason=f"找不到与标注文件同名的图像：{annotation_path}",
                    source="format",
                )
            )
        else:
            pairs.append((image_path, annotation_path))
    annotation_stems = {path.with_suffix("") for path in dataset_dir.rglob("*.json")}
    for extension in image_extensions:
        for image_path in dataset_dir.rglob(f"*{extension}"):
            if image_path.with_suffix("") not in annotation_stems:
                dataset_issues.append(
                    Issue(
                        issue_type="MISSING_ANNOTATION",
                        severity="FAIL",
                        reason=f"找不到与图像同名的 JSON 标注文件：{image_path}",
                        source="format",
                    )
                )
    return pairs, dataset_issues


def load_labelme(
    image_path: Path,
    annotation_path: Path,
    allowed_labels: set[str],
    min_polygon_area: float,
) -> tuple[list[Instance], list[Issue], tuple[int, int]]:
    issues: list[Issue] = []
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [], [Issue("INVALID_JSON", "FAIL", f"读取 JSON 标注文件失败：{error}", source="format")], (0, 0)

    with Image.open(image_path) as image:
        width, height = image.size
    if data.get("imageWidth") != width or data.get("imageHeight") != height:
        issues.append(
            Issue(
                "IMAGE_SIZE_MISMATCH",
                "FAIL",
                f"JSON 记录的尺寸为 {data.get('imageWidth')}×{data.get('imageHeight')}，实际图像尺寸为 {width}×{height}",
                source="format",
            )
        )
    if Path(str(data.get("imagePath", ""))).name != image_path.name:
        issues.append(
            Issue(
                "IMAGE_PATH_MISMATCH",
                "FAIL",
                f"JSON 中的 imagePath 为 {data.get('imagePath')!r}，实际图像文件名为 {image_path.name!r}",
                source="format",
            )
        )

    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return (
            [],
            issues
            + [
                Issue(
                    "INVALID_SHAPES", "FAIL", "JSON 中的 shapes 必须是列表", source="format"
                )
            ],
            (width, height),
        )

    instances: list[Instance] = []
    for shape_index, shape in enumerate(shapes, start=1):
        if not isinstance(shape, dict):
            issues.append(
                Issue(
                    "INVALID_SHAPE",
                    "FAIL",
                    "标注列表中存在不是合法对象结构的内容",
                    [shape_index],
                    source="format",
                )
            )
            continue
        label = shape.get("label")
        if not isinstance(label, str) or label not in allowed_labels:
            issues.append(
                Issue(
                    "UNKNOWN_LABEL",
                    "FAIL",
                    f"类别 {label!r} 未配置，或类别名称大小写不正确",
                    [shape_index],
                    source="format",
                )
            )
        if shape.get("shape_type", "polygon") != "polygon":
            issues.append(
                Issue(
                    "INVALID_SHAPE_TYPE",
                    "FAIL",
                    f"类别 {label!r} 的 shape_type 不是 polygon",
                    [shape_index],
                    source="format",
                )
            )
            continue
        points = shape.get("points")
        valid_points = (
            isinstance(points, list)
            and len(points) >= 3
            and all(
                isinstance(p, list)
                and len(p) == 2
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)
                for p in points
            )
        )
        if not valid_points:
            issues.append(
                Issue(
                    "INVALID_POLYGON",
                    "FAIL",
                    f"类别 {label!r} 的多边形至少需要 3 个有效坐标点",
                    [shape_index],
                    source="format",
                )
            )
            continue
        outside = [
            (x, y) for x, y in points if x < 0 or x > width or y < 0 or y > height
        ]
        point_box = (
            max(0, min(width, int(min(point[0] for point in points)))),
            max(0, min(height, int(min(point[1] for point in points)))),
            max(0, min(width, int(max(point[0] for point in points)) + 1)),
            max(0, min(height, int(max(point[1] for point in points)) + 1)),
        )
        if outside:
            issues.append(
                Issue(
                    "POINT_OUT_OF_BOUNDS",
                    "FAIL",
                    f"类别 {label!r} 有 {len(outside)} 个多边形点超出图像边界",
                    [shape_index],
                    box=point_box,
                    source="format",
                )
            )
        contour = cv2.contourArea(np.asarray(points, dtype="float32"))
        mask, box = polygon_to_cropped_mask(points, width, height)
        area = int(mask.sum())
        if contour < min_polygon_area or area == 0:
            reason = (
                f"类别 {label!r} 的多边形裁剪到图像范围后为空"
                if area == 0
                else f"类别 {label!r} 的多边形面积仅为 {contour:.2f} 像素"
            )
            issues.append(
                Issue(
                    "EMPTY_OR_TINY_POLYGON",
                    "FAIL",
                    reason,
                    [shape_index],
                    box=point_box,
                    source="format",
                )
            )
            continue
        instances.append(
            Instance(shape_index, str(label), mask, box, area, points=points)
        )
    return instances, issues, (width, height)
