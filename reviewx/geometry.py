from __future__ import annotations

import cv2
import numpy as np

from .types import Instance


def polygon_to_mask(points: list[list[float]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim == 2 and polygon.shape[0] >= 3 and polygon.shape[1] == 2:
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def polygon_to_cropped_mask(
    points: list[list[float]], width: int, height: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    polygon = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32)
    x1 = max(0, int(polygon[:, 0].min()))
    y1 = max(0, int(polygon[:, 1].min()))
    x2 = min(width, int(polygon[:, 0].max()) + 1)
    y2 = min(height, int(polygon[:, 1].max()) + 1)
    if x1 >= x2 or y1 >= y2:
        return np.zeros((0, 0), dtype=bool), (0, 0, 0, 0)
    shifted = polygon - np.asarray([x1, y1], dtype=np.int32)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted], 1)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return mask.astype(bool), (0, 0, 0, 0)
    local_x1, local_y1 = int(xs.min()), int(ys.min())
    local_x2, local_y2 = int(xs.max()) + 1, int(ys.max()) + 1
    return (
        mask[local_y1:local_y2, local_x1:local_x2].astype(bool),
        (x1 + local_x1, y1 + local_y1, x1 + local_x2, y1 + local_y2),
    )


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return intersection / union if union else 0.0


def intersection_over_smaller(left: np.ndarray, right: np.ndarray) -> float:
    denominator = min(int(left.sum()), int(right.sum()))
    return int(np.logical_and(left, right).sum()) / denominator if denominator else 0.0


def instance_intersection(left: Instance, right: Instance) -> int:
    x1 = max(left.box[0], right.box[0])
    y1 = max(left.box[1], right.box[1])
    x2 = min(left.box[2], right.box[2])
    y2 = min(left.box[3], right.box[3])
    if x1 >= x2 or y1 >= y2:
        return 0
    left_crop = left.mask[
        y1 - left.box[1] : y2 - left.box[1],
        x1 - left.box[0] : x2 - left.box[0],
    ]
    right_crop = right.mask[
        y1 - right.box[1] : y2 - right.box[1],
        x1 - right.box[0] : x2 - right.box[0],
    ]
    return int(np.logical_and(left_crop, right_crop).sum())


def instance_iou(left: Instance, right: Instance) -> float:
    intersection = instance_intersection(left, right)
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


def instance_intersection_over_smaller(left: Instance, right: Instance) -> float:
    denominator = min(left.area, right.area)
    return instance_intersection(left, right) / denominator if denominator else 0.0


def deduplicate_predictions(
    instances: list[Instance], threshold: float = 0.8
) -> list[Instance]:
    kept: list[Instance] = []
    for candidate in sorted(
        instances, key=lambda item: item.score or 0.0, reverse=True
    ):
        if all(instance_iou(candidate, prior) < threshold for prior in kept):
            kept.append(candidate)
    for index, instance in enumerate(kept, start=1):
        instance.instance_id = index
    return kept


def greedy_match(
    human: list[Instance], predicted: list[Instance], minimum_iou: float = 0.05
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates = [
        (instance_iou(h, p), hi, pi)
        for hi, h in enumerate(human)
        for pi, p in enumerate(predicted)
    ]
    used_human: set[int] = set()
    used_predicted: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, hi, pi in sorted(candidates, reverse=True):
        if iou < minimum_iou:
            break
        if hi not in used_human and pi not in used_predicted:
            matches.append((hi, pi, iou))
            used_human.add(hi)
            used_predicted.add(pi)
    return (
        matches,
        [index for index in range(len(human)) if index not in used_human],
        [index for index in range(len(predicted)) if index not in used_predicted],
    )
