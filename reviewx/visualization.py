from __future__ import annotations

import textwrap
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .geometry import instance_iou
from .types import ImageAudit, Instance, Issue


def _annotation_font(image_size: tuple[int, int], scale: float = 0.025):
    font_size = max(16, int(min(image_size) * scale))
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _expanded_crop(
    box: tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
    minimum_size: int = 180,
) -> tuple[int, int, int, int]:
    width, height = image_size
    if box is None:
        return (0, 0, width, height)
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    crop_width = min(width, max(minimum_size, int(box_width * 2.0)))
    crop_height = min(height, max(minimum_size, int(box_height * 2.0)))
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    crop_x1 = min(max(0, center_x - crop_width // 2), width - crop_width)
    crop_y1 = min(max(0, center_y - crop_height // 2), height - crop_height)
    return (
        crop_x1,
        crop_y1,
        crop_x1 + crop_width,
        crop_y1 + crop_height,
    )


def _mask_in_crop(
    instance: Instance, crop_box: tuple[int, int, int, int]
) -> np.ndarray:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    result = np.zeros((crop_y2 - crop_y1, crop_x2 - crop_x1), dtype=bool)
    x1 = max(crop_x1, instance.box[0])
    y1 = max(crop_y1, instance.box[1])
    x2 = min(crop_x2, instance.box[2])
    y2 = min(crop_y2, instance.box[3])
    if x1 >= x2 or y1 >= y2:
        return result
    result[y1 - crop_y1 : y2 - crop_y1, x1 - crop_x1 : x2 - crop_x1] = (
        instance.mask[
            y1 - instance.box[1] : y2 - instance.box[1],
            x1 - instance.box[0] : x2 - instance.box[0],
        ]
    )
    return result


def _blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    if not mask.any():
        return
    image[mask] = image[mask] * 0.45 + np.asarray(color, dtype=np.float32) * 0.55


def _draw_mask_outline(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    draw = ImageDraw.Draw(image)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        points = [(int(p[0][0]), int(p[0][1])) for p in contour]
        if len(points) >= 2:
            draw.line(points + [points[0]], fill=color, width=width)


def _panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    contained = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (28, 28, 28))
    canvas.paste(
        contained,
        ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2),
    )
    return canvas


def _select_model_instance(
    issue: Issue,
    instances: list[Instance],
    predictions: dict[str, list[Instance]] | None,
) -> Instance | None:
    if not predictions or not issue.model_class:
        return None
    candidates = predictions.get(issue.model_class, [])
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.box == issue.box:
            return candidate
    relevant = [item for item in instances if item.instance_id in issue.instance_ids]
    if relevant:
        return max(
            candidates,
            key=lambda candidate: max(
                instance_iou(candidate, human) for human in relevant
            ),
        )
    return max(candidates, key=lambda item: item.score or 0.0)


def render_audit(
    image_path: Path,
    instances: list[Instance],
    audit: ImageAudit,
    output_path: Path,
    predictions: dict[str, list[Instance]] | None = None,
) -> None:
    """Render a lightweight full-image locator overview."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = _annotation_font(image.size)
    issue_ids = {value for issue in audit.issues for value in issue.instance_ids}
    for instance in instances:
        color = (
            (255, 50, 50, 220)
            if instance.instance_id in issue_ids
            else (0, 220, 255, 150)
        )
        points = [tuple(point) for point in instance.points]
        if len(points) >= 3:
            draw.line(points + [points[0]], fill=color, width=3)
        x1, y1, _, _ = instance.box
        draw.text(
            (x1 + 2, max(0, y1 - 20)),
            instance.label,
            fill=color,
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 220),
        )
    for number, issue in enumerate(audit.issues, start=1):
        if issue.box is None:
            continue
        x1, y1, x2, y2 = issue.box
        draw.rectangle((x1, y1, x2, y2), outline=(255, 215, 0, 255), width=4)
        draw.text(
            (x1 + 2, y1 + 2),
            f"问题{number}：{issue.display_name}",
            fill=(255, 215, 0, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_issue_evidence(
    image_path: Path,
    instances: list[Instance],
    issue: Issue,
    issue_number: int,
    output_path: Path,
    predictions: dict[str, list[Instance]] | None = None,
) -> None:
    """Render a four-panel, zoomed evidence card for one issue."""
    full_image = Image.open(image_path).convert("RGB")
    crop_box = _expanded_crop(issue.box, full_image.size)
    original = full_image.crop(crop_box)
    original_array = np.asarray(original).astype(np.float32)
    relevant = [item for item in instances if item.instance_id in issue.instance_ids]
    model_instance = _select_model_instance(issue, instances, predictions)

    original_panel = original.copy()
    if issue.box is not None:
        draw = ImageDraw.Draw(original_panel)
        draw.rectangle(
            (
                issue.box[0] - crop_box[0],
                issue.box[1] - crop_box[1],
                issue.box[2] - crop_box[0],
                issue.box[3] - crop_box[1],
            ),
            outline=(255, 215, 0),
            width=4,
        )

    human_array = original_array.copy()
    human_masks: list[np.ndarray] = []
    human_targets = relevant
    colors = [(255, 55, 55), (30, 144, 255), (255, 165, 0), (50, 205, 50)]
    for index, instance in enumerate(human_targets):
        mask = _mask_in_crop(instance, crop_box)
        human_masks.append(mask)
        _blend_mask(human_array, mask, colors[index % len(colors)])
    human_panel = Image.fromarray(np.clip(human_array, 0, 255).astype(np.uint8))
    for index, mask in enumerate(human_masks):
        _draw_mask_outline(human_panel, mask, colors[index % len(colors)])

    model_array = original_array.copy()
    model_mask = None
    if model_instance is not None:
        model_mask = _mask_in_crop(model_instance, crop_box)
        _blend_mask(model_array, model_mask, (255, 0, 255))
    model_panel = Image.fromarray(np.clip(model_array, 0, 255).astype(np.uint8))
    if model_mask is not None:
        _draw_mask_outline(model_panel, model_mask, (255, 0, 255))

    difference = original_array.copy()
    human_union = (
        np.logical_or.reduce(human_masks)
        if human_masks
        else np.zeros(original_array.shape[:2], dtype=bool)
    )
    if model_mask is not None:
        common = np.logical_and(human_union, model_mask)
        human_only = np.logical_and(human_union, ~model_mask)
        model_only = np.logical_and(model_mask, ~human_union)
        _blend_mask(difference, common, (0, 210, 80))
        _blend_mask(difference, human_only, (255, 50, 50))
        _blend_mask(difference, model_only, (255, 0, 255))
    elif len(human_masks) >= 2:
        overlap = np.logical_and(human_masks[0], human_masks[1])
        _blend_mask(difference, human_masks[0], (255, 50, 50))
        _blend_mask(difference, human_masks[1], (30, 144, 255))
        _blend_mask(difference, overlap, (255, 215, 0))
    elif human_masks:
        _blend_mask(difference, human_union, (255, 50, 50))
    difference_panel = Image.fromarray(np.clip(difference, 0, 255).astype(np.uint8))

    if model_instance is not None:
        panels = [original_panel, human_panel, model_panel, difference_panel]
        titles = [
            "1. 原始局部图（黄色框为问题位置）",
            "2. 人工标注",
            f"3. SAM 3：{model_instance.label}",
            "4. 差异图：绿=一致，红=仅人工，洋红=仅模型",
        ]
    elif len(human_masks) >= 2:
        separated_panels = []
        for index, mask in enumerate(human_masks[:2]):
            separated = original_array.copy()
            color = colors[index]
            _blend_mask(separated, mask, color)
            separated_image = Image.fromarray(
                np.clip(separated, 0, 255).astype(np.uint8)
            )
            _draw_mask_outline(separated_image, mask, color)
            separated_panels.append(separated_image)
        left_label = human_targets[0].label
        right_label = human_targets[1].label
        panels = [original_panel, *separated_panels, difference_panel]
        titles = [
            "1. 原始局部图（黄色框为问题位置）",
            f"2. 人工标注：{left_label}",
            f"3. 人工标注：{right_label}",
            f"4. 重叠图：红={left_label}，蓝={right_label}，黄=重叠",
        ]
    elif human_masks:
        panels = [original_panel, human_panel]
        titles = [
            "1. 原始局部图（黄色框为问题位置）",
            f"2. 人工标注：{human_targets[0].label}",
        ]
    else:
        panels = [original_panel]
        titles = ["原始局部图（黄色框为问题位置）"]

    four_panel = len(panels) == 4
    canvas_height = 760 if four_panel else 470
    canvas = Image.new("RGB", (900, canvas_height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    title_font = _annotation_font(canvas.size, 0.034)
    text_font = _annotation_font(canvas.size, 0.022)
    small_font = _annotation_font(canvas.size, 0.018)
    draw.text(
        (24, 14),
        f"问题 {issue_number}：{issue.display_name}（{issue.severity}）",
        fill=(180, 20, 20),
        font=title_font,
    )
    reason_lines = textwrap.wrap(issue.reason, width=48) or [issue.reason]
    draw.multiline_text(
        (24, 58), "\n".join(reason_lines[:2]), fill=(25, 25, 25), font=text_font
    )
    metrics = []
    label_counts = Counter(item.label for item in relevant)
    if label_counts:
        human_labels = "、".join(
            label if count == 1 else f"{label}×{count}"
            for label, count in label_counts.items()
        )
        metrics.append(f"人工类别：{human_labels}")
    if issue.model_class:
        metrics.append(f"模型类别：{issue.model_class}")
    if issue.model_score is not None:
        metrics.append(f"模型置信度：{issue.model_score:.3f}")
    if issue.mask_iou is not None:
        metrics.append(f"掩码交并比：{issue.mask_iou:.3f}")
    draw.text((24, 116), "　".join(metrics), fill=(70, 70, 70), font=small_font)

    panel_size = (420, 250)
    if four_panel:
        positions = [(24, 175), (456, 175), (24, 465), (456, 465)]
    elif len(panels) == 2:
        positions = [(24, 175), (456, 175)]
    else:
        positions = [(240, 175)]
    for panel_image, title, (x, y) in zip(panels, titles, positions):
        draw.text((x, y - 28), title, fill=(20, 20, 20), font=small_font)
        canvas.paste(_panel(panel_image, panel_size), (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
