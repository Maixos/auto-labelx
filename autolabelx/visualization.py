from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_annotation_font(image_size: tuple[int, int]) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    width, height = image_size
    font_size = max(18, int(min(width, height) * 0.03))
    font_candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in font_candidates:
        if os.path.isfile(font_path):
            try:
                return ImageFont.truetype(font_path, size=font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_annotation_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    x1, y1 = position
    padding_x = 8
    padding_y = 6
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    x1 = max(0, float(x1))
    y1 = max(0, float(y1) - text_height - padding_y * 2)
    x2 = x1 + text_width + padding_x * 2
    y2 = y1 + text_height + padding_y * 2
    draw.rectangle((x1, y1, x2, y2), fill=(0, 255, 0))
    draw.text((x1 + padding_x, y1 + padding_y), text, fill=(0, 0, 0), font=font)


def draw_instance_preview(
    image_rgb: np.ndarray,
    boxes_xyxy,
    labels: list[str],
    scores,
    masks,
) -> Image.Image:
    overlay = image_rgb.copy().astype(np.float32)
    rng = np.random.default_rng(42)

    for mask in masks:
        color = rng.integers(64, 256, size=3, dtype=np.uint8)
        binary_mask = mask.cpu().numpy()[0].astype(bool)
        overlay[binary_mask] = overlay[binary_mask] * 0.45 + color * 0.55

    preview = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(preview)
    font = get_annotation_font(preview.size)

    for box, label, score in zip(boxes_xyxy, labels, scores):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        _draw_annotation_text(draw, (x1, y1), f"{label} {float(score):.2f}", font)
    return preview


def draw_detection_annotations_preview(
    image_rgb: np.ndarray,
    annotations: list[dict],
) -> Image.Image:
    preview = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(preview)
    font = get_annotation_font(preview.size)

    for annotation in annotations:
        box = annotation.get("box", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        label = str(annotation.get("class_name", "object"))
        score = float(annotation.get("score", 0.0))
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        _draw_annotation_text(draw, (x1, y1), f"{label} {score:.2f}", font)
    return preview


def draw_instance_annotations_preview(
    image_rgb: np.ndarray,
    annotations: list[dict],
    class_colors: dict[str, tuple[int, int, int]],
) -> Image.Image:
    overlay = image_rgb.copy().astype(np.float32)

    for annotation in annotations:
        class_name = str(annotation.get("class_name", "object"))
        color = np.asarray(class_colors.get(class_name, (0, 255, 0)), dtype=np.float32)
        for polygon in annotation.get("polygon", []):
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if points.shape[0] < 3:
                continue
            mask_canvas = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_canvas, [points.astype(np.int32)], 1)
            binary_mask = mask_canvas.astype(bool)
            overlay[binary_mask] = overlay[binary_mask] * 0.45 + color * 0.55

    preview = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(preview)
    font = get_annotation_font(preview.size)

    for annotation in annotations:
        box = annotation.get("box", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        label = str(annotation.get("class_name", "object"))
        score = float(annotation.get("score", 0.0))
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        _draw_annotation_text(draw, (x1, y1), f"{label} {score:.2f}", font)
    return preview


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def colorize_semantic_mask(
    semantic_mask: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]],
    ignore_index: int,
) -> Image.Image:
    color_mask = np.zeros((*semantic_mask.shape, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        color_mask[semantic_mask == class_id] = color
    color_mask[semantic_mask == ignore_index] = (0, 0, 0)
    return Image.fromarray(color_mask, mode="RGB")


def build_side_by_side_visualization(
    rgb_path: Path,
    semantic_mask: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]],
    ignore_index: int,
) -> Image.Image:
    rgb_image = Image.open(rgb_path).convert("RGB")
    color_mask = colorize_semantic_mask(semantic_mask, class_colors, ignore_index)
    if rgb_image.size != color_mask.size:
        raise ValueError(
            f"RGB image size {rgb_image.size} does not match mask size {color_mask.size} for {rgb_path.name}"
        )
    canvas = Image.new("RGB", (rgb_image.width * 2, rgb_image.height))
    canvas.paste(rgb_image, (0, 0))
    canvas.paste(color_mask, (rgb_image.width, 0))
    return canvas


def polygons_to_overlay(image_rgb: np.ndarray, polygons: list[list[float]], color: tuple[int, int, int]) -> np.ndarray:
    overlay = image_rgb.copy().astype(np.float32)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 3:
            continue
        mask_canvas = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_canvas, [points.astype(np.int32)], 1)
        binary_mask = mask_canvas.astype(bool)
        overlay[binary_mask] = overlay[binary_mask] * 0.45 + np.asarray(color) * 0.55
    return overlay.astype(np.uint8)
