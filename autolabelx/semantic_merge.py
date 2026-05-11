from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import AutoSegConfig, ClassConfig
from visualization import build_side_by_side_visualization, save_image


def polygon_to_mask(polygon: list[float], width: int, height: int) -> np.ndarray | None:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    mask_image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon([tuple(map(float, point)) for point in points], fill=1)
    return np.array(mask_image, dtype=bool)


def load_record(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def apply_annotation(
    semantic_mask: np.ndarray,
    score_map: np.ndarray,
    priority_map: np.ndarray,
    annotation: dict,
    class_config: ClassConfig,
    class_priority: int,
    width: int,
    height: int,
) -> None:
    score = float(annotation.get("score", 0.0))
    for polygon in annotation.get("polygon", []):
        polygon_mask = polygon_to_mask(polygon, width, height)
        if polygon_mask is None:
            continue
        better_score = polygon_mask & (score > score_map)
        tie_break = polygon_mask & np.isclose(score, score_map) & (class_priority > priority_map)
        update_mask = better_score | tie_break
        if not np.any(update_mask):
            continue
        semantic_mask[update_mask] = class_config.class_id
        score_map[update_mask] = score
        priority_map[update_mask] = class_priority


def merge_semantic_masks(
    config: AutoSegConfig,
    class_annotation_dirs: dict[str, Path],
    semantic_dir: Path,
    semantic_vis_dir: Path,
    image_names: list[str],
) -> None:
    semantic_dir.mkdir(parents=True, exist_ok=True)
    semantic_vis_dir.mkdir(parents=True, exist_ok=True)

    class_by_name = {class_config.name: class_config for class_config in config.classes}
    class_colors = {class_config.class_id: class_config.color for class_config in config.classes}

    for index, image_name in enumerate(image_names, start=1):
        image_stem = Path(image_name).stem
        records = {}
        first_record = None
        for class_config in config.classes:
            record_path = class_annotation_dirs[class_config.name] / f"{image_stem}.json"
            record = load_record(record_path)
            records[class_config.name] = record
            if first_record is None:
                first_record = record

        width = int(first_record["image_width"])
        height = int(first_record["image_height"])
        semantic_mask = np.full((height, width), config.ignore_index, dtype=np.uint8)
        score_map = np.full((height, width), -np.inf, dtype=np.float32)
        priority_map = np.full((height, width), -1, dtype=np.int16)

        for priority, class_config in enumerate(config.classes):
            record = records[class_config.name]
            for annotation in record.get("annotations", []):
                apply_annotation(
                    semantic_mask=semantic_mask,
                    score_map=score_map,
                    priority_map=priority_map,
                    annotation=annotation,
                    class_config=class_by_name[class_config.name],
                    class_priority=priority,
                    width=width,
                    height=height,
                )

        rgb_path = config.image_dir / image_name
        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing RGB image: {rgb_path}")

        Image.fromarray(semantic_mask, mode="L").save(semantic_dir / f"{image_stem}.png")
        vis = build_side_by_side_visualization(
            rgb_path=rgb_path,
            semantic_mask=semantic_mask,
            class_colors=class_colors,
            ignore_index=config.ignore_index,
        )
        save_image(vis, semantic_vis_dir / f"{image_stem}.png")
        print(f"[semantic {index}/{len(image_names)}] saved {image_stem}.png")
