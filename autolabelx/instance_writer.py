from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import ClassConfig
from io_utils import write_json
from inference import InstancePrediction


def mask_to_polygons(mask: np.ndarray, min_area: float = 1.0) -> list[list[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        contour = contour.reshape(-1, 2)
        if contour.shape[0] < 3:
            continue
        polygons.append(contour.astype(float).reshape(-1).tolist())
    return polygons


def build_instance_record(
    image_path: Path,
    image_size: tuple[int, int],
    class_config: ClassConfig,
    prediction: InstancePrediction,
) -> dict:
    width, height = image_size
    annotations = []
    for idx, (box, score) in enumerate(zip(prediction.boxes_xyxy, prediction.scores), start=1):
        instance_mask = prediction.masks[idx - 1].cpu().numpy()[0].astype(np.uint8)
        annotations.append(
            {
                "id": idx,
                "label": class_config.name,
                "prompt_hits": prediction.labels[idx - 1] if idx - 1 < len(prediction.labels) else class_config.name,
                "score": float(score),
                "box": [float(value) for value in box.tolist()],
                "polygon": mask_to_polygons(instance_mask),
            }
        )

    return {
        "image_file": image_path.name,
        "image_path": str(image_path),
        "class_id": class_config.class_id,
        "class_name": class_config.name,
        "prompts": class_config.prompts,
        "image_width": width,
        "image_height": height,
        "annotations": annotations,
    }


def build_detection_record(
    image_path: Path,
    image_size: tuple[int, int],
    annotations: list[dict],
) -> dict:
    width, height = image_size
    return {
        "image_file": image_path.name,
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "annotations": annotations,
    }


def build_instance_segmentation_record(
    image_path: Path,
    image_size: tuple[int, int],
    annotations: list[dict],
) -> dict:
    width, height = image_size
    return {
        "image_file": image_path.name,
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "annotations": annotations,
    }


def convert_class_annotations_to_detection_annotations(
    class_record: dict,
) -> list[dict]:
    annotations = []
    for annotation in class_record.get("annotations", []):
        annotations.append(
            {
                "id": int(annotation["id"]),
                "class_id": int(class_record["class_id"]),
                "class_name": class_record["class_name"],
                "score": float(annotation["score"]),
                "box": list(annotation["box"]),
            }
        )
    return annotations


def save_instance_record(record: dict, path: Path) -> None:
    write_json(path, record)
