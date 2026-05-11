from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from config import AutoSegConfig
from inference import Sam3AutoSegInferencer
from instance_writer import (
    build_detection_record,
    build_instance_record,
    build_instance_segmentation_record,
    convert_class_annotations_to_detection_annotations,
    save_instance_record,
)
from io_utils import (
    create_symlink,
    ensure_dir,
    list_images,
    load_image,
    slugify,
    timestamp_now,
    write_json,
)
from semantic_merge import merge_semantic_masks
from visualization import (
    draw_detection_annotations_preview,
    draw_instance_annotations_preview,
    draw_instance_preview,
    save_image,
)


def _build_run_dirs(config: AutoSegConfig) -> dict[str, Path]:
    run_dir = ensure_dir(config.output_root / timestamp_now())
    return {
        "run_dir": run_dir,
        "images_dir": ensure_dir(run_dir / "images"),
        "classes_dir": ensure_dir(run_dir / "classes"),
        "detection_annotations_dir": ensure_dir(run_dir / "detection" / "annotations"),
        "detection_visualizations_dir": ensure_dir(
            run_dir / "detection" / "visualizations"
        ),
        "instance_annotations_dir": ensure_dir(
            run_dir / "instance_segmentation" / "annotations"
        ),
        "instance_visualizations_dir": ensure_dir(
            run_dir / "instance_segmentation" / "visualizations"
        ),
        "semantic_annotations_dir": ensure_dir(
            run_dir / "semantic_segmentation" / "annotations"
        ),
        "semantic_visualizations_dir": ensure_dir(
            run_dir / "semantic_segmentation" / "visualizations"
        ),
    }


def _class_dirs(classes_root: Path, class_name: str) -> dict[str, Path]:
    class_root = ensure_dir(classes_root / slugify(class_name))
    return {
        "root": class_root,
        "annotations": ensure_dir(class_root / "annotations"),
        "visualizations": ensure_dir(class_root / "visualizations"),
    }


def run_pipeline(config: AutoSegConfig) -> Path:
    image_paths = list_images(config.image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {config.image_dir}")
    if config.max_images is not None:
        image_paths = image_paths[: int(config.max_images)]

    run_dirs = _build_run_dirs(config)
    manifest = {
        "image_dir": str(config.image_dir),
        "output_root": str(config.output_root),
        "checkpoint": str(config.checkpoint),
        "device": config.device,
        "resolution": config.resolution,
        "ignore_index": config.ignore_index,
        "image_count": len(image_paths),
        "classes": [
            {
                "id": class_config.class_id,
                "name": class_config.name,
                "prompts": class_config.prompts,
                "color_rgb": list(class_config.color),
                "confidence_threshold": class_config.confidence_threshold,
            }
            for class_config in config.classes
        ],
    }
    write_json(run_dirs["run_dir"] / "manifest.json", manifest)

    inferencer = Sam3AutoSegInferencer(
        config.checkpoint,
        device=config.device,
        confidence_threshold=config.classes[0].confidence_threshold,
        resolution=config.resolution,
    )

    class_annotation_dirs: dict[str, Path] = {}
    image_names = [image_path.name for image_path in image_paths]
    image_sizes: dict[str, tuple[int, int]] = {}
    detection_annotations_by_image: dict[str, list[dict]] = {
        image_path.name: [] for image_path in image_paths
    }
    instance_annotations_by_image: dict[str, list[dict]] = {
        image_path.name: [] for image_path in image_paths
    }
    class_colors = {
        class_config.name: class_config.color for class_config in config.classes
    }

    for image_path in image_paths:
        create_symlink(image_path, run_dirs["images_dir"] / image_path.name)

    for class_config in config.classes:
        dirs = _class_dirs(run_dirs["classes_dir"], class_config.name)
        class_annotation_dirs[class_config.name] = dirs["annotations"]
        print(
            f"[class] {class_config.name} :: threshold={class_config.confidence_threshold} :: prompts={class_config.prompts}"
        )

        for image_path in tqdm(image_paths, desc=f"instance::{class_config.name}"):
            image_pil, image_rgb = load_image(image_path)
            prediction = inferencer.predict_with_threshold(
                image_pil,
                class_config.prompts,
                class_config.confidence_threshold,
            )
            record = build_instance_record(
                image_path=image_path,
                image_size=image_pil.size,
                class_config=class_config,
                prediction=prediction,
            )
            image_sizes[image_path.name] = image_pil.size
            save_instance_record(record, dirs["annotations"] / f"{image_path.stem}.json")

            class_detection_annotations = convert_class_annotations_to_detection_annotations(
                record
            )
            next_detection_id = len(detection_annotations_by_image[image_path.name]) + 1
            for annotation in class_detection_annotations:
                annotation["id"] = next_detection_id
                next_detection_id += 1
                detection_annotations_by_image[image_path.name].append(annotation)

            next_instance_id = len(instance_annotations_by_image[image_path.name]) + 1
            for annotation in record.get("annotations", []):
                merged_annotation = {
                    "id": next_instance_id,
                    "class_id": int(record["class_id"]),
                    "class_name": record["class_name"],
                    "prompt_hits": annotation.get("prompt_hits", record["class_name"]),
                    "score": float(annotation["score"]),
                    "box": list(annotation["box"]),
                    "polygon": list(annotation.get("polygon", [])),
                }
                next_instance_id += 1
                instance_annotations_by_image[image_path.name].append(merged_annotation)

            preview = draw_instance_preview(
                image_rgb=image_rgb,
                boxes_xyxy=prediction.boxes_xyxy,
                labels=[class_config.name] * int(prediction.scores.shape[0]),
                scores=prediction.scores,
                masks=prediction.masks,
            )
            save_image(preview, dirs["visualizations"] / image_path.name)

    for image_path in image_paths:
        image_name = image_path.name
        image_size = image_sizes[image_name]
        _, image_rgb = load_image(image_path)

        detection_record = build_detection_record(
            image_path=image_path,
            image_size=image_size,
            annotations=detection_annotations_by_image[image_name],
        )
        save_instance_record(
            detection_record,
            run_dirs["detection_annotations_dir"] / f"{image_path.stem}.json",
        )
        detection_preview = draw_detection_annotations_preview(
            image_rgb=image_rgb,
            annotations=detection_annotations_by_image[image_name],
        )
        save_image(
            detection_preview,
            run_dirs["detection_visualizations_dir"] / image_name,
        )

        instance_record = build_instance_segmentation_record(
            image_path=image_path,
            image_size=image_size,
            annotations=instance_annotations_by_image[image_name],
        )
        save_instance_record(
            instance_record,
            run_dirs["instance_annotations_dir"] / f"{image_path.stem}.json",
        )
        instance_preview = draw_instance_annotations_preview(
            image_rgb=image_rgb,
            annotations=instance_annotations_by_image[image_name],
            class_colors=class_colors,
        )
        save_image(
            instance_preview,
            run_dirs["instance_visualizations_dir"] / image_name,
        )

    merge_semantic_masks(
        config=config,
        class_annotation_dirs=class_annotation_dirs,
        semantic_dir=run_dirs["semantic_annotations_dir"],
        semantic_vis_dir=run_dirs["semantic_visualizations_dir"],
        image_names=image_names,
    )
    return run_dirs["run_dir"]
