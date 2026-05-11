from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ClassConfig:
    class_id: int
    name: str
    prompts: list[str]
    color: tuple[int, int, int]
    confidence_threshold: float


@dataclass(frozen=True)
class AutoSegConfig:
    image_dir: Path
    output_root: Path
    checkpoint: Path
    device: str
    resolution: int
    ignore_index: int
    classes: list[ClassConfig]
    max_images: int | None = None


DEFAULT_COLORS = [
    (70, 130, 180),
    (190, 153, 153),
    (128, 64, 128),
    (0, 0, 142),
    (119, 11, 32),
    (220, 20, 60),
    (107, 142, 35),
    (255, 127, 80),
    (46, 139, 87),
    (255, 215, 0),
    (176, 196, 222),
    (205, 92, 92),
]


def _load_raw_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        raise ValueError("Config file must be .json, .yaml, or .yml")

    if not isinstance(data, dict):
        raise ValueError("Config root must be an object")
    return data


def _normalize_prompts(value: Any, class_name: str) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        prompts = [stripped] if stripped else []
    elif isinstance(value, list):
        prompts = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"Prompt list for class '{class_name}' must contain strings")
            stripped = item.strip()
            if stripped:
                prompts.append(stripped)
    else:
        raise ValueError(f"Prompt for class '{class_name}' must be a string or list of strings")

    if not prompts:
        raise ValueError(f"Class '{class_name}' has no valid prompts")
    return prompts


def _resolve_color(
    class_id: int,
    class_name: str,
    explicit_colors: dict[str, Any],
    index: int,
) -> tuple[int, int, int]:
    raw = explicit_colors.get(str(class_id))
    if raw is None:
        raw = explicit_colors.get(class_name)
    if raw is None:
        return DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"Color for class '{class_name}' must be a 3-element list")
    color = tuple(int(channel) for channel in raw)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError(f"Color for class '{class_name}' must be within [0, 255]")
    return color


def _parse_inline_color(raw: Any, class_name: str) -> tuple[int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"Color for class '{class_name}' must be a 3-element list")
    color = tuple(int(channel) for channel in raw)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError(f"Color for class '{class_name}' must be within [0, 255]")
    return color


def _resolve_project_path(value: str | os.PathLike[str], *, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _parse_classes(
    raw: dict[str, Any],
    *,
    override_confidence_threshold: float | None = None,
) -> list[ClassConfig]:
    classes_section = raw.get("classes")
    if classes_section is not None:
        if not isinstance(classes_section, list):
            raise ValueError("'classes' must be a list")
        classes: list[ClassConfig] = []
        for index, item in enumerate(classes_section):
            if not isinstance(item, dict):
                raise ValueError("Each item in 'classes' must be an object")
            class_id = int(item["id"])
            name = str(item.get("name", item.get("class_name", ""))).strip()
            if not name:
                raise ValueError(f"Class name for id {class_id} is empty")
            prompt_value = item.get("prompts", item.get("prompt"))
            if prompt_value is None:
                raise ValueError(f"Class '{name}' must define 'prompts' or 'prompt'")
            prompts = _normalize_prompts(prompt_value, name)
            raw_color = item.get("color_rgb", item.get("color"))
            color = (
                _parse_inline_color(raw_color, name)
                if raw_color is not None
                else DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
            )
            if override_confidence_threshold is not None:
                class_confidence_threshold = float(override_confidence_threshold)
            elif "confidence_threshold" in item:
                class_confidence_threshold = float(item["confidence_threshold"])
            else:
                raise ValueError(
                    f"Class '{name}' must define its own confidence_threshold"
                )
            classes.append(
                ClassConfig(
                    class_id=class_id,
                    name=name,
                    prompts=prompts,
                    color=color,
                    confidence_threshold=class_confidence_threshold,
                )
            )
        return classes

    seg_classes = raw.get("SEG_CLASSES")
    class_prompts = raw.get("CLASS_PROMPTS")
    class_colors = raw.get("CLASS_COLORS", {})
    if not isinstance(seg_classes, dict) or not isinstance(class_prompts, dict):
        raise ValueError("Config must define either 'classes' or both 'SEG_CLASSES' and 'CLASS_PROMPTS'")

    if override_confidence_threshold is None:
        raise ValueError(
            "Legacy SEG_CLASSES/CLASS_PROMPTS format requires a global override confidence_threshold"
        )

    classes = []
    for index, (raw_class_id, raw_name) in enumerate(
        sorted(seg_classes.items(), key=lambda item: int(item[0]))
    ):
        class_id = int(raw_class_id)
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"Class name for id {class_id} is empty")
        if class_id == int(raw.get("ignore_index", 255)) or name.lower() == "ignore":
            continue
        if name not in class_prompts:
            raise ValueError(f"Missing prompt for class '{name}'")
        prompts = _normalize_prompts(class_prompts[name], name)
        color = _resolve_color(class_id, name, class_colors, index)
        classes.append(
            ClassConfig(
                class_id=class_id,
                name=name,
                prompts=prompts,
                color=color,
                confidence_threshold=float(override_confidence_threshold),
            )
        )
    return classes


def load_config(
    config_path: str | os.PathLike[str],
    *,
    checkpoint: str | None = None,
    output_root: str | None = None,
    device: str | None = None,
    confidence_threshold: float | None = None,
    max_images: int | None = None,
    resolution: int | None = None,
) -> AutoSegConfig:
    path = Path(config_path).expanduser().resolve()
    raw = _load_raw_config(path)
    project_root = PROJECT_ROOT

    image_dir_value = raw.get("image_dir", raw.get("root"))
    if not image_dir_value:
        raise ValueError("Config must define 'image_dir' or 'root'")
    image_dir = _resolve_project_path(image_dir_value, project_root=project_root)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    resolved_output_root = output_root or raw.get("output_root")
    if resolved_output_root:
        output_dir = _resolve_project_path(
            resolved_output_root, project_root=project_root
        )
    else:
        output_dir = project_root / "runs"

    resolved_checkpoint = (
        checkpoint
        or raw.get("checkpoint")
        or raw.get("checkpoint_path")
        or os.getenv("SAM3_CHECKPOINT_PATH")
    )
    if not resolved_checkpoint:
        raise ValueError("Checkpoint path must be provided via config, CLI, or SAM3_CHECKPOINT_PATH")
    checkpoint = _resolve_project_path(
        resolved_checkpoint, project_root=project_root
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    classes = _parse_classes(
        raw, override_confidence_threshold=confidence_threshold
    )
    if not classes:
        raise ValueError("No valid classes found in config")

    id_counts: dict[int, int] = {}
    name_counts: dict[str, int] = {}
    for item in classes:
        id_counts[item.class_id] = id_counts.get(item.class_id, 0) + 1
        name_counts[item.name] = name_counts.get(item.name, 0) + 1

    duplicate_ids = {class_id for class_id, count in id_counts.items() if count > 1}
    if duplicate_ids:
        raise ValueError(f"Duplicate class ids found: {sorted(duplicate_ids)}")
    duplicate_names = {name for name, count in name_counts.items() if count > 1}
    if duplicate_names:
        raise ValueError(f"Duplicate class names found: {sorted(duplicate_names)}")

    return AutoSegConfig(
        image_dir=image_dir,
        output_root=output_dir,
        checkpoint=checkpoint,
        device=device or str(raw.get("device", "cuda")),
        resolution=int(resolution if resolution is not None else raw.get("resolution", 1008)),
        ignore_index=int(raw.get("ignore_index", 255)),
        classes=classes,
        max_images=(
            int(max_images)
            if max_images is not None
            else (int(raw["max_images"]) if raw.get("max_images") is not None else None)
        ),
    )
