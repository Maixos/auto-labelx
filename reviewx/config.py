from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AuditClass:
    name: str
    prompts: tuple[str, ...]
    model_audit: bool
    confidence_threshold: float


@dataclass(frozen=True)
class AuditConfig:
    dataset_dir: Path
    output_dir: Path
    checkpoint: Path | None
    device: str
    resolution: int
    classes: tuple[AuditClass, ...]
    min_polygon_area: float
    duplicate_iou: float
    match_iou: float
    low_mask_iou: float
    missing_min_area_ratio: float
    visualize_pass: bool

    @property
    def class_names(self) -> set[str]:
        return {item.name for item in self.classes}

    @property
    def model_classes(self) -> tuple[AuditClass, ...]:
        return tuple(item for item in self.classes if item.model_audit)


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _number(raw: dict[str, Any], key: str, default: float) -> float:
    value = float(raw.get(key, default))
    if not 0 <= value <= 1 and key != "min_polygon_area":
        raise ValueError(f"配置项 {key!r} 必须在 0 到 1 之间")
    return value


def load_audit_config(path: str | Path) -> AuditConfig:
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("审核配置文件的最外层必须是 JSON 对象")
    base = config_path.parent

    raw_classes = raw.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("审核配置必须包含非空的 classes 列表")
    classes: list[AuditClass] = []
    for index, item in enumerate(raw_classes):
        if not isinstance(item, dict):
            raise ValueError(f"classes[{index}] 必须是 JSON 对象")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"classes[{index}].name 不能为空")
        prompts = item.get("prompts", [])
        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(prompts, list) or not all(
            isinstance(value, str) and value.strip() for value in prompts
        ):
            raise ValueError(f"类别 {name!r} 的 prompts 配置不合法")
        model_audit = bool(item.get("model_audit", False))
        if model_audit and not prompts:
            raise ValueError(f"启用模型审核的类别 {name!r} 至少需要一个提示词")
        classes.append(
            AuditClass(
                name=name,
                prompts=tuple(value.strip() for value in prompts),
                model_audit=model_audit,
                confidence_threshold=float(item.get("confidence_threshold", 0.5)),
            )
        )
    names = [item.name for item in classes]
    if len(names) != len(set(names)):
        raise ValueError("类别名称不能重复，并且区分大小写")

    checkpoint_value = raw.get("checkpoint")
    checkpoint = _resolve(str(checkpoint_value), base) if checkpoint_value else None
    return AuditConfig(
        dataset_dir=_resolve(str(raw["dataset_dir"]), base),
        output_dir=(
            _resolve(str(raw["output_dir"]), base)
            if raw.get("output_dir")
            else PROJECT_ROOT / "runs" / "reviewx"
        ),
        checkpoint=checkpoint,
        device=str(raw.get("device", "cuda")),
        resolution=int(raw.get("resolution", 1008)),
        classes=tuple(classes),
        min_polygon_area=float(raw.get("min_polygon_area", 4.0)),
        duplicate_iou=_number(raw, "duplicate_iou", 0.9),
        match_iou=_number(raw, "match_iou", 0.5),
        low_mask_iou=_number(raw, "low_mask_iou", 0.65),
        missing_min_area_ratio=_number(raw, "missing_min_area_ratio", 0.0002),
        visualize_pass=bool(raw.get("visualize_pass", False)),
    )
