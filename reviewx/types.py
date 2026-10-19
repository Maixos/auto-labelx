from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


ISSUE_TYPE_NAMES = {
    "MISSING_IMAGE": "缺少对应图像",
    "MISSING_ANNOTATION": "缺少对应标注文件",
    "INVALID_JSON": "标注文件无法读取",
    "IMAGE_SIZE_MISMATCH": "图像尺寸不一致",
    "IMAGE_PATH_MISMATCH": "图像文件名不一致",
    "INVALID_SHAPES": "标注实例列表格式错误",
    "INVALID_SHAPE": "标注实例格式错误",
    "UNKNOWN_LABEL": "未知或大小写错误的类别",
    "INVALID_SHAPE_TYPE": "标注形状类型错误",
    "INVALID_POLYGON": "多边形格式错误",
    "POINT_OUT_OF_BOUNDS": "多边形点超出图像边界",
    "EMPTY_OR_TINY_POLYGON": "多边形为空或面积过小",
    "DUPLICATE_INSTANCE": "疑似重复标注",
    "CROSS_CLASS_DUPLICATE": "不同类别使用了近似相同的掩码",
    "MASK_MISMATCH": "人工掩码与模型掩码差异较大",
    "POSSIBLE_MISSING_LABEL": "疑似漏标",
    "POSSIBLE_WRONG_CLASS": "疑似类别错误",
    "POSSIBLE_SPLIT_INSTANCE": "疑似同一目标被拆成多个实例",
    "POSSIBLE_MERGED_INSTANCES": "疑似多个目标被合成一个实例",
}

SOURCE_NAMES = {
    "format": "数据格式检查",
    "rules": "确定性规则检查",
    "sam3": "SAM 3 模型交叉检查",
}

CHECK_NAMES = {
    "labelme_schema": "LabelMe 文件结构",
    "image_pair_and_size": "图像与标注配对及尺寸",
    "class_name": "类别名称和大小写",
    "polygon_geometry": "多边形坐标、面积和边界",
    "duplicate_and_overlap": "重复标注和高重叠标注",
    "sam3_missing_label": "SAM 3 疑似漏标检查",
    "sam3_class_consistency": "SAM 3 类别一致性检查",
    "sam3_mask_iou": "SAM 3 掩码交并比检查",
    "sam3_split_merge": "SAM 3 实例拆分与合并检查",
}


def issue_type_name(code: str) -> str:
    return ISSUE_TYPE_NAMES.get(code, code)


def source_name(code: str) -> str:
    return SOURCE_NAMES.get(code, code)


def check_name(code: str) -> str:
    return CHECK_NAMES.get(code, code)


@dataclass
class Instance:
    instance_id: int
    label: str
    mask: np.ndarray
    box: tuple[int, int, int, int]
    area: int
    points: list[list[float]] = field(default_factory=list)
    score: float | None = None
    prompt: str | None = None


@dataclass
class Issue:
    issue_type: str
    severity: str
    reason: str
    instance_ids: list[int] = field(default_factory=list)
    model_class: str | None = None
    model_score: float | None = None
    mask_iou: float | None = None
    box: tuple[int, int, int, int] | None = None
    source: str = "rules"
    evidence_visualization: str | None = None

    @property
    def display_name(self) -> str:
        return issue_type_name(self.issue_type)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["issue_code"] = result.pop("issue_type")
        result["issue_type"] = self.display_name
        result["source_code"] = result.pop("source")
        result["source"] = source_name(self.source)
        if result["box"] is not None:
            result["box"] = list(result["box"])
        return result


@dataclass
class ImageAudit:
    image_path: str
    annotation_path: str
    status: str
    instance_count: int
    checked_model_classes: list[str]
    issues: list[Issue]
    checks_performed: list[str]
    visualization: str | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["issues"] = [issue.to_dict() for issue in self.issues]
        result["check_codes"] = result.pop("checks_performed")
        result["checks_performed"] = [
            check_name(code) for code in self.checks_performed
        ]
        return result
