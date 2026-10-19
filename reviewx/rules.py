from __future__ import annotations

from collections import defaultdict

from .config import AuditConfig
from .geometry import (
    greedy_match,
    instance_intersection_over_smaller,
    instance_iou,
)
from .types import Instance, Issue


def static_overlap_issues(
    instances: list[Instance], config: AuditConfig
) -> list[Issue]:
    issues: list[Issue] = []
    for left_index, left in enumerate(instances):
        for right in instances[left_index + 1 :]:
            iou = instance_iou(left, right)
            if left.label == right.label and iou >= config.duplicate_iou:
                issues.append(
                    Issue(
                        "DUPLICATE_INSTANCE",
                        "REVIEW",
                        f"同类别实例的掩码交并比为 {iou:.3f}，达到重复标注阈值 {config.duplicate_iou:.3f}，请确认是否重复画了同一目标",
                        [left.instance_id, right.instance_id],
                        mask_iou=iou,
                        box=left.box,
                    )
                )
            elif left.label != right.label and iou >= config.duplicate_iou:
                issues.append(
                    Issue(
                        "CROSS_CLASS_DUPLICATE",
                        "REVIEW",
                        f"类别 {left.label} 与 {right.label} 使用了近似相同的掩码，交并比为 {iou:.3f}，请确认哪个类别正确",
                        [left.instance_id, right.instance_id],
                        mask_iou=iou,
                        box=left.box,
                    )
                )
    return issues


def compare_with_model(
    human: list[Instance],
    predictions: dict[str, list[Instance]],
    config: AuditConfig,
    image_area: int,
) -> list[Issue]:
    issues: list[Issue] = []
    humans_by_class: dict[str, list[Instance]] = defaultdict(list)
    for instance in human:
        humans_by_class[instance.label].append(instance)

    matched_human_ids: set[int] = set()
    for class_config in config.model_classes:
        class_human = humans_by_class[class_config.name]
        class_predicted = predictions.get(class_config.name, [])
        matches, _, unmatched_predicted = greedy_match(class_human, class_predicted)
        for human_index, predicted_index, iou in matches:
            h = class_human[human_index]
            p = class_predicted[predicted_index]
            matched_human_ids.add(h.instance_id)
            if iou < config.low_mask_iou:
                issues.append(
                    Issue(
                        "MASK_MISMATCH",
                        "REVIEW",
                        f"人工标注类别 {class_config.name} 与 SAM 3 提示词“{p.prompt}”得到的掩码交并比仅为 {iou:.3f}，请检查掩码是否偏大、偏小、偏移或缺失",
                        [h.instance_id],
                        class_config.name,
                        p.score,
                        iou,
                        h.box,
                        "sam3",
                    )
                )
        for predicted_index in unmatched_predicted:
            p = class_predicted[predicted_index]
            if p.area / image_area < config.missing_min_area_ratio:
                continue
            best_any = max((instance_iou(p, h) for h in human), default=0.0)
            if best_any < config.match_iou:
                issues.append(
                    Issue(
                        "POSSIBLE_MISSING_LABEL",
                        "REVIEW",
                        f"SAM 3 通过提示词“{p.prompt}”发现类别 {class_config.name} 的目标，但没有匹配到人工标注；模型置信度为 {p.score:.3f}，掩码面积为 {p.area} 像素",
                        model_class=class_config.name,
                        model_score=p.score,
                        mask_iou=best_any,
                        box=p.box,
                        source="sam3",
                    )
                )

    # A strong alternate-class overlap is useful evidence, but remains REVIEW.
    for h in human:
        if (
            h.label not in {item.name for item in config.model_classes}
            or h.instance_id in matched_human_ids
        ):
            continue
        candidates = [
            (instance_iou(h, p), p)
            for name, values in predictions.items()
            if name != h.label
            for p in values
        ]
        if candidates:
            iou, p = max(candidates, key=lambda value: value[0])
            if iou >= config.match_iou:
                issues.append(
                    Issue(
                        "POSSIBLE_WRONG_CLASS",
                        "REVIEW",
                        f"人工类别 {h.label} 没有匹配到同类别模型结果，却与模型预测类别 {p.label} 的掩码高度重合，交并比为 {iou:.3f}，请核对类别",
                        [h.instance_id],
                        p.label,
                        p.score,
                        iou,
                        h.box,
                        "sam3",
                    )
                )
    issues.extend(_split_merge_issues(human, predictions, config))
    return issues


def _split_merge_issues(
    human: list[Instance], predictions: dict[str, list[Instance]], config: AuditConfig
) -> list[Issue]:
    issues: list[Issue] = []
    for name, predicted in predictions.items():
        class_human = [item for item in human if item.label == name]
        for p in predicted:
            overlaps = [
                h
                for h in class_human
                if instance_intersection_over_smaller(h, p) >= 0.7
            ]
            if len(overlaps) >= 2:
                issues.append(
                    Issue(
                        "POSSIBLE_SPLIT_INSTANCE",
                        "REVIEW",
                        f"一个 SAM 3 类别 {name} 的掩码大范围覆盖了 {len(overlaps)} 个人工实例，请确认同一目标是否被错误拆分",
                        [h.instance_id for h in overlaps],
                        name,
                        p.score,
                        box=p.box,
                        source="sam3",
                    )
                )
        for h in class_human:
            overlaps = [
                p for p in predicted if instance_intersection_over_smaller(h, p) >= 0.7
            ]
            if len(overlaps) >= 2:
                issues.append(
                    Issue(
                        "POSSIBLE_MERGED_INSTANCES",
                        "REVIEW",
                        f"一个人工类别 {name} 的实例大范围包含了 {len(overlaps)} 个 SAM 3 实例，请确认多个目标是否被错误合并",
                        [h.instance_id],
                        name,
                        max(p.score or 0 for p in overlaps),
                        box=h.box,
                        source="sam3",
                    )
                )
    return issues
