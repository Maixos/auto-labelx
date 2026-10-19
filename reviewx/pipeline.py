from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # Keep deterministic/static audit usable in minimal environments.

    def tqdm(iterable, **_kwargs):
        return iterable


from .config import AuditConfig
from .labelme import find_pairs, load_labelme
from .model import PredictionCache, Sam3AuditPredictor, model_signature
from .report import write_reports
from .rules import compare_with_model, static_overlap_issues
from .types import ImageAudit, check_name
from .visualization import render_audit, render_issue_evidence


def run_audit(
    config: AuditConfig,
    *,
    static_only: bool = False,
    max_images: int | None = None,
    no_visualizations: bool = False,
) -> Path:
    if not config.dataset_dir.is_dir():
        raise FileNotFoundError(f"找不到数据集目录：{config.dataset_dir}")
    pairs, dataset_issues = find_pairs(config.dataset_dir)
    if max_images is not None:
        pairs = pairs[:max_images]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    model_enabled = not static_only
    predictor: Sam3AuditPredictor | None = None
    cache = (
        None
        if static_only
        else PredictionCache(config.output_dir / "cache", model_signature(config))
    )
    checked_model_classes = (
        [] if static_only else [item.name for item in config.model_classes]
    )
    audits: list[ImageAudit] = []
    label_counts: Counter[str] = Counter()
    for image_path, annotation_path in tqdm(pairs, desc="reviewx"):
        relative_image = image_path.relative_to(config.dataset_dir)
        relative_annotation = annotation_path.relative_to(config.dataset_dir)
        instances, issues, (width, height) = load_labelme(
            image_path, annotation_path, config.class_names, config.min_polygon_area
        )
        label_counts.update(item.label for item in instances)
        issues.extend(static_overlap_issues(instances, config))
        checks = [
            "labelme_schema",
            "image_pair_and_size",
            "class_name",
            "polygon_geometry",
            "duplicate_and_overlap",
        ]
        predictions = None
        if model_enabled and not any(
            issue.severity == "FAIL" for issue in issues
        ):
            assert cache is not None
            predictions = cache.load(relative_image)
            if predictions is None:
                if predictor is None:
                    predictor = Sam3AuditPredictor(config)
                predictions = predictor.predict(image_path, config.model_classes)
                cache.save(relative_image, predictions, (height, width))
            issues.extend(
                compare_with_model(instances, predictions, config, width * height)
            )
            checks.extend(
                [
                    "sam3_missing_label",
                    "sam3_class_consistency",
                    "sam3_mask_iou",
                    "sam3_split_merge",
                ]
            )
        status = (
            "FAIL"
            if any(issue.severity == "FAIL" for issue in issues)
            else "REVIEW" if issues else "PASS"
        )
        audit = ImageAudit(
            image_path=relative_image.as_posix(),
            annotation_path=relative_annotation.as_posix(),
            status=status,
            instance_count=len(instances),
            checked_model_classes=checked_model_classes,
            issues=issues,
            checks_performed=checks,
        )
        if not no_visualizations and (issues or config.visualize_pass):
            image_visual_dir = (
                output_dir
                / "visualizations"
                / relative_image.parent
                / relative_image.stem
            )
            visual_path = image_visual_dir / "overview.jpg"
            render_audit(
                image_path,
                instances,
                audit,
                visual_path,
                predictions=predictions,
            )
            audit.visualization = visual_path.relative_to(output_dir).as_posix()
            for issue_number, issue in enumerate(issues, start=1):
                evidence_path = image_visual_dir / f"issue_{issue_number:03d}.jpg"
                render_issue_evidence(
                    image_path,
                    instances,
                    issue,
                    issue_number,
                    evidence_path,
                    predictions=predictions,
                )
                issue.evidence_visualization = evidence_path.relative_to(
                    output_dir
                ).as_posix()
        audits.append(audit)

    metadata = {
        "dataset_dir": str(config.dataset_dir),
        "mode": "static-only" if static_only else "static+sam3",
        "checked_model_classes": checked_model_classes,
        "unchecked_model_classes": sorted(
            config.class_names - set(checked_model_classes)
        ),
        "check_codes": sorted(
            {check for audit in audits for check in audit.checks_performed}
        ),
        "checks_performed": [
            check_name(code)
            for code in sorted(
                {check for audit in audits for check in audit.checks_performed}
            )
        ],
        "label_instance_counts": dict(sorted(label_counts.items())),
        "configured_but_absent_labels": sorted(config.class_names - set(label_counts)),
        "pass_scope_note": "PASS 只表示本报告列出的检查项和模型审核类别没有发现问题，不代表未启用模型审核的类别在语义上一定正确。",
    }
    write_reports(output_dir, audits, dataset_issues, metadata)
    return output_dir
