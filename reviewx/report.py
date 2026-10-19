from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

from .types import ImageAudit, Issue, check_name, issue_type_name, source_name


def write_reports(
    output_dir: Path,
    audits: list[ImageAudit],
    dataset_issues: list[Issue],
    metadata: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(item.status for item in audits)
    issue_code_counts = Counter(
        issue.issue_type for audit in audits for issue in audit.issues
    )
    issue_code_counts.update(issue.issue_type for issue in dataset_issues)
    issue_counts = Counter()
    for code, count in issue_code_counts.items():
        issue_counts[issue_type_name(code)] += count
    summary = {
        **metadata,
        "image_count": len(audits),
        "status_counts": dict(sorted(status_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_code_counts": dict(sorted(issue_code_counts.items())),
        "dataset_issues": [item.to_dict() for item in dataset_issues],
    }
    (output_dir / "report.json").write_text(
        json.dumps(
            {"summary": summary, "images": [item.to_dict() for item in audits]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "images.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "图片路径",
                "标注路径",
                "审核状态",
                "人工实例数",
                "问题数",
                "问题类型",
                "SAM3已审核类别",
                "已执行检查",
                "可视化路径",
            ],
        )
        writer.writeheader()
        for audit in audits:
            writer.writerow(
                {
                    "图片路径": audit.image_path,
                    "标注路径": audit.annotation_path,
                    "审核状态": audit.status,
                    "人工实例数": audit.instance_count,
                    "问题数": len(audit.issues),
                    "问题类型": "；".join(
                        issue.display_name for issue in audit.issues
                    ),
                    "SAM3已审核类别": "；".join(audit.checked_model_classes),
                    "已执行检查": "；".join(
                        check_name(code) for code in audit.checks_performed
                    ),
                    "可视化路径": audit.visualization or "",
                }
            )
    with (output_dir / "issues.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "图片路径",
            "审核状态",
            "问题代码",
            "问题类型",
            "严重程度",
            "人工实例编号",
            "问题原因",
            "SAM3预测类别",
            "SAM3置信度",
            "掩码交并比",
            "问题框坐标",
            "问题来源",
            "问题证据图",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for audit in audits:
            for issue in audit.issues:
                writer.writerow(
                    {
                        "图片路径": audit.image_path,
                        "审核状态": audit.status,
                        "问题代码": issue.issue_type,
                        "问题类型": issue.display_name,
                        "严重程度": issue.severity,
                        "人工实例编号": "；".join(map(str, issue.instance_ids)),
                        "问题原因": issue.reason,
                        "SAM3预测类别": issue.model_class or "",
                        "SAM3置信度": (
                            ""
                            if issue.model_score is None
                            else f"{issue.model_score:.6f}"
                        ),
                        "掩码交并比": (
                            "" if issue.mask_iou is None else f"{issue.mask_iou:.6f}"
                        ),
                        "问题框坐标": (
                            "" if issue.box is None else ",".join(map(str, issue.box))
                        ),
                        "问题来源": source_name(issue.source),
                        "问题证据图": issue.evidence_visualization or "",
                    }
                )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_html_report(output_dir, audits, summary)
    return summary


def _write_html_report(
    output_dir: Path, audits: list[ImageAudit], summary: dict
) -> None:
    issue_names = sorted(
        {issue.display_name for audit in audits for issue in audit.issues}
    )
    status_counts = summary.get("status_counts", {})
    cards: list[str] = []
    for audit in audits:
        for index, issue in enumerate(audit.issues, start=1):
            evidence = (
                f'<a href="{html.escape(issue.evidence_visualization)}" target="_blank">'
                f'<img src="{html.escape(issue.evidence_visualization)}" loading="lazy" alt="问题证据图"></a>'
                if issue.evidence_visualization
                else '<div class="no-image">本次运行没有生成问题证据图</div>'
            )
            overview = (
                f'<a href="{html.escape(audit.visualization)}" target="_blank">查看整图定位</a>'
                if audit.visualization
                else ""
            )
            metrics = []
            if issue.instance_ids:
                metrics.append("人工实例：" + "、".join(map(str, issue.instance_ids)))
            if issue.model_class:
                metrics.append(f"模型类别：{html.escape(issue.model_class)}")
            if issue.model_score is not None:
                metrics.append(f"模型置信度：{issue.model_score:.3f}")
            if issue.mask_iou is not None:
                metrics.append(f"掩码交并比：{issue.mask_iou:.3f}")
            cards.append(
                f'''<article class="issue-card" data-status="{audit.status}" data-type="{html.escape(issue.display_name)}">
  <div class="card-head"><span class="status {audit.status.lower()}">{audit.status}</span>
  <strong>{html.escape(issue.display_name)}</strong><span>问题 {index}</span></div>
  <div class="path">{html.escape(audit.image_path)}</div>
  <p>{html.escape(issue.reason)}</p>
  <div class="metrics">{'　'.join(metrics)}</div>
  <div class="links">{overview}</div>
  {evidence}
</article>'''
            )

    options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
        for name in issue_names
    )
    issue_total = sum(summary.get("issue_counts", {}).values())
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>实例分割标注审核报告</title>
<style>
body{{margin:0;background:#f3f5f7;color:#202124;font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#17212b;color:white;padding:18px 28px;box-shadow:0 2px 8px #0004}}
h1{{font-size:22px;margin:0 0 12px}} .summary{{display:flex;gap:12px;flex-wrap:wrap}}
.pill{{background:#ffffff1c;border:1px solid #ffffff38;border-radius:20px;padding:6px 12px}}
.filters{{margin-top:14px;display:flex;gap:10px;flex-wrap:wrap}} select,input{{padding:8px 10px;border-radius:6px;border:1px solid #9aa0a6}}
main{{max-width:1500px;margin:22px auto;padding:0 18px;display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px}}
.issue-card{{background:white;border-radius:10px;padding:16px;box-shadow:0 1px 5px #0002}}
.card-head{{display:flex;align-items:center;gap:12px;font-size:18px}} .card-head span:last-child{{margin-left:auto;color:#666;font-size:14px}}
.status{{padding:4px 9px;border-radius:5px;color:white;font-size:13px}} .fail{{background:#c62828}} .review{{background:#ef6c00}} .pass{{background:#2e7d32}}
.path{{font-family:monospace;color:#455a64;margin-top:10px;overflow-wrap:anywhere}} p{{line-height:1.55}}
.metrics{{color:#5f6368;margin:8px 0}} .links a{{color:#1565c0}} img{{width:100%;margin-top:12px;border:1px solid #ddd;border-radius:6px}}
.no-image{{margin-top:12px;background:#eee;padding:30px;text-align:center;color:#666}}
#empty{{display:none;text-align:center;padding:60px;font-size:18px;color:#666}}
</style></head><body>
<header><h1>实例分割标注审核报告</h1>
<div class="summary"><span class="pill">图片：{summary.get('image_count', 0)}</span>
<span class="pill">通过：{status_counts.get('PASS', 0)}</span>
<span class="pill">需复核：{status_counts.get('REVIEW', 0)}</span>
<span class="pill">失败：{status_counts.get('FAIL', 0)}</span>
<span class="pill">问题：{issue_total}</span></div>
<div class="filters"><select id="status"><option value="">全部状态</option><option>FAIL</option><option>REVIEW</option></select>
<select id="type"><option value="">全部问题类型</option>{options}</select>
<input id="search" placeholder="按图片路径搜索"></div></header>
<main id="cards">{''.join(cards)}</main><div id="empty">没有符合筛选条件的问题</div>
<script>
const statusEl=document.getElementById('status'),typeEl=document.getElementById('type'),searchEl=document.getElementById('search');
function filter(){{let shown=0;document.querySelectorAll('.issue-card').forEach(card=>{{
 const ok=(!statusEl.value||card.dataset.status===statusEl.value)&&(!typeEl.value||card.dataset.type===typeEl.value)&&card.textContent.toLowerCase().includes(searchEl.value.toLowerCase());
 card.style.display=ok?'block':'none';if(ok)shown++;}});document.getElementById('empty').style.display=shown?'none':'block';}}
[statusEl,typeEl,searchEl].forEach(el=>el.addEventListener('input',filter));
</script></body></html>'''
    (output_dir / "index.html").write_text(page, encoding="utf-8")
