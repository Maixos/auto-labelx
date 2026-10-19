from __future__ import annotations

import argparse

from reviewx.config import load_audit_config
from reviewx.pipeline import run_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="reviewx",
        description="使用确定性规则和 SAM 3 审核 LabelMe 实例分割标注。"
    )
    parser.add_argument("--config", required=True, help="审核配置 JSON 路径")
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="只执行格式和几何检查，不加载 SAM 3",
    )
    parser.add_argument(
        "--max-images", type=int, default=None, help="限制图片数量，用于小样本测试"
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="不生成问题可视化图片",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run_audit(
        load_audit_config(args.config),
        static_only=args.static_only,
        max_images=args.max_images,
        no_visualizations=args.no_visualizations,
    )
    print(f"标注审核完成，结果目录：{output_dir}")


if __name__ == "__main__":
    main()
