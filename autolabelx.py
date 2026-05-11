from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent / "autolabelx"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent automatic semantic segmentation pipeline built on SAM3."
    )
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Override SAM3 checkpoint path.")
    parser.add_argument("--output-root", default=None, help="Override output root directory.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Override prediction confidence threshold.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override processor resolution.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for how many images to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from config import load_config
    from pipeline import run_pipeline

    config = load_config(
        args.config,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        max_images=args.max_images,
        resolution=args.resolution,
    )
    run_dir = run_pipeline(config)
    print(f"Auto semantic labeling completed: {run_dir}")


if __name__ == "__main__":
    main()
