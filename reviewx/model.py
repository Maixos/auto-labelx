from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import AuditClass, AuditConfig
from .geometry import deduplicate_predictions, mask_box
from .types import Instance


def model_signature(config: AuditConfig) -> str:
    checkpoint_stat = (
        config.checkpoint.stat()
        if config.checkpoint and config.checkpoint.is_file()
        else None
    )
    payload = {
        "checkpoint": str(config.checkpoint),
        "checkpoint_size": checkpoint_stat.st_size if checkpoint_stat else None,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns if checkpoint_stat else None,
        "resolution": config.resolution,
        "classes": [
            {
                "name": item.name,
                "prompts": item.prompts,
                "threshold": item.confidence_threshold,
            }
            for item in config.model_classes
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class PredictionCache:
    def __init__(self, cache_dir: Path, signature: str) -> None:
        self.root = cache_dir / signature
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, relative_image_path: Path) -> Path:
        digest = hashlib.sha1(relative_image_path.as_posix().encode()).hexdigest()
        return self.root / f"{digest}.npz"

    def load(self, relative_image_path: Path) -> dict[str, list[Instance]] | None:
        path = self.path_for(relative_image_path)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            labels = data["labels"].tolist()
            scores = data["scores"].tolist()
            prompts = data["prompts"].tolist()
            boxes = data["boxes"].tolist()
            result: dict[str, list[Instance]] = {}
            for index, (label, score, prompt, box) in enumerate(
                zip(labels, scores, prompts, boxes), start=1
            ):
                mask = data[f"mask_{index - 1}"].astype(bool)
                result.setdefault(str(label), []).append(
                    Instance(
                        index,
                        str(label),
                        mask,
                        tuple(int(v) for v in box),
                        int(mask.sum()),
                        score=float(score),
                        prompt=str(prompt),
                    )
                )
        return result

    def save(
        self,
        relative_image_path: Path,
        predictions: dict[str, list[Instance]],
        shape: tuple[int, int],
    ) -> None:
        flat = [instance for values in predictions.values() for instance in values]
        payload = dict(
            labels=np.asarray([item.label for item in flat]),
            scores=np.asarray([item.score for item in flat], dtype=np.float32),
            prompts=np.asarray([item.prompt or "" for item in flat]),
            boxes=np.asarray([item.box for item in flat], dtype=np.int32).reshape(
                -1, 4
            ),
        )
        payload.update({f"mask_{index}": item.mask for index, item in enumerate(flat)})
        np.savez_compressed(self.path_for(relative_image_path), **payload)


class Sam3AuditPredictor:
    """Runs all configured text prompts from one image embedding."""

    def __init__(self, config: AuditConfig) -> None:
        if config.checkpoint is None or not config.checkpoint.is_file():
            raise FileNotFoundError(f"找不到 SAM 3 权重文件：{config.checkpoint}")
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.torch = torch
        self.device = config.device
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("配置要求使用 CUDA，但当前环境无法使用 GPU")
        self.model = build_sam3_image_model(
            checkpoint_path=str(config.checkpoint),
            load_from_HF=False,
            device=config.device,
        )
        self.processor = Sam3Processor(
            self.model,
            resolution=config.resolution,
            device=config.device,
            confidence_threshold=0.5,
        )

    def predict(
        self, image_path: Path, classes: tuple[AuditClass, ...]
    ) -> dict[str, list[Instance]]:
        image = Image.open(image_path).convert("RGB")
        if self.device.startswith("cuda"):
            context = self.torch.autocast("cuda", dtype=self.torch.bfloat16)
        else:
            from contextlib import nullcontext

            context = nullcontext()
        result: dict[str, list[Instance]] = {}
        with context:
            state = self.processor.set_image(image)
            for class_config in classes:
                collected: list[Instance] = []
                self.processor.set_confidence_threshold(
                    class_config.confidence_threshold, state
                )
                for prompt in class_config.prompts:
                    self.processor.reset_all_prompts(state)
                    output = self.processor.set_text_prompt(prompt=prompt, state=state)
                    masks = output["masks"].detach().cpu().numpy()
                    scores = output["scores"].detach().float().cpu().numpy()
                    for mask, score in zip(masks, scores):
                        binary = mask[0].astype(bool)
                        box = mask_box(binary)
                        x1, y1, x2, y2 = box
                        cropped = binary[y1:y2, x1:x2]
                        collected.append(
                            Instance(
                                0,
                                class_config.name,
                                cropped,
                                box,
                                int(cropped.sum()),
                                score=float(score),
                                prompt=prompt,
                            )
                        )
                result[class_config.name] = deduplicate_predictions(collected)
        return result
