from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def configure_autocast(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@dataclass
class InstancePrediction:
    boxes_xyxy: torch.Tensor
    scores: torch.Tensor
    masks: torch.Tensor
    labels: list[str]


class Sam3AutoSegInferencer:
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str,
        confidence_threshold: float,
        resolution: int,
    ) -> None:
        self.device = device
        self.autocast_context = configure_autocast(device)
        self.model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False,
            device=device,
        )
        self.processor = Sam3Processor(
            self.model,
            resolution=resolution,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    def predict(self, image_pil: Image.Image, prompts: list[str]) -> InstancePrediction:
        return self.predict_with_threshold(
            image_pil=image_pil,
            prompts=prompts,
            confidence_threshold=self.processor.confidence_threshold,
        )

    def predict_with_threshold(
        self,
        image_pil: Image.Image,
        prompts: list[str],
        confidence_threshold: float,
    ) -> InstancePrediction:
        width, height = image_pil.size
        with self.autocast_context:
            self.processor.set_confidence_threshold(confidence_threshold)
            state = self.processor.set_image(image_pil)
            boxes_list = []
            masks_list = []
            scores_list = []
            labels_list: list[str] = []

            for prompt in prompts:
                self.processor.reset_all_prompts(state)
                output = self.processor.set_text_prompt(prompt=prompt, state=state)
                prompt_boxes = output["boxes"].detach().float().cpu()
                prompt_scores = output["scores"].detach().float().cpu()
                prompt_masks = output["masks"]
                if prompt_boxes.numel() == 0:
                    continue
                boxes_list.append(prompt_boxes)
                scores_list.append(prompt_scores)
                masks_list.append(prompt_masks)
                labels_list.extend([prompt] * int(prompt_scores.shape[0]))

        if boxes_list:
            boxes_xyxy = torch.cat(boxes_list, dim=0)
            scores = torch.cat(scores_list, dim=0)
            masks = torch.cat(masks_list, dim=0)
        else:
            boxes_xyxy = torch.empty((0, 4), dtype=torch.float32)
            scores = torch.empty((0,), dtype=torch.float32)
            masks = torch.empty((0, 1, height, width), dtype=torch.bool)

        return InstancePrediction(
            boxes_xyxy=boxes_xyxy,
            scores=scores,
            masks=masks,
            labels=labels_list,
        )
