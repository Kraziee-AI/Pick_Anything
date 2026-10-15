"""Lazy GroundingDINO inference through the installed Transformers interface."""

from __future__ import annotations

from pathlib import Path

import torch

from .language import goal_caption
from .masks import select_detection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = PROJECT_ROOT / "outputs/grounded_pick/cache/huggingface"
DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


def workspace_output(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"输出与缓存路径必须位于 Pick_Anything 内：{resolved}")
    return resolved


class GroundingDino:
    def __init__(
        self, model_name: str = DEFAULT_MODEL, device: str = "cpu",
        cache_dir: str = str(DEFAULT_CACHE), local_files_only: bool = True,
        box_threshold: float = 0.25, text_threshold: float = 0.25,
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if not (0 < box_threshold < 1 and 0 < text_threshold < 1):
            raise ValueError("Grounding thresholds must be between 0 and 1")
        self.device = torch.device(device)
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.model_name = model_name
        cache = workspace_output(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        kwargs = {"cache_dir": str(cache), "local_files_only": local_files_only}
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, **kwargs)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_name, disable_custom_kernels=True, **kwargs,
            )
        except OSError as exc:
            raise RuntimeError(
                "GroundingDINO 权重不可用。先运行 scripts/ground_orange_image.py --download，"
                f"或用 --grounding-model 指定本地 Transformers 模型目录。原始错误：{exc}"
            ) from exc
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, rgb: torch.Tensor, instruction: str, caption: str = "") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Only RGB and language enter this function; rgb is uint8 (N,H,W,3)."""
        from PIL import Image

        caption = goal_caption(instruction, caption)
        boxes, valid, confidence = [], [], []
        # Single-image inference limits peak memory; intended for a few play envs.
        for frame in rgb:
            pil = Image.fromarray(frame.detach().cpu().numpy())
            inputs = self.processor(images=pil, text=caption, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            result = self.processor.post_process_grounded_object_detection(
                outputs, input_ids=inputs.input_ids, threshold=self.box_threshold,
                text_threshold=self.text_threshold, target_sizes=[(pil.height, pil.width)],
            )[0]
            box, found, score = select_detection(result, rgb.device)
            boxes.append(box)
            valid.append(found)
            confidence.append(score)
        return (torch.stack(boxes), torch.tensor(valid, device=rgb.device, dtype=torch.bool),
                torch.tensor(confidence, device=rgb.device))
