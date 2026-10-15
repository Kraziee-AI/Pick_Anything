"""Download/preflight GroundingDINO or ground an RGB image without launching Isaac."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounded_pick_cli import configure_caches


def main():
    configure_caches()
    import numpy as np
    import torch
    from PIL import Image
    from pick_anything.grounded_pick.grounding import DEFAULT_CACHE, DEFAULT_MODEL, GroundingDino, workspace_output
    from pick_anything.grounded_pick.masks import boxes_to_masks
    from pick_anything.grounded_pick.language import goal_caption

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="输入 RGB 图片；不传图片时只检查模型加载")
    parser.add_argument("--instruction", default="抓橘子")
    parser.add_argument("--caption", default="", help="自由英文目标描述；可作为中文指令的检测词")
    parser.add_argument("--grounding-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--download", action="store_true", help="允许首次下载到本仓库的专用缓存")
    parser.add_argument("--output-dir", default="outputs/grounded_pick/image_check")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    args = parser.parse_args()
    caption = goal_caption(args.instruction, args.caption)
    detector = GroundingDino(
        args.grounding_model, args.device, str(DEFAULT_CACHE), not args.download,
        args.box_threshold, args.text_threshold,
    )
    print(f"GroundingDINO ready: {args.grounding_model}")
    if args.image is None:
        return
    image = Image.open(args.image).convert("RGB")
    rgb = torch.from_numpy(np.array(image))[None]
    boxes, found, confidence = detector.predict(rgb, args.instruction, caption=caption)
    mask = boxes_to_masks(boxes, found, image.height, image.width)
    directory = workspace_output(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    image.save(directory / "rgb.png")
    Image.fromarray((mask[0, 0] * 255).byte().numpy()).save(directory / "mask.png")
    metadata = {"instruction": args.instruction, "caption": caption,
                "box": boxes[0].tolist(), "detected": bool(found[0]), "confidence": float(confidence[0])}
    (directory / "grounding.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
