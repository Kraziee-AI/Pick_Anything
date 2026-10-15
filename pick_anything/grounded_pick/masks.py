"""Binary rectangular goal masks, matching the GCRL_UR10e implementation.

GT uses a sphere approximation of the current orange. It describes a projected
goal region, including occluded pixels, rather than a visible-instance segment.
All quaternions follow IsaacLab 6.1's (x, y, z, w) convention.
"""

from __future__ import annotations

import re

import torch


def orange_caption(instruction: str) -> str:
    """Parse the single supported goal; reject other fruits rather than relabel them."""
    text = instruction.strip().lower()
    if not text or re.search(r"apple|pear|lemon|banana|苹果|梨|柠檬|香蕉", text):
        raise ValueError("当前版本只支持橘子，例如：抓橘子 / pick up the orange。")
    if any(word in text for word in ("橘子", "橙子")) or text in ("橘", "橙"):
        return "orange fruit."
    if not re.search(r"\borange\b", text):
        raise ValueError("语言指令必须指定橘子（orange）；尚未支持其他目标。")
    # Keep English spatial descriptions for the detector, strip action verbs.
    text = re.sub(r"^(?:please\s+)?(?:pick\s+up|pick|grab|grasp)\s+", "", text)
    text = re.sub(r"^(?:the|an|a)\s+", "", text)
    text = re.split(r"\s+(?:and|then)\s+", text)[0].strip(" .!。！")
    if re.search(r"\borange\s+(?:box|cup|bottle|robot)\b", text):
        raise ValueError("orange 必须指水果橘子，当前未支持其他橙色物体。")
    return ("orange fruit" if text == "orange" else text) + "."


def quat_xyzw_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = torch.nn.functional.normalize(quat, dim=-1)
    x, y, z, w = quat.unbind(-1)
    return torch.stack(
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
         2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
         2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def sphere_boxes(
    centers_w: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_ros: torch.Tensor,
    intrinsics: torch.Tensor,
    radius: float,
    near: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tight perspective bounds of a sphere in pixel xyxy; behind/near => invalid.

    Camera ROS axes: +X right, +Y down, +Z forward. Objects intersecting the
    near plane are treated as unavailable rather than exploding into a full mask.
    """
    if radius <= 0 or near <= 0:
        raise ValueError("radius and near must be positive")
    rot = quat_xyzw_matrix(camera_quat_ros)
    xyz = torch.einsum("nji,nj->ni", rot, centers_w - camera_pos_w)
    z = xyz[:, 2]
    valid = torch.isfinite(xyz).all(-1) & (z > radius + near)
    denominator = (z.square() - radius**2).clamp_min(1e-8)
    bounds = []
    for axis in (0, 1):
        a = xyz[:, axis]
        tangent = radius * (a.square() + z.square() - radius**2).clamp_min(0).sqrt()
        lower = (a * z - tangent) / denominator
        upper = (a * z + tangent) / denominator
        focal, principal = intrinsics[:, axis, axis], intrinsics[:, axis, 2]
        bounds.append((lower * focal + principal, upper * focal + principal))
    boxes = torch.stack([bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]], -1)
    valid &= torch.isfinite(boxes).all(-1)
    return torch.where(valid[:, None], boxes, 0.0), valid


def boxes_to_masks(boxes: torch.Tensor, valid: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Rasterize pixel xyxy bounds at pixel centers; output (N,1,H,W), exactly 0/1."""
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    x = torch.arange(width, device=boxes.device, dtype=boxes.dtype)[None, None, :] + 0.5
    y = torch.arange(height, device=boxes.device, dtype=boxes.dtype)[None, :, None] + 0.5
    finite = torch.isfinite(boxes).all(-1) & valid
    x0, y0, x1, y1 = boxes.unbind(-1)
    mask = ((x >= x0[:, None, None]) & (x < x1[:, None, None])
            & (y >= y0[:, None, None]) & (y < y1[:, None, None]) & finite[:, None, None])
    return mask[:, None].float()


def select_detection(result: dict, device: torch.device | str) -> tuple[torch.Tensor, bool, float]:
    """Select one finite, nondegenerate box, or return an empty goal (never GT)."""
    boxes = result["boxes"].to(device=device, dtype=torch.float32)
    scores = result["scores"].to(device=device, dtype=torch.float32)
    good = torch.isfinite(boxes).all(-1) & torch.isfinite(scores)
    good &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not bool(good.any()):
        return torch.zeros(4, device=device), False, 0.0
    index = scores.masked_fill(~good, -torch.inf).argmax()
    return boxes[index], True, float(scores[index])
