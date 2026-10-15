"""Conservative oriented bounds for GT goal rectangles; xyzw quaternions."""

import torch

from pick_anything.grounded_pick.masks import quat_xyzw_matrix

# Bounding half extents about geometric centres (metres), not rigid-body origins.
HALF_EXTENTS = {
    "orange": (0.0281, 0.0281, 0.0281),
    "yellow_duck": (0.025, 0.039, 0.031),
    "blue_cube": (0.025, 0.025, 0.025),
    "red_cylinder": (0.021, 0.021, 0.030),
}


def oriented_boxes(centers, quats, half_extents, camera_pos, camera_quats, intrinsics, near=0.001):
    signs = centers.new_tensor([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ])
    corners = torch.einsum("nij,nkj->nki", quat_xyzw_matrix(quats), half_extents[:, None] * signs)
    corners += centers[:, None] - camera_pos[:, None]
    xyz = torch.einsum("nji,nkj->nki", quat_xyzw_matrix(camera_quats), corners)
    valid = torch.isfinite(xyz).all(dim=(1, 2)) & (xyz[..., 2] > near).all(-1)
    uv = xyz[..., :2] / xyz[..., 2:].clamp_min(near)
    uv = uv * torch.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1]], -1)[:, None]
    uv += intrinsics[:, :2, 2][:, None]
    bounds = torch.cat([uv.amin(1), uv.amax(1)], -1)
    valid &= torch.isfinite(bounds).all(-1)
    return torch.where(valid[:, None], bounds, 0), valid
