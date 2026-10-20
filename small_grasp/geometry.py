"""Camera-optical geometry and target filtering; no simulator state or robot API."""
import numpy as np


def load_binary_mask(path):
    from PIL import Image
    values = np.asarray(Image.open(path))
    if values.ndim != 2 or not np.isin(values, [0, 1, 255]).all():
        raise ValueError('Target mask must be a single-channel binary image (0/1 or 0/255), not an RGB overlay')
    return values > 0


def validate_intrinsics(data, shape):
    height, width = shape
    values = np.array([data[k] for k in ('fx', 'fy', 'cx', 'cy', 'depth_scale')], dtype=float)
    if not np.isfinite(values).all() or min(values[0], values[1], values[4]) <= 0:
        raise ValueError('Intrinsics and depth_scale must be finite, with positive focal lengths and scale')
    if data.get('width') != width or data.get('height') != height:
        raise ValueError('Intrinsics resolution must match the registered RGB/depth images')
    return dict(data)


def backproject(depth_raw, intrinsics, max_depth=2.0):
    z = np.asarray(depth_raw, dtype=np.float32) / intrinsics['depth_scale']
    v, u = np.indices(z.shape)
    points = np.stack([(u - intrinsics['cx']) * z / intrinsics['fx'],
                       (v - intrinsics['cy']) * z / intrinsics['fy'], z], -1).astype(np.float32)
    valid = np.isfinite(points).all(-1) & (z > 0) & (z < max_depth)
    return points, valid


def target_crop(mask, padding=20):
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return np.zeros_like(mask, dtype=bool)
    crop = np.zeros_like(mask, dtype=bool)
    crop[max(0, yy.min() - padding):yy.max() + padding + 1,
         max(0, xx.min() - padding):xx.max() + padding + 1] = True
    return crop


def network_input_mask(mask, valid_depth, region='bbox', padding=20):
    """Select inference input; collision checks must still use the full scene."""
    if region == 'target-mask':
        return np.asarray(mask, dtype=bool) & valid_depth
    if region == 'bbox':
        return target_crop(mask, padding) & valid_depth
    raise ValueError(f'Unknown network input region: {region}')


def target_membership(centres, depth_m, target_mask, intrinsics, tolerance=0.04):
    """Projected centre must overlap target pixels and agree with visible depth."""
    xyz = np.asarray(centres)
    z = xyz[:, 2]
    finite = np.isfinite(xyz).all(-1) & (z > 0)
    u, v = np.zeros(len(xyz), dtype=int), np.zeros(len(xyz), dtype=int)
    u[finite] = np.rint(xyz[finite, 0] * intrinsics['fx'] / z[finite] + intrinsics['cx']).astype(int)
    v[finite] = np.rint(xyz[finite, 1] * intrinsics['fy'] / z[finite] + intrinsics['cy']).astype(int)
    valid = finite & (u >= 0) & (u < target_mask.shape[1]) & (v >= 0) & (v < target_mask.shape[0])
    ids = np.flatnonzero(valid)
    valid[ids] &= target_mask[v[ids], u[ids]]
    valid[ids] &= np.isfinite(depth_m[v[ids], u[ids]]) & (depth_m[v[ids], u[ids]] > 0)
    valid[ids] &= np.abs(z[ids] - depth_m[v[ids], u[ids]]) <= tolerance
    return valid


def validate_transform(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('T_world_camera must be a finite 4x4 matrix')
    rotation = matrix[:3, :3]
    if not (np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError('T_world_camera must be a proper rigid SE(3) transform')
    return matrix


def pose_matrix(row):
    matrix = np.eye(4)
    matrix[:3, :3] = row[4:13].reshape(3, 3)
    matrix[:3, 3] = row[13:16]
    return validate_transform(matrix)


def diverse_grasps(rows, top_k, distance=0.01, angle_deg=15):
    indices, _ = diverse_grasp_indices(rows, top_k, distance, angle_deg)
    return np.asarray(rows[indices], dtype=np.float32).reshape(-1, 17)


def diverse_grasp_indices(rows, top_k, distance=0.01, angle_deg=15):
    """Return source row indices and a reason for each unselected row."""
    if top_k < 1:
        raise ValueError('top_k must be positive')
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 17:
        raise ValueError('Grasp rows must have shape N x 17')
    kept = []
    reasons = np.full(len(rows), 'beyond_top_k', dtype='<U32')
    cosine = np.cos(np.deg2rad(angle_deg))
    for index in np.argsort(-rows[:, 0], kind='stable'):
        row = rows[index]
        rotation = row[4:13].reshape(3, 3)
        duplicate = False
        for other_index in kept:
            other = rows[other_index]
            other_rotation = other[4:13].reshape(3, 3)
            if (np.linalg.norm(row[13:16] - other[13:16]) < distance
                    and np.dot(rotation[:, 0], other_rotation[:, 0]) > cosine
                    and abs(np.dot(rotation[:, 1], other_rotation[:, 1])) > cosine):
                duplicate = True
                break
        if duplicate:
            reasons[index] = 'duplicate_pose'
        elif len(kept) < top_k:
            kept.append(index)
            reasons[index] = 'selected'
    return np.asarray(kept, dtype=np.int64), reasons
