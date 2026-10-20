"""Inference-only PointNet2 forward operations using installed PyTorch.

Ball/cylinder queries preserve source point order, strict boundaries and first
neighbour padding of the upstream CUDA kernels. FPS starts at zero and excludes
points with squared norm <= 1e-3, as upstream. Equal-distance FPS/3NN tie ordering
can differ from CUDA parallel reduction. No claim of bitwise equivalence.
"""
import torch


def furthest_point_sampling(xyz, count):
    batch, points, _ = xyz.shape
    if points == 0 or count < 1:
        raise ValueError('FPS requires nonempty points and positive count')
    indices = torch.zeros((batch, count), dtype=torch.long, device=xyz.device)
    distance = torch.full((batch, points), 1e10, device=xyz.device)
    valid = xyz.square().sum(-1) > 1e-3
    old = indices[:, 0]
    rows = torch.arange(batch, device=xyz.device)
    for step in range(1, count):
        delta = xyz - xyz[rows, old][:, None]
        distance = torch.minimum(distance, delta.square().sum(-1))
        old = torch.where(valid, distance, -1.0).argmax(-1)
        indices[:, step] = old
    return indices.int()


def gather_points(features, indices):
    return features.gather(2, indices.long()[:, None].expand(-1, features.shape[1], -1))


def group_points(features, indices):
    batch, channels, _ = features.shape
    return features.gather(2, indices.long().reshape(batch, 1, -1).expand(-1, channels, -1)).reshape(
        batch, channels, indices.shape[1], indices.shape[2])


def three_nn(unknown, known):
    distances, indices = [], []
    for chunk in unknown.split(128, dim=1):
        squared = (chunk[:, :, None] - known[:, None]).square().sum(-1)
        dist, idx = squared.topk(3, dim=-1, largest=False, sorted=True)
        distances.append(dist)
        indices.append(idx)
    return torch.cat(distances, 1), torch.cat(indices, 1).int()


def three_interpolate(features, indices, weights):
    return (group_points(features, indices) * weights[:, None]).sum(-1)


def _first_neighbours(valid, count):
    size = valid.shape[-1]
    order = torch.arange(size, device=valid.device).expand(valid.shape)
    values = torch.where(valid, order, size).topk(min(count, size), dim=-1, largest=False, sorted=True).values
    first = values[..., :1].clamp_max(size - 1)
    first = torch.where(values[..., :1] == size, 0, first)
    values = torch.where(values == size, first, values)
    if count > size:
        values = torch.cat([values, first.expand(*first.shape[:-1], count - size)], -1)
    return values.int()


def ball_query(centres, xyz, radius, count):
    groups = []
    for chunk in centres.split(64, dim=1):
        squared = (xyz[:, None] - chunk[:, :, None]).square().sum(-1)
        groups.append(_first_neighbours(squared < radius * radius, count))
    return torch.cat(groups, 1)


def cylinder_query(centres, xyz, rotations, radius, hmin, hmax, count):
    groups = []
    rotations = rotations.reshape(*centres.shape[:2], 3, 3)
    for start in range(0, centres.shape[1], 64):
        chunk = centres[:, start:start + 64]
        local = torch.matmul(xyz[:, None] - chunk[:, :, None], rotations[:, start:start + 64])
        valid = (local[..., 1:].square().sum(-1) < radius * radius)
        valid &= (local[..., 0] > hmin) & (local[..., 0] < hmax)
        groups.append(_first_neighbours(valid, count))
    return torch.cat(groups, 1)
