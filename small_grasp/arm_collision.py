"""Conservative, sampled SO101 capsule checks against RGB-D and the table.

Capsules join rigid-body origins; they are an approximation, not mesh planning.
All positions and radii use world metres. No simulator object pose enters this module.
"""
from dataclasses import dataclass

import numpy as np

from .geometry import backproject, load_binary_mask, validate_intrinsics, validate_transform


# Rigid bodies in the SO101 USD, from base to jaw. Radii include a small margin.
# The camera housing, wires and grasped object's swept volume are not represented.
LINKS = (
    ('base', 'shoulder', .035),
    ('shoulder', 'upper_arm', .032),
    ('upper_arm', 'lower_arm', .027),
    ('lower_arm', 'wrist', .027),
    ('wrist', 'gripper', .028),
    ('gripper', 'jaw', .018),
)
FIXED_PAD_BODY_M = np.array([-.01289, -.00022, -.0986])
MOVING_PAD_BODY_M = np.array([-.0068, -.0766, .0189])
# Computed from the repository USD using the two pad reference points above.
# These Euclidean distances are NOT calibrated parallel-jaw inner gaps.
CLOSED_PAD_SPAN_M = .030171974752842008
OPEN_PAD_SPAN_M = .1209006333584636
TABLE_Z_M = .041
TABLE_XY_BOUNDS = ((-.145, .855), (-.677, -.227))
TABLE_MARGIN_M = .003
POINT_MARGIN_M = .006
JOINT_SAMPLE_RAD = .08


@dataclass(frozen=True)
class Capsule:
    name: str
    start: np.ndarray
    end: np.ndarray
    radius: float


def capsules_from_bodies(bodies):
    """Construct the same named arm geometry at every FK sample."""
    missing = sorted({name for a, b, _ in LINKS for name in (a, b)} - bodies.keys())
    if missing:
        raise ValueError('SO101 FK missing collision bodies: ' + ', '.join(missing))
    return [Capsule(f'{a}:{b}', np.asarray(bodies[a], dtype=float),
                    np.asarray(bodies[b], dtype=float), radius) for a, b, radius in LINKS]


def add_finger_capsules(capsules, poses):
    """Include both physical pad offsets from the SO101 body/jaw origins."""
    from scipy.spatial.transform import Rotation

    for name, offset, parent in (('fixed_pad', FIXED_PAD_BODY_M, 'gripper'),
                                 ('moving_pad', MOVING_PAD_BODY_M, 'jaw')):
        pose = np.asarray(poses[parent], dtype=float)
        tip = pose[:3] + Rotation.from_quat(pose[3:7]).apply(offset)
        capsules.append(Capsule(f'{parent}:{name}', pose[:3], tip, .006))
    return capsules


def pad_span_from_poses(poses):
    """Distance between the same two pad contact reference points used by SO101."""
    from scipy.spatial.transform import Rotation

    def pad(name, offset):
        pose = np.asarray(poses[name], dtype=float)
        return pose[:3] + Rotation.from_quat(pose[3:7]).apply(offset)

    return float(np.linalg.norm(pad('gripper', FIXED_PAD_BODY_M) -
                                pad('jaw', MOVING_PAD_BODY_M)))


def width_rejection(width_m):
    """Validate network width without inventing a single-jaw opening mapping."""
    try:
        valid = np.isfinite(float(width_m)) and float(width_m) > 0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        return 'invalid_candidate_width'
    return None


def _closest_segment_points(a, b, c, d):
    """Return squared distance between finite 3-D segments."""
    u, v, w = b - a, d - c, a - c
    uu, uv, vv, uw, vw = u @ u, u @ v, v @ v, u @ w, v @ w
    denom = uu * vv - uv * uv
    if denom > 1e-14:
        s = np.clip((uv * vw - vv * uw) / denom, 0., 1.)
    else:
        s = 0.
    t = np.clip((uv * s + vw) / vv, 0., 1.) if vv > 1e-14 else 0.
    s = np.clip((uv * t - uw) / uu, 0., 1.) if uu > 1e-14 else 0.
    t = np.clip((uv * s + vw) / vv, 0., 1.) if vv > 1e-14 else 0.
    return float(np.sum((a + s * u - c - t * v) ** 2))


def point_segment_distance(points, start, end):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    delta = end - start
    length2 = float(delta @ delta)
    t = np.clip(((points - start) @ delta) / length2, 0., 1.) if length2 > 1e-14 else np.zeros(len(points))
    return np.linalg.norm(points - (start + t[:, None] * delta), axis=1)


def _table_hit(capsule, table_z):
    """Check the finite 1.0 x 0.45 m table cuboid, expanded by capsule radius."""
    low_t, high_t = 0., 1.
    delta = capsule.end - capsule.start
    for axis, (low, high) in enumerate(TABLE_XY_BOUNDS):
        low -= capsule.radius
        high += capsule.radius
        if abs(delta[axis]) < 1e-12:
            if not low <= capsule.start[axis] <= high:
                return False
            continue
        t0, t1 = sorted(((low - capsule.start[axis]) / delta[axis],
                         (high - capsule.start[axis]) / delta[axis]))
        low_t, high_t = max(low_t, t0), min(high_t, t1)
        if low_t > high_t:
            return False
    z = min(capsule.start[2] + low_t * delta[2], capsule.start[2] + high_t * delta[2])
    return z - capsule.radius < table_z + TABLE_MARGIN_M


def check_capsules(capsules, obstacles, target, *, table_z=TABLE_Z_M):
    """Return first collision with reason; only the gripper/jaw may touch target."""
    for i, capsule in enumerate(capsules):
        if i > 0 and _table_hit(capsule, table_z):
            return {'reason': 'table', 'link': capsule.name}
        for j in range(i + 2, len(capsules)):
            other = capsules[j]
            # The last jaw capsule shares the gripper body with the prior link.
            if (set(capsule.name.split(':')) & set(other.name.split(':')) or
                    all(any(part in name for part in ('gripper', 'jaw', 'pad'))
                        for name in (capsule.name, other.name))):
                continue
            distance2 = _closest_segment_points(capsule.start, capsule.end, other.start, other.end)
            if distance2 < (capsule.radius + other.radius) ** 2:
                return {'reason': 'self', 'link': capsule.name, 'other_link': other.name}
        if len(obstacles) and np.any(point_segment_distance(obstacles, capsule.start, capsule.end)
                                     < capsule.radius + POINT_MARGIN_M):
            return {'reason': 'observed_obstacle', 'link': capsule.name}
        if capsule.name not in ('gripper:jaw', 'gripper:fixed_pad', 'jaw:moving_pad') and len(target) and np.any(
                point_segment_distance(target, capsule.start, capsule.end) < capsule.radius + POINT_MARGIN_M):
            return {'reason': 'target_contact_outside_gripper', 'link': capsule.name}
    return None


def joint_samples(start, end, arm_indices, max_delta=JOINT_SAMPLE_RAD):
    """Include both endpoints; sample at most max_delta on every arm joint."""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    count = max(1, int(np.ceil(np.max(np.abs(b[arm_indices] - a[arm_indices])) / max_delta)))
    return [a + (b - a) * (step / count) for step in range(count + 1)]


def load_observed_cloud(view_dir, *, stride=3):
    """Load non-target and target RGB-D points in world frame from a saved view."""
    import json
    from pathlib import Path

    view = Path(view_dir)
    depth = np.load(view / 'depth.npy')
    intrinsics = validate_intrinsics(json.loads((view / 'intrinsics.json').read_text()), depth.shape)
    transform = validate_transform(json.loads((view / 'world_from_camera.json').read_text()))
    mask = load_binary_mask(view / 'target_mask.png')
    if mask.shape != depth.shape:
        raise ValueError('Target mask and depth resolution differ')
    cloud, valid = backproject(depth, intrinsics)
    sampled = np.zeros_like(valid)
    sampled[::stride, ::stride] = True
    valid &= sampled
    world = cloud @ transform[:3, :3].T + transform[:3, 3]
    # The table plane has its own analytic check. Keep elevated visible points.
    obstacle = world[valid & ~mask & (world[..., 2] > TABLE_Z_M + .012)]
    target = world[valid & mask]
    if not len(target):
        raise ValueError('No valid target depth for SO101 collision check')
    return obstacle.astype(float), target.astype(float)


def remove_robot_returns(points, current_capsules, margin=.012):
    """Remove the robot's own returns from the initial wrist RGB-D image."""
    keep = np.ones(len(points), dtype=bool)
    for capsule in current_capsules:
        keep &= point_segment_distance(points, capsule.start, capsule.end) > capsule.radius + margin
    return points[keep]


def load_capture_joint_positions(view_dir, joint_names):
    """Load only robot joint state from the image's capture instant.

    The named robot-only file takes precedence. Older evaluation_state.json
    files are supported solely for their joint_position field; object state is
    neither returned nor used by the collision planner.
    """
    import json
    from pathlib import Path

    view = Path(view_dir)
    names = list(joint_names)
    if len(names) != len(set(names)):
        raise ValueError('Robot joint names contain duplicates')
    robot_file = view / 'robot_joint_positions.json'
    if robot_file.exists():
        data = json.loads(robot_file.read_text())
        saved_names = data.get('joint_names')
        saved_positions = data.get('joint_position', data.get('joint_positions'))
        if (not isinstance(saved_names, list) or not isinstance(saved_positions, list)
                or len(saved_names) != len(saved_positions)
                or len(saved_names) != len(set(saved_names)) or set(saved_names) != set(names)):
            raise ValueError('Captured robot joint names do not match executor')
        by_name = dict(zip(saved_names, saved_positions, strict=True))
        q = np.asarray([by_name[name] for name in names], dtype=float)
        source = 'robot_joint_positions.json'
    else:
        data = json.loads((view / 'evaluation_state.json').read_text())
        if 'joint_position' not in data:
            raise ValueError('Legacy capture lacks robot joint_position')
        q = np.asarray(data['joint_position'], dtype=float)
        source = 'evaluation_state.json:joint_position'
    if q.shape != (len(names),) or not np.isfinite(q).all():
        raise ValueError('Captured robot joint positions must be finite and match robot joints')
    return q, source


def rejection_counts(attempts):
    """Count rejected candidate orientations; per-candidate detail stays in attempts."""
    counts = {}
    for attempt in attempts:
        reason = attempt.get('rejected')
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return counts
