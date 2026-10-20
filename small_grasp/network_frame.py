"""Experimental model input coordinates; exported poses stay in real camera axes.

Only model input points and complete decoded poses change basis. It never
projects a prediction onto a preferred direction or uses object ground truth.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def network_from_camera(profile, frame):
    if profile not in ('legacy', 'open-jaws') or frame not in ('camera', 'legacy-optical'):
        raise ValueError('Unknown camera profile or model frame')
    if frame == 'camera' or profile == 'legacy':
        return np.eye(3)
    from .wrist_view import OPEN_JAW_YAW_DEG, OPEN_JAW_PITCH_DEG
    basis = np.diag([1., -1., -1.])
    yaw = Rotation.from_rotvec([0., np.deg2rad(OPEN_JAW_YAW_DEG), 0.]).as_matrix()
    pitch = Rotation.from_rotvec([np.deg2rad(OPEN_JAW_PITCH_DEG), 0., 0.]).as_matrix()
    return basis @ yaw @ pitch @ basis


def frame_metadata(profile, frame):
    """A missing profile is permitted only for unchanged camera coordinates."""
    if profile is None and frame == 'camera':
        transform = np.eye(3)
    else:
        transform = network_from_camera(profile, frame)
    return {'frame': frame, 'wrist_camera_profile': profile,
            'model_from_camera': transform.tolist(),
            'camera_from_model': transform.T.tolist(),
            'point_transform': 'p_model = model_from_camera @ p_camera',
            'pose_inverse': 'p_camera = camera_from_model @ p_model; R_camera_grasp = camera_from_model @ R_model_grasp',
            'exported_pose_frame': 'camera_optical',
            'calibration_assumption': 'same camera origin; known wrist_view mount rotation only; live intrinsics and extrinsics unchanged'}


def grasps_to_camera(rows, model_from_camera):
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 17:
        raise ValueError('Expected N x 17 grasp rows')
    transform = np.asarray(model_from_camera)
    if transform.shape != (3, 3) or not np.isfinite(transform).all() or not np.allclose(
            transform.T @ transform, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(transform), 1.):
        raise ValueError('Model transform must be a proper rotation')
    result = rows.copy()
    if np.array_equal(transform, np.eye(3)):
        return result
    result[:, 13:16] = rows[:, 13:16] @ transform
    result[:, 4:13] = (transform.T @ rows[:, 4:13].reshape(-1, 3, 3)).reshape(-1, 9)
    return result
