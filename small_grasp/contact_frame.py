"""Contact-frame calibration for a single rotating SO101 jaw."""
import numpy as np
from scipy.spatial.transform import Rotation


def contact_frame(fixed, moving, pivot, axis, current_angle, diameter):
    fixed, moving, pivot, axis = [np.asarray(value, dtype=float) for value in (fixed, moving, pivot, axis)]
    axis = axis / np.linalg.norm(axis)

    def pad(angle):
        return pivot + Rotation.from_rotvec(axis * (angle - current_angle)).apply(moving - pivot)

    low, high = 0., float(current_angle)
    if not np.linalg.norm(pad(low) - fixed) < diameter < np.linalg.norm(pad(high) - fixed):
        raise ValueError('Estimated target diameter is outside the calibrated jaw pad span')
    for _ in range(24):
        angle = (low + high) / 2
        if np.linalg.norm(pad(angle) - fixed) < diameter:
            low = angle
        else:
            high = angle
    angle = (low + high) / 2
    moving_contact = pad(angle)
    width = moving_contact - fixed
    width /= np.linalg.norm(width)
    midpoint = (fixed + moving_contact) / 2
    approach = midpoint - np.dot(midpoint, width) * width
    approach /= np.linalg.norm(approach)
    return np.stack([approach, width, np.cross(approach, width)], axis=1), {
        'contact_joint_rad': angle, 'contact_midpoint_body_m': midpoint.tolist(),
        'contact_diameter_m': diameter, 'method': 'jaw_contact_width_not_open_width'}
