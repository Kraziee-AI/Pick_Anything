"""Robot-only reference-line geometry; no object size or measured body rotation."""
import numpy as np
from scipy.spatial.transform import Rotation

from .gripper_geometry import FIXED_REFERENCE_M, MOVING_REFERENCE_M

REACHABLE_LINE_BETAS = (1., .75, .5, .25, 0.)
REACHABLE_LINE_NOMINAL_ORIENTATION_DEG = 13.


class PlannedClosingLine:
    def __init__(self, geometry, centre, initial_planned_rotation, initial_jaw_rad, tool_offset, beta=1.):
        self.geometry = geometry
        self.centre = np.array(centre, dtype=float, copy=True)
        self.rotation0 = np.array(initial_planned_rotation, dtype=float, copy=True)
        self.tool = np.array(tool_offset, dtype=float, copy=True)
        if (self.centre.shape != (3,) or self.tool.shape != (3,) or self.rotation0.shape != (3, 3) or
                not all(np.isfinite(x).all() for x in (self.centre, self.rotation0, self.tool)) or
                not np.isfinite(initial_jaw_rad) or
                not np.allclose(self.rotation0.T@self.rotation0, np.eye(3), atol=1e-6) or
                not np.isclose(np.linalg.det(self.rotation0), 1., atol=1e-6)):
            raise ValueError('Invalid planned closing-line pose')
        self.jaw0 = float(initial_jaw_rad)
        if not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError('Invalid closing-line compensation fraction')
        self.beta = float(beta)
        self.frame0 = self.rotation0@self.axes(self.jaw0)

    def references(self, jaw):
        if not np.isfinite(jaw):
            raise ValueError('Invalid closing-line jaw angle')
        return FIXED_REFERENCE_M, self.geometry.moving_to_fixed(MOVING_REFERENCE_M, float(jaw))

    def axes(self, jaw):
        fixed, moving = self.references(jaw)
        width = moving-fixed
        span = np.linalg.norm(width)
        if not np.isfinite(span) or span <= 1e-9:
            raise ValueError('Degenerate robot closing references')
        width = width/span
        approach = self.tool-np.dot(self.tool, width)*width
        norm = np.linalg.norm(approach)
        if not np.isfinite(norm) or norm <= 1e-9:
            raise ValueError('Degenerate robot closing approach')
        approach = approach/norm
        return np.stack([approach, width, np.cross(approach, width)], axis=1)

    def pose(self, jaw):
        fixed, moving = self.references(jaw)
        midpoint = (fixed+moving)/2
        # Return the original input exactly at the initial point, avoiding
        # numerical bootstrap changes to the planned achieved orientation.
        full_rotation = self.rotation0.copy() if jaw == self.jaw0 else self.frame0@self.axes(jaw).T
        if jaw == self.jaw0 or self.beta == 0.:
            rotation = self.rotation0.copy()
        elif self.beta == 1.:
            rotation = full_rotation  # preserve existing planned-line arithmetic exactly
        else:
            log = Rotation.from_matrix(self.rotation0.T@full_rotation).as_rotvec()
            rotation = self.rotation0@Rotation.from_rotvec(self.beta*log).as_matrix()
        desired = self.centre+rotation@(self.tool-midpoint)
        origin = desired-rotation@self.tool
        fixed_world = origin+rotation@fixed
        moving_world = origin+rotation@moving
        span = float(np.linalg.norm(moving-fixed))
        evidence = {'jaw_rad': float(jaw), 'span_m': span, 'reference_midpoint_body_m': midpoint.tolist(),
            'requested_rotation_world': rotation.tolist(), 'requested_tool_world_m': desired.tolist(),
            'fixed_reference_world_m': fixed_world.tolist(), 'moving_reference_world_m': moving_world.tolist(),
            'fixed_line_identity_error_m': float(np.linalg.norm(fixed_world-(self.centre-span/2*self.frame0[:, 1]))),
            'moving_line_identity_error_m': float(np.linalg.norm(moving_world-(self.centre+span/2*self.frame0[:, 1])))}
        width_world = rotation@((moving-fixed)/span)
        evidence.update(compensation_beta=self.beta,
            closing_line_residual_deg=float(np.degrees(np.arctan2(
                np.linalg.norm(np.cross(width_world, self.frame0[:, 1])), width_world@self.frame0[:, 1]))),
            full_compensation_angle_deg=float(np.degrees(Rotation.from_matrix(self.rotation0.T@full_rotation).magnitude())),
            applied_compensation_angle_deg=float(np.degrees(Rotation.from_matrix(self.rotation0.T@rotation).magnitude())),
            exact_line_preservation_claim=self.beta == 1.)
        return desired, rotation, evidence

    def observed_residual(self, actual_rotation, actual_jaw):
        """Write-only robot telemetry; never used to redefine a desired pose."""
        rotation = np.asarray(actual_rotation, dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.isfinite(actual_jaw):
            return {'status': 'unavailable', 'actual_closing_line_residual_deg': None}
        fixed, moving = self.references(actual_jaw)
        width = rotation@((moving-fixed)/np.linalg.norm(moving-fixed))
        residual = np.degrees(np.arctan2(np.linalg.norm(np.cross(width, self.frame0[:, 1])), width@self.frame0[:, 1]))
        return {'status': 'available', 'selected_beta': self.beta, 'actual_jaw_rad': float(actual_jaw),
                'actual_reference_width_world': width.tolist(), 'initial_planned_width_world': self.frame0[:, 1].tolist(),
                'actual_closing_line_residual_deg': float(residual),
                'role': 'robot-only angular diagnostic; never control or scoring input'}
