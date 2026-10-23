"""Independent privileged scoring for registered single objects; never a grasp target."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = 'target_contact_lift_8cm_5steps_v1'
POCKET = (.0408, -.0002, -.0709)
THRESHOLDS = {'rise_gt_m': .08, 'consecutive_steps': 5, 'distance_lt_m': .045,
              'claw_force_gt_n': 2.2, 'object_force_gt_n': .1,
              'grip_gt_rad': .05, 'grip_lt_rad': 1.6, 'relative_speed_lt_m_s': .15}


def center_kinematics(position, quaternion_xyzw, linear_velocity, angular_velocity,
                      center_offset, com_position=None):
    """Rotate asset-local geometry center; angular velocity is world-frame."""
    position = np.asarray(position, dtype=float)
    center = position + Rotation.from_quat(quaternion_xyzw).apply(center_offset)
    com = position if com_position is None else np.asarray(com_position, dtype=float)
    velocity = np.asarray(linear_velocity) + np.cross(angular_velocity, center - com)
    return center, velocity


def contact_sample(center, velocity, pocket, gripper_velocity, grip, claw_force, object_force):
    center = np.asarray(center, dtype=float)
    distance = float(np.linalg.norm(center - pocket))
    relative_speed = float(np.linalg.norm(velocity - gripper_velocity))
    values = [*center, distance, relative_speed, grip, claw_force, object_force]
    finite = all(math.isfinite(float(value)) for value in values)
    held = bool(finite and distance < .045 and claw_force > 2.2 and object_force > .1
                and .05 < grip < 1.6 and relative_speed < .15)
    return {'target_center_world_m': center.tolist(), 'target_center_z': float(center[2]),
            # Historical executor telemetry key only. Never passed as target_centre.
            'orange_z': float(center[2]), 'distance': distance, 'relative_speed_m_s': relative_speed,
            'grip': float(grip), 'claw_force': float(claw_force), 'object_force': float(object_force),
            'held': held, 'score_schema': SCHEMA}


class LiftTracker:
    """Once-per-physics-control-step evidence; no success-triggered reset."""
    def __init__(self, baseline_center_z, step_dt):
        if not math.isfinite(baseline_center_z) or step_dt <= 0:
            raise ValueError('A finite settled reference and positive step duration are required')
        self.baseline = float(baseline_center_z)
        self.step_dt = float(step_dt)
        self.rows = []
        self.run = self.maximum_run = 0
        self.success = False

    def observe(self, step, sample):
        if self.rows and step <= self.rows[-1]['step']:
            return
        if self.rows and step != self.rows[-1]['step'] + 1:
            self.run = 0
        rise = sample['target_center_z'] - self.baseline
        self.run = self.run + 1 if sample['held'] and rise > .08 else 0
        self.maximum_run = max(self.maximum_run, self.run)
        self.success |= self.run >= 5
        self.rows.append({'step': int(step), **sample, 'rise_m': rise, 'held_lift_run': self.run})

    def summary(self, execution_trace=()):
        hold_steps = {row['step'] for row in execution_trace if row.get('phase') == 'lift_hold'}
        held_rows = [row for row in self.rows if row['step'] in hold_steps]
        final_run = 0
        for row in held_rows:
            final_run = final_run + 1 if row['held'] and row['rise_m'] > .08 else 0
        return {'score_schema': SCHEMA, 'score_thresholds': dict(THRESHOLDS),
                'score_role': 'privileged_evaluation_only_not_used_for_control',
                'settled_center_z_m': self.baseline,
                'generic_grasp_lift_success': bool(self.success),
                'generic_end_hold_success': bool(held_rows and held_rows[-1]['held']
                                               and held_rows[-1]['rise_m'] > .08),
                'lift_hold_samples': len(held_rows), 'final_hold_consecutive_steps': final_run,
                'final_hold_duration_s': final_run * self.step_dt,
                'max_held_lift_run': self.maximum_run,
                'generic_max_rise_m': max((r['rise_m'] for r in self.rows), default=0.),
                'place_evaluated': False, 'place_success': None, 'original_task_success': None}


def sample_env(env, spec):
    """Read GT only here, outside perception/control; robot force is a real sensor."""
    from .sim_eval import tensor
    from isaaclab.utils.math import quat_apply
    robot, obj = env.scene['robot'], env.scene['target_object']
    body = int(robot.find_bodies(['gripper'])[0][0])
    joint = int(robot.find_joints(['gripper'])[0][0])
    def array(value):
        return tensor(value)[0].detach().cpu().numpy()
    com = getattr(obj.data, 'root_com_pos_w', obj.data.root_pos_w)
    center, velocity = center_kinematics(array(obj.data.root_pos_w), array(obj.data.root_quat_w),
        array(obj.data.root_lin_vel_w), array(obj.data.root_ang_vel_w), spec['center_offset_m'], array(com))
    position = tensor(robot.data.body_pos_w)[:, body]
    pocket = position + quat_apply(tensor(robot.data.body_quat_w)[:, body], position.new_tensor(POCKET)[None])
    sensor = env.scene.sensors['robot_contact']
    columns = [i for i, name in enumerate(sensor.body_names) if name in ('gripper', 'moving_jaw_so101_v1')]
    if not columns:
        raise RuntimeError('Missing robot claw contact bodies')
    claw_force = float(tensor(sensor.data.net_forces_w)[0, columns].norm(dim=-1).amax())
    object_force = float(tensor(env.scene.sensors['target_contact'].data.net_forces_w)[0].norm(dim=-1).sum())
    return contact_sample(center, velocity, array(pocket), array(robot.data.body_lin_vel_w)[body],
                          float(tensor(robot.data.joint_pos)[0, joint]), claw_force, object_force)
