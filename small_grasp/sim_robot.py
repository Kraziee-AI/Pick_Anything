"""SO101 execution adapter; perception plans use only candidates/RGB-D/robot state."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .sim_eval import tensor
from .grasp_hold import BoundedContactHold, HoldConfig

POCKET = (.0408, -.0002, -.0709)
CONFIG = {'ik_seeds': 16, 'ik_iterations': 80, 'ik_position_tolerance_m': .01,
          'ik_orientation_tolerance_deg': 35, 'pregrasp_distance_m': .05,
          'lift_distance_m': .12, 'open_joint_rad': 1.4, 'closed_joint_rad': .05,
          'contact_hold_preload_rad': .08, 'max_candidates': 20}
SERVO_CONFIG = {'position_tolerance_m': .012, 'orientation_tolerance_deg': 15.,
                'arm_joint_tolerance_rad': .14, 'arrival_consecutive_steps': 3,
                'arrival_max_steps': 45, 'cartesian_waypoint_spacing_m': .008,
                'cartesian_steps_per_waypoint': 8, 'cartesian_max_waypoints': 12,
                'approach_arrival_arm_speed_rad_s': .05,
                'cartesian_max_joint_delta_rad': .15, 'cartesian_joint_distance_weight_m': .002,
                'mesh_table_margin_m': .008, 'mesh_box_margin_m': .012,
                'mesh_max_vertical_correction_m': .008, 'approach_clearance_z_m': .20,
                'hold_preload_rad': .02, 'open_joint_rad': 1.4,
                'opening_policy': 'wide_first_feasible', 'opening_options_rad': [1.4, 1., .8],
                'opening_adjust_steps': 45, 'close_policy': 'contact_stall',
                'close_contact_force_n': 8., 'close_velocity_ratio': .8,
                'close_command_error_rad': .03, 'close_contact_consecutive_steps': 3,
                'bounded_hold_enabled': False, 'soft_limit_fallback_enabled': True,
                'soft_limit_fallback_weight': 100., 'lift_fixed_jaw_target_enabled': False}


def _soft_limit_dof_arrays(q_starts, qd_starts, dof_dimensions, coordinates, soft_limits, lower, upper):
    """Map robot scalar joint coordinates to Newton DOF-bound arrays explicitly."""
    lower, upper = np.array(lower, dtype=np.float32, copy=True), np.array(upper, dtype=np.float32, copy=True)
    dimensions = np.asarray(dof_dimensions).sum(axis=1)
    q_starts, qd_starts = np.asarray(q_starts), np.asarray(qd_starts)
    coordinates, soft_limits = np.asarray(coordinates), np.asarray(soft_limits)
    if soft_limits.shape != (len(coordinates), 2) or not np.isfinite(soft_limits).all() or (soft_limits[:, 0] > soft_limits[:, 1]).any():
        raise ValueError('Invalid robot soft joint limits for IK fallback')
    mapping = []
    for coordinate, bounds in zip(coordinates, soft_limits):
        matches = np.flatnonzero((q_starts[:len(dimensions)] == coordinate) & (dimensions == 1))
        if len(matches) != 1:
            raise ValueError(f'IK fallback requires one scalar joint for coordinate {coordinate}')
        dof = int(qd_starts[matches[0]])
        if dof < 0 or dof >= len(lower) or dof >= len(upper) or dof in mapping:
            raise ValueError(f'Invalid or duplicate IK fallback DOF {dof}')
        lower[dof], upper[dof] = bounds
        mapping.append(dof)
    return lower, upper, mapping


class _ClosureContactLatch:
    """Latch a jaw target using only measured robot contact and motion."""
    def __init__(self, nominal_rate_rad_s, policy='legacy'):
        if policy not in ('legacy', 'no_reopen'):
            raise ValueError('Unsupported closure latch policy')
        self.policy = policy
        self.nominal_rate = abs(float(nominal_rate_rad_s))
        self.consecutive = 0
        self.target = None

    def observe(self, force_n, joint_rad, command_rad, velocity_rad_s):
        if self.target is not None:
            return None
        values = np.array([force_n, joint_rad, command_rad, velocity_rad_s, self.nominal_rate])
        qualifies = (np.isfinite(values).all() and self.nominal_rate > 0 and
                     force_n >= SERVO_CONFIG['close_contact_force_n'] and
                     abs(velocity_rad_s) <= SERVO_CONFIG['close_velocity_ratio'] * self.nominal_rate and
                     joint_rad - command_rad > SERVO_CONFIG['close_command_error_rad'])
        if self.policy == 'no_reopen':
            qualifies = qualifies and command_rad >= CONFIG['closed_joint_rad']
        self.consecutive = self.consecutive + 1 if qualifies else 0
        if self.consecutive < SERVO_CONFIG['close_contact_consecutive_steps']:
            return None
        self.target = max(CONFIG['closed_joint_rad'], joint_rad - SERVO_CONFIG['hold_preload_rad'])
        legacy_target = self.target
        if self.policy == 'no_reopen':
            # Preserve the already issued closing command. Do not consume any
            # additional closure, preload, force or actuator budget here.
            self.target = max(CONFIG['closed_joint_rad'], min(command_rad, self.target))
        return {'event': 'contact_stall_latched', 'claw_force_n': float(force_n),
                'joint_position_rad': float(joint_rad), 'previous_joint_target_rad': float(command_rad),
                'joint_velocity_rad_s': float(velocity_rad_s), 'nominal_close_rate_rad_s': self.nominal_rate,
                'command_error_rad': float(joint_rad - command_rad),
                'consecutive_contact_steps': self.consecutive, 'locked_joint_target_rad': self.target,
                'latch_policy': self.policy, 'legacy_joint_target_rad': legacy_target}


class _GripperMeshClearance:
    """Actual robot collision submeshes against table and observed box points.

    Each authored collision submesh is convexified separately, matching a
    conservative collision representation without enclosing the jaw opening.
    """
    def __init__(self, geometry, box_points=()):
        from scipy.spatial import ConvexHull
        self.geometry = geometry
        self.box_points = np.asarray(box_points, dtype=float).reshape(-1, 3)
        self.parts = []
        for label, (triangles, names) in geometry.meshes.items():
            names = np.asarray(names)
            for name in np.unique(names):
                vertices = np.unique(triangles[names == name].reshape(-1, 3), axis=0)
                hull = ConvexHull(vertices)
                self.parts.append((label, name, vertices[hull.vertices], hull.equations))

    def check(self, pocket, rotation, joint_rad, *, table_buffer_m=0.):
        from .arm_collision import TABLE_XY_BOUNDS, TABLE_Z_M
        rotation, pocket = np.asarray(rotation), np.asarray(pocket)
        points_fixed = (self.box_points - pocket) @ rotation + np.asarray(POCKET)
        g = self.geometry
        moving_rotation = g.parent_rotation @ Rotation.from_rotvec(g.axis * joint_rad).as_matrix() @ g.child_rotation.T
        points_moving = (points_fixed - g.parent_pivot) @ moving_rotation + g.child_pivot
        minimum_z = np.inf
        minimum_evidence = {}
        for label, name, vertices, equations in self.parts:
            vertices_fixed = g.moving_to_fixed(vertices, joint_rad) if label == 'moving' else vertices
            world = (vertices_fixed - POCKET) @ rotation.T + pocket
            over_table = np.ones(len(world), dtype=bool)
            for axis, (low, high) in enumerate(TABLE_XY_BOUNDS):
                over_table &= (world[:, axis] >= low) & (world[:, axis] <= high)
            if over_table.any():
                selected_world = world[over_table]
                lowest = selected_world[int(np.argmin(selected_world[:, 2]))]
                if float(lowest[2]) < minimum_z:
                    minimum_z = float(lowest[2])
                    minimum_evidence = {'mesh': str(name), 'body': label,
                                        'minimum_vertex_world_m': lowest.tolist()}
            local = points_moving if label == 'moving' else points_fixed
            margin = SERVO_CONFIG['mesh_box_margin_m']
            near = local[(local >= vertices.min(0) - margin).all(1) &
                         (local <= vertices.max(0) + margin).all(1)]
            if len(near) and ((near @ equations[:, :3].T + equations[:, 3]) <= margin).all(1).any():
                return {'reason': 'observed_box_mesh', 'mesh': str(name), 'body': label}
        required_margin = SERVO_CONFIG['mesh_table_margin_m'] + table_buffer_m
        if minimum_z < TABLE_Z_M + required_margin:
            return {'reason': 'table_mesh', 'minimum_mesh_z_m': minimum_z,
                    'required_vertical_correction_m': TABLE_Z_M + required_margin - minimum_z,
                    'table_top_z_m': TABLE_Z_M, 'table_margin_m': SERVO_CONFIG['mesh_table_margin_m'],
                    'planning_table_buffer_m': table_buffer_m, 'required_table_margin_m': required_margin,
                    'surface_clearance_m': minimum_z - TABLE_Z_M, **minimum_evidence}
        return None


class GraspExecutor:
    def __init__(self, env, scorer, directory, closure='first_contact', controller='baseline', destination=None, target_centre=None, contact_frame='open', collision_check='off', observation_dir=None, arm_tracking_feedback=False, candidate_contact_ranking=False, candidate_contact_policy='soft8', closure_latch_policy='legacy', close_tracking_integral=False, close_integral_transfer='reset', close_frame_policy='fixed-body'):
        self.env, self.scorer, self.directory = env, scorer, Path(directory)
        self.robot = env.scene['robot']
        self.device = env.device
        self.body = int(self.robot.find_bodies(['gripper'])[0][0])
        self.jaw = int(self.robot.find_bodies(['moving_jaw_so101_v1'])[0][0])
        self.grip = int(self.robot.find_joints(['gripper'])[0][0])
        self.arm = self.robot.find_joints(['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'], preserve_order=True)[0]
        self.baseline = scorer()['orange_z']
        self.trace = []
        self.max_rise = 0.0
        self.hold_run = 0
        self.lift_success = False
        self.executed = False
        self.ik_attempts = []
        self.closure = closure
        self.controller = controller
        if controller not in ('baseline', 'repaired', 'servo'):
            raise ValueError('controller must be baseline, repaired or servo')
        if arm_tracking_feedback and controller != 'servo':
            raise ValueError('arm tracking feedback requires the servo controller')
        self.arm_tracking_feedback = bool(arm_tracking_feedback)
        if (closure_latch_policy not in ('legacy', 'no_reopen') or
                (closure_latch_policy != 'legacy' and (controller != 'servo' or not arm_tracking_feedback))):
            raise ValueError('closure latch policy requires servo arm tracking feedback and a supported policy')
        self.closure_latch_policy = closure_latch_policy
        if close_tracking_integral and (controller != 'servo' or not arm_tracking_feedback):
            raise ValueError('close tracking integral requires servo arm tracking feedback')
        self.close_tracking_integral = bool(close_tracking_integral)
        if (close_integral_transfer not in ('reset', 'adjacent') or
                (close_integral_transfer == 'adjacent' and not close_tracking_integral)):
            raise ValueError('adjacent integral transfer requires enabled servo close integral')
        self.close_integral_transfer = close_integral_transfer
        if (close_frame_policy not in ('fixed-body', 'planned-line', 'reachable-line') or
                (close_frame_policy in ('planned-line', 'reachable-line') and (controller != 'servo' or not arm_tracking_feedback or
                 contact_frame != 'open' or close_tracking_integral or close_integral_transfer != 'reset' or
                 closure_latch_policy != 'legacy'))):
            raise ValueError('planned-line/reachable-line requires servo tracking, open frame, integral off, reset transfer and legacy latch')
        self.close_frame_policy = close_frame_policy
        self._last_closure_rotation = None
        self._last_closure_tool = None
        self._reachable_closing_line = None
        self._pending_close_integral_transfer = None
        if candidate_contact_ranking and controller != 'servo':
            raise ValueError('candidate contact ranking requires the servo controller')
        self.candidate_contact_ranking = bool(candidate_contact_ranking)
        if candidate_contact_policy not in ('soft8', 'band_first') or (candidate_contact_policy != 'soft8' and not candidate_contact_ranking):
            raise ValueError('candidate contact policy requires enabled ranking and a supported policy')
        self.candidate_contact_policy = candidate_contact_policy
        self.candidate_contact_events = []
        self.arm_tracking_events = []
        if self.arm_tracking_feedback:
            from .arm_tracking import BoundedArmTrackingFeedback
            self._arm_tracking = BoundedArmTrackingFeedback(self.arm, close_integral=self.close_tracking_integral,
                integral_transfer=self.close_integral_transfer)
            self._last_nominal_target = None
            self._last_issued_target = None
        if controller == 'servo' and target_centre is not None:
            raise ValueError('servo uses the network grasp point; target_centre overrides are unsupported')
        self.motion_goal = None
        self.motion_diagnostics = []
        self.close_ik_attempts = []
        self.close_control_events = []
        self.hold_control_events = []
        self._grasp_hold = None
        self._grasp_hold_last_target_rad = None
        self.motion_failure = None
        self.servo_routes = {}
        self.servo_descents = {}
        self.opening_plan_stages = []
        self._configure_tool()
        self.destination = destination
        self.target_centre = target_centre
        self.contact_frame = contact_frame
        if collision_check not in ('off', 'capsule'):
            raise ValueError('collision_check must be off or capsule')
        if collision_check == 'capsule' and controller not in ('repaired', 'servo'):
            raise ValueError('SO101 capsule checking requires the repaired planner')
        self.collision_check = collision_check
        self.observation_dir = Path(observation_dir) if observation_dir is not None else self.directory
        self.collision_evidence = {'mode': collision_check, 'geometry': 'SO101 body-origin capsules',
                                   'observation': str(self.observation_dir),
                                   'preflight_phases': ['approach', 'descend', 'nominal_close', 'nominal_lift']
                                       if collision_check == 'capsule' else [],
                                   'runtime_guard_phases': ['open', 'approach', 'descend', 'close_centered',
                                                            'hold', 'lift', 'lift_hold']
                                       if collision_check == 'capsule' else [],
                                   'unchecked_phases': ['settle', 'scan', 'transport', 'place', 'release', 'retreat']
                                       if collision_check == 'capsule' else []}
        self.collision_ready = True
        self.runtime_collision = None
        self._build_ik()
        self.mesh_evidence = {'enabled': controller == 'servo'}
        if controller == 'servo':
            try:
                self._init_mesh_clearance()
            except (OSError, ValueError) as error:
                self.collision_ready = False
                self.mesh_evidence['error'] = f'{type(error).__name__}: {error}'
        if collision_check == 'capsule':
            try:
                self._init_collision()
            except (OSError, ValueError) as error:
                self.collision_ready = False
                self.collision_evidence['error'] = f'{type(error).__name__}: {error}'

    def _init_mesh_clearance(self):
        from .gripper_geometry import SO101GripperGeometry
        from .geometry import backproject, load_binary_mask
        box_points = np.empty((0, 3))
        if self.destination is not None:
            view = self._box_observation()
            self.mesh_evidence['box_observation'] = str(view)
            mask = load_binary_mask(view / 'box_mask.png')
            depth = np.load(view / 'depth.npy')
            points, valid = backproject(depth, json.loads((view / 'intrinsics.json').read_text()))
            transform = np.asarray(json.loads((view / 'world_from_camera.json').read_text()))
            box_points = points[valid & mask] @ transform[:3, :3].T + transform[:3, 3]
            # Reject background leakage while retaining visible side walls.
            destination = np.asarray(self.destination)
            box_points = box_points[(np.abs(box_points[:, :2] - destination[:2]) < .14).all(1) &
                                    (box_points[:, 2] > destination[2] - .005) &
                                    (box_points[:, 2] < destination[2] + .09)]
            if not len(box_points):
                raise ValueError('No observed box depth points for servo mesh clearance')
            _, indices = np.unique(np.floor(box_points / .005).astype(np.int64), axis=0, return_index=True)
            box_points = box_points[indices]
            visible_count = len(box_points)
            # A single view can miss the opposite wall. Complete only the
            # observed axis-aligned tray boundary; keep its interior hollow.
            low, high = np.quantile(box_points, [.01, .99], axis=0)
            low[:2] -= .003
            high[:2] += .003
            low[2] = destination[2]
            high[2] += .005
            axes = [np.linspace(low[i], high[i], max(2, int(np.ceil((high[i] - low[i]) / .005)) + 1))
                    for i in range(3)]
            walls = []
            for axis in (0, 1):
                other = 1 - axis
                horizontal, height = np.meshgrid(axes[other], axes[2])
                for boundary in (low[axis], high[axis]):
                    points = np.empty((horizontal.size, 3))
                    points[:, axis] = boundary
                    points[:, other] = horizontal.ravel()
                    points[:, 2] = height.ravel()
                    walls.append(points)
            box_points = np.concatenate([box_points, *walls])
            self.mesh_evidence.update(visible_box_points=visible_count,
                inferred_box_wall_bounds_world_m=[low.tolist(), high.tolist()],
                box_completion_assumption='axis-aligned hollow tray; four walls from RGBD 1/99 percentiles plus 3mm xy and 5mm z')
        geometry = getattr(self, 'gripper_geometry', None) or SO101GripperGeometry(
            variants=getattr(self.robot.cfg.spawn, 'variants', None))
        self.mesh_clearance = _GripperMeshClearance(geometry, box_points)
        if getattr(self, 'arm_tracking_feedback', False):
            from .closure_envelope import FixedMidpointClosureEnvelope
            self.closure_envelope = FixedMidpointClosureEnvelope(geometry, self.mesh_clearance.parts)
        self.mesh_state = self.model.state()
        self.mesh_evidence.update(geometry='authored collision submeshes separately convexified',
                                  robot_geometry=geometry.evidence,
                                  observed_box_points=len(box_points), observation=str(self.observation_dir),
                                  table_margin_m=SERVO_CONFIG['mesh_table_margin_m'],
                                  box_margin_m=SERVO_CONFIG['mesh_box_margin_m'],
                                  unchecked='arm links, occluded box surfaces, held object, placement')

    def _configure_tool(self, opening=None):
        """Keep opening, grasp axes and IK reference point in the same frame."""
        self.open_joint_rad = (SERVO_CONFIG['open_joint_rad'] if opening is None else float(opening)) if self.controller == 'servo' else CONFIG['open_joint_rad']
        self.tool_offset_b = np.asarray(POCKET, dtype=float)
        self.tool_evidence = {'control_reference': 'fixed_pocket',
            'control_tool_offset_body_m': self.tool_offset_b.tolist(), 'commanded_open_joint_rad': self.open_joint_rad,
            'fixed_scoring_pocket_offset_body_m': list(POCKET),
            'live_robot_usd_source': str(self.robot.cfg.spawn.usd_path),
            'live_robot_usd_variants': getattr(self.robot.cfg.spawn, 'variants', None)}
        if self.controller == 'servo':
            from .gripper_geometry import SO101GripperGeometry, FIXED_REFERENCE_M, MOVING_REFERENCE_M
            self.gripper_geometry = getattr(self, 'gripper_geometry', None) or SO101GripperGeometry(
                variants=getattr(self.robot.cfg.spawn, 'variants', None))
            # Preserve the exact v4 wide reference; narrow openings require a
            # different midpoint rather than reusing the wide scoring pocket.
            if self.open_joint_rad != CONFIG['open_joint_rad']:
                moving = self.gripper_geometry.moving_to_fixed(MOVING_REFERENCE_M, self.open_joint_rad)
                self.tool_offset_b = (FIXED_REFERENCE_M + moving) / 2
            self.tool_evidence.update(control_reference=('fixed_pocket' if self.open_joint_rad == CONFIG['open_joint_rad']
                                                        else 'pad_reference_midpoint_at_commanded_opening'),
                control_tool_offset_body_m=self.tool_offset_b.tolist(),
                fixed_score_minus_control_tool_body_m=(np.asarray(POCKET) - self.tool_offset_b).tolist(),
                geometry_usd_source=str(self.gripper_geometry.usd_path),
                robot_geometry=self.gripper_geometry.evidence,
                geometry_and_live_usd_basename_match=Path(self.robot.cfg.spawn.usd_path).name == self.gripper_geometry.usd_path.name,
                geometry_semantics='USD robot reference points; not object center or calibrated universal inner gap')

    def _switch_opening_tool(self, opening):
        """Update the existing solver objective, preserving its model and RNG."""
        import warp as wp
        self._configure_tool(opening)
        self.objective.position_objective.link_offset = wp.vec3(*self.tool_offset_b)
        self.objective.offset = wp.transformf(wp.vec3(*self.tool_offset_b), wp.quat_identity())

    def _plan_servo_openings(self, candidates):
        """Exhaust the original primary/opening policy before any tail planning.

        This routine runs before approach/descent/closure. An execution failure
        after selection never returns here for a second physical grasp.
        """
        from .arm_collision import rejection_counts
        primary_stop = min(len(candidates), CONFIG['max_candidates'])
        pools = [('primary', 0, primary_stop)]
        if len(candidates) > primary_stop:
            pools.append(('tail', primary_stop, len(candidates)))
        for candidate_stage, start, stop in pools:
            for opening_index, opening in enumerate(SERVO_CONFIG['opening_options_rad']):
                stage_index = len(self.opening_plan_stages)
                stage = {'stage_index': stage_index, 'opening_option_index': opening_index,
                    'candidate_stage': candidate_stage, 'candidate_index_start': start,
                    'candidate_index_stop': stop, 'open_joint_rad': opening}
                if self.open_joint_rad != opening:
                    self._switch_opening_tool(opening)
                    opened = tensor(self.robot.data.joint_pos).clone()
                    opened[:, self.grip] = opening
                    if not self._move(opened, 'open_adjust', SERVO_CONFIG['opening_adjust_steps']):
                        stage.update(outcome='opening_adjust_terminated', tool=dict(self.tool_evidence))
                        self.opening_plan_stages.append(stage)
                        return None
                midpoint, tool = self._pad_midpoint()
                self.tool_evidence.update(actual_open_joint_rad=float(tensor(self.robot.data.joint_pos)[0, self.grip]),
                    actual_open_reference_midpoint_error_m=float(np.linalg.norm(midpoint - tool)))
                self.servo_routes.clear()
                self.servo_descents.clear()
                attempts_start = len(self.ik_attempts)
                axes = self._grasp_axes_in_body()
                # Keep the primary call and its 20-candidate budget unchanged.
                # Tail indices remain global for cost, caches, trace and execution.
                plan = (self._plan_repaired(candidates, axes) if candidate_stage == 'primary' else
                        self._plan_repaired(candidates, axes, candidate_start=start, candidate_stop=stop))
                attempts = self.ik_attempts[attempts_start:]
                for attempt in attempts:
                    candidate = candidates[attempt['candidate']]
                    attempt.update(opening_stage_index=stage_index, opening_option_index=opening_index,
                        candidate_stage=candidate_stage, candidate_rank=candidate.get('rank', attempt['candidate']),
                        raw_index=candidate.get('raw_index'), open_joint_rad=opening,
                        control_tool_offset_body_m=self.tool_offset_b.tolist(),
                        control_tool_reference=self.tool_evidence['control_reference'])
                stage.update(outcome='selected' if plan is not None else 'all_candidate_plans_rejected',
                    tool=dict(self.tool_evidence), ik_attempts_start=attempts_start,
                    ik_attempts_end=len(self.ik_attempts), rejection_counts=rejection_counts(attempts))
                if plan is not None:
                    stage.update(selected_candidate=plan[1], selected_candidate_id=plan[2].get('candidate_id', plan[1]))
                self.opening_plan_stages.append(stage)
                if plan is not None:
                    return plan
        return None

    def _box_observation(self):
        """Resolve the accepted box frame independently of the target frame."""
        scan = self.directory / 'scan.json'
        if not scan.exists():
            return self.observation_dir
        records = json.loads(scan.read_text()).get('views', [])
        for record in reversed(records):
            quality = record.get('box_quality', {})
            base = np.asarray(quality.get('world_base_m', []), dtype=float)
            if quality.get('accepted') and base.shape == (3,) and np.allclose(
                    base, np.asarray(self.destination), atol=1e-5, rtol=0.):
                return Path(record['view'])
        raise ValueError('No accepted box observation matches the destination in scan.json')

    def _mesh_at_q(self, q, *, table_buffer_m=0.):
        import newton
        import warp as wp
        coordinates = wp.to_torch(self.model.joint_q).cpu().numpy().copy()
        values = q.detach().cpu().numpy().reshape(-1) if hasattr(q, 'detach') else np.asarray(q).reshape(-1)
        coordinates[self.coordinates.cpu().numpy()] = values
        joint_q = wp.array(coordinates, dtype=wp.float32, device=self.device)
        newton.eval_fk(self.model, joint_q, self.model.joint_qd, self.mesh_state)
        pose = wp.to_torch(self.mesh_state.body_q)[self.ik_body].cpu().numpy()
        rotation = Rotation.from_quat(pose[3:]).as_matrix()
        if table_buffer_m:
            return self.mesh_clearance.check(pose[:3] + rotation @ POCKET, rotation, values[self.grip],
                                             table_buffer_m=table_buffer_m)
        return self.mesh_clearance.check(pose[:3] + rotation @ POCKET, rotation, values[self.grip])

    def _mesh_path(self, start, end, phase, *, table_buffer_m=0.):
        from .arm_collision import joint_samples
        a = start.detach().cpu().numpy().reshape(-1)
        b = end.detach().cpu().numpy().reshape(-1)
        for index, q in enumerate(joint_samples(a, b, self.arm, max_delta=.06)):
            collision = self._mesh_at_q(q, table_buffer_m=table_buffer_m) if table_buffer_m else self._mesh_at_q(q)
            if collision:
                return {'phase': phase, 'sample': index, **collision}
        return None

    def _planning_table_buffer(self):
        from .arm_tracking import PLANNING_TABLE_BUFFER_M
        return PLANNING_TABLE_BUFFER_M if getattr(self, 'arm_tracking_feedback', False) else 0.

    def _planning_mesh_at_q(self, q):
        buffer = self._planning_table_buffer()
        return self._mesh_at_q(q, table_buffer_m=buffer) if buffer else self._mesh_at_q(q)

    def _planning_mesh_path(self, start, end, phase):
        buffer = self._planning_table_buffer()
        return self._mesh_path(start, end, phase, table_buffer_m=buffer) if buffer else self._mesh_path(start, end, phase)

    def _planning_closure_envelope(self, point, rotation):
        """Cheap full-jaw geometry screening before committing a candidate."""
        from .arm_collision import TABLE_Z_M
        evidence = self.closure_envelope.minimum(point, rotation, self.open_joint_rad, CONFIG['closed_joint_rad'])
        margin = SERVO_CONFIG['mesh_table_margin_m'] + self._planning_table_buffer()
        correction = max(0., TABLE_Z_M + margin - evidence['minimum_mesh_z_m'])
        evidence.update(required_table_margin_m=margin, planning_table_buffer_m=self._planning_table_buffer(),
            required_vertical_correction_m=correction, valid=correction == 0.)
        return evidence

    def _contact_band_diagnostic(self, point, rotation):
        from .candidate_contact import ContactBandConfig, contact_band_cost, prepare_target_cloud
        config = ContactBandConfig(enabled=getattr(self, 'candidate_contact_ranking', False),
                                   policy=getattr(self, 'candidate_contact_policy', 'soft8'))
        if not config.enabled:
            return contact_band_cost(None, None, None, None, config=config)
        if not hasattr(self, '_candidate_target_cloud'):
            from .geometry import validate_transform
            self._candidate_target_cloud = None
            source = self.directory / 'proposals/point_clouds.npz'
            calibration = self.observation_dir / 'world_from_camera.json'
            try:
                transform = validate_transform(json.loads(calibration.read_text()))
                with np.load(source) as data:
                    points = np.asarray(data['target_xyz_camera_m'], dtype=float)
                self._candidate_target_cloud = prepare_target_cloud(points @ transform[:3, :3].T + transform[:3, 3])
                self._candidate_cloud_source = {'target_cloud': str(source), 'actual_camera_calibration': str(calibration),
                    'semantics': 'actual target-mask RGB-D points; observed surfaces are not object centers'}
            except (OSError, ValueError, KeyError) as error:
                self._candidate_cloud_source = {'error': f'{type(error).__name__}: {error}', 'status': 'missing_target_cloud'}
        diagnostic = contact_band_cost(self._candidate_target_cloud, point, rotation, self.tool_offset_b, config=config)
        diagnostic['source'] = self._candidate_cloud_source
        return diagnostic

    def _build_ik(self):
        import warp as wp
        import newton
        import isaaclab.sim as sim_utils
        from isaaclab import cloner
        from isaaclab_newton.cloner import copy_newton_clone_source
        from isaaclab_newton.ik import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg, NewtonIKSolver, NewtonIKSolverCfg
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        resolved = cloner.query.path_to_source(plan, self.robot.cfg.prim_path)
        source = copy_newton_clone_source(resolved[0])
        prototype = newton.ModelBuilder(up_axis=source.up_axis)
        origin = -tensor(self.env.scene.env_origins)[0]
        prototype.add_builder(source, xform=wp.transform(wp.vec3(*origin.tolist()), wp.quat_identity()))
        self.model = prototype.finalize(device=self.device)
        bodies = [str(x).rsplit('/', 1)[-1] for x in self.model.body_label]
        self.ik_body = bodies.index('gripper')
        names = [str(x).rsplit('/', 1)[-1] for x in self.model.joint_label]
        self.jaw_joint_index = names.index('gripper')
        starts = wp.to_torch(self.model.joint_q_start).long()
        self.coordinates = torch.tensor([int(starts[names.index(x)]) for x in self.robot.joint_names], device=self.device)
        self.seed = wp.to_torch(self.model.joint_q)[None].clone()
        self.solver = NewtonIKSolver(NewtonIKSolverCfg(optimizer='lm', jacobian_mode='analytic', sampler='gauss',
             n_seeds=CONFIG['ik_seeds'], noise_std=.4, iterations=CONFIG['ik_iterations'], lambda_initial=.1,
             rng_seed=self.env.cfg.seed), model=self.model, num_envs=1, device=self.device,
             objectives=[NewtonIKPoseObjectiveCfg(body_name='gripper', name='grasp', body_offset_pos=tuple(self.tool_offset_b),
                         position_weight=100., rotation_weight=3.), NewtonIKJointLimitObjectiveCfg(weight=1.)],
             link_resolver=lambda _: self.ik_body)
        self.objective = self.solver.objectives_by_name['grasp']
        if self.controller == 'servo':
            self.ik_limit_objective = self.solver.objectives[-1].objective
            limits = tensor(self.robot.data.soft_joint_pos_limits)[0].cpu().numpy()
            lower, upper, dofs = _soft_limit_dof_arrays(
                wp.to_torch(self.model.joint_q_start).cpu().numpy(),
                wp.to_torch(self.model.joint_qd_start).cpu().numpy(),
                wp.to_torch(self.model.joint_dof_dim).cpu().numpy(),
                self.coordinates.cpu().numpy(), limits,
                wp.to_torch(self.ik_limit_objective.joint_limit_lower).cpu().numpy(),
                wp.to_torch(self.ik_limit_objective.joint_limit_upper).cpu().numpy())
            self.ik_soft_lower = wp.array(lower, dtype=wp.float32, device=self.device)
            self.ik_soft_upper = wp.array(upper, dtype=wp.float32, device=self.device)
            self.ik_fallback_parameters = {'policy': 'one retry only after all primary seeds fail original gates',
                'robot_joint_names': list(self.robot.joint_names), 'robot_q_coordinates': self.coordinates.cpu().tolist(),
                'robot_dof_indices': dofs, 'robot_soft_limits_rad': limits.tolist(),
                'units': 'radians; six SO101 revolute joints',
                'weight': SERVO_CONFIG['soft_limit_fallback_weight'],
                'seeds': CONFIG['ik_seeds'], 'iterations': CONFIG['ik_iterations'],
                'bounds_source': 'robot.data.soft_joint_pos_limits; all other model DOFs preserve original bounds'}

    def _init_collision(self):
        """Use only saved RGB-D, named robot bodies and robot FK for planning."""
        from .arm_collision import LINKS, load_capture_joint_positions, load_observed_cloud, remove_robot_returns

        names = [str(label).rsplit('/', 1)[-1] for label in self.model.body_label]
        aliases = {'jaw': ('jaw', 'moving_jaw_so101_v1')}
        required = {item for a, b, _ in LINKS for item in (a, b)}
        self.collision_body_ids = {}
        for name in required:
            matches = [names.index(alias) for alias in aliases.get(name, (name,)) if alias in names]
            if len(matches) != 1:
                raise ValueError(f'Expected one SO101 collision body {name}, found {matches}')
            self.collision_body_ids[name] = matches[0]
        self.collision_state = self.model.state()
        current = tensor(self.robot.data.joint_pos)[0].cpu().numpy()
        current_poses = self._fk_body_poses(current)
        real_positions = tensor(self.robot.data.body_pos_w)[0].cpu().numpy()
        real_quats = tensor(self.robot.data.body_quat_w)[0].cpu().numpy()
        pose_errors, angles = [], []
        for name, predicted in current_poses.items():
            if name == 'jaw':
                actual_index = self.jaw
            else:
                actual_ids = self.robot.find_bodies([name])[0]
                if len(actual_ids) != 1:
                    raise ValueError(f'Expected one live SO101 body {name}')
                actual_index = int(actual_ids[0])
            pose_errors.append(float(np.linalg.norm(predicted[:3] - real_positions[actual_index])))
            dot = abs(float(np.dot(predicted[3:7], real_quats[actual_index])))
            angles.append(float(np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1)))))
        self.collision_evidence.update(fk_world_position_error_max_m=max(pose_errors),
                                       fk_world_orientation_error_max_deg=max(angles))
        if max(pose_errors) > .025 or max(angles) > 10:
            raise ValueError('Newton FK and live SO101 body world poses disagree')
        capture_q, source = load_capture_joint_positions(self.observation_dir, self.robot.joint_names)
        limits = tensor(self.robot.data.soft_joint_pos_limits)[0].cpu().numpy()
        soft_excess = np.maximum(np.maximum(limits[:, 0] - capture_q, capture_q - limits[:, 1]), 0.)
        self.collision_evidence['capture_soft_limit_excess_rad'] = {
            name: float(excess) for name, excess in zip(self.robot.joint_names, soft_excess, strict=True)}
        # This is a measured capture pose, not a planning target. Keep its
        # actual values for FK even when a reset lies beyond planning soft limits.
        capture_capsules = self._fk_capsules(capture_q)
        obstacles, target = load_observed_cloud(self.observation_dir)
        self.collision_obstacles = remove_robot_returns(obstacles, capture_capsules)
        self.collision_target = target
        self.collision_evidence.update(capture_robot_state_source=source,
                                       capture_to_execution_joint_delta_max_rad=float(np.max(np.abs(capture_q - current))),
                                       observed_obstacle_points=len(obstacles),
                                       after_robot_exclusion=len(self.collision_obstacles),
                                       target_points=len(target), sample_joint_rad=.08,
                                       point_stride_px=3, table_z_m=.041,
                                       links=[{'from': a, 'to': b, 'radius_m': radius} for a, b, radius in LINKS],
                                       pad_radius_m=.006)

    def _fk_body_poses(self, q):
        """Evaluate Newton FK at one six-joint configuration, in world coordinates."""
        import newton
        import warp as wp

        coordinates = wp.to_torch(self.model.joint_q).cpu().numpy().copy()
        coordinates[self.coordinates.cpu().numpy()] = np.asarray(q)
        joint_q = wp.array(coordinates, dtype=wp.float32, device=self.device)
        newton.eval_fk(self.model, joint_q, self.model.joint_qd, self.collision_state)
        poses = wp.to_torch(self.collision_state.body_q).cpu().numpy()
        return {name: poses[index] for name, index in self.collision_body_ids.items()}

    def _fk_capsules(self, q):
        from .arm_collision import add_finger_capsules, capsules_from_bodies

        named = self._fk_body_poses(q)
        return add_finger_capsules(capsules_from_bodies({name: pose[:3] for name, pose in named.items()}), named)

    def _check_path(self, start, end, phase):
        if self.collision_check == 'off':
            return None
        from .arm_collision import check_capsules, joint_samples

        start_q = start[0].detach().cpu().numpy() if hasattr(start, 'detach') else np.asarray(start).reshape(-1)
        end_q = end[0].detach().cpu().numpy() if hasattr(end, 'detach') else np.asarray(end).reshape(-1)
        sampled_joints = np.r_[np.asarray(self.arm, dtype=int), self.grip]
        samples = joint_samples(start_q, end_q, sampled_joints)
        for step, q in enumerate(samples):
            collision = check_capsules(self._fk_capsules(q), self.collision_obstacles, self.collision_target)
            if collision:
                return {'phase': phase, 'sample': step, 'sample_count': len(samples), **collision}
        return None

    def _solve(self, position, rotation, seed=None, prefer_near=False):
        import warp as wp
        position = torch.as_tensor(position, dtype=torch.float32, device=self.device).reshape(1, 3)
        quat = torch.tensor(Rotation.from_matrix(rotation).as_quat(), dtype=torch.float32, device=self.device).reshape(1, 4)
        self.objective.position_objective.set_target_positions(wp.from_torch(position.contiguous(), dtype=wp.vec3))
        self.objective.rotation_objective.set_target_rotations(wp.from_torch(quat.contiguous(), dtype=wp.vec4))
        initial = self.seed.clone()
        initial[:, self.coordinates] = tensor(self.robot.data.joint_pos) if seed is None else seed
        self.solver.solve(wp.from_torch(initial.contiguous(), dtype=wp.float32))
        evaluate = lambda: self._ik_result(position, quat, seed, prefer_near)
        result, diagnostic = evaluate()
        return self._maybe_soft_limit_fallback(
            wp.from_torch(initial.contiguous(), dtype=wp.float32), result, diagnostic, evaluate)

    def _maybe_soft_limit_fallback(self, initial, result, diagnostic, evaluate):
        """One bounded-budget retry; never leave the primary objective changed."""
        diagnostic.update(ik_source='primary', fallback_attempted=False)
        if (result is not None or self.controller != 'servo' or
                not SERVO_CONFIG['soft_limit_fallback_enabled']):
            return result, diagnostic
        primary = dict(diagnostic)
        objective = self.ik_limit_objective
        original = objective.joint_limit_lower, objective.joint_limit_upper, objective.weight
        try:
            objective.joint_limit_lower, objective.joint_limit_upper = self.ik_soft_lower, self.ik_soft_upper
            objective.weight = SERVO_CONFIG['soft_limit_fallback_weight']
            self.solver.solve(initial)
            result, diagnostic = evaluate()
        except (RuntimeError, ValueError) as error:
            result, diagnostic = None, dict(primary)
            diagnostic['fallback_error'] = f'{type(error).__name__}: {error}'
        finally:
            objective.joint_limit_lower, objective.joint_limit_upper, objective.weight = original
        diagnostic.update(ik_source='soft_limit_fallback', fallback_attempted=True,
                          primary_ik=primary, fallback_parameters=dict(self.ik_fallback_parameters))
        return result, diagnostic

    def _ik_result(self, position, quat, seed, prefer_near):
        """Apply the identical FK, joint, pose and continuity gates to either solve."""
        import warp as wp
        from isaaclab.utils.math import quat_apply
        count = CONFIG['ik_seeds']
        q = wp.to_torch(self.solver.joint_q).reshape(count, -1)[:, self.coordinates]
        poses = wp.to_torch(self.solver.solver.body_q).reshape(count, -1, 7)[:, self.ik_body]
        actual = poses[:, :3] + quat_apply(poses[:, 3:], poses.new_tensor(self.tool_offset_b).expand(count, 3))
        error = (actual - position).norm(dim=-1)
        dot = (poses[:, 3:] * quat).sum(-1).abs().clamp(0, 1)
        angle = torch.rad2deg(2 * torch.acos(dot))
        limits = tensor(self.robot.data.soft_joint_pos_limits)[0]
        gates = {'finite_joints': torch.isfinite(q).all(-1),
                 'joint_limits': (q >= limits[:, 0] - .001).all(-1) & (q <= limits[:, 1] + .001).all(-1),
                 'position': error < CONFIG['ik_position_tolerance_m'],
                 'orientation': angle < CONFIG['ik_orientation_tolerance_deg']}
        joint_distance = joint_delta = None
        if prefer_near:
            reference = tensor(self.robot.data.joint_pos) if seed is None else seed
            differences = q[:, self.arm] - reference[:, self.arm]
            joint_distance = differences.norm(dim=-1)
            joint_delta = differences.abs().max(dim=-1).values
            gates['joint_continuity'] = joint_delta <= SERVO_CONFIG['cartesian_max_joint_delta_rad']
        valid = gates['finite_joints'] & gates['joint_limits'] & gates['position'] & gates['orientation']
        if prefer_near:
            valid &= gates['joint_continuity']
        costs = error + .0002 * angle
        if prefer_near:
            costs += SERVO_CONFIG['cartesian_joint_distance_weight_m'] * joint_distance
        idx = int(torch.where(valid, costs, torch.inf).argmin())
        achieved_position = actual[idx].cpu().numpy()
        achieved_quat = poses[idx, 3:].cpu().numpy()
        diagnostic = {'position_error_m': float(error[idx]) if torch.isfinite(error[idx]) else None,
                      'orientation_error_deg': float(angle[idx]) if torch.isfinite(angle[idx]) else None,
                      'achieved_pocket_world_m': achieved_position.tolist() if np.isfinite(achieved_position).all() else None,
                      'requested_pocket_world_m': position[0].cpu().numpy().tolist(),
                      'tool_reference': self.tool_evidence['control_reference'],
                      'valid': bool(valid.any())}
        diagnostic['selected_seed_gates'] = {name: bool(mask[idx]) for name, mask in gates.items()}
        diagnostic['passing_seed_counts'] = {name: int(mask.sum()) for name, mask in gates.items()}
        if prefer_near:
            diagnostic.update(joint_continuity_required=True,
                              selected_max_arm_delta_rad=float(joint_delta[idx]) if torch.isfinite(joint_delta[idx]) else None,
                              selected_arm_distance_rad=float(joint_distance[idx]) if torch.isfinite(joint_distance[idx]) else None)
        diagnostic['joint_soft_limit_excess_max_rad'] = float(torch.maximum(
            torch.maximum(limits[:, 0] - q[idx], q[idx] - limits[:, 1]), q.new_zeros(q.shape[1])).max()) if torch.isfinite(q[idx]).all() else None
        diagnostic['achieved_rotation'] = (Rotation.from_quat(achieved_quat).as_matrix().tolist()
                                           if np.isfinite(achieved_quat).all() and np.linalg.norm(achieved_quat) > 0 else None)
        result = q[idx:idx + 1].clone()
        return result if valid.any() else None, diagnostic

    def _start_grasp_hold(self, latched_target_rad, phase='close_centered'):
        """Arm upkeep only from a completed robot contact-stall closure."""
        if (self.controller != 'servo' or not SERVO_CONFIG['bounded_hold_enabled'] or
                latched_target_rad is None or not self.env.tracking or not self.collision_ready):
            return False
        self._grasp_hold = BoundedContactHold(latched_target_rad)
        self._grasp_hold_last_target_rad = self._grasp_hold.target
        if not hasattr(self, 'hold_control_events'):
            self.hold_control_events = []
        self.hold_control_events.append({'event': 'bounded_hold_started', 'phase': phase,
            'observation_step': int(self.env.episode_length_buf[0]),
            'initial_joint_target_rad': self._grasp_hold.target,
            'active_phases': ['hold', 'lift', 'lift_hold'],
            'config': asdict(self._grasp_hold.config)})
        return True

    def _stop_grasp_hold(self, phase, reason, target=None):
        """Stop upkeep, optionally carrying its final jaw command into placement."""
        hold = getattr(self, '_grasp_hold', None)
        if hold is not None:
            self.hold_control_events.append({'event': 'bounded_hold_stopped', 'phase': phase,
                'reason': reason, 'observation_step': int(self.env.episode_length_buf[0]),
                'final_joint_target_rad': hold.target,
                'extra_closure_rad': hold.initial_target - hold.target})
            self._grasp_hold = None
        last_target = getattr(self, '_grasp_hold_last_target_rad', None)
        if target is not None and last_target is not None:
            target = target.clone()
            target[:, self.grip] = last_target
        return target

    def _apply_grasp_hold(self, target, phase):
        hold = getattr(self, '_grasp_hold', None)
        if hold is None:
            return target
        if self.controller != 'servo' or not SERVO_CONFIG['bounded_hold_enabled']:
            self._stop_grasp_hold(phase, 'hold_disabled')
            return target
        if phase not in ('hold', 'lift', 'lift_hold'):
            self._stop_grasp_hold(phase, 'phase_outside_scope')
            return target
        # Only robot aggregate contact and joint measurements are read here.
        # Object pose, held score and diagnostic contact partners are excluded.
        force = float(self.scorer()['claw_force'])
        joint = float(tensor(self.robot.data.joint_pos)[0, self.grip])
        velocity = float(tensor(self.robot.data.joint_vel)[0, self.grip])
        decision = hold.update(force, joint, velocity, float(self.env.step_dt))
        commanded = target.clone()
        commanded[:, self.grip] = decision.target_rad
        self._grasp_hold_last_target_rad = decision.target_rad
        finite = lambda value: float(value) if np.isfinite(value) else None
        self.hold_control_events.append({'event': 'bounded_hold_observation', 'phase': phase,
            'observation_step': int(self.env.episode_length_buf[0]),
            'reason': decision.reason, 'claw_force_n': finite(force),
            'joint_position_rad': finite(joint), 'joint_velocity_rad_s': finite(velocity),
            'caller_joint_target_rad': finite(float(target[0, self.grip])),
            'joint_target_rad': decision.target_rad, 'extra_closure_rad': decision.extra_closure_rad})
        return commanded

    def _tracking_command(self, nominal, phase):
        """Approve bounded feedback through the existing geometric safety gates."""
        transfer = getattr(self, '_pending_close_integral_transfer', None)
        self._pending_close_integral_transfer = None
        if not getattr(self, 'arm_tracking_feedback', False):
            return nominal
        q = tensor(self.robot.data.joint_pos)
        active_phase = phase in ('approach_high', 'approach', 'descend', 'close_align', 'close_centered', 'hold', 'lift', 'lift_hold')
        # The unmodified nominal command is not a valid fallback from an
        # already unsafe measured origin. Stop before issuing another action.
        origin_collision = self._mesh_at_q(q) if active_phase else None
        nominal_collision = None
        if active_phase and origin_collision is None:
            nominal_collision = self._check_path(q, nominal, phase + '_arm_tracking_nominal')
            if nominal_collision is None:
                nominal_collision = self._mesh_path(q, nominal, phase + '_arm_tracking_nominal')
        if origin_collision is not None or nominal_collision is not None:
            collision = ({'phase': phase + '_arm_tracking_origin', 'sample': 0, **origin_collision}
                         if origin_collision is not None else nominal_collision)
            stop_reason = 'origin_collision_stop' if origin_collision is not None else 'nominal_path_collision_stop'
            self._arm_tracking.reset()
            self.runtime_collision = collision
            self.collision_ready = False
            self.motion_failure = phase + ('_arm_tracking_origin_collision' if origin_collision else '_arm_tracking_nominal_collision')
            self.mesh_evidence['runtime_stop'] = collision
            self.arm_tracking_events.append({'step': int(self.env.episode_length_buf[0]), 'phase': phase,
                'tracking_phase': True, 'accepted': False, 'reason': stop_reason,
                'collision': collision, 'nominal_joint_target_rad': nominal[0].tolist(),
                'proposed_joint_target_rad': None, 'applied_joint_target_rad': None,
                'nominal_minus_actual_arm_rad': (nominal[0, self.arm] - q[0, self.arm]).tolist(),
                'proposed_arm_bias_rad': [0.] * 5, 'applied_arm_bias_rad': [0.] * 5,
                'command_issued': False, 'nominal_cache_updated': False})
            self.trace.append({'phase': phase, 'event': 'runtime_collision_rejected',
                'step': int(self.env.episode_length_buf[0]), 'command_issued': False,
                'joint_position': q[0].tolist(), 'nominal_joint_target': nominal[0].tolist(),
                'collision': collision})
            return None
        context = {}
        if getattr(self, 'close_tracking_integral', False):
            context = {'phase': phase, 'dt_s': float(self.env.step_dt),
                       'arm_velocity': tensor(self.robot.data.joint_vel)[0, self.arm].cpu().numpy(),
                       'integral_transfer': transfer}
        decision = self._arm_tracking.propose(nominal[0].detach().cpu().numpy(),
            q[0].detach().cpu().numpy(), tensor(self.robot.data.soft_joint_pos_limits)[0].cpu().numpy(),
            tracking=active_phase, **context)
        command = nominal.new_tensor(decision.command)[None]
        collision = None
        accepted = decision.reason == 'feedback_proposed'
        if accepted and decision.active:
            collision = self._check_path(q, command, phase + '_arm_tracking')
            if collision is None:
                collision = self._mesh_path(q, command, phase + '_arm_tracking')
            accepted = collision is None
        self._arm_tracking.commit(decision, accepted)
        # Only the close phase can add I. Elsewhere keep the original safe
        # nominal fallback, already approved by the origin and nominal gates.
        integral_stop = (getattr(self, 'close_tracking_integral', False) and
                         phase == 'close_centered' and not accepted)
        self.arm_tracking_events.append({'step': int(self.env.episode_length_buf[0]), 'phase': phase,
            'tracking_phase': active_phase, 'accepted': accepted,
            'reason': ('collision_rejected' if collision else decision.reason), 'collision': collision,
            'nominal_joint_target_rad': decision.nominal.tolist(),
            'proposed_joint_target_rad': decision.command.tolist(),
            'applied_joint_target_rad': None if integral_stop else (decision.command if accepted else decision.nominal).tolist(),
            'nominal_minus_actual_arm_rad': decision.error.tolist(),
            'proposed_arm_bias_rad': decision.bias.tolist(),
            'applied_arm_bias_rad': (decision.bias if accepted else np.zeros(5)).tolist(),
            'nominal_cache_updated': False,
            'close_tracking_integral': {'enabled': getattr(self, 'close_tracking_integral', False),
                'reason': decision.integral_reason, 'stationary_nominal_steps': decision.stationary_steps,
                'proposed_integral_rad': decision.integral.tolist() if decision.integral is not None else [0.] * 5,
                'committed_integral_rad': self._arm_tracking.integral.tolist(),
                'transfer_used': decision.integral_transfer_used,
                'transfer_context': ({'session': transfer.session, 'from_index': transfer.from_index,
                                      'to_index': transfer.to_index} if transfer is not None else None)},
            **({'command_issued': False} if integral_stop else {})})
        if integral_stop:
            self.motion_failure = phase + '_close_tracking_integral_rejected'
            self.collision_ready = False
            self.runtime_collision = collision or {'reason': decision.reason}
            self.mesh_evidence['runtime_stop'] = self.runtime_collision
            return None
        return command if accepted else nominal

    def _step(self, target, phase):
        if not self.env.tracking or not self.collision_ready:
            self._stop_grasp_hold(phase, 'execution_unavailable')
            return False
        target = self._apply_grasp_hold(target, phase)
        q = tensor(self.robot.data.joint_pos)
        if self.collision_check == 'capsule' and phase in self.collision_evidence['runtime_guard_phases']:
            collision = self._check_path(q, target, phase + '_command')
            if collision:
                self.runtime_collision = collision
                self.collision_evidence['runtime_stop'] = collision
                self.collision_ready = False
                self.trace.append({'phase': phase, 'event': 'runtime_collision_rejected',
                                   'step': int(self.env.episode_length_buf[0]),
                                   'joint_position': q[0].tolist(), 'joint_target': target[0].tolist(),
                                   'collision': collision})
                self._stop_grasp_hold(phase, 'runtime_collision_rejected')
                return False
        nominal_target = target
        target = self._tracking_command(nominal_target, phase)
        if target is None:
            self._stop_grasp_hold(phase, 'arm_tracking_origin_collision')
            return False
        action = torch.zeros((1, 6), device=self.device)
        # Standalone instance uses absolute JointPositionActionCfg(scale=1).
        action[:, :5] = target[:, self.arm]
        action[:, 5] = target[:, self.grip]
        try:
            self.env.step(action)
        except BaseException:
            self._stop_grasp_hold(phase, 'step_exception')
            raise
        if getattr(self, 'arm_tracking_feedback', False):
            # Nominal and issued targets are distinct: inheriting the biased
            # issued command would apply the feedback offset a second time.
            # Only a successfully issued step may advance either cache.
            self._last_nominal_target = nominal_target.detach().clone()
            self._last_issued_target = target.detach().clone()
            if self.arm_tracking_events:
                self.arm_tracking_events[-1]['nominal_cache_updated'] = True
        self.executed = True
        if self.env.tracking:
            sample = self.scorer()
            rise = sample['orange_z'] - self.baseline
            self.max_rise = max(self.max_rise, rise)
            self.hold_run = self.hold_run + 1 if sample['held'] and rise > .08 else 0
            self.lift_success |= self.hold_run >= 5
            extra = {}
            if self.controller in ('repaired', 'servo'):
                midpoint, pocket = self._pad_midpoint()
                body_rotation = self._body_rotation()
                score_pocket = pocket + body_rotation @ (np.asarray(POCKET) - self.tool_offset_b)
                extra = {'pad_midpoint_world_m': midpoint.tolist(), 'pocket_world_m': pocket.tolist(),
                         'control_tool_world_m': pocket.tolist(), 'fixed_scoring_pocket_world_m': score_pocket.tolist(),
                         'control_tool_reference': self.tool_evidence['control_reference'],
                         'body_rotation_world': self._body_rotation().tolist(),
                         'joint_velocity_rad_s': tensor(self.robot.data.joint_vel)[0].tolist()}
                reachable_line = getattr(self, '_reachable_closing_line', None)
                if reachable_line is not None:
                    extra['reachable_line_observation'] = reachable_line.observed_residual(
                        body_rotation, float(tensor(self.robot.data.joint_pos)[0, self.grip]))
                if self.motion_goal is not None:
                    goal_position, goal_rotation = self.motion_goal
                    extra.update(requested_pocket_world_m=np.asarray(goal_position).tolist(),
                                 requested_rotation_world=np.asarray(goal_rotation).tolist(),
                                 pocket_tracking_error_m=float(np.linalg.norm(pocket - goal_position)),
                                 orientation_tracking_error_deg=float(np.rad2deg(Rotation.from_matrix(
                                     np.asarray(goal_rotation).T @ self._body_rotation()).magnitude())))
            if getattr(self, 'arm_tracking_feedback', False):
                extra.update(nominal_joint_target=nominal_target[0].tolist(),
                             arm_tracking_bias_rad=self._arm_tracking.bias.tolist(), nominal_cache_updated=True)
            self.trace.append({'phase': phase, 'step': int(self.env.episode_length_buf[0]), 'rise_m': rise,
                               'held_lift_run': self.hold_run,
                               'joint_position': tensor(self.robot.data.joint_pos)[0].tolist(),
                               'joint_target': target[0].tolist(),
                               'max_joint_error_rad': float((tensor(self.robot.data.joint_pos) - target).abs().max()), **sample, **extra})
        if not self.env.tracking:
            self._stop_grasp_hold(phase, 'environment_terminated')
        return self.env.tracking

    def _move(self, target, phase, steps=60, start=None):
        caller_start = start is not None
        cached = getattr(self, '_last_nominal_target', None) if getattr(self, 'arm_tracking_feedback', False) else None
        if start is None and cached is not None:
            start = cached.clone()
            start_source = 'last_issued_nominal'
        else:
            start = tensor(self.robot.data.joint_pos).clone() if start is None else start.clone()
            start_source = 'explicit_caller' if caller_start else 'measured_bootstrap'
        if getattr(self, 'arm_tracking_feedback', False):
            measured = tensor(self.robot.data.joint_pos).clone()
            issued = getattr(self, '_last_issued_target', None)
            self.motion_diagnostics.append({'phase': 'motion_start', 'event': 'nominal_move_start',
                'motion_phase': phase, 'start_source': start_source, 'move_steps': steps,
                'interpolation_start_rad': start[0].tolist(), 'measured_joint_position_rad': measured[0].tolist(),
                'last_issued_nominal_rad': cached[0].tolist() if cached is not None else None,
                'last_issued_command_rad': issued[0].tolist() if issued is not None else None,
                'start_minus_measured_rad': (start - measured)[0].tolist()})
        if (self.controller == 'servo' and SERVO_CONFIG['lift_fixed_jaw_target_enabled'] and
                phase in ('lift', 'lift_hold')):
            # The goal already carries the issued close/hold jaw command.
            # Reusing the actual blocked angle as an interpolation start would
            # briefly reopen the command. Preserve that same goal from step 1.
            actual = float(tensor(self.robot.data.joint_pos)[0, self.grip])
            goal = float(target[0, self.grip])
            if not hasattr(self, 'motion_diagnostics'):
                self.motion_diagnostics = []
            self.motion_diagnostics.append({'event': 'lift_fixed_jaw_target', 'phase': phase,
                'scope': ['lift', 'lift_hold'], 'move_steps': steps,
                'jaw_target_source': 'unchanged caller lift goal inherited from close/hold',
                'start_source': 'caller_copy' if caller_start else 'measured_robot_copy',
                'jaw_start_before_override_rad': float(start[0, self.grip]),
                'jaw_actual_at_move_start_rad': actual, 'fixed_jaw_target_rad': goal,
                'first_jaw_command_rad': goal, 'first_actual_minus_command_rad': actual - goal})
            start[:, self.grip] = target[:, self.grip]
        # Placement inherits the last jaw command without interpolating back
        # toward a lagging actual angle. Arm interpolation is unchanged, and
        # release still interpolates the opening normally with upkeep stopped.
        if (phase in ('transport', 'place') and
                getattr(self, '_grasp_hold_last_target_rad', None) is not None):
            start[:, self.grip] = target[:, self.grip]
        ramp_steps = max(1, steps - (2 if self.controller == 'servo' else 10))
        for step in range(1, steps + 1):
            if not self._step(torch.lerp(start, target, min(1., step / ramp_steps)), phase):
                return False
        return True

    def _arrive(self, target, point, rotation, phase):
        """Wait for measured robot arrival; target scores never enter this gate."""
        self.motion_goal = (np.asarray(point), np.asarray(rotation))
        consecutive = 0
        log = {'phase': phase, 'requested_pocket_world_m': np.asarray(point).tolist(),
               'requested_rotation_world': np.asarray(rotation).tolist()}
        for step in range(SERVO_CONFIG['arrival_max_steps']):
            if not self.env.tracking or not self.collision_ready:
                log['terminated'] = True
                break
            _, pocket = self._pad_midpoint()
            position_error = float(np.linalg.norm(pocket - point))
            angle = float(np.rad2deg(Rotation.from_matrix(np.asarray(rotation).T @ self._body_rotation()).magnitude()))
            arm_error = float((tensor(self.robot.data.joint_pos)[:, self.arm] - target[:, self.arm]).abs().max())
            measure_speed = (phase == 'approach' and self.controller == 'servo') or (
                getattr(self, 'arm_tracking_feedback', False) and phase in ('descend', 'close_align'))
            arm_speed = float(tensor(self.robot.data.joint_vel)[:, self.arm].abs().max()) if measure_speed else None
            arrived = (position_error <= SERVO_CONFIG['position_tolerance_m'] and
                       angle <= SERVO_CONFIG['orientation_tolerance_deg'] and
                       arm_error <= SERVO_CONFIG['arm_joint_tolerance_rad'])
            if arm_speed is not None:
                arrived &= arm_speed <= SERVO_CONFIG['approach_arrival_arm_speed_rad_s']
            consecutive = consecutive + 1 if arrived else 0
            log.update(position_error_m=position_error, orientation_error_deg=angle,
                       max_arm_joint_error_rad=arm_error, settle_steps=step,
                       consecutive_arrival_steps=consecutive)
            if arm_speed is not None:
                log['max_arm_joint_speed_rad_s'] = arm_speed
            if consecutive >= SERVO_CONFIG['arrival_consecutive_steps']:
                log['arrived'] = True
                self.motion_diagnostics.append(log)
                return True
            if not self._step(target, phase):
                break
        log['arrived'] = False
        self.motion_diagnostics.append(log)
        self.motion_failure = phase + '_arrival_timeout'
        return False

    def _plan_cartesian_descent(self, start, seed, point, rotation, phase='descend_preflight'):
        """Solve and check a complete descent without issuing robot commands."""
        start = np.asarray(start)
        count = min(SERVO_CONFIG['cartesian_max_waypoints'], max(1, int(np.ceil(
            np.linalg.norm(np.asarray(point) - start) / SERVO_CONFIG['cartesian_waypoint_spacing_m']))))
        seed = seed.clone()
        log = {'phase': phase, 'start_pocket_world_m': start.tolist(), 'waypoints': []}
        # Solve the complete finite path before issuing its first command.
        path = []
        for index in range(1, count + 1):
            desired = start + (np.asarray(point) - start) * index / count
            q, diagnostic = self._solve(desired, rotation, seed, prefer_near=True)
            log['waypoints'].append({'index': index, **diagnostic})
            if q is None:
                log['failure_reason'] = 'descend_cartesian_ik'
                return None, log
            q[:, self.grip] = self.open_joint_rad
            # Also enforce continuity here so a custom solver adapter cannot
            # silently bypass the path's finite command bound.
            delta = float((q[:, self.arm] - seed[:, self.arm]).abs().max())
            log['waypoints'][-1].update(max_arm_delta_rad=delta, joint_target=q[0].tolist())
            if delta > SERVO_CONFIG['cartesian_max_joint_delta_rad']:
                log['failure_reason'] = 'descend_cartesian_joint_jump'
                return None, log
            collision = self._check_path(seed, q, phase)
            if collision is None and self.controller == 'servo':
                collision = self._planning_mesh_path(seed, q, phase)
            if collision:
                log['collision'] = collision
                log['failure_reason'] = 'descend_cartesian_collision'
                return None, log
            path.append((desired, q))
            seed = q
        log['valid'] = True
        return path, log

    def _cartesian_descend(self, point, rotation, grasp_q, prepared=None):
        """Execute the accepted continuous path, checking the measured connector."""
        current = tensor(self.robot.data.joint_pos).clone()
        if prepared is None:
            path, log = self._plan_cartesian_descent(self._pad_midpoint()[1], current, point, rotation)
            self.motion_diagnostics.append(log)
            if path is None:
                self.motion_failure = log['failure_reason']
                return False
        else:
            path = prepared
            collision = self._check_path(current, path[0][1], 'descend_connect')
            if collision is None and self.controller == 'servo':
                collision = self._planning_mesh_path(current, path[0][1], 'descend_connect')
            connector_delta = float((current[:, self.arm] - path[0][1][:, self.arm]).abs().max())
            log = {'phase': 'descend_prepared', 'waypoint_count': len(path),
                   'connector_max_arm_delta_rad': connector_delta, 'collision': collision}
            self.motion_diagnostics.append(log)
            if collision or connector_delta > SERVO_CONFIG['cartesian_max_joint_delta_rad']:
                self.motion_failure = 'descend_connector_collision' if collision else 'descend_connector_joint_jump'
                return False
        cached = getattr(self, '_last_nominal_target', None) if getattr(self, 'arm_tracking_feedback', False) else None
        command_start = cached.clone() if cached is not None else current
        if getattr(self, 'arm_tracking_feedback', False):
            self.motion_diagnostics.append({'phase': 'descend_connector', 'event': 'nominal_connector_start',
                'start_source': 'last_issued_nominal' if cached is not None else 'measured_bootstrap',
                'interpolation_start_rad': command_start[0].tolist(),
                'measured_joint_position_rad': current[0].tolist(),
                'start_minus_measured_rad': (command_start - current)[0].tolist()})
        for desired, q in path:
            self.motion_goal = (desired, rotation)
            if not self._move(q, 'descend', SERVO_CONFIG['cartesian_steps_per_waypoint'], start=command_start):
                return False
            command_start = q
            if getattr(self, 'arm_tracking_feedback', False):
                if not self._arrive(q, desired, rotation, 'descend'):
                    return False
        # The final joint target belongs to the continuous seed path rather than
        # the independent candidate solve; measure arrival against that target.
        return self._arrive(path[-1][1], point, rotation, 'descend')

    def _grasp_axes_in_body(self):
        from isaaclab.utils.math import quat_apply, quat_apply_inverse
        positions = tensor(self.robot.data.body_pos_w)
        quats = tensor(self.robot.data.body_quat_w)
        fixed = positions[:, self.body] + quat_apply(quats[:, self.body], positions.new_tensor([[-.01289, -.00022, -.0986]]))
        moving = positions[:, self.jaw] + quat_apply(quats[:, self.jaw], positions.new_tensor([[-.0068, -.0766, .0189]]))
        if self.controller in ('repaired', 'servo') and self.contact_frame == 'contact':
            import warp as wp
            from .contact_frame import contact_frame
            fixed_b = quat_apply_inverse(quats[:, self.body], fixed - positions[:, self.body])[0].cpu().numpy()
            moving_b = quat_apply_inverse(quats[:, self.body], moving - positions[:, self.body])[0].cpu().numpy()
            pivot = wp.to_torch(self.model.joint_X_p)[self.jaw_joint_index, :3].cpu().numpy()
            qd_start = int(wp.to_torch(self.model.joint_qd_start)[self.jaw_joint_index])
            axis_child = wp.to_torch(self.model.joint_axis)[qd_start:qd_start + 1]
            axis_world = quat_apply(quats[:, self.jaw], axis_child)
            axis_b = quat_apply_inverse(quats[:, self.body], axis_world)[0].cpu().numpy()
            axes, calibration = contact_frame(fixed_b, moving_b, pivot, axis_b,
                 float(tensor(self.robot.data.joint_pos)[0, self.grip]), .056)
            (self.directory / 'contact_frame.json').write_text(json.dumps(calibration, indent=2) + '\n')
            return axes
        width = quat_apply_inverse(quats[:, self.body], moving - fixed)[0].cpu().numpy()
        width /= np.linalg.norm(width)
        approach = self.tool_offset_b.copy()
        approach -= np.dot(approach, width) * width
        approach /= np.linalg.norm(approach)
        return np.stack([approach, width, np.cross(approach, width)], axis=1)

    def _pad_midpoint(self):
        from isaaclab.utils.math import quat_apply
        positions = tensor(self.robot.data.body_pos_w)
        quats = tensor(self.robot.data.body_quat_w)
        fixed = positions[:, self.body] + quat_apply(quats[:, self.body], positions.new_tensor([[-.01289, -.00022, -.0986]]))
        moving = positions[:, self.jaw] + quat_apply(quats[:, self.jaw], positions.new_tensor([[-.0068, -.0766, .0189]]))
        pocket = positions[:, self.body] + quat_apply(quats[:, self.body], positions.new_tensor(self.tool_offset_b[None]))
        return (fixed + moving)[0].cpu().numpy() / 2, pocket[0].cpu().numpy()

    def _body_rotation(self):
        return Rotation.from_quat(tensor(self.robot.data.body_quat_w)[0, self.body].cpu().numpy()).as_matrix()

    def _closure_rotation(self, planned_rotation):
        """Keep measured tracking error out of the next nominal pose target."""
        actual = self._body_rotation()
        if not getattr(self, 'arm_tracking_feedback', False):
            return actual
        planned = np.array(planned_rotation, dtype=float, copy=True)
        self.motion_diagnostics.append({'phase': 'closure_orientation',
            'event': 'planned_achieved_rotation_preserved',
            'nominal_rotation_world': planned.tolist(), 'measured_rotation_world': actual.tolist(),
            'measured_orientation_error_deg': float(np.degrees(Rotation.from_matrix(planned.T @ actual).magnitude())),
            'source': 'selected candidate achieved body rotation used for descent; measured pose is diagnostic only'})
        return planned

    def _plan_repaired(self, candidates, axes, *, candidate_start=0, candidate_stop=None):
        """Prefer reachable orientation and check both actual open pad heights."""
        from isaaclab.utils.math import quat_apply_inverse
        from .arm_collision import width_rejection
        positions = tensor(self.robot.data.body_pos_w)
        midpoint, pocket = self._pad_midpoint()
        midpoint_b = quat_apply_inverse(tensor(self.robot.data.body_quat_w)[:, self.body],
            positions.new_tensor(midpoint)[None] - positions[:, self.body])[0].cpu().numpy()
        fixed_b = np.array([-.01289, -.00022, -.0986])
        moving_b = midpoint_b * 2 - fixed_b
        plans = []
        contact_priorities = {}
        start_q = tensor(self.robot.data.joint_pos).clone()
        stop = min(len(candidates), CONFIG['max_candidates']) if candidate_stop is None else min(len(candidates), candidate_stop)
        for index in range(candidate_start, stop):
            candidate = candidates[index]
            width_reason = width_rejection(candidate.get('width_m'))
            if width_reason:
                self.ik_attempts.append({'candidate': index, 'candidate_id': candidate.get('candidate_id', index),
                                         'rejected': width_reason})
                continue
            pose = np.asarray(candidate['T_world_grasp'])
            point = pose[:3, 3] + pose[:3, 0] * candidate['depth_m']
            if self.target_centre is not None:
                point = np.asarray(self.target_centre)
            prepoint = point - pose[:3, 0] * CONFIG['pregrasp_distance_m']
            if point[2] < .05 or prepoint[2] < .07:
                self.ik_attempts.append({'candidate': index, 'candidate_id': candidate.get('candidate_id', index),
                                         'rejected': 'approach_below_table_clearance'})
                continue
            for flipped in (False, True):
                grasp_r = pose[:3, :3] @ (np.diag([1, -1, -1]) if flipped else np.eye(3))
                rotation = grasp_r @ axes.T
                grasp_q, gd = self._solve(point, rotation)
                pre_q, pd = self._solve(prepoint, rotation, grasp_q) if grasp_q is not None else (None, {})
                entry = {'candidate': index, 'candidate_id': candidate.get('candidate_id', index),
                         'candidate_width_m': float(candidate['width_m']),
                         'symmetric_flip': flipped, 'grasp': gd, 'pregrasp': pd}
                self.ik_attempts.append(entry)
                if grasp_q is None or pre_q is None:
                    entry['rejected'] = 'grasp_ik' if grasp_q is None else 'pregrasp_ik'
                    continue
                achieved = np.asarray(gd['achieved_rotation'])
                pad_z = [float((point + achieved @ (pad - self.tool_offset_b))[2]) for pad in (fixed_b, moving_b)]
                entry['open_pad_z_m'] = pad_z
                if min(pad_z) < .044 and self.controller != 'servo':
                    correction = .046 - min(pad_z)
                    if self.contact_frame != 'contact' or correction > .012:
                        entry['rejected'] = 'actual_ik_pad_below_table_clearance'
                        continue
                    shifted = point + np.array([0, 0, correction])
                    grasp_q, gd = self._solve(shifted, rotation, grasp_q)
                    pre_q, pd = self._solve(prepoint + [0, 0, correction], rotation, grasp_q) if grasp_q is not None else (None, {})
                    entry.update(grasp=gd, pregrasp=pd, vertical_clearance_correction_m=correction)
                    if grasp_q is None or pre_q is None:
                        entry['rejected'] = 'grasp_ik_after_clearance' if grasp_q is None else 'pregrasp_ik_after_clearance'
                        continue
                    achieved = np.asarray(gd['achieved_rotation'])
                    pad_z = [float((shifted + achieved @ (pad - self.tool_offset_b))[2]) for pad in (fixed_b, moving_b)]
                    entry['open_pad_z_m'] = pad_z
                    if min(pad_z) < .044:
                        entry['rejected'] = 'actual_ik_pad_below_table_clearance'
                        continue
                    point_for_plan = shifted
                else:
                    point_for_plan = point
                if self.controller == 'servo':
                    entry['network_pocket_world_m'] = point.tolist()
                    grasp_q[:, self.grip] = self.open_joint_rad
                    pre_q[:, self.grip] = self.open_joint_rad
                    entry['planning_table_buffer_m'] = self._planning_table_buffer()
                    entry['required_table_margin_m'] = SERVO_CONFIG['mesh_table_margin_m'] + self._planning_table_buffer()
                    collision = self._planning_mesh_at_q(grasp_q)
                    correction = collision.get('required_vertical_correction_m', 0.) if collision else 0.
                    if collision and collision['reason'] == 'table_mesh' and correction <= SERVO_CONFIG['mesh_max_vertical_correction_m']:
                        # A finite explicit adaptation of the network endpoint,
                        # retaining its approach and grasp orientation.
                        correction = min(SERVO_CONFIG['mesh_max_vertical_correction_m'], correction + .0001)
                        shifted = point + [0, 0, correction]
                        grasp_q, gd = self._solve(shifted, rotation, grasp_q)
                        pre_q, pd = self._solve(prepoint + [0, 0, correction], rotation, grasp_q) if grasp_q is not None else (None, {})
                        entry.update(grasp=gd, pregrasp=pd, mesh_vertical_correction_m=correction)
                        if grasp_q is None or pre_q is None:
                            entry['rejected'] = 'mesh_clearance_correction_ik'
                            continue
                        grasp_q[:, self.grip] = self.open_joint_rad
                        pre_q[:, self.grip] = self.open_joint_rad
                        point_for_plan = shifted
                        achieved = np.asarray(gd['achieved_rotation'])
                        collision = self._planning_mesh_at_q(grasp_q)
                    entry['mesh_grasp_collision'] = collision
                    entry['executed_pocket_offset_m'] = (point_for_plan - point).tolist()
                    if collision:
                        entry['rejected'] = collision['reason']
                        continue
                    if getattr(self, 'arm_tracking_feedback', False):
                        envelope = self._planning_closure_envelope(point_for_plan, achieved)
                        entry['closure_envelope_before_correction'] = envelope
                        if not envelope['valid']:
                            total = float(point_for_plan[2] - point[2] + envelope['required_vertical_correction_m'] + .0001)
                            if total > SERVO_CONFIG['mesh_max_vertical_correction_m']:
                                entry.update(rejected='closure_envelope_vertical_correction_exceeds_limit',
                                    required_total_vertical_correction_m=total)
                                continue
                            shifted = point + [0, 0, total]
                            grasp_q, gd = self._solve(shifted, rotation, grasp_q)
                            pre_q, pd = self._solve(prepoint + [0, 0, total], rotation, grasp_q) if grasp_q is not None else (None, {})
                            entry.update(grasp=gd, pregrasp=pd, closure_envelope_vertical_correction_m=total)
                            if grasp_q is None or pre_q is None:
                                entry['rejected'] = 'closure_envelope_correction_ik'
                                continue
                            grasp_q[:, self.grip] = self.open_joint_rad
                            pre_q[:, self.grip] = self.open_joint_rad
                            point_for_plan = shifted
                            achieved = np.asarray(gd['achieved_rotation'])
                            collision = self._planning_mesh_at_q(grasp_q)
                            envelope = self._planning_closure_envelope(point_for_plan, achieved)
                            entry.update(mesh_grasp_collision=collision,
                                executed_pocket_offset_m=(point_for_plan - point).tolist())
                        entry['closure_envelope'] = envelope
                        if collision or not envelope['valid']:
                            entry['rejected'] = 'closure_envelope_clearance_after_ik'
                            continue
                    highpoint = point_for_plan - pose[:3, 0] * CONFIG['pregrasp_distance_m']
                    highpoint[2] = max(highpoint[2], SERVO_CONFIG['approach_clearance_z_m'])
                    high_q, high_diagnostic = self._solve(highpoint, achieved, pre_q)
                    entry['high_approach_ik'] = high_diagnostic
                    if high_q is None:
                        entry['rejected'] = 'high_approach_ik'
                        continue
                    high_q[:, self.grip] = self.open_joint_rad
                    collision = self._planning_mesh_path(start_q, high_q, 'approach_high')
                    if collision is None:
                        collision = self._planning_mesh_path(high_q, pre_q, 'approach')
                    entry['mesh_approach_collision'] = collision
                    if collision:
                        entry['rejected'] = collision['reason']
                        continue
                    self.servo_routes[(index, flipped)] = (high_q, highpoint)
                if self.destination is not None and self.controller != 'servo':
                    box = np.asarray(self.destination)
                    tool = np.array([point_for_plan - achieved @ POCKET,
                         point_for_plan + achieved @ (fixed_b - POCKET), point_for_plan + achieved @ (moving_b - POCKET)])
                    inside = (np.abs(tool[:, :2] - box[:2]) < .092).all(-1) & (tool[:, 2] < box[2] + .065)
                    if inside.any():
                        entry['rejected'] = 'observed_box_tool_clearance'
                        continue
                if self.collision_check == 'capsule':
                    pre_q[:, self.grip] = self.open_joint_rad
                    grasp_q[:, self.grip] = self.open_joint_rad
                    for phase, source, goal in (('approach', start_q, pre_q), ('descend', pre_q, grasp_q)):
                        collision = self._check_path(source, goal, phase)
                        if collision:
                            entry['rejected'] = 'arm_collision'
                            entry['collision'] = collision
                            break
                    if 'rejected' in entry:
                        continue
                if self.collision_check == 'capsule' or self.controller == 'servo':
                    lift_q, lift_diagnostic = self._solve(point_for_plan + [0, 0, CONFIG['lift_distance_m']],
                                                          rotation, grasp_q)
                    entry['nominal_lift'] = lift_diagnostic
                    if lift_q is None:
                        entry['rejected'] = 'nominal_lift_ik'
                        continue
                if self.collision_check == 'capsule':
                    lift_q[:, self.grip] = CONFIG['closed_joint_rad']
                    closed_grasp = grasp_q.clone()
                    closed_grasp[:, self.grip] = CONFIG['closed_joint_rad']
                    collision = self._check_path(grasp_q, closed_grasp, 'nominal_close')
                    if collision is None:
                        collision = self._check_path(closed_grasp, lift_q, 'nominal_lift')
                    if collision:
                        entry['rejected'] = 'arm_collision'
                        entry['collision'] = collision
                        continue
                if self.controller == 'servo':
                    descent, descent_diagnostic = self._plan_cartesian_descent(
                        np.asarray(pd['achieved_pocket_world_m']), pre_q, point_for_plan, achieved)
                    entry['descend_preflight'] = descent_diagnostic
                    if descent is None:
                        entry['rejected'] = descent_diagnostic['failure_reason']
                        continue
                    self.servo_descents[(index, flipped)] = descent
                cost = max(gd['orientation_error_deg'], pd['orientation_error_deg']) + .1 * index
                entry['planning_cost_base'] = cost
                if getattr(self, 'candidate_contact_ranking', False):
                    from .candidate_contact import ContactBandConfig, contact_priority_key
                    contact = self._contact_band_diagnostic(point_for_plan, achieved)
                    policy = getattr(self, 'candidate_contact_policy', 'soft8')
                    priority = contact_priority_key(cost, contact, config=ContactBandConfig(enabled=True, policy=policy))
                    contact_priorities[(index, flipped)] = priority
                    cost += contact['penalty_deg_equivalent']
                    entry['candidate_contact'] = contact
                    entry['candidate_contact_policy'] = policy
                    entry['candidate_contact_priority'] = list(priority)
                entry['planning_cost'] = cost
                entry['accepted'] = True
                plans.append((cost, index, candidate, point_for_plan, grasp_q, pre_q, achieved, flipped))
        key = ((lambda x: contact_priorities[(x[1], x[-1])]) if getattr(self, 'candidate_contact_ranking', False)
               else (lambda x: x[0]))
        selected = min(plans, key=key) if plans else None
        if getattr(self, 'candidate_contact_ranking', False) and plans:
            costs = {(e['candidate'], e['symmetric_flip']): e['planning_cost_base']
                     for e in self.ik_attempts if e.get('accepted') and 'planning_cost_base' in e}
            baseline = min(plans, key=lambda x: costs[(x[1], x[-1])])
            self.candidate_contact_events.append({'open_joint_rad': self.open_joint_rad,
                'candidate_start': candidate_start, 'candidate_stop': stop,
                'baseline_candidate': baseline[1], 'baseline_candidate_id': baseline[2].get('candidate_id', baseline[1]),
                'baseline_symmetric_flip': baseline[-1],
                'selected_candidate': selected[1], 'selected_candidate_id': selected[2].get('candidate_id', selected[1]),
                'selected_symmetric_flip': selected[-1],
                'policy': getattr(self, 'candidate_contact_policy', 'soft8'),
                'selected_priority': list(contact_priorities[(selected[1], selected[-1])]),
                'semantics': 'soft heuristic cost units; same feasible plans and unchanged candidate poses'})
        return selected

    def _close_centered(self, centre, rotation):
        """Track pad midpoint from robot kinematics while the asymmetric jaw closes."""
        if getattr(self, 'arm_tracking_feedback', False):
            return self._close_centered_synchronized(centre, rotation)
        target = tensor(self.robot.data.joint_pos).clone()
        start_grip = float(target[0, self.grip])
        adaptive = self.controller == 'servo' and SERVO_CONFIG['close_policy'] == 'contact_stall'
        if self.controller == 'servo' and SERVO_CONFIG['close_policy'] not in ('timed', 'contact_stall'):
            raise ValueError('servo close_policy must be timed or contact_stall')
        latch = _ClosureContactLatch((start_grip - CONFIG['closed_joint_rad']) / (55 * self.env.step_dt)) if adaptive else None
        for step in range(70):
            if not self.env.tracking:
                break
            if step % 5 == 0:
                midpoint, pocket = self._pad_midpoint()
                desired = np.asarray(centre) + pocket - midpoint
                q, diagnostic = self._solve(desired, rotation, target)
                self.close_ik_attempts.append({'close_step': step, 'requested_pocket_world_m': desired.tolist(),
                                               'pad_midpoint_world_m': midpoint.tolist(), **diagnostic})
                if q is not None:
                    target[:, self.arm] = q[:, self.arm]
            if latch is not None:
                event = latch.observe(float(self.scorer()['claw_force']),
                    float(tensor(self.robot.data.joint_pos)[0, self.grip]), float(target[0, self.grip]),
                    float(tensor(self.robot.data.joint_vel)[0, self.grip]))
                if event is not None:
                    self.close_control_events.append({'close_step': step,
                        'step': int(self.env.episode_length_buf[0]), **event})
            target[:, self.grip] = latch.target if latch is not None and latch.target is not None else max(
                CONFIG['closed_joint_rad'], start_grip +
                (CONFIG['closed_joint_rad'] - start_grip) * min(1., (step + 1) / 55))
            if not self._step(target, 'close_centered'):
                break
        if self.env.tracking and not (latch is not None and latch.target is not None):
            # Once the sustained closure has seated the fruit, hold a small
            # preload beyond the measured blocked joint, not the early touch.
            actual = float(tensor(self.robot.data.joint_pos)[0, self.grip])
            preload = SERVO_CONFIG['hold_preload_rad'] if self.controller == 'servo' else .10
            target[:, self.grip] = max(CONFIG['closed_joint_rad'], actual - preload)
        if latch is not None and latch.target is None:
            self.close_control_events.append({'event': 'contact_stall_not_observed',
                'final_joint_target_rad': float(target[0, self.grip])})
        if latch is not None and latch.target is not None:
            self._start_grasp_hold(latch.target)
        return target

    def _prepare_synchronized_closure(self, centre, rotation, start, count):
        """Preflight every nominal jaw increment at a fixed observed grasp center."""
        if getattr(self, 'close_frame_policy', 'fixed-body') == 'reachable-line':
            return self._prepare_reachable_line_closure(centre, rotation, start, count)
        if getattr(self, 'close_frame_policy', 'fixed-body') == 'planned-line':
            return self._prepare_planned_line_closure(centre, rotation, start, count)
        from .gripper_geometry import FIXED_REFERENCE_M, MOVING_REFERENCE_M
        seed = start.clone()
        path, evidence = [], {'phase': 'close_preflight', 'waypoints': [], 'table_margin_m': SERVO_CONFIG['mesh_table_margin_m'],
            'future_required_table_margin_m': SERVO_CONFIG['mesh_table_margin_m'] + self._planning_table_buffer(),
            'origin_runtime_margin_m': SERVO_CONFIG['mesh_table_margin_m']}
        for index, jaw in enumerate(np.linspace(float(start[0, self.grip]), CONFIG['closed_joint_rad'], count + 1)):
            midpoint_b = (FIXED_REFERENCE_M + self.gripper_geometry.moving_to_fixed(MOVING_REFERENCE_M, jaw)) / 2
            desired = np.asarray(centre) + np.asarray(rotation) @ (self.tool_offset_b - midpoint_b)
            q, diagnostic = self._solve(desired, rotation, seed, prefer_near=True)
            item = {'index': index, 'jaw_joint_target_rad': float(jaw), **diagnostic}
            evidence['waypoints'].append(item)
            self.close_ik_attempts.append({'close_step': index, 'preflight': True,
                'requested_pocket_world_m': desired.tolist(), **diagnostic})
            if q is None:
                evidence['failure_reason'] = 'close_preflight_ik'
                return None, evidence
            q[:, self.grip] = float(jaw)
            delta = float((q[:, self.arm] - seed[:, self.arm]).abs().max())
            item.update(max_arm_delta_rad=delta, joint_target=q[0].tolist())
            if delta > SERVO_CONFIG['cartesian_max_joint_delta_rad']:
                evidence['failure_reason'] = 'close_preflight_joint_jump'
                return None, evidence
            # The measured origin requires the original runtime clearance.
            # Every future nominal endpoint, and paths joining such endpoints,
            # must preserve the separate tracking reserve as well.
            future_collision = self._planning_mesh_at_q(q)
            item['future_endpoint_collision'] = future_collision
            if future_collision:
                evidence.update(failure_reason='close_preflight_tracking_clearance', collision=future_collision)
                return None, evidence
            collision = self._check_path(seed, q, 'close_preflight')
            if collision is None:
                # This connector begins at measured robot state. Use the
                # original runtime margin, including both segment endpoints.
                collision = (self._mesh_path(seed, q, 'close_preflight') if index == 0 else
                             self._planning_mesh_path(seed, q, 'close_preflight'))
            if collision:
                evidence.update(failure_reason='close_preflight_collision', collision=collision)
                return None, evidence
            path.append((desired, q))
            seed = q
        evidence['valid'] = True
        return path, evidence

    def _prepare_planned_line_closure(self, centre, rotation, start, count):
        from .closing_line import PlannedClosingLine
        line = PlannedClosingLine(self.gripper_geometry, centre, rotation,
                                  float(start[0, self.grip]), self.tool_offset_b)
        evidence = {'phase': 'close_preflight', 'close_frame_policy': 'planned-line', 'waypoints': [],
            'table_margin_m': SERVO_CONFIG['mesh_table_margin_m'],
            'future_required_table_margin_m': SERVO_CONFIG['mesh_table_margin_m']+self._planning_table_buffer(),
            'origin_runtime_margin_m': SERVO_CONFIG['mesh_table_margin_m'],
            'initial_planned_rotation': line.rotation0.tolist(), 'initial_measured_jaw_rad': line.jaw0,
            'preserved_planned_line_frame_world': line.frame0.tolist(), 'canonical_centre_world_m': line.centre.tolist(),
            'point_shift_added_m': [0., 0., 0.],
            'geometry_part_count': len(self.mesh_clearance.parts),
            'sweep': 'fresh full robot FK mesh endpoints and all joint-sampled connecting paths; no fixed-R envelope exemption'}
        if len(self.mesh_clearance.parts) != 18:
            evidence['failure_reason'] = 'close_planned_line_geometry_coverage'
            return None, evidence
        seed = start.clone(); path = []
        for index, jaw in enumerate(np.linspace(line.jaw0, CONFIG['closed_joint_rad'], count+1)):
            desired, requested_rotation, geometry = line.pose(float(jaw))
            q, diagnostic = self._solve(desired, requested_rotation, seed, prefer_near=True)
            item = {'index': index, 'jaw_joint_target_rad': float(jaw), 'line_geometry': geometry, **diagnostic}
            evidence['waypoints'].append(item)
            self.close_ik_attempts.append({'close_step': index, 'preflight': True,
                'close_frame_policy': 'planned-line', 'requested_pocket_world_m': desired.tolist(),
                'requested_rotation_world': requested_rotation.tolist(), **diagnostic})
            if q is None:
                evidence['failure_reason'] = 'close_preflight_ik'
                return None, evidence
            q[:, self.grip] = float(jaw)
            delta = float((q[:, self.arm]-seed[:, self.arm]).abs().max())
            item.update(max_arm_delta_rad=delta, joint_target=q[0].tolist())
            if delta > SERVO_CONFIG['cartesian_max_joint_delta_rad']:
                evidence['failure_reason'] = 'close_preflight_joint_jump'
                return None, evidence
            future_collision = self._planning_mesh_at_q(q)
            item['future_endpoint_collision'] = future_collision
            if future_collision:
                evidence.update(failure_reason='close_preflight_tracking_clearance', collision=future_collision)
                return None, evidence
            collision = self._check_path(seed, q, 'close_planned_line_preflight')
            if collision is None:
                collision = (self._mesh_path(seed, q, 'close_planned_line_preflight') if index == 0 else
                             self._planning_mesh_path(seed, q, 'close_planned_line_preflight'))
            if collision:
                evidence.update(failure_reason='close_preflight_collision', collision=collision)
                return None, evidence
            path.append((desired, q, requested_rotation))
            seed = q
        evidence['valid'] = True
        return path, evidence

    def _prepare_reachable_line_closure(self, centre, rotation, start, count):
        from .closing_line import PlannedClosingLine, REACHABLE_LINE_BETAS, REACHABLE_LINE_NOMINAL_ORIENTATION_DEG
        self._reachable_closing_line = None
        evidence = {'phase': 'close_preflight', 'close_frame_policy': 'reachable-line',
            'beta_candidates': list(REACHABLE_LINE_BETAS), 'beta_attempts': [], 'waypoints': [],
            'nominal_orientation_limit_deg': REACHABLE_LINE_NOMINAL_ORIENTATION_DEG,
            'runtime_orientation_limit_deg': SERVO_CONFIG['orientation_tolerance_deg'],
            'orientation_reserve_deg': SERVO_CONFIG['orientation_tolerance_deg']-REACHABLE_LINE_NOMINAL_ORIENTATION_DEG,
            'nominal_position_limit_m': CONFIG['ik_position_tolerance_m'],
            'table_margin_m': SERVO_CONFIG['mesh_table_margin_m'],
            'future_required_table_margin_m': SERVO_CONFIG['mesh_table_margin_m']+self._planning_table_buffer(),
            'origin_runtime_margin_m': SERVO_CONFIG['mesh_table_margin_m'],
            'selected_beta': None, 'selection': 'first complete feasible path in fixed descending beta list'}
        for beta in REACHABLE_LINE_BETAS:
            path, attempt = self._prepare_reachable_line_beta(centre, rotation, start, count, beta)
            evidence['beta_attempts'].append(attempt)
            if path is not None:
                self._reachable_closing_line = PlannedClosingLine(self.gripper_geometry, centre, rotation,
                    float(start[0, self.grip]), self.tool_offset_b, beta=beta)
                evidence.update(valid=True, selected_beta=beta, waypoints=attempt['waypoints'],
                    initial_planned_rotation=attempt['initial_planned_rotation'],
                    initial_measured_jaw_rad=attempt['initial_measured_jaw_rad'],
                    preserved_planned_line_frame_world=attempt['preserved_planned_line_frame_world'],
                    canonical_centre_world_m=attempt['canonical_centre_world_m'],
                    point_shift_added_m=[0., 0., 0.], geometry_part_count=attempt['geometry_part_count'])
                return path, evidence
        evidence.update(valid=False, failure_reason='close_reachable_line_no_feasible_beta')
        return None, evidence

    def _prepare_reachable_line_beta(self, centre, rotation, start, count, beta):
        from .closing_line import PlannedClosingLine, REACHABLE_LINE_NOMINAL_ORIENTATION_DEG
        line = PlannedClosingLine(self.gripper_geometry, centre, rotation,
                                  float(start[0, self.grip]), self.tool_offset_b, beta=beta)
        evidence = {'beta': beta, 'valid': False, 'waypoints': [],
            'initial_planned_rotation': line.rotation0.tolist(), 'initial_measured_jaw_rad': line.jaw0,
            'preserved_planned_line_frame_world': line.frame0.tolist(), 'canonical_centre_world_m': line.centre.tolist(),
            'geometry_part_count': len(self.mesh_clearance.parts), 'point_shift_added_m': [0., 0., 0.],
            'nominal_orientation_limit_deg': REACHABLE_LINE_NOMINAL_ORIENTATION_DEG,
            'nominal_position_limit_m': CONFIG['ik_position_tolerance_m'],
            'sweep': 'fresh original full FK 18-hull endpoints and joint-sampled paths for this beta'}
        if len(self.mesh_clearance.parts) != 18:
            evidence['failure_reason'] = 'close_planned_line_geometry_coverage'
            return None, evidence
        path = []; seed = start.clone()
        # Inspect all 56 points even after a rejected point, recording exactly
        # which poses were evaluated. No incomplete path can ever be selected.
        for index, jaw in enumerate(np.linspace(line.jaw0, CONFIG['closed_joint_rad'], count+1)):
            desired, requested_rotation, geometry = line.pose(float(jaw))
            q, diagnostic = self._solve(desired, requested_rotation, seed, prefer_near=True)
            diagnostic = dict(diagnostic)
            unavailable = []
            for key in ('position_error_m', 'orientation_error_deg'):
                value = diagnostic.get(key)
                if (value is None or isinstance(value, bool) or
                        not isinstance(value, (int, float, np.number)) or not np.isfinite(value)):
                    diagnostic[key] = None  # absence is explicit; never fabricate a finite residual
                    unavailable.append(key)
            item = {'index': index, 'jaw_joint_target_rad': float(jaw), 'line_geometry': geometry,
                    **diagnostic, 'failures': [], 'path_point_valid': False,
                    'unavailable_nominal_pose_fields': unavailable}
            evidence['waypoints'].append(item)
            self.close_ik_attempts.append({'close_step': index, 'preflight': True,
                'close_frame_policy': 'reachable-line', 'compensation_beta': beta,
                'requested_pocket_world_m': desired.tolist(), 'requested_rotation_world': requested_rotation.tolist(),
                'nominal_orientation_limit_deg': REACHABLE_LINE_NOMINAL_ORIENTATION_DEG, **diagnostic})
            if q is None or diagnostic.get('valid') is not True:
                item['failures'].append('close_preflight_ik')
                continue  # subsequent original IK calls seed from the last returned valid q
            position_error = diagnostic.get('position_error_m')
            orientation_error = diagnostic.get('orientation_error_deg')
            pose_available = (position_error is not None and orientation_error is not None and
                np.isfinite([position_error, orientation_error]).all() and position_error >= 0 and orientation_error >= 0)
            pose_passed = (pose_available and position_error < CONFIG['ik_position_tolerance_m'] and
                           orientation_error <= REACHABLE_LINE_NOMINAL_ORIENTATION_DEG)
            item['nominal_pose_gate'] = {'available': bool(pose_available), 'passed': bool(pose_passed),
                'position_error_m': position_error, 'orientation_error_deg': orientation_error,
                'position_limit_m': CONFIG['ik_position_tolerance_m'],
                'orientation_limit_deg': REACHABLE_LINE_NOMINAL_ORIENTATION_DEG}
            if not pose_passed:
                item['failures'].append('close_preflight_nominal_pose' if pose_available else 'close_preflight_nominal_pose_unavailable')
            q[:, self.grip] = float(jaw)
            delta = float((q[:, self.arm]-seed[:, self.arm]).abs().max())
            item.update(max_arm_delta_rad=delta, joint_target=q[0].tolist())
            if delta > SERVO_CONFIG['cartesian_max_joint_delta_rad']:
                item['failures'].append('close_preflight_joint_jump')
            future_collision = self._planning_mesh_at_q(q)
            item['future_endpoint_collision'] = future_collision
            if future_collision:
                item['failures'].append('close_preflight_tracking_clearance')
            collision = self._check_path(seed, q, 'close_reachable_line_preflight')
            if collision is None:
                collision = (self._mesh_path(seed, q, 'close_reachable_line_preflight') if index == 0 else
                             self._planning_mesh_path(seed, q, 'close_reachable_line_preflight'))
            item['connector_collision'] = collision
            if collision:
                item['failures'].append('close_preflight_collision')
            if not item['failures']:
                item['path_point_valid'] = True
                path.append((desired, q, requested_rotation))
            seed = q
        failed = [item for item in evidence['waypoints'] if item['failures']]
        if failed:
            evidence.update(failure_reason=failed[0]['failures'][0], first_failed_index=failed[0]['index'],
                            failed_point_count=len(failed))
            return None, evidence
        evidence['valid'] = True
        return path, evidence

    def _closure_followup_rotation(self, original_rotation):
        if getattr(self, 'close_frame_policy', 'fixed-body') not in ('planned-line', 'reachable-line'):
            return original_rotation
        if self._last_closure_rotation is None or self._last_closure_tool is None:
            raise RuntimeError('planned-line closure completed without an issued planned pose')
        rotation = self._last_closure_rotation.copy()
        self.motion_goal = (self._last_closure_tool.copy(), rotation)
        self.motion_diagnostics.append({'phase': 'hold_lift_orientation', 'event': 'last_planned_closure_rotation_inherited',
            'requested_rotation_world': rotation.tolist(), 'requested_tool_world_m': self._last_closure_tool.tolist(),
            'source': 'last successfully issued planned closure waypoint; measured body rotation excluded'})
        return rotation

    def _close_centered_synchronized(self, centre, rotation):
        # Adjacent transfer owns only this invocation of this preflight path.
        # Cleanup runs on normal completion, timeout, safety stop or exception.
        try:
            return self._close_centered_synchronized_impl(centre, rotation)
        finally:
            if getattr(self, 'close_integral_transfer', 'reset') == 'adjacent':
                self._pending_close_integral_transfer = None
                self._arm_tracking.end_close()

    def _close_centered_synchronized_impl(self, centre, rotation):
        from .arm_tracking import MotionSynchronizationConfig
        config = MotionSynchronizationConfig()
        start = tensor(self.robot.data.joint_pos).clone()
        planned_line = getattr(self, 'close_frame_policy', 'fixed-body') in ('planned-line', 'reachable-line')
        if planned_line:
            self._last_closure_rotation = self._last_closure_tool = None
        path, evidence = self._prepare_synchronized_closure(centre, rotation, start, config.close_advance_steps)
        self.motion_diagnostics.append(evidence)
        if path is None:
            self.motion_failure = evidence['failure_reason']
            return None
        adjacent_transfer = getattr(self, 'close_integral_transfer', 'reset') == 'adjacent'
        if adjacent_transfer and evidence.get('valid') is not True:
            self.motion_failure = 'close_integral_transfer_preflight_unavailable'
            return None
        desired, target = path[0][:2]
        if planned_line:
            rotation = path[0][2]
        self.motion_goal = (desired, rotation)
        if not self._move(target, 'close_align', config.close_alignment_move_steps) or not self._arrive(target, desired, rotation, 'close_align'):
            if not self.motion_failure:
                self.motion_failure = 'close_alignment_failed'
            return None
        if adjacent_transfer:
            transfer_path = [q[0].detach().cpu().numpy().copy() for _, q in path]
            self._arm_tracking.begin_close(transfer_path)
        latch = _ClosureContactLatch((float(start[0, self.grip]) - CONFIG['closed_joint_rad']) /
                                      (config.close_advance_steps * self.env.step_dt),
                                      policy=getattr(self, 'closure_latch_policy', 'legacy'))
        progress = stable = final_steps = 0
        for elapsed in range(config.close_max_steps):
            if not self.env.tracking or not self.collision_ready:
                return None
            actual = tensor(self.robot.data.joint_pos)
            velocity = tensor(self.robot.data.joint_vel)
            # Feedback modifies only arm joints. For no_reopen explicitly use
            # the last successfully issued jaw command, never the measured jaw
            # or a future precomputed waypoint as the retained command.
            issued = getattr(self, '_last_issued_target', None)
            if latch.policy == 'no_reopen' and issued is None:
                self.motion_failure = 'close_latch_command_unavailable'
                return None
            jaw_command = (float(issued[0, self.grip]) if latch.policy == 'no_reopen'
                           else float(target[0, self.grip]))
            event = latch.observe(float(self.scorer()['claw_force']), float(actual[0, self.grip]),
                                  jaw_command, float(velocity[0, self.grip]))
            if event is not None:
                self.close_control_events.append({'close_step': progress, 'elapsed_close_steps': elapsed,
                    'step': int(self.env.episode_length_buf[0]), **event})
            _, pocket = self._pad_midpoint()
            position_error = float(np.linalg.norm(pocket - desired))
            angle = float(np.degrees(Rotation.from_matrix(np.asarray(rotation).T @ self._body_rotation()).magnitude()))
            arm_error = float((actual[:, self.arm] - target[:, self.arm]).abs().max())
            speed = float(velocity[:, self.arm].abs().max())
            settled = (position_error <= SERVO_CONFIG['position_tolerance_m'] and
                angle <= SERVO_CONFIG['orientation_tolerance_deg'] and arm_error <= SERVO_CONFIG['arm_joint_tolerance_rad'] and
                speed <= config.arm_speed_tolerance_rad_s)
            stable = stable + 1 if settled else 0
            advanced = False
            if latch.target is None and progress < config.close_advance_steps and stable >= config.consecutive_steps:
                previous_progress = progress
                previous_target = target
                progress += 1
                desired, target = path[progress][:2]
                if planned_line:
                    rotation = path[progress][2]
                if adjacent_transfer:
                    token = self._arm_tracking.authorize_adjacent_transfer(transfer_path,
                        previous_progress, progress, previous_target[0].detach().cpu().numpy(),
                        target[0].detach().cpu().numpy())
                    self._pending_close_integral_transfer = token
                    self.close_control_events.append({'event': 'integral_transfer_authorized',
                        'step': int(self.env.episode_length_buf[0]), 'elapsed_close_steps': elapsed,
                        'authorized': token is not None, 'path_preflight_valid': True,
                        'session': token.session if token is not None else None,
                        'from_index': previous_progress, 'to_index': progress,
                        'settled': settled, 'consecutive_settled_steps': stable,
                        'latched': False, 'previous_nominal': previous_target[0].tolist(),
                        'next_nominal': target[0].tolist(), 'accepted_integral_before_transfer_rad': self._arm_tracking.integral.tolist()})
                stable = 0
                advanced = True
            elif latch.target is not None:
                if adjacent_transfer:
                    self._pending_close_integral_transfer = None
                    self._arm_tracking.clear_integral()
                # Arm reference follows actual stopped jaw geometry; the locked
                # jaw command remains exactly the original contact-stall rule.
                index = int(np.argmin([abs(float(item[1][0, self.grip]) - float(actual[0, self.grip])) for item in path]))
                desired, prepared = path[index][:2]
                if planned_line:
                    rotation = path[index][2]
                target = prepared.clone()
                target[:, self.grip] = latch.target
            finished = latch.target is not None or progress == config.close_advance_steps
            final_steps = final_steps + 1 if finished else 0
            self.close_control_events.append({'event': 'closure_synchronization',
                'step': int(self.env.episode_length_buf[0]), 'elapsed_close_steps': elapsed,
                'jaw_advance_steps': progress, 'jaw_advanced': advanced,
                'consecutive_settled_steps': stable, 'arm_settled': settled,
                'position_error_m': position_error, 'orientation_error_deg': angle,
                'max_arm_joint_error_rad': arm_error, 'max_arm_joint_speed_rad_s': speed,
                'nominal_jaw_target_rad': float(target[0, self.grip]), 'latched': latch.target is not None,
                **({'planned_line_target': {'waypoint_index': index if latch.target is not None else progress,
                    'requested_rotation_world': np.asarray(rotation).tolist(),
                    'requested_tool_world_m': np.asarray(desired).tolist(),
                    'gate_timing': 'published gate precedes any waypoint advance; target here is next issued nominal'}}
                   if planned_line else {})})
            self.motion_goal = (desired, rotation)
            if not self._step(target, 'close_centered'):
                return None
            if planned_line:
                self._last_closure_rotation = np.array(rotation, copy=True)
                self._last_closure_tool = np.array(desired, copy=True)
            if finished and final_steps >= config.close_final_hold_steps:
                if latch.target is None:
                    target[:, self.grip] = max(CONFIG['closed_joint_rad'],
                        float(tensor(self.robot.data.joint_pos)[0, self.grip]) - SERVO_CONFIG['hold_preload_rad'])
                    self.close_control_events.append({'event': 'contact_stall_not_observed',
                        'final_joint_target_rad': float(target[0, self.grip])})
                else:
                    self._start_grasp_hold(latch.target)
                if not planned_line:
                    self.motion_goal = None
                return target
        self.motion_failure = 'close_synchronization_timeout'
        self.motion_diagnostics.append({'phase': 'close_centered', 'arrived': False,
            'failure_reason': self.motion_failure, 'elapsed_close_steps': config.close_max_steps,
            'jaw_advance_steps': progress, 'max_close_steps': config.close_max_steps})
        return None

    def execute(self, candidates, scene):
        if getattr(self, 'arm_tracking_feedback', False):
            self._arm_tracking.reset()
            self._last_nominal_target = None
            self._last_issued_target = None
        self._stop_grasp_hold('execute_start', 'execution_reset')
        self._grasp_hold_last_target_rad = None
        try:
            return self._execute(candidates, scene)
        finally:
            self._stop_grasp_hold('execute_end', 'execution_finished')

    def _execute(self, candidates, scene):
        from .arm_collision import width_rejection
        failure = 'open_gripper' if self.collision_ready else 'collision_check_unavailable'
        chosen = None
        opened = tensor(self.robot.data.joint_pos).clone()
        opened[:, self.grip] = self.open_joint_rad
        if self.collision_ready and self._move(opened, 'open', 70):
            if self.controller == 'servo':
                midpoint, control_tool = self._pad_midpoint()
                self.tool_evidence.update(actual_open_joint_rad=float(tensor(self.robot.data.joint_pos)[0, self.grip]),
                    actual_open_reference_midpoint_error_m=float(np.linalg.norm(midpoint - control_tool)))
            axes = self._grasp_axes_in_body()
            failure = 'ik_unreachable'
            planned = (self._plan_servo_openings(candidates) if self.controller == 'servo' else
                       self._plan_repaired(candidates, axes) if self.controller == 'repaired' else None)
            if self.controller in ('repaired', 'servo') and planned is None:
                failure = 'all_candidate_plans_rejected'
            pool = list(enumerate(candidates[:CONFIG['max_candidates']]))
            if self.controller in ('repaired', 'servo'):
                pool = [(planned[1], planned[2])] if planned else []
            for index, candidate in pool:
                width_reason = width_rejection(candidate.get('width_m'))
                if width_reason:
                    self.ik_attempts.append({'candidate': index, 'candidate_id': candidate.get('candidate_id', index),
                                             'rejected': width_reason})
                    continue
                pose = np.asarray(candidate['T_world_grasp'])
                rotation = pose[:3, :3] @ axes.T
                point = pose[:3, 3] + pose[:3, 0] * candidate['depth_m']
                prepoint = point - pose[:3, 0] * CONFIG['pregrasp_distance_m']
                if not planned and (point[2] < .05 or prepoint[2] < .07):
                    self.ik_attempts.append({'candidate': index, 'rejected': 'approach_below_table_clearance'})
                    continue
                if planned:
                    _, _, _, point, grasp_q, pre_q, rotation, flipped = planned
                else:
                    grasp_q, gd = self._solve(point, rotation)
                    pre_q, pd = self._solve(prepoint, rotation, grasp_q) if grasp_q is not None else (None, {})
                    self.ik_attempts.append({'candidate': index, 'grasp': gd, 'pregrasp': pd})
                if grasp_q is not None and pre_q is not None:
                    chosen = {'candidate': index, 'candidate_id': candidate.get('candidate_id', index),
                              'candidate_rank': candidate.get('rank', index), 'raw_index': candidate.get('raw_index'),
                              'candidate_width_m': float(candidate['width_m']),
                              'world_pocket_m': point.tolist(), 'rotation': rotation.tolist(),
                              'world_pocket_reference': self.tool_evidence['control_reference']}
                    if planned:
                        chosen.update(symmetric_flip=planned[-1], planning_cost=planned[0],
                                      point_source='language_mask_depth_sphere_fit' if self.target_centre is not None else 'network_with_pad_clearance',
                                      contact_frame=self.contact_frame)
                        if self.controller == 'servo':
                            selected_entry = next(item for item in reversed(self.ik_attempts)
                                if item.get('accepted') and item['candidate'] == index and item['symmetric_flip'] == planned[-1])
                            chosen.update(network_pocket_world_m=selected_entry['network_pocket_world_m'],
                                          executed_pocket_offset_m=selected_entry['executed_pocket_offset_m'],
                                          open_joint_rad=self.open_joint_rad,
                                          opening_stage_index=selected_entry['opening_stage_index'],
                                          candidate_stage=selected_entry.get('candidate_stage', 'primary'),
                                          descent_waypoint_count=len(self.servo_descents[(index, planned[-1])]),
                                          point_source='network_with_explicit_mesh_clearance')
                    pre_q[:, self.grip] = self.open_joint_rad
                    grasp_q[:, self.grip] = self.open_joint_rad
                    if self.controller == 'servo':
                        # Preserve the selected network approach axis, including
                        # any observed-pad vertical clearance correction.
                        planned_prepoint = point - pose[:3, 0] * CONFIG['pregrasp_distance_m']
                        high_q, highpoint = self.servo_routes[(index, planned[-1])]
                        self.motion_goal = (highpoint, rotation)
                        high_ok = self._move(high_q, 'approach_high', 60)
                        self.motion_goal = (planned_prepoint, rotation)
                        approached = high_ok and self._move(pre_q, 'approach') and self._arrive(
                            pre_q, planned_prepoint, rotation, 'approach')
                        descended = approached and self._cartesian_descend(point, rotation, grasp_q,
                            prepared=self.servo_descents[(index, planned[-1])])
                    else:
                        descended = self._move(pre_q, 'approach') and self._move(grasp_q, 'descend', 60)
                    if descended:
                        failure = 'close_or_lift'
                        if self.controller in ('repaired', 'servo'):
                            self.motion_goal = None
                            rotation = self._closure_rotation(rotation)
                            closed = self._close_centered(point, rotation)
                            if closed is None:
                                failure = self.motion_failure or 'close_synchronization_failed'
                                break
                            rotation = self._closure_followup_rotation(rotation)
                        else:
                            closed = tensor(self.robot.data.joint_pos).clone()
                            closed[:, self.grip] = CONFIG['closed_joint_rad']
                            for _ in range(70):
                                if not self._step(closed, 'close'):
                                    break
                                measured = self.scorer()
                                if self.closure == 'first_contact' and measured['claw_force'] > 3.0 and .1 < measured['grip'] < 1.3:
                                    closed[:, self.grip] = measured['grip'] - CONFIG['contact_hold_preload_rad']
                                    break
                        for _ in range(15):
                            if not self._step(closed, 'hold'):
                                break
                        if self.env.tracking and self.collision_ready:
                            # Use the current jaw command in lift IK/path
                            # seeds too; the caller's close tensor was not
                            # modified by per-step upkeep.
                            if getattr(self, '_grasp_hold_last_target_rad', None) is not None:
                                closed = closed.clone()
                                closed[:, self.grip] = self._grasp_hold_last_target_rad
                            lift_origin = self._pad_midpoint()[1] if self.controller in ('repaired', 'servo') else point
                            liftpoint = lift_origin + np.array([0, 0, CONFIG['lift_distance_m']])
                            lift_q, ld = self._solve(liftpoint, rotation, closed)
                            chosen['lift_ik'] = ld
                            if lift_q is None:
                                failure = 'lift_ik_rejected'
                            if lift_q is not None:
                                if self.controller == 'servo':
                                    self.motion_goal = (liftpoint, rotation)
                                lift_q[:, self.grip] = closed[:, self.grip]
                                collision = self._check_path(tensor(self.robot.data.joint_pos).clone(), lift_q, 'lift')
                                chosen['lift_collision'] = collision
                                if collision:
                                    failure = 'lift_collision_rejected'
                                    break
                                self._move(lift_q, 'lift', 90)
                                for _ in range(30):
                                    if not self._step(lift_q, 'lift_hold'):
                                        break
                                lift_q = self._stop_grasp_hold('lift_hold_end', 'scope_complete', lift_q)
                                # Repaired control never branches on privileged
                                # target pose/held score. Robot contact alone is
                                # available to its placement decision.
                                ready = self.lift_success if self.controller == 'baseline' else self.scorer()['claw_force'] > 2.2
                                if scene == 'single' and ready and self.env.tracking:
                                    self.motion_goal = None
                                    failure = self._place(lift_q, rotation)
                    elif self.motion_failure:
                        failure = self.motion_failure
                    break
        terminal = self.env.terminal_result
        success = bool(terminal and terminal.get('success'))
        if success:
            failure = None
        elif terminal:
            failure = '+'.join(k for k, v in terminal.items() if v)
        elif self.runtime_collision:
            failure = 'runtime_collision_rejected'
        self.lift_success |= bool(scene == 'multi' and success)
        (self.directory / 'execution_trace.json').write_text(json.dumps(self.trace, indent=2, allow_nan=False) + '\n')
        from .arm_collision import CLOSED_PAD_SPAN_M, OPEN_PAD_SPAN_M, rejection_counts
        from .arm_tracking import MotionSynchronizationConfig
        return {'success': success, 'grasp_lift_success': self.lift_success, 'robot_executed': self.executed,
                'failure_stage': failure, 'max_rise_m': self.max_rise, 'chosen': chosen,
                'ik_attempts': self.ik_attempts, 'rejection_counts': rejection_counts(self.ik_attempts),
                'motion_diagnostics': self.motion_diagnostics, 'close_ik_attempts': self.close_ik_attempts,
                'arm_tracking_feedback': {
                    'enabled': getattr(self, 'arm_tracking_feedback', False),
                    'config': asdict(self._arm_tracking.config) if getattr(self, 'arm_tracking_feedback', False) else None,
                    'planning_table_buffer_m': self._planning_table_buffer(),
                    'motion_synchronization': asdict(MotionSynchronizationConfig()) if getattr(self, 'arm_tracking_feedback', False) else None,
                    'nominal_transition_policy': 'last_issued_nominal' if getattr(self, 'arm_tracking_feedback', False) else 'measured_joint_position',
                    'closure_latch_policy': getattr(self, 'closure_latch_policy', 'legacy'),
                    'close_integral_transfer': getattr(self, 'close_integral_transfer', 'reset'),
                    'close_frame_policy': getattr(self, 'close_frame_policy', 'fixed-body'),
                    'close_tracking_integral': {'enabled': getattr(self, 'close_tracking_integral', False),
                        'gain_per_s': .2, 'stationary_nominal_steps': 3, 'arm_speed_tolerance_rad_s': .05,
                        'max_dt_s': .05, 'phase': 'close_centered',
                        'reset_on_waypoint_change': getattr(self, 'close_integral_transfer', 'reset') == 'reset',
                        'waypoint_transfer': getattr(self, 'close_integral_transfer', 'reset'),
                        'unqualified_waypoint_change': 'reset',
                        'total_bias_bound_rad': .025, 'total_bias_slew_rad_per_step': .004,
                        'antiwindup': 'conditional_integration', 'commit': 'safety_accept_only',
                        'rejection': 'stop_before_next_action_in_close_centered',
                        'outside_close_rejection': 'original_safe_nominal_fallback'},
                    'arrival_reference': 'unchanged nominal IK tool pose and joint target',
                    'source': 'robot nominal and measured joints only; no target pose, class or score',
                    'events': getattr(self, 'arm_tracking_events', [])},
                'candidate_contact_ranking': {'enabled': getattr(self, 'candidate_contact_ranking', False),
                    'policy': getattr(self, 'candidate_contact_policy', 'soft8'),
                    'events': getattr(self, 'candidate_contact_events', [])},
                'closure_geometry_preflight': {
                    'enabled': getattr(self, 'arm_tracking_feedback', False),
                    'method': 'analytic full-jaw constant-pad-midpoint hull vertex height envelope before candidate commitment',
                    'future_required_table_margin_m': SERVO_CONFIG['mesh_table_margin_m'] + self._planning_table_buffer(),
                    'origin_runtime_margin_m': SERVO_CONFIG['mesh_table_margin_m'],
                    'max_total_vertical_correction_m': SERVO_CONFIG['mesh_max_vertical_correction_m'],
                    'correction_policy': 'joint open-FK and full-closure clearance; one local re-solve then recheck or reject',
                    'table_xy_assumption': 'analytic screening conservatively counts every gripper vertex over table'},
                'close_control_events': self.close_control_events,
                'hold_control_evidence': {
                    'enabled': self.controller == 'servo' and SERVO_CONFIG['bounded_hold_enabled'],
                    'active_phases': ['hold', 'lift', 'lift_hold'],
                    'requires_contact_stall_latch': True, 'config': asdict(HoldConfig()),
                    'force_semantics': 'robot aggregate total contact; not object-filtered pressure',
                    'events': getattr(self, 'hold_control_events', [])},
                'collision_evidence': self.collision_evidence,
                'mesh_clearance_evidence': self.mesh_evidence,
                'tool_evidence': self.tool_evidence,
                'opening_plan_stages': self.opening_plan_stages,
                'width_evidence': {'candidate_width_semantics': 'GraspNet parallel-jaw inner gap',
                    'so101_reference_point_distance_closed_m': CLOSED_PAD_SPAN_M,
                    'so101_reference_point_distance_open_m': OPEN_PAD_SPAN_M,
                    'so101_reference_joint_angles_rad': [CONFIG['closed_joint_rad'], CONFIG['open_joint_rad']],
                    'reference_open_angle_is_actual_command': self.open_joint_rad == CONFIG['open_joint_rad'],
                    'actual_open_joint_rad': self.open_joint_rad,
                    'control': 'fixed_open_then_existing_contact_closure',
                    'limitation': 'Reference-point distances are not inner gaps; candidate width to SO101 joint mapping is uncalibrated.'},
                'execution_config': {**CONFIG, 'open_joint_rad': self.open_joint_rad,
                    'candidate_pool_count': len(candidates), 'primary_candidate_limit': CONFIG['max_candidates'],
                    'tool_offset_body_m': self.tool_offset_b.tolist(),
                    'controller': self.controller, 'collision_check': self.collision_check,
                    'arm_tracking_planning_table_buffer_m': self._planning_table_buffer(),
                    'required_planning_table_margin_m': SERVO_CONFIG['mesh_table_margin_m'] + self._planning_table_buffer(),
                    'motion_synchronization': asdict(MotionSynchronizationConfig()) if getattr(self, 'arm_tracking_feedback', False) else None,
                    'full_closure_envelope_enabled': getattr(self, 'arm_tracking_feedback', False),
                    'nominal_move_start_policy': 'last_issued_nominal' if getattr(self, 'arm_tracking_feedback', False) else 'measured_joint_position',
                    'closure_orientation_policy': 'planned_achieved' if getattr(self, 'arm_tracking_feedback', False) else 'measured_body_rotation',
                    'closure_latch_policy': getattr(self, 'closure_latch_policy', 'legacy'),
                    'close_integral_transfer': getattr(self, 'close_integral_transfer', 'reset'),
                    'close_frame_policy': getattr(self, 'close_frame_policy', 'fixed-body'),
                    'close_tracking_integral': {'enabled': getattr(self, 'close_tracking_integral', False),
                        'gain_per_s': .2, 'stationary_nominal_steps': 3, 'arm_speed_tolerance_rad_s': .05,
                        'max_dt_s': .05, 'phase': 'close_centered',
                        'reset_on_waypoint_change': getattr(self, 'close_integral_transfer', 'reset') == 'reset',
                        'waypoint_transfer': getattr(self, 'close_integral_transfer', 'reset'),
                        'unqualified_waypoint_change': 'reset',
                        'total_bias_bound_rad': .025, 'total_bias_slew_rad_per_step': .004,
                        'antiwindup': 'conditional_integration', 'commit': 'safety_accept_only',
                        'rejection': 'stop_before_next_action_in_close_centered',
                        'outside_close_rejection': 'original_safe_nominal_fallback'},
                    'servo': SERVO_CONFIG if self.controller == 'servo' else None,
                    'closure': ('centre_tracking_contact_stall_latch' if self.controller == 'servo' and
                                SERVO_CONFIG['close_policy'] == 'contact_stall' else
                                'centre_tracking_with_blocked_angle_preload' if self.controller in ('repaired', 'servo') else self.closure)}}

    def _place(self, held_q, rotation):
        held_q = self._stop_grasp_hold('place_entry', 'placement_started', held_q)
        if self.controller in ('repaired', 'servo'):
            return self._place_repaired(held_q, rotation)
        # Estimate the destination from the INITIAL RGB-D image, never object state.
        from PIL import Image
        from .grounding import predict_mask, default_model
        from .geometry import backproject
        rgb = Image.open(self.directory / 'rgb.png').convert('RGB')
        mask, detection = predict_mask(rgb, 'cardboard box', default_model(), self.device)
        (self.directory / 'box_detection.json').write_text(json.dumps(detection, indent=2) + '\n')
        depth = np.load(self.directory / 'depth.npy')
        intrinsics = json.loads((self.directory / 'intrinsics.json').read_text())
        cloud, valid = backproject(depth, intrinsics)
        points = cloud[valid & mask]
        if not len(points):
            return 'place_box_not_detected'
        world_from_camera = np.asarray(json.loads((self.directory / 'world_from_camera.json').read_text()))
        points = points @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
        box = np.median(points, axis=0)
        for phase, dz in [('transport', .10), ('place', .025)]:
            q, diagnostic = self._solve(box + [0, 0, dz], rotation, held_q)
            if q is None:
                return phase + '_ik'
            q[:, self.grip] = held_q[:, self.grip]
            if not self._move(q, phase, 90):
                return phase + '_terminated'
            held_q = q
        opened = held_q.clone()
        opened[:, self.grip] = self.open_joint_rad
        if not self._move(opened, 'release', 80):
            return 'release_terminated'
        retreat, _ = self._solve(box + [0, 0, .12], rotation, opened)
        if retreat is not None:
            retreat[:, self.grip] = self.open_joint_rad
            self._move(retreat, 'retreat', 90)
        for _ in range(45):
            if not self._step(retreat if retreat is not None else opened, 'settle'):
                break
        return 'placement_incomplete'

    def _place_repaired(self, held_q, rotation):
        held_q = self._stop_grasp_hold('place_repaired_entry', 'placement_started', held_q)
        if self.destination is None:
            return 'place_box_not_observed'
        box = np.asarray(self.destination)
        log = {'world_base_m': box.tolist(), 'ik': []}
        from isaaclab.utils.math import quat_apply_inverse
        positions = tensor(self.robot.data.body_pos_w)
        midpoint, pocket = self._pad_midpoint()
        midpoint_b = quat_apply_inverse(tensor(self.robot.data.body_quat_w)[:, self.body],
            positions.new_tensor(midpoint)[None] - positions[:, self.body])[0].cpu().numpy()
        rotation = self._body_rotation()
        start = midpoint.copy()
        # Match the arm's shoulder yaw as it moves around its fixed base.
        # Rotating about world Z preserves the grip's tilt relative to gravity.
        base_xy = tensor(self.robot.data.root_pos_w)[0, :2].cpu().numpy()
        a = start[:2] - base_xy
        b = box[:2] - base_xy
        yaw = np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b))
        targets = []
        for fraction in (1/3, 2/3, 1.):
            desired_mid = start * (1 - fraction) + (box + [0, 0, .14]) * fraction
            desired_r = Rotation.from_rotvec([0, 0, yaw * fraction]).as_matrix() @ rotation
            targets.append(('transport', desired_mid, desired_r, 45))
        targets.append(('place', box + [0, 0, .04], targets[-1][2], 100))
        for phase, desired_mid, desired_r, steps in targets:
            desired_pocket = desired_mid + desired_r @ (self.tool_offset_b - midpoint_b)
            q, diagnostic = self._solve(desired_pocket, desired_r, held_q)
            log['ik'].append({'phase': phase, **diagnostic})
            (self.directory / 'placement.json').write_text(json.dumps(log, indent=2) + '\n')
            if q is None:
                (self.directory / 'placement.json').write_text(json.dumps(log, indent=2) + '\n')
                return phase + '_ik'
            q[:, self.grip] = held_q[:, self.grip]
            if not self._move(q, phase, steps):
                return phase + '_terminated'
            held_q = q
            for _ in range(15):
                if not self._step(q, phase + '_hold'):
                    return phase + '_terminated'
        opened = held_q.clone()
        opened[:, self.grip] = self.open_joint_rad
        if not self._move(opened, 'release', 65):
            return 'release_terminated'
        # After opening, use this controller's open reference midpoint again.
        retreat, diagnostic = self._solve(box + [0, 0, .17], targets[-1][2], opened)
        log['ik'].append({'phase': 'retreat', **diagnostic})
        (self.directory / 'placement.json').write_text(json.dumps(log, indent=2) + '\n')
        if retreat is None:
            return 'retreat_ik'
        retreat[:, self.grip] = self.open_joint_rad
        self._move(retreat, 'retreat', 100)
        for _ in range(60):
            if not self._step(retreat, 'settle'):
                break
        return 'placement_incomplete'
