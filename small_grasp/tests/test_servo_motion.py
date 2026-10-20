"""CPU regression checks for measured arrival and finite Cartesian execution."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from small_grasp.sim_robot import GraspExecutor, SERVO_CONFIG, _GripperMeshClearance, _ClosureContactLatch, POCKET


class ServoMotionTests(unittest.TestCase):
    def executor(self):
        executor = GraspExecutor.__new__(GraspExecutor)
        executor.controller = 'servo'
        executor.open_joint_rad = SERVO_CONFIG['open_joint_rad']
        executor.tool_offset_b = np.asarray(POCKET)
        executor.tool_evidence = {'control_reference': 'fixed_pocket'}
        executor.env = SimpleNamespace(tracking=True)
        executor.robot = SimpleNamespace(data=SimpleNamespace(
            joint_pos=torch.zeros((1, 6)), body_pos_w=torch.zeros((1, 2, 3)),
            body_quat_w=torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]])))
        executor.arm, executor.grip, executor.body = [0, 1, 2, 3, 4], 5, 0
        executor.collision_ready, executor.collision_check = True, 'off'
        executor.motion_diagnostics, executor.ik_attempts = [], []
        executor.motion_failure = None
        executor.servo_routes = {}
        executor.servo_descents = {}
        executor.opening_plan_stages = []
        executor._mesh_at_q = Mock(return_value=None)
        executor._mesh_path = Mock(return_value=None)
        executor._body_rotation = Mock(return_value=np.eye(3))
        executor.scorer = Mock(side_effect=AssertionError('GT/scoring must not control motion'))
        return executor

    def test_arrival_uses_live_tcp_and_requires_consecutive_samples(self):
        executor = self.executor()
        executor._pad_midpoint = Mock(side_effect=[(np.zeros(3), np.array([.03, 0, 0])),
            (np.zeros(3), np.zeros(3)), (np.zeros(3), np.zeros(3)), (np.zeros(3), np.zeros(3))])
        executor._step = Mock(return_value=True)
        self.assertTrue(executor._arrive(torch.zeros((1, 6)), np.zeros(3), np.eye(3), 'descend'))
        self.assertEqual(executor._step.call_count, 3)
        self.assertEqual(executor.motion_diagnostics[-1]['consecutive_arrival_steps'], 3)
        executor.scorer.assert_not_called()

    def test_arrival_rejects_persistent_real_error_after_finite_wait(self):
        executor = self.executor()
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.array([.03, 0, 0])))
        executor._step = Mock(return_value=True)
        self.assertFalse(executor._arrive(torch.zeros((1, 6)), np.zeros(3), np.eye(3), 'descend'))
        self.assertEqual(executor._step.call_count, SERVO_CONFIG['arrival_max_steps'])
        self.assertEqual(executor.motion_failure, 'descend_arrival_timeout')
        executor.scorer.assert_not_called()

    def test_cartesian_path_is_pre_solved_straight_and_successively_seeded(self):
        executor = self.executor()
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.array([0., 0., .2])))
        requests = []
        def solve(point, rotation, seed, prefer_near=False):
            self.assertTrue(prefer_near)
            requests.append((np.array(point), seed.clone()))
            return torch.full((1, 6), len(requests) * .01), {'valid': True}
        executor._solve = solve
        executor._check_path = Mock(return_value=None)
        executor._move = Mock(return_value=True)
        executor._arrive = Mock(return_value=True)
        goal = np.array([.04, 0., .2])
        self.assertTrue(executor._cartesian_descend(goal, np.eye(3), torch.zeros((1, 6))))
        self.assertEqual(len(requests), 5)
        for index, (point, seed) in enumerate(requests):
            np.testing.assert_allclose(point, [(.008 * (index + 1)), 0., .2])
            np.testing.assert_allclose(seed[0, :5], index * .01)
        self.assertEqual(executor._move.call_count, 5)
        executor.scorer.assert_not_called()

    def test_failed_cartesian_ik_issues_no_partial_descent(self):
        executor = self.executor()
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.array([0., 0., .2])))
        executor._solve = Mock(side_effect=[(torch.zeros((1, 6)), {'valid': True}), (None, {'valid': False})])
        executor._check_path = Mock(return_value=None)
        executor._move = Mock()
        self.assertFalse(executor._cartesian_descend(np.array([.04, 0., .2]), np.eye(3), torch.zeros((1, 6))))
        executor._move.assert_not_called()
        self.assertEqual(executor.motion_failure, 'descend_cartesian_ik')

    def test_lift_preflight_is_not_tied_to_capsule_switch(self):
        executor = self.executor()
        executor.target_centre, executor.destination, executor.contact_frame = None, None, 'open'
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.zeros(3)))
        count = 0
        def solve(point, rotation, seed=None):
            nonlocal count
            count += 1
            if point[2] > .3:
                return None, {'valid': False, 'selected_seed_gates': {'joint_limits': False}}
            return torch.zeros((1, 6)), {'valid': True, 'achieved_rotation': np.eye(3).tolist(),
                                       'orientation_error_deg': 0.}
        executor._solve = solve
        executor._check_path = Mock(side_effect=AssertionError('capsules are disabled'))
        pose = np.eye(4)
        pose[:3, 3] = [.25, 0., .2]
        math_module = SimpleNamespace(quat_apply_inverse=lambda _quat, vector: vector)
        with patch.dict(sys.modules, {'isaaclab.utils.math': math_module}):
            planned = executor._plan_repaired([{'T_world_grasp': pose, 'depth_m': 0., 'width_m': .075}], np.eye(3))
        self.assertIsNone(planned)
        self.assertEqual(count, 8)
        self.assertTrue(all(item['rejected'] == 'nominal_lift_ik' for item in executor.ik_attempts))
        executor.scorer.assert_not_called()

    def test_separate_mesh_hulls_preserve_open_gap_and_detect_observed_box(self):
        from scipy.spatial import ConvexHull
        vertices = np.array([[x, y, z] for x in (-.01, .01) for y in (-.01, .01) for z in (-.01, .01)])
        triangles = vertices[ConvexHull(vertices).simplices]
        geometry = SimpleNamespace(meshes={label: (triangles, ['cube'] * len(triangles))
                                           for label in ('fixed', 'moving')},
            parent_pivot=np.array([.1, 0., 0.]), child_pivot=np.zeros(3),
            parent_rotation=np.eye(3), child_rotation=np.eye(3), axis=np.array([0., 0., 1.]),
            moving_to_fixed=lambda points, q: points + [.1, 0., 0.])
        pocket = np.asarray(POCKET) + [0., -.4, .3]
        # A point in the gap must not collide with an enclosing whole-gripper hull.
        clearance = _GripperMeshClearance(geometry, [[.05, -.4, .3]])
        self.assertIsNone(clearance.check(pocket, np.eye(3), 0.))
        clearance.box_points = np.array([[.1, -.4, .3]])
        self.assertEqual(clearance.check(pocket, np.eye(3), 0.)['reason'], 'observed_box_mesh')
        clearance.box_points = np.empty((0, 3))
        low_pocket = np.asarray(POCKET) + [0., -.4, .055]
        table = clearance.check(low_pocket, np.eye(3), 0.)
        self.assertEqual(table['reason'], 'table_mesh')
        self.assertAlmostEqual(table['required_vertical_correction_m'], .004)

    def test_contact_latch_requires_slowdown_and_consecutive_tracking_error(self):
        latch = _ClosureContactLatch(.74)
        for _ in range(8):
            self.assertIsNone(latch.observe(12., .6, .5, -.7))
        self.assertEqual(latch.consecutive, 0)
        self.assertIsNone(latch.observe(10., .6, .55, -.3))
        self.assertIsNone(latch.observe(10., .6, .55, -.3))
        # Losing sustained contact resets the whole consecutive gate.
        self.assertIsNone(latch.observe(0., .6, .55, -.3))
        for _ in range(2):
            self.assertIsNone(latch.observe(10., .6, .55, -.3))
        event = latch.observe(10., .6, .55, -.3)
        self.assertEqual(event['event'], 'contact_stall_latched')
        self.assertAlmostEqual(latch.target, .58)
        self.assertIsNone(latch.observe(50., .4, .05, 0.))
        self.assertAlmostEqual(latch.target, .58)

    def test_centered_closure_keeps_budget_and_locked_target_without_gt(self):
        executor = self.executor()
        executor.env.step_dt = 1 / 30
        executor.env.episode_length_buf = torch.zeros(1, dtype=torch.long)
        executor.robot.data.joint_pos[0, executor.grip] = 1.4
        executor.robot.data.joint_vel = torch.zeros((1, 6))
        executor.close_ik_attempts, executor.close_control_events = [], []
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.zeros(3)))
        executor._solve = Mock(return_value=(None, {'valid': False}))
        class RobotOnlySample(dict):
            def __getitem__(self, key):
                if key != 'claw_force':
                    raise AssertionError('Closure read a non-robot score')
                return 10.
        executor.scorer = Mock(return_value=RobotOnlySample())
        commands = []
        def step(target, phase):
            self.assertEqual(phase, 'close_centered')
            command = float(target[0, executor.grip])
            previous = float(executor.robot.data.joint_pos[0, executor.grip])
            actual = max(.6, command)
            executor.robot.data.joint_pos[0, executor.grip] = actual
            executor.robot.data.joint_vel[0, executor.grip] = (actual - previous) / executor.env.step_dt
            executor.env.episode_length_buf += 1
            commands.append(command)
            return True
        executor._step = step
        closed = executor._close_centered(np.zeros(3), np.eye(3))
        self.assertEqual(len(commands), 70)
        self.assertEqual(executor._solve.call_count, 14)
        self.assertEqual(len(executor.close_control_events), 1)
        event = executor.close_control_events[0]
        self.assertEqual(event['event'], 'contact_stall_latched')
        self.assertAlmostEqual(float(closed[0, executor.grip]), .58, places=6)
        self.assertTrue(all(abs(command - .58) < 1e-6 for command in commands[event['close_step']:]))

    def test_box_frame_is_latest_accepted_matching_destination_not_target(self):
        executor = self.executor()
        with tempfile.TemporaryDirectory() as temporary:
            executor.directory = Path(temporary)
            executor.observation_dir = executor.directory / 'target'
            executor.destination = [.4, -.4, .041]
            records = [{'view': str(executor.observation_dir), 'box_quality': {'accepted': False}},
                {'view': str(executor.directory / 'box_01'), 'box_quality': {
                    'accepted': True, 'world_base_m': executor.destination}},
                {'view': str(executor.directory / 'box_02'), 'box_quality': {
                    'accepted': True, 'world_base_m': [.6, -.4, .041]}},
                {'view': str(executor.directory / 'box_03'), 'box_quality': {
                    'accepted': True, 'world_base_m': executor.destination}}]
            (executor.directory / 'scan.json').write_text(json.dumps({'views': records}))
            self.assertEqual(executor._box_observation(), executor.directory / 'box_03')
            self.assertEqual(executor.observation_dir, executor.directory / 'target')
            executor.destination = [.7, -.4, .041]
            with self.assertRaisesRegex(ValueError, 'No accepted box observation'):
                executor._box_observation()

    def test_narrow_open_tool_matches_usd_midpoint_and_world_grasp_axes(self):
        from scipy.spatial.transform import Rotation
        from small_grasp.gripper_geometry import DEFAULT_USD, FIXED_REFERENCE_M, MOVING_REFERENCE_M
        executor = self.executor()
        executor.robot.cfg = SimpleNamespace(spawn=SimpleNamespace(usd_path=str(DEFAULT_USD)))
        executor._configure_tool(.8)
        self.assertEqual(executor.open_joint_rad, .8)
        np.testing.assert_allclose(executor.tool_offset_b, [.028760935076280567, -.00015989457591530503,
                                                          -.09012291161221168], atol=1e-10)
        self.assertAlmostEqual(np.linalg.norm(np.asarray(POCKET) - executor.tool_offset_b), .022681733256818697)
        g = executor.gripper_geometry
        joint_rotation = g.parent_rotation @ Rotation.from_rotvec(g.axis * .8).as_matrix() @ g.child_rotation.T
        body_rotation = Rotation.from_euler('xyz', [.3, -.2, .5]).as_matrix()
        body_position = np.array([.2, -.4, .2])
        jaw_position = body_position + body_rotation @ (g.parent_pivot - joint_rotation @ g.child_pivot)
        executor.jaw, executor.contact_frame = 1, 'open'
        executor.robot.data.body_pos_w = torch.tensor([[body_position.tolist(), jaw_position.tolist()]], dtype=torch.float64)
        executor.robot.data.body_quat_w = torch.tensor([[Rotation.from_matrix(body_rotation).as_quat().tolist(),
            Rotation.from_matrix(body_rotation @ joint_rotation).as_quat().tolist()]], dtype=torch.float64)
        def apply(quaternions, points):
            result = Rotation.from_quat(quaternions.numpy()).apply(points.numpy())
            return torch.as_tensor(result, dtype=points.dtype)
        def inverse(quaternions, points):
            result = Rotation.from_quat(quaternions.numpy()).inv().apply(points.numpy())
            return torch.as_tensor(result, dtype=points.dtype)
        with patch.dict(sys.modules, {'isaaclab.utils.math': SimpleNamespace(quat_apply=apply, quat_apply_inverse=inverse)}):
            midpoint, tool = executor._pad_midpoint()
            axes_body = executor._grasp_axes_in_body()
        np.testing.assert_allclose(tool, midpoint, atol=1e-9)
        np.testing.assert_allclose(tool, body_position + body_rotation @ executor.tool_offset_b, atol=1e-9)
        np.testing.assert_allclose(axes_body.T @ axes_body, np.eye(3), atol=1e-9)
        network_rotation = Rotation.from_euler('xyz', [.2, .4, -.3]).as_matrix()
        planned_rotation = network_rotation @ axes_body.T
        network_point = np.array([.25, -.4, .07])
        planned_body = network_point - planned_rotation @ executor.tool_offset_b
        fixed = planned_body + planned_rotation @ FIXED_REFERENCE_M
        moving = planned_body + planned_rotation @ g.moving_to_fixed(MOVING_REFERENCE_M, .8)
        np.testing.assert_allclose((fixed + moving) / 2, network_point, atol=1e-9)
        np.testing.assert_allclose(planned_rotation @ axes_body, network_rotation, atol=1e-9)
        executor._configure_tool(1.4)
        self.assertIs(executor.gripper_geometry, g)
        np.testing.assert_array_equal(executor.tool_offset_b, POCKET)
        self.assertEqual(executor.tool_evidence['control_reference'], 'fixed_pocket')
        # Repaired remains the original opening and fixed reference point.
        executor.controller = 'repaired'
        executor._configure_tool()
        self.assertEqual(executor.open_joint_rad, 1.4)
        np.testing.assert_array_equal(executor.tool_offset_b, POCKET)

    def test_mesh_fk_keeps_fixed_pocket_while_control_tool_is_different(self):
        executor = self.executor()
        executor.tool_offset_b = np.array([.02, 0., -.09])
        executor.device, executor.ik_body = 'cpu', 0
        executor.coordinates = torch.arange(6)
        executor.model = SimpleNamespace(joint_q=torch.zeros(6), joint_qd=torch.zeros(6))
        executor.mesh_state = SimpleNamespace(body_q=torch.tensor([[.2, -.4, .3, 0., 0., 0., 1.]]))
        executor.mesh_clearance = SimpleNamespace(check=Mock(return_value=None))
        warp = SimpleNamespace(to_torch=lambda value: value, float32=torch.float32,
                               array=lambda value, **kwargs: torch.as_tensor(value))
        with patch.dict(sys.modules, {'newton': SimpleNamespace(eval_fk=Mock()), 'warp': warp}):
            GraspExecutor._mesh_at_q(executor, torch.zeros((1, 6)))
        pocket = executor.mesh_clearance.check.call_args.args[0]
        np.testing.assert_allclose(pocket, [.2 + POCKET[0], -.4 + POCKET[1], .3 + POCKET[2]], atol=1e-7)
        self.assertGreater(np.linalg.norm(pocket - ([.2, -.4, .3] + executor.tool_offset_b)), .01)

    def test_wide_feasible_plan_does_not_try_narrow_opening(self):
        executor = self.executor()
        executor.robot.data.joint_pos[0, executor.grip] = 1.4
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.zeros(3)))
        executor._grasp_axes_in_body = Mock(return_value=np.eye(3))
        selected = (0., 0, {'candidate_id': 'wide'}, np.zeros(3), None, None, np.eye(3), False)
        executor._plan_repaired = Mock(return_value=selected)
        executor._switch_opening_tool, executor._move = Mock(), Mock()
        self.assertIs(executor._plan_servo_openings([selected[2]]), selected)
        executor._switch_opening_tool.assert_not_called()
        executor._move.assert_not_called()
        executor._plan_repaired.assert_called_once()
        self.assertEqual(executor.opening_plan_stages[0]['outcome'], 'selected')
        self.assertEqual(executor.opening_plan_stages[0]['open_joint_rad'], 1.4)

    def test_narrow_opening_requires_all_previous_plans_rejected(self):
        executor = self.executor()
        executor.robot.data.joint_pos[0, executor.grip] = 1.4
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.zeros(3)))
        executor._grasp_axes_in_body = Mock(return_value=np.eye(3))
        selected = (0., 0, {'candidate_id': 'narrow'}, np.zeros(3), None, None, np.eye(3), False)
        planned_openings = []
        def plan(_candidates, _axes):
            planned_openings.append(executor.open_joint_rad)
            executor.ik_attempts.append({'candidate': 0, 'rejected': 'table_mesh'})
            return selected if executor.open_joint_rad == .8 else None
        executor._plan_repaired = plan
        def switch(opening):
            executor.open_joint_rad = opening
        executor._switch_opening_tool = Mock(side_effect=switch)
        def move(target, phase, steps):
            self.assertEqual(phase, 'open_adjust')
            self.assertEqual(steps, SERVO_CONFIG['opening_adjust_steps'])
            executor.robot.data.joint_pos.copy_(target)
            return True
        executor._move = Mock(side_effect=move)
        self.assertIs(executor._plan_servo_openings([selected[2]]), selected)
        self.assertEqual(planned_openings, [1.4, 1., .8])
        self.assertEqual([c.args[0] for c in executor._switch_opening_tool.call_args_list], [1., .8])
        self.assertEqual([s['outcome'] for s in executor.opening_plan_stages],
                         ['all_candidate_plans_rejected', 'all_candidate_plans_rejected', 'selected'])
        self.assertEqual([a['open_joint_rad'] for a in executor.ik_attempts], [1.4, 1., .8])

    def test_selected_plan_execution_failure_does_not_retry_grasp_or_opening(self):
        executor = self.executor()
        executor.robot.data.joint_pos[0, executor.grip] = 1.4
        executor.env.terminal_result = None
        executor.runtime_collision = None
        executor.target_centre, executor.destination, executor.contact_frame = None, None, 'open'
        executor.closure = 'first_contact'
        executor.collision_evidence, executor.mesh_evidence = {}, {}
        executor.trace, executor.close_ik_attempts, executor.close_control_events = [], [], []
        executor.lift_success = executor.executed = False
        executor.max_rise = 0.
        pose = np.eye(4)
        pose[:3, 3] = [.25, -.4, .2]
        candidate = {'candidate_id': 'selected', 'T_world_grasp': pose.tolist(), 'depth_m': 0., 'width_m': .075}
        q = torch.zeros((1, 6))
        selected = (0., 0, candidate, pose[:3, 3], q.clone(), q.clone(), np.eye(3), False)
        executor.ik_attempts = [{'candidate': 0, 'candidate_id': 'selected', 'accepted': True,
            'symmetric_flip': False, 'network_pocket_world_m': pose[:3, 3].tolist(),
            'executed_pocket_offset_m': [0., 0., 0.], 'opening_stage_index': 0}]
        executor.servo_routes[(0, False)] = (q.clone(), np.array([.25, -.4, .2]))
        executor.servo_descents[(0, False)] = [(pose[:3, 3], q.clone())]
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.zeros(3)))
        executor._grasp_axes_in_body = Mock(return_value=np.eye(3))
        executor._plan_servo_openings = Mock(return_value=selected)
        executor._switch_opening_tool = Mock()
        executor._move, executor._arrive, executor._cartesian_descend = Mock(return_value=True), Mock(return_value=True), Mock(return_value=True)
        executor._close_centered = Mock(return_value=q.clone())
        def stop(_target, _phase):
            executor.env.tracking = False
            return False
        executor._step = stop
        with tempfile.TemporaryDirectory() as temporary:
            executor.directory = Path(temporary)
            result = executor.execute([candidate], 'single')
        executor._plan_servo_openings.assert_called_once()
        executor._cartesian_descend.assert_called_once()
        executor._close_centered.assert_called_once()
        executor._switch_opening_tool.assert_not_called()
        self.assertFalse(result['grasp_lift_success'])

    def test_approach_waits_for_measured_velocity_settle(self):
        executor = self.executor()
        executor.robot.data.joint_vel = torch.full((1, 6), .2)
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.zeros(3)))
        calls = []
        def settle(_target, _phase):
            calls.append(1)
            executor.robot.data.joint_vel.fill_(.07 if len(calls) == 1 else .02)
            return True
        executor._step = settle
        self.assertTrue(executor._arrive(torch.zeros((1, 6)), np.zeros(3), np.eye(3), 'approach'))
        self.assertEqual(len(calls), 4)
        self.assertLess(executor.motion_diagnostics[-1]['max_arm_joint_speed_rad_s'], .05)
        executor.scorer.assert_not_called()

    def test_prepared_path_does_not_resolve_and_commands_remain_continuous(self):
        executor = self.executor()
        executor._solve = Mock(side_effect=AssertionError('Accepted path must not be re-solved after approach'))
        executor._check_path = Mock(return_value=None)
        executor._arrive = Mock(return_value=True)
        commands = []
        # Deliberately leave measured joints lagging at zero throughout the test.
        executor._step = lambda target, phase: commands.append(target.clone()) or True
        goals = []
        for position in (.04, .08):
            q = torch.full((1, 6), position)
            q[:, executor.grip] = executor.open_joint_rad
            goals.append((np.array([position, 0., .2]), q))
        self.assertTrue(executor._cartesian_descend(goals[-1][0], np.eye(3), goals[-1][1], prepared=goals))
        executor._solve.assert_not_called()
        self.assertEqual(len(commands), 16)
        self.assertAlmostEqual(float(commands[7][0, 0]), .04, places=6)
        self.assertGreater(float(commands[8][0, 0]), .04)
        self.assertAlmostEqual(float(commands[-1][0, 0]), .08, places=6)
        executor.scorer.assert_not_called()

    def test_cartesian_joint_jump_is_rejected_before_any_command(self):
        executor = self.executor()
        executor._solve = Mock(return_value=(torch.full((1, 6), .3), {'valid': True}))
        executor._move = Mock()
        path, log = executor._plan_cartesian_descent(np.array([0., 0., .2]), torch.zeros((1, 6)),
            np.array([.008, 0., .2]), np.eye(3))
        self.assertIsNone(path)
        self.assertEqual(log['failure_reason'], 'descend_cartesian_joint_jump')
        executor._move.assert_not_called()
        executor.scorer.assert_not_called()

    def test_undescendable_candidate_is_rejected_before_plan_selection(self):
        executor = self.executor()
        executor.target_centre, executor.destination, executor.contact_frame = None, None, 'open'
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.zeros(3)))
        def solve(point, rotation, seed=None, **kwargs):
            return torch.zeros((1, 6)), {'valid': True, 'achieved_rotation': np.eye(3).tolist(),
                'orientation_error_deg': 0., 'achieved_pocket_world_m': np.asarray(point).tolist()}
        executor._solve = solve
        executor._move = Mock(side_effect=AssertionError('Static planning must not command movement'))
        def descent(start, seed, point, rotation):
            if point[0] < .26:
                return None, {'failure_reason': 'descend_cartesian_collision', 'collision': {'reason': 'observed_box_mesh'}}
            return [(point, seed)], {'valid': True}
        executor._plan_cartesian_descent = Mock(side_effect=descent)
        candidates = []
        for x in (.25, .3):
            pose = np.eye(4)
            pose[:3, 3] = [x, 0., .2]
            candidates.append({'T_world_grasp': pose, 'depth_m': 0., 'width_m': .075})
        math_module = SimpleNamespace(quat_apply_inverse=lambda _quat, vector: vector)
        with patch.dict(sys.modules, {'isaaclab.utils.math': math_module}):
            plan = executor._plan_repaired(candidates, np.eye(3))
        self.assertEqual(plan[1], 1)
        self.assertTrue(all(x.get('rejected') == 'descend_cartesian_collision'
                            for x in executor.ik_attempts if x['candidate'] == 0))
        self.assertNotIn((0, False), executor.servo_descents)
        self.assertIn((1, False), executor.servo_descents)
        executor._move.assert_not_called()
        executor.scorer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
