"""CPU checks of servo upkeep lifecycle and actual outgoing joint commands."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from small_grasp.sim_robot import GraspExecutor, POCKET, SERVO_CONFIG


class HoldIntegrationTests(unittest.TestCase):
    def setUp(self):
        enabled = patch.dict(SERVO_CONFIG, bounded_hold_enabled=True)
        enabled.start()
        self.addCleanup(enabled.stop)

    def executor(self, force=8.):
        executor = GraspExecutor.__new__(GraspExecutor)
        executor.controller, executor.device = 'servo', 'cpu'
        executor.arm, executor.grip, executor.body = list(range(5)), 5, 0
        executor.robot = SimpleNamespace(data=SimpleNamespace(
            joint_pos=torch.tensor([[0., 0., 0., 0., 0., .41]]), joint_vel=torch.zeros((1, 6))))
        executor.env = SimpleNamespace(tracking=True, step_dt=1 / 30,
                                      episode_length_buf=torch.zeros(1, dtype=torch.long))
        executor.collision_ready, executor.collision_check = True, 'off'
        executor._grasp_hold, executor._grasp_hold_last_target_rad = None, None
        executor.hold_control_events, executor.trace = [], []
        executor.close_ik_attempts, executor.close_control_events = [], []
        executor.baseline, executor.max_rise, executor.hold_run = 0., 0., 0
        executor.lift_success, executor.executed = False, False
        executor.motion_goal, executor.tool_offset_b = None, np.asarray(POCKET)
        executor.tool_evidence = {'control_reference': 'fixed_pocket'}
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.zeros(3)))
        executor._body_rotation = Mock(return_value=np.eye(3))
        executor.scorer = Mock(side_effect=lambda: {
            'orange_z': 0., 'held': False, 'distance': 1.,
            'grip': float(executor.robot.data.joint_pos[0, 5]), 'claw_force': force})
        executor.commands = []
        def step(action):
            executor.commands.append(action.clone())
            executor.robot.data.joint_pos.copy_(action)
            executor.env.episode_length_buf += 1
        executor.env.step = step
        return executor

    def test_disabled_unlatched_and_other_controllers_preserve_old_command(self):
        for controller, enabled, latch in [('servo', False, .4), ('servo', True, None),
                                            ('repaired', True, .4), ('baseline', True, .4)]:
            executor = self.executor()
            executor.controller = controller
            caller = torch.full((1, 6), .7)
            with patch.dict(SERVO_CONFIG, bounded_hold_enabled=enabled):
                self.assertFalse(executor._start_grasp_hold(latch))
                self.assertIs(executor._apply_grasp_hold(caller, 'lift'), caller)
                executor.scorer.assert_not_called()
                self.assertTrue(executor._step(caller, 'lift'))
            self.assertEqual(executor.scorer.call_count, 1)  # Original post-step scoring only.
            torch.testing.assert_close(executor.commands[0], caller)
            self.assertEqual(executor.hold_control_events, [])

    def test_real_closure_only_arms_after_stall_latch(self):
        for force, expect_active in ((0., False), (10., True)):
            executor = self.executor(force)
            executor.robot.data.joint_pos[0, 5] = 1.4
            executor._solve = Mock(return_value=(None, {'valid': False}))
            commands = []
            def close_step(target, phase):
                self.assertEqual(phase, 'close_centered')
                self.assertIsNone(executor._grasp_hold)  # No upkeep during close.
                previous = float(executor.robot.data.joint_pos[0, 5])
                actual = max(.6, float(target[0, 5]))
                executor.robot.data.joint_pos[0, 5] = actual
                executor.robot.data.joint_vel[0, 5] = (actual - previous) * 30
                executor.env.episode_length_buf += 1
                commands.append(target.clone())
                return True
            executor._step = close_step
            executor._close_centered(np.zeros(3), np.eye(3))
            self.assertEqual(len(commands), 70)
            self.assertEqual(executor._grasp_hold is not None, expect_active)
            if expect_active:
                initial = executor.close_control_events[0]['locked_joint_target_rad']
                self.assertAlmostEqual(executor._grasp_hold.initial_target, initial)

    def test_move_overrides_each_jaw_command_without_mutating_caller_or_arm(self):
        executor = self.executor()
        executor._start_grasp_hold(.4)
        target = torch.full((1, 6), .2)
        target[0, 5] = .8  # Deliberately stale interpolation goal.
        original = target.clone()
        self.assertTrue(executor._move(target, 'lift', 100))
        torch.testing.assert_close(target, original)
        jaws = [float(command[0, 5]) for command in executor.commands]
        self.assertTrue(all(b <= a + 1e-7 for a, b in zip(jaws, jaws[1:])))
        self.assertAlmostEqual(jaws[0], .4, places=6)
        self.assertAlmostEqual(jaws[-1], .36, places=6)
        self.assertAlmostEqual(float(executor.commands[0][0, 0]), .2 / 98, places=6)
        self.assertAlmostEqual(float(executor.commands[-1][0, 0]), .2, places=6)
        observations = [event for event in executor.hold_control_events
                        if event['event'] == 'bounded_hold_observation']
        self.assertEqual(len(observations), 100)
        self.assertEqual(observations[-1]['reason'], 'closure_budget_exhausted')
        self.assertLessEqual(observations[-1]['extra_closure_rad'], .04 + 1e-12)
        for key in ('claw_force_n', 'joint_position_rad', 'joint_velocity_rad_s',
                    'joint_target_rad', 'caller_joint_target_rad', 'phase', 'observation_step'):
            self.assertIn(key, observations[-1])

    def test_only_three_phases_use_robot_only_sample_and_release_stops(self):
        executor = self.executor()
        class RobotOnly(dict):
            def __getitem__(self, key):
                if key != 'claw_force':
                    raise AssertionError('Privileged signal requested by upkeep')
                return 8.
        executor.scorer = Mock(return_value=RobotOnly())
        executor._start_grasp_hold(.4)
        caller = torch.full((1, 6), .7)
        for phase in ('hold', 'lift', 'lift_hold'):
            commanded = executor._apply_grasp_hold(caller, phase)
            self.assertIsNot(commanded, caller)
            self.assertLessEqual(float(commanded[0, 5]), .4 + 1e-7)
        calls = executor.scorer.call_count
        self.assertIs(executor._apply_grasp_hold(caller, 'release'), caller)
        self.assertIsNone(executor._grasp_hold)
        self.assertEqual(executor.scorer.call_count, calls)
        self.assertIs(executor._apply_grasp_hold(caller, 'hold'), caller)
        self.assertEqual(executor.hold_control_events[-1]['phase'], 'release')

    def test_no_contact_never_chases_closed(self):
        executor = self.executor(force=0)
        executor._start_grasp_hold(.4)
        target = torch.full((1, 6), .1)
        for _ in range(100):
            executor._step(target, 'lift')
        self.assertTrue(all(abs(float(command[0, 5]) - .4) < 1e-7
                            for command in executor.commands))
        self.assertEqual(executor.hold_control_events[-1]['reason'], 'contact_absent')
        self.assertEqual(executor.hold_control_events[-1]['extra_closure_rad'], 0.)

    def test_all_other_phases_stop_without_reading_contact(self):
        for phase in ('open', 'approach', 'descend', 'close_centered', 'transport',
                      'place', 'release', 'retreat', 'settle', 'transport_hold'):
            executor = self.executor()
            executor._start_grasp_hold(.4)
            caller = torch.full((1, 6), .7)
            self.assertIs(executor._apply_grasp_hold(caller, phase), caller)
            self.assertIsNone(executor._grasp_hold)
            executor.scorer.assert_not_called()
            self.assertEqual(executor.hold_control_events[-1]['reason'], 'phase_outside_scope')

    def test_final_target_passes_to_place_and_release_opens_normally(self):
        executor = self.executor()
        executor._start_grasp_hold(.4)
        stale = torch.full((1, 6), .4)
        executor._move(stale, 'lift', 12)
        last = executor._grasp_hold.target
        carried = executor._stop_grasp_hold('lift_hold_end', 'scope_complete', stale)
        self.assertIsNone(executor._grasp_hold)
        self.assertAlmostEqual(float(carried[0, 5]), last, places=6)
        torch.testing.assert_close(stale, torch.full((1, 6), .4))
        executor._place_repaired = Mock(return_value='placement_incomplete')
        executor._place(stale, np.eye(3))
        place_input = executor._place_repaired.call_args.args[0]
        self.assertAlmostEqual(float(place_input[0, 5]), last, places=6)
        # Even with physical jaw lag, transport preserves the last command.
        executor.robot.data.joint_pos[0, 5] = last + .025
        before = len(executor.commands)
        executor._move(place_input, 'transport', 3)
        self.assertTrue(all(abs(float(command[0, 5]) - last) < 1e-7
                            for command in executor.commands[before:]))
        opened = place_input.clone()
        opened[:, 5] = 1.4
        executor._move(opened, 'release', 4)
        self.assertAlmostEqual(float(executor.commands[-1][0, 5]), 1.4, places=6)
        self.assertIsNone(executor._grasp_hold)
        # Also cover a direct _place call while upkeep is still active.
        executor._start_grasp_hold(.37)
        executor._place(stale, np.eye(3))
        self.assertIsNone(executor._grasp_hold)
        self.assertAlmostEqual(float(executor._place_repaired.call_args.args[0][0, 5]), .37, places=6)

    def test_direct_repaired_placement_entry_and_failures_clear_hold(self):
        executor = self.executor()
        executor._start_grasp_hold(.4)
        executor.destination = None
        self.assertEqual(executor._place_repaired(torch.zeros((1, 6)), np.eye(3)),
                         'place_box_not_observed')
        self.assertIsNone(executor._grasp_hold)
        self.assertEqual(executor.hold_control_events[-1]['phase'], 'place_repaired_entry')
        executor._start_grasp_hold(.4)
        executor.env.step = Mock(side_effect=RuntimeError('test step error'))
        with self.assertRaisesRegex(RuntimeError, 'test step error'):
            executor._step(torch.zeros((1, 6)), 'hold')
        self.assertIsNone(executor._grasp_hold)
        self.assertEqual(executor.hold_control_events[-1]['reason'], 'step_exception')
        executor._start_grasp_hold(.4)
        executor.env.tracking = False
        self.assertFalse(executor._step(torch.zeros((1, 6)), 'hold'))
        self.assertIsNone(executor._grasp_hold)

    def test_environment_termination_and_execute_finally_clear_hold(self):
        executor = self.executor()
        executor._start_grasp_hold(.4)
        executor.env.step = lambda _action: setattr(executor.env, 'tracking', False)
        self.assertFalse(executor._step(torch.zeros((1, 6)), 'lift'))
        self.assertIsNone(executor._grasp_hold)
        self.assertEqual(executor.hold_control_events[-1]['reason'], 'environment_terminated')
        executor.env.tracking = True
        def fail_after_arming(*_args):
            executor._start_grasp_hold(.4)
            raise RuntimeError('test execute error')
        executor._execute = fail_after_arming
        with self.assertRaisesRegex(RuntimeError, 'test execute error'):
            executor.execute([], 'single')
        self.assertIsNone(executor._grasp_hold)
        self.assertEqual(executor.hold_control_events[-1]['reason'], 'execution_finished')

    def test_real_execute_return_evidence_includes_finally_stop_when_lift_ik_fails(self):
        executor = self.executor()
        executor.env.terminal_result = None
        executor.open_joint_rad = 1.4
        executor.runtime_collision = None
        executor.target_centre, executor.destination, executor.contact_frame = None, None, 'open'
        executor.closure = 'first_contact'
        executor.collision_evidence, executor.mesh_evidence = {}, {}
        executor.motion_diagnostics, executor.opening_plan_stages = [], []
        pose = np.eye(4)
        pose[:3, 3] = [.25, -.4, .2]
        candidate = {'candidate_id': 'test', 'T_world_grasp': pose.tolist(), 'depth_m': 0., 'width_m': .075}
        q = torch.zeros((1, 6))
        selected = (0., 0, candidate, pose[:3, 3], q.clone(), q.clone(), np.eye(3), False)
        executor.ik_attempts = [{'candidate': 0, 'accepted': True, 'symmetric_flip': False,
            'network_pocket_world_m': pose[:3, 3].tolist(), 'executed_pocket_offset_m': [0., 0., 0.],
            'opening_stage_index': 0}]
        executor.servo_routes = {(0, False): (q.clone(), np.array([.25, -.4, .2]))}
        executor.servo_descents = {(0, False): [(pose[:3, 3], q.clone())]}
        executor._grasp_axes_in_body = Mock(return_value=np.eye(3))
        executor._plan_servo_openings = Mock(return_value=selected)
        executor._move, executor._arrive = Mock(return_value=True), Mock(return_value=True)
        executor._cartesian_descend = Mock(return_value=True)
        executor._solve = Mock(return_value=(None, {'valid': False}))
        def close(*_args):
            target = torch.zeros((1, 6))
            target[:, 5] = .4
            executor._start_grasp_hold(.4)
            return target
        executor._close_centered = close
        with tempfile.TemporaryDirectory() as temporary:
            executor.directory = Path(temporary)
            result = executor.execute([candidate], 'single')
            trace = json.loads((executor.directory / 'execution_trace.json').read_text())
        self.assertEqual(len(trace), 15)
        self.assertFalse(result['chosen']['lift_ik']['valid'])
        events = result['hold_control_evidence']['events']
        self.assertIs(events, executor.hold_control_events)
        self.assertEqual(events[-1]['event'], 'bounded_hold_stopped')
        self.assertEqual(events[-1]['phase'], 'execute_end')
        self.assertIsNone(executor._grasp_hold)
        self.assertAlmostEqual(executor._grasp_hold_last_target_rad,
                               events[-1]['final_joint_target_rad'])
        # Returned evidence remains valid JSON after execute's finally block.
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
