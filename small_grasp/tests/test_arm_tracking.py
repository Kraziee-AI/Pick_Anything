import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from small_grasp.arm_tracking import BoundedArmTrackingFeedback


class TestArmTracking(unittest.TestCase):
    def setUp(self):
        self.controller = BoundedArmTrackingFeedback([0, 1, 2, 3, 4])
        self.limits = np.tile([-1., 1.], (6, 1))

    def test_constant_load_dynamic_plant_reduces_offset_without_changing_jaw(self):
        # A separate first-order plant with a load offset. This establishes the
        # feedback sign/stability, not the outcome of an Isaac grasp.
        nominal = np.array([0., 0., .3, 0., 0., .7])
        load = np.array([-.004, .008, .04, .004, -.001, 0.])
        actual = nominal + load
        previous = np.zeros(5)
        for _ in range(100):
            decision = self.controller.propose(nominal, actual, self.limits)
            self.assertTrue(decision.active)
            self.assertEqual(decision.command[-1], nominal[-1])
            self.assertLessEqual(np.max(np.abs(decision.bias)), .025 + 1e-12)
            self.assertLessEqual(np.max(np.abs(decision.bias - previous)), .004 + 1e-12)
            self.controller.commit(decision, True)
            actual += .2 * (decision.command + load - actual)
            previous = decision.bias.copy()
        self.assertLess(np.linalg.norm(actual[:5] - nominal[:5]), .7 * np.linalg.norm(load[:5]))
        np.testing.assert_array_equal(decision.nominal, nominal)

    def test_reversing_motion_has_bounded_bias_and_no_integral_windup(self):
        nominal = np.zeros(6)
        actual = np.array([0., 0., .1, 0., 0., 0.])
        for _ in range(12):
            decision = self.controller.propose(nominal, actual, self.limits)
            self.controller.commit(decision, True)
        self.assertAlmostEqual(decision.bias[2], -.025)
        prior = decision.bias.copy()
        actual[2] = -.1
        decision = self.controller.propose(nominal, actual, self.limits)
        self.assertLessEqual(np.max(np.abs(decision.bias-prior)), .004 + 1e-12)
        for _ in range(20):
            self.controller.commit(decision, True)
            decision = self.controller.propose(nominal, actual, self.limits)
        self.assertAlmostEqual(decision.bias[2], .025)

    def test_rejected_limits_and_path_do_not_persist_a_bias(self):
        nominal = np.array([.999, 0., 0., 0., 0., .7])
        actual = nominal.copy(); actual[0] -= .05
        decision = self.controller.propose(nominal, actual, self.limits)
        self.assertEqual(decision.reason, 'compensated_command_outside_soft_limits')
        np.testing.assert_array_equal(decision.command, nominal)
        nominal[0] = 0.; actual[0] = -.05
        decision = self.controller.propose(nominal, actual, self.limits)
        self.controller.commit(decision, False)  # executor collision rejection
        np.testing.assert_array_equal(self.controller.bias, np.zeros(5))
        actual[0] = -.2
        decision = self.controller.propose(nominal, actual, self.limits)
        self.assertEqual(decision.reason, 'compensated_command_discontinuous')

    def test_nonfinite_feedback_is_rejected(self):
        decision = self.controller.propose(np.zeros(6), np.full(6, np.nan), self.limits)
        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, 'nonfinite_state_or_invalid_limits')

    def test_phase_exit_decays_bias_without_step_jump(self):
        nominal, actual = np.zeros(6), np.array([0., 0., .08, 0., 0., 0.])
        for _ in range(10):
            decision = self.controller.propose(nominal, actual, self.limits)
            self.controller.commit(decision, True)
        prior = decision.bias.copy()
        for _ in range(7):
            decision = self.controller.propose(nominal, actual, self.limits, tracking=False)
            self.assertLessEqual(np.max(np.abs(decision.bias-prior)), .004+1e-12)
            self.controller.commit(decision, True)
            prior = decision.bias.copy()
        np.testing.assert_array_equal(prior, np.zeros(5))

    def test_executor_approves_both_safety_paths_and_logs_nominal_reference(self):
        from small_grasp.sim_robot import GraspExecutor
        executor = GraspExecutor.__new__(GraspExecutor)
        executor.arm_tracking_feedback = True
        executor._arm_tracking = self.controller
        executor.arm_tracking_events = []
        executor.arm = [0, 1, 2, 3, 4]
        executor.env = SimpleNamespace(episode_length_buf=torch.tensor([3]))
        executor.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([[0., 0., .04, 0., 0., .7]]),
            soft_joint_pos_limits=torch.tensor(self.limits)[None]))
        executor._check_path = Mock(return_value=None)
        executor._mesh_at_q = Mock(return_value=None)
        executor._mesh_path = Mock(return_value=None)
        executor.scorer = Mock(side_effect=AssertionError('feedback must never read scoring'))
        nominal = torch.tensor([[0., 0., 0., 0., 0., .7]])
        result = executor._tracking_command(nominal, 'descend')
        self.assertLess(float(result[0, 2]), 0)
        torch.testing.assert_close(nominal, torch.tensor([[0., 0., 0., 0., 0., .7]]))
        self.assertEqual(executor._check_path.call_count, 2)  # nominal and compensated paths
        self.assertEqual(executor._mesh_path.call_count, 2)
        self.assertTrue(executor.arm_tracking_events[-1]['accepted'])
        # An unsafe new command cannot escape through the old nominal path.
        executor._mesh_path.side_effect = [None, {'reason': 'table_mesh'}]
        result = executor._tracking_command(nominal, 'descend')
        self.assertIs(result, nominal)
        self.assertEqual(executor.arm_tracking_events[-1]['reason'], 'collision_rejected')
        np.testing.assert_array_equal(self.controller.bias, np.zeros(5))
        executor.scorer.assert_not_called()

    def test_executor_unsafe_measured_origin_stops_even_with_zero_bias(self):
        from small_grasp.sim_robot import GraspExecutor
        executor = GraspExecutor.__new__(GraspExecutor)
        executor.arm_tracking_feedback = True
        executor._arm_tracking = self.controller
        executor.arm_tracking_events, executor.trace, executor.mesh_evidence = [], [], {}
        executor.arm, executor.grip = [0, 1, 2, 3, 4], 5
        executor.collision_ready, executor.collision_check = True, 'off'
        executor.env = SimpleNamespace(tracking=True, episode_length_buf=torch.tensor([3]), step=Mock())
        nominal = torch.zeros((1, 6))
        executor.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=nominal.clone()))
        executor._mesh_at_q = Mock(return_value={'reason': 'table_mesh', 'minimum_mesh_z_m': .0489995,
            'mesh': 'moving/Cube_02', 'body': 'moving', 'minimum_vertex_world_m': [.34, -.39, .0489995]})
        executor._check_path, executor._mesh_path = Mock(), Mock()
        executor._apply_grasp_hold = lambda target, phase: target
        executor._stop_grasp_hold = Mock()
        self.assertFalse(executor._step(nominal, 'descend'))
        executor.env.step.assert_not_called()
        executor._check_path.assert_not_called()
        executor._mesh_path.assert_not_called()
        self.assertFalse(executor.collision_ready)
        self.assertEqual(executor.motion_failure, 'descend_arm_tracking_origin_collision')
        self.assertEqual(executor.arm_tracking_events[-1]['reason'], 'origin_collision_stop')
        self.assertIsNone(executor.arm_tracking_events[-1]['applied_joint_target_rad'])
        self.assertFalse(executor.trace[-1]['command_issued'])

    def test_executor_off_returns_identical_tensor_without_state_or_geometry_reads(self):
        from small_grasp.sim_robot import GraspExecutor
        executor = GraspExecutor.__new__(GraspExecutor)
        nominal = torch.tensor([[0., 0., .2, 0., 0., .7]])
        self.assertIs(executor._tracking_command(nominal, 'descend'), nominal)


if __name__ == '__main__':
    unittest.main()
