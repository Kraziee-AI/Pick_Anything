"""CPU checks for bounded-budget IK retry, unchanged gates and DOF mapping."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from small_grasp.sim_robot import GraspExecutor, SERVO_CONFIG, _soft_limit_dof_arrays


class SoftLimitFallbackTests(unittest.TestCase):
    def executor(self):
        executor = GraspExecutor.__new__(GraspExecutor)
        executor.controller = 'servo'
        executor.ik_limit_objective = SimpleNamespace(joint_limit_lower=object(), joint_limit_upper=object(), weight=1.)
        executor.ik_soft_lower, executor.ik_soft_upper = object(), object()
        executor.ik_fallback_parameters = {'weight': 100., 'seeds': 16, 'iterations': 80,
                                          'robot_dof_indices': [6, 7, 8, 9, 10, 11]}
        executor.solver = Mock()
        executor.scorer = Mock(side_effect=AssertionError('GT/scoring is not an IK input'))
        return executor

    def test_primary_valid_disabled_and_other_controllers_never_retry(self):
        for controller, enabled, valid in [('servo', True, True), ('servo', False, False),
                                            ('repaired', True, False), ('baseline', True, False)]:
            with self.subTest(controller=controller, enabled=enabled, valid=valid):
                executor = self.executor()
                executor.controller = controller
                result = object() if valid else None
                evaluate = Mock()
                with patch.dict(SERVO_CONFIG, soft_limit_fallback_enabled=enabled):
                    returned, diagnostic = executor._maybe_soft_limit_fallback(object(), result, {'valid': valid}, evaluate)
                self.assertIs(returned, result)
                self.assertEqual(diagnostic['ik_source'], 'primary')
                self.assertFalse(diagnostic['fallback_attempted'])
                executor.solver.solve.assert_not_called()
                evaluate.assert_not_called()

    def test_single_retry_preserves_primary_evidence_and_restores_objective(self):
        executor = self.executor()
        objective = executor.ik_limit_objective
        original = objective.joint_limit_lower, objective.joint_limit_upper, objective.weight
        initial, result = object(), object()
        def solve(received):
            self.assertIs(received, initial)
            self.assertIs(objective.joint_limit_lower, executor.ik_soft_lower)
            self.assertIs(objective.joint_limit_upper, executor.ik_soft_upper)
            self.assertEqual(objective.weight, 100.)
        executor.solver.solve.side_effect = solve
        primary = {'valid': False, 'passing_seed_counts': {'joint_limits': 0}}
        evaluate = Mock(return_value=(result, {'valid': True, 'passing_seed_counts': {'joint_limits': 16}}))
        returned, log = executor._maybe_soft_limit_fallback(initial, None, primary, evaluate)
        self.assertIs(returned, result)
        self.assertEqual(log['primary_ik']['passing_seed_counts']['joint_limits'], 0)
        self.assertEqual(log['passing_seed_counts']['joint_limits'], 16)
        self.assertTrue(log['fallback_attempted'])
        self.assertEqual(log['ik_source'], 'soft_limit_fallback')
        self.assertEqual((objective.joint_limit_lower, objective.joint_limit_upper, objective.weight), original)
        executor.solver.solve.assert_called_once_with(initial)
        evaluate.assert_called_once()
        executor.scorer.assert_not_called()

    def test_failed_retry_stays_failed_without_loop_and_restores_after_exceptions(self):
        for exception in [None, RuntimeError('CPU solve failed'), ValueError('FK evaluation failed'), KeyError('fatal')]:
            with self.subTest(exception=exception):
                executor = self.executor()
                objective = executor.ik_limit_objective
                original = objective.joint_limit_lower, objective.joint_limit_upper, objective.weight
                evaluate = Mock(return_value=(None, {'valid': False}))
                if exception is not None:
                    evaluate.side_effect = exception
                if isinstance(exception, KeyError):
                    with self.assertRaises(KeyError):
                        executor._maybe_soft_limit_fallback(object(), None, {'valid': False}, evaluate)
                else:
                    returned, log = executor._maybe_soft_limit_fallback(object(), None, {'valid': False}, evaluate)
                    self.assertIsNone(returned)
                    self.assertFalse(log['valid'])
                    if exception is not None:
                        self.assertIn(type(exception).__name__, log['fallback_error'])
                self.assertEqual((objective.joint_limit_lower, objective.joint_limit_upper, objective.weight), original)
                self.assertEqual(executor.solver.solve.call_count, 1)

    def test_soft_limits_map_coordinates_to_dofs_in_robot_order(self):
        lower = np.full(9, -10.)
        upper = np.full(9, 10.)
        # A floating root has seven coordinates but six DOFs; robot order is permuted.
        bounds = np.array([[-.8, .9], [-.2, .3], [-1.2, 1.3]])
        lo, hi, dofs = _soft_limit_dof_arrays([0, 7, 8, 9, 10], [0, 6, 7, 8, 9],
            [[3, 3], [0, 1], [0, 1], [0, 1]], [8, 7, 9], bounds, lower, upper)
        self.assertEqual(dofs, [7, 6, 8])
        np.testing.assert_allclose(lo[dofs], bounds[:, 0])
        np.testing.assert_allclose(hi[dofs], bounds[:, 1])
        np.testing.assert_array_equal(lo[:6], lower[:6])
        np.testing.assert_array_equal(lower, np.full(9, -10.))
        with self.assertRaises(ValueError):
            _soft_limit_dof_arrays([0, 7], [0, 6], [[3, 3]], [0], [[-1, 1]], lower, upper)
        with self.assertRaises(ValueError):
            _soft_limit_dof_arrays([0, 7, 8], [0, 6, 7], [[3, 3], [0, 1]],
                                  [7, 7], [[-1, 1], [-1, 1]], lower, upper)

    def test_shared_fk_evaluator_keeps_every_original_gate(self):
        executor = self.executor()
        executor.coordinates = torch.arange(6)
        executor.arm, executor.ik_body = list(range(5)), 0
        executor.tool_offset_b = np.zeros(3)
        executor.tool_evidence = {'control_reference': 'test_robot_tool'}
        executor.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros((1, 6)),
            soft_joint_pos_limits=torch.tensor([[[-1., 1.]] * 6])))
        warp = SimpleNamespace(to_torch=lambda value: value)
        math = SimpleNamespace(quat_apply=lambda quaternion, value: value)
        for failure in ['position', 'orientation', 'joint_limits', 'finite_joints', 'joint_continuity', None]:
            with self.subTest(failure=failure):
                q = torch.zeros((16, 6))
                poses = torch.zeros((16, 1, 7))
                poses[:, :, 6] = 1.
                if failure == 'position':
                    poses[:, :, 0] = .011
                elif failure == 'orientation':
                    poses[:, :, 5] = np.sin(np.deg2rad(20.))
                    poses[:, :, 6] = np.cos(np.deg2rad(20.))
                elif failure == 'joint_limits':
                    q[:, 5] = 1.002
                elif failure == 'finite_joints':
                    q[:, 5] = float('nan')
                elif failure == 'joint_continuity':
                    q[:, 0] = .151
                executor.solver = SimpleNamespace(joint_q=q, solver=SimpleNamespace(body_q=poses))
                with patch.dict(sys.modules, {'warp': warp, 'isaaclab.utils.math': math}):
                    returned, log = executor._ik_result(torch.zeros((1, 3)), torch.tensor([[0., 0., 0., 1.]]),
                                                        torch.zeros((1, 6)), True)
                self.assertEqual(log['valid'], failure is None)
                self.assertEqual(returned is not None, failure is None)
                if failure is not None:
                    self.assertEqual(log['passing_seed_counts'][failure], 0)


if __name__ == '__main__':
    unittest.main()
