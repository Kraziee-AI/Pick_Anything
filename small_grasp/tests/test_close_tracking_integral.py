"""Bounded close-only PI against an independent loaded dynamic plant."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from small_grasp.arm_tracking import BoundedArmTrackingFeedback


class CloseTrackingIntegralTests(unittest.TestCase):
    def setUp(self):
        self.limits = np.tile([-1., 1.], (6, 1))
        self.nominal = np.array([0., 0., .3, 0., 0., .7])

    def propose(self, controller, actual, phase='close_centered', speed=0., nominal=None):
        return controller.propose(self.nominal if nominal is None else nominal, actual, self.limits,
            phase=phase, dt_s=1/30, arm_velocity=np.full(5, speed))

    def test_dynamic_load_offset_reduced_with_original_total_command_bounds(self):
        errors = []
        for enabled in (False, True):
            c = BoundedArmTrackingFeedback(range(5), close_integral=enabled)
            load = np.array([0., 0., -.03, 0., 0., 0.])
            actual = self.nominal + load
            for _ in range(360):
                prior = c.bias.copy()
                d = self.propose(c, actual)
                self.assertLessEqual(np.max(np.abs(d.bias)), .025 + 1e-12)
                self.assertLessEqual(np.max(np.abs(d.bias - prior)), .004 + 1e-12)
                self.assertEqual(d.command[5], self.nominal[5])
                c.commit(d, True)
                actual += .2 * (d.command + load - actual)
            errors.append(abs(actual[2] - self.nominal[2]))
        self.assertLess(errors[1], .3 * errors[0])

    def test_nominal_debounce_speed_gate_waypoint_reset_and_bounded_phase_exit(self):
        c = BoundedArmTrackingFeedback(range(5), close_integral=True)
        actual = self.nominal.copy(); actual[2] -= .03
        for i in range(3):
            d = self.propose(c, actual); c.commit(d, True)
            np.testing.assert_array_equal(c.integral, np.zeros(5))
        d = self.propose(c, actual, speed=.051); c.commit(d, True)
        self.assertEqual(d.integral_reason, 'arm_moving')
        np.testing.assert_array_equal(c.integral, np.zeros(5))
        for _ in range(40):
            d = self.propose(c, actual); c.commit(d, True)
        self.assertGreater(c.integral[2], 0)
        changed = self.nominal.copy(); changed[5] -= .01
        prior = c.bias.copy()
        d = self.propose(c, actual, nominal=changed)
        self.assertEqual(d.integral_reason, 'nominal_changed')
        np.testing.assert_array_equal(d.integral, np.zeros(5))
        self.assertLessEqual(np.max(np.abs(d.bias - prior)), .004 + 1e-12)
        c.commit(d, True)
        for _ in range(40):
            d = self.propose(c, actual, nominal=changed); c.commit(d, True)
        prior = c.bias.copy()
        d = self.propose(c, actual, phase='hold', nominal=changed)
        np.testing.assert_array_equal(d.integral, np.zeros(5))
        self.assertEqual(d.integral_reason, 'outside_close_phase')
        self.assertLessEqual(np.max(np.abs(d.bias - prior)), .004 + 1e-12)

    def test_conditional_antiwindup_and_reversal_do_not_store_saturated_error(self):
        c = BoundedArmTrackingFeedback(range(5), close_integral=True)
        actual = self.nominal.copy(); actual[2] -= .1
        for _ in range(300):
            d = self.propose(c, actual); c.commit(d, True)
        self.assertAlmostEqual(c.bias[2], .025)
        self.assertEqual(c.integral[2], 0.)
        self.assertEqual(d.integral_reason, 'conditional_antiwindup')
        actual[2] = self.nominal[2] + .1
        for _ in range(20):
            prior = c.bias.copy(); d = self.propose(c, actual)
            self.assertLessEqual(np.max(np.abs(d.bias - prior)), .004 + 1e-12)
            c.commit(d, True)
        self.assertAlmostEqual(c.bias[2], -.025)
        self.assertEqual(c.integral[2], 0.)

    def test_proposals_commit_only_after_acceptance_and_invalid_input_resets(self):
        c = BoundedArmTrackingFeedback(range(5), close_integral=True)
        actual = self.nominal.copy(); actual[2] -= .03
        for _ in range(10):
            d = self.propose(c, actual); c.commit(d, True)
        prior = c.integral.copy()
        d = self.propose(c, actual)
        self.assertGreater(d.integral[2], prior[2])
        np.testing.assert_array_equal(c.integral, prior)
        c.commit(d, False)
        np.testing.assert_array_equal(c.integral, np.zeros(5))
        np.testing.assert_array_equal(c.bias, np.zeros(5))
        self.assertIsNone(c.integral_nominal)
        for dt, vel in [(float('nan'), np.zeros(5)), (.1, np.zeros(5)), (1/30, np.full(5, np.nan))]:
            d = c.propose(self.nominal, actual, self.limits, phase='close_centered', dt_s=dt, arm_velocity=vel)
            self.assertEqual(d.reason, 'invalid_close_integral_observation')
            c.commit(d, False)
            self.assertIsNone(c.integral_nominal)

    def test_off_ignores_integral_context_and_preserves_exact_original_commands(self):
        a, b = BoundedArmTrackingFeedback(range(5)), BoundedArmTrackingFeedback(range(5), close_integral=False)
        actual = self.nominal.copy(); actual[2] -= .03
        for _ in range(20):
            da = a.propose(self.nominal, actual, self.limits)
            db = b.propose(self.nominal, actual, self.limits, phase='close_centered', dt_s=float('nan'), arm_velocity=None)
            np.testing.assert_array_equal(da.command, db.command)
            a.commit(da, True); b.commit(db, True)

    def test_executor_integral_rejected_path_stops_before_new_command(self):
        from small_grasp.sim_robot import GraspExecutor
        e = GraspExecutor.__new__(GraspExecutor)
        e.arm_tracking_feedback = e.close_tracking_integral = True
        e.arm = list(range(5)); e._arm_tracking = BoundedArmTrackingFeedback(e.arm, close_integral=True)
        e.env = SimpleNamespace(episode_length_buf=torch.tensor([3]), step_dt=1/30)
        actual = torch.tensor(self.nominal)[None]; actual[0, 2] -= .03
        e.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=actual, joint_vel=torch.zeros((1, 6)),
            soft_joint_pos_limits=torch.tensor(self.limits)[None]))
        e._mesh_at_q = Mock(return_value=None); e._check_path = Mock(return_value=None)
        e._mesh_path = Mock(side_effect=[None, {'reason': 'table_mesh'}])
        e.arm_tracking_events, e.mesh_evidence = [], {}
        e.scorer = Mock(side_effect=AssertionError('No target scoring in PI'))
        self.assertIsNone(e._tracking_command(torch.tensor(self.nominal)[None], 'close_centered'))
        self.assertFalse(e.collision_ready)
        self.assertIsNone(e.arm_tracking_events[-1]['applied_joint_target_rad'])
        self.assertFalse(e.arm_tracking_events[-1]['command_issued'])
        np.testing.assert_array_equal(e._arm_tracking.integral, np.zeros(5))
        e.scorer.assert_not_called()

    def executor(self, actual, limits=None, enabled=True):
        from small_grasp.sim_robot import GraspExecutor
        e = GraspExecutor.__new__(GraspExecutor)
        e.arm_tracking_feedback, e.close_tracking_integral = True, enabled
        e.arm, e.grip = list(range(5)), 5
        e._arm_tracking = BoundedArmTrackingFeedback(e.arm, close_integral=enabled)
        e.env = SimpleNamespace(episode_length_buf=torch.tensor([3]), step_dt=1/30,
                                tracking=True, step=Mock())
        e.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor(actual)[None],
            joint_vel=torch.zeros((1, 6)),
            soft_joint_pos_limits=torch.tensor(self.limits if limits is None else limits)[None]))
        e._mesh_at_q, e._check_path, e._mesh_path = (Mock(return_value=None) for _ in range(3))
        e.arm_tracking_events, e.mesh_evidence, e.trace = [], {}, []
        e.collision_ready, e.collision_check = True, 'off'
        e._apply_grasp_hold = lambda target, phase: target
        e._stop_grasp_hold = Mock()
        return e

    def test_observed_nonclose_soft_limit_rejections_keep_original_safe_fallback(self):
        # Wrist coordinates from cube6002/6004 step185 and cube6001 step668.
        # Other coordinates are neutral: the observed rejection is the wrist's
        # positive P offset exceeding its unchanged robot soft upper limit.
        for phase, nominal_wrist, actual_wrist, prior_bias in [
            ('approach_high', 1.623409628868103, 1.6180803775787354, .0021616454124450683),
            ('approach_high', 1.623407006263733, 1.618034839630127, .0021825666427612305),
            ('lift', 1.614526391029358, 1.5836129188537598, .014957391738891601),
        ]:
            with self.subTest(phase=phase, nominal=nominal_wrist):
                nominal, actual = np.zeros(6), np.zeros(6)
                nominal[3], actual[3] = nominal_wrist, actual_wrist
                limits = np.tile([-3., 3.], (6, 1)); limits[3] = [-1.6248986721038818, 1.6248986721038818]
                self.assertTrue(((nominal >= limits[:, 0]) & (nominal <= limits[:, 1])).all())
                outputs = []
                for enabled in (False, True):
                    e = self.executor(actual, limits, enabled)
                    e._arm_tracking.bias[3] = prior_bias
                    target = torch.tensor(nominal)[None]
                    result = e._tracking_command(target, phase)
                    self.assertIs(result, target)
                    self.assertTrue(e.collision_ready)
                    event = e.arm_tracking_events[-1]
                    self.assertEqual(event['reason'], 'compensated_command_outside_soft_limits')
                    self.assertEqual(event['applied_joint_target_rad'], nominal.tolist())
                    e._mesh_at_q.assert_called_once()
                    e._check_path.assert_called_once()
                    e._mesh_path.assert_called_once()
                    np.testing.assert_array_equal(e._arm_tracking.bias, np.zeros(5))
                    np.testing.assert_array_equal(e._arm_tracking.integral, np.zeros(5))
                    outputs.append(result)
                torch.testing.assert_close(*outputs, rtol=0, atol=0)

    def test_nonclose_fallback_still_requires_safe_origin_and_both_nominal_paths(self):
        for rejected_gate in ('origin', 'capsule_path', 'mesh_path'):
            with self.subTest(gate=rejected_gate):
                e = self.executor(self.nominal)
                gate = {'origin': e._mesh_at_q, 'capsule_path': e._check_path,
                        'mesh_path': e._mesh_path}[rejected_gate]
                gate.return_value = {'reason': 'table_mesh'}
                e._arm_tracking.propose = Mock(side_effect=AssertionError('unsafe origin/path'))
                self.assertFalse(e._step(torch.tensor(self.nominal)[None], 'approach_high'))
                e.env.step.assert_not_called()
                e._arm_tracking.propose.assert_not_called()
                self.assertFalse(e.collision_ready)
                self.assertFalse(e.arm_tracking_events[-1]['command_issued'])
                self.assertIsNone(e.arm_tracking_events[-1]['applied_joint_target_rad'])

    def test_nonclose_compensated_collision_reverts_only_after_nominal_gates_pass(self):
        actual = self.nominal.copy(); actual[2] -= .03
        for phase in ('approach_high', 'approach', 'descend', 'close_align', 'hold', 'lift', 'lift_hold'):
            with self.subTest(phase=phase):
                e = self.executor(actual)
                e._mesh_path.side_effect = [None, {'reason': 'table_mesh'}]
                target = torch.tensor(self.nominal)[None]
                self.assertIs(e._tracking_command(target, phase), target)
                self.assertTrue(e.collision_ready)
                self.assertEqual(e.arm_tracking_events[-1]['reason'], 'collision_rejected')
                self.assertEqual(e._check_path.call_count, 2)
                self.assertEqual(e._mesh_path.call_count, 2)

    def test_close_soft_limit_and_invalid_integral_observation_still_stop(self):
        for invalid in ('soft_limit', 'velocity'):
            with self.subTest(invalid=invalid):
                actual = self.nominal.copy(); actual[2] -= .03
                e = self.executor(actual)
                if invalid == 'soft_limit':
                    e.robot.data.soft_joint_pos_limits[0, 2, 1] = self.nominal[2] + .001
                else:
                    e.robot.data.joint_vel[:] = float('nan')
                self.assertFalse(e._step(torch.tensor(self.nominal)[None], 'close_centered'))
                e.env.step.assert_not_called()
                self.assertFalse(e.collision_ready)
                self.assertFalse(e.arm_tracking_events[-1]['command_issued'])
                self.assertIsNone(e.arm_tracking_events[-1]['applied_joint_target_rad'])
                np.testing.assert_array_equal(e._arm_tracking.integral, np.zeros(5))


if __name__ == '__main__':
    unittest.main()
