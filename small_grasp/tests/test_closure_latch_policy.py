"""Robot-only latch invariants, issued-command provenance and safe integration."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from small_grasp.sim_robot import CONFIG, _ClosureContactLatch
from small_grasp.tests import test_motion_synchronization as synchronized_tests


class ClosureLatchPolicyTests(unittest.TestCase):
    def test_legacy_and_no_reopen_have_identical_contact_stall_trigger(self):
        for policy, expected in [('legacy', .48), ('no_reopen', .46)]:
            latch = _ClosureContactLatch(.5, policy)
            for force, joint, command, velocity in [(7.9, .5, .46, 0), (10, .5, .48, 0), (10, .5, .46, .41)]:
                self.assertIsNone(latch.observe(force, joint, command, velocity))
                self.assertEqual(latch.consecutive, 0)
            self.assertIsNone(latch.observe(10, .5, .46, 0))
            self.assertIsNone(latch.observe(10, .5, .46, 0))
            event = latch.observe(10, .5, .46, 0)
            self.assertAlmostEqual(latch.target, expected)
            self.assertAlmostEqual(event['legacy_joint_target_rad'], .48)
            self.assertEqual(event['latch_policy'], policy)
            self.assertIsNone(latch.observe(0, .1, .05, 0))
            self.assertAlmostEqual(latch.target, expected)

    def test_no_reopen_rejects_invalid_observations_and_respects_closed_boundary(self):
        for bad in (float('nan'), float('inf'), -float('inf'), .049):
            latch = _ClosureContactLatch(.5, 'no_reopen')
            for _ in range(5):
                self.assertIsNone(latch.observe(10, .5, bad, 0))
            self.assertIsNone(latch.target)
        for field in (0, 1, 3):
            sample = [10., .5, .46, 0.]
            sample[field] = float('nan')
            latch = _ClosureContactLatch(.5, 'no_reopen')
            for _ in range(5):
                self.assertIsNone(latch.observe(*sample))
        latch = _ClosureContactLatch(.5, 'no_reopen')
        for _ in range(3):
            latch.observe(10, .1, CONFIG['closed_joint_rad'], 0)
        self.assertEqual(latch.target, CONFIG['closed_joint_rad'])
        with self.assertRaises(ValueError):
            _ClosureContactLatch(.5, 'unsupported')

    def executor(self, issued=True):
        e = synchronized_tests.MotionSynchronizationTests().executor()
        e.closure_latch_policy = 'no_reopen'
        e._move = Mock(return_value=True)
        e._arrive = Mock(return_value=True)
        path = []
        for jaw in np.linspace(1.4, .05, 56):
            q = torch.zeros((1, 6)); q[0, 5] = float(jaw)
            path.append((np.zeros(3), q))
        e._prepare_synchronized_closure = Mock(return_value=(path, {'valid': True}))
        if issued:
            # Deliberately differs from path[0]: only this value was issued.
            e._last_issued_target = path[0][1].clone()
            e._last_issued_target[0, 5] = .86
        e.robot.data.joint_pos[0, 5] = .9
        class RobotOnly(dict):
            def __getitem__(self, key):
                if key != 'claw_force':
                    raise AssertionError('Only robot aggregate contact is available')
                return 10.
        e.scorer = Mock(return_value=RobotOnly())
        e.issued = []
        def step(target, phase):
            e.issued.append(target.clone())
            if e._last_issued_target is not None:
                e._last_issued_target = target.clone()
                # Keep the established stall command during debounce.
                if len(e.issued) < 3:
                    e._last_issued_target[0, 5] = .86
            e.env.episode_length_buf += 1
            return True
        e._step = step
        return e

    def test_synchronized_latch_retains_actual_issued_jaw_and_existing_arm_reference(self):
        e = self.executor()
        result = e._close_centered_synchronized(np.zeros(3), np.eye(3))
        self.assertIsNotNone(result)
        latch = next(x for x in e.close_control_events if x['event'] == 'contact_stall_latched')
        self.assertAlmostEqual(latch['previous_joint_target_rad'], .86, places=6)
        self.assertAlmostEqual(latch['locked_joint_target_rad'], .86, places=6)
        self.assertAlmostEqual(latch['legacy_joint_target_rad'], .88, places=6)
        self.assertTrue(all(abs(float(x[0, 5]) - .86) < 1e-6 for x in e.issued[-15:]))
        self.assertTrue(all(torch.equal(x[0, :5], torch.zeros(5)) for x in e.issued))

    def test_missing_issued_command_or_failed_preflight_stops_before_new_action(self):
        e = self.executor(issued=False)
        self.assertIsNone(e._close_centered_synchronized(np.zeros(3), np.eye(3)))
        self.assertEqual(e.motion_failure, 'close_latch_command_unavailable')
        self.assertEqual(e.issued, [])
        e = self.executor()
        e._prepare_synchronized_closure.return_value = None, {'failure_reason': 'close_preflight_collision'}
        self.assertIsNone(e._close_centered_synchronized(np.zeros(3), np.eye(3)))
        self.assertEqual(e.motion_failure, 'close_preflight_collision')
        e._move.assert_not_called()
        self.assertEqual(e.issued, [])

    def test_disabled_legacy_control_ignores_last_issued_override(self):
        helper = synchronized_tests.MotionSynchronizationTests()
        for kwargs in ({}, {'contact': True}, {'forever_fast': True}):
            default, result, commands, _ = helper.run_closure(**kwargs)
            explicit = helper.executor()
            explicit.closure_latch_policy = 'legacy'
            explicit._last_issued_target = torch.full((1, 6), .7)
            with patch.object(helper, 'executor', return_value=explicit):
                explicit, other_result, other_commands, _ = helper.run_closure(**kwargs)
            self.assertEqual(default.close_control_events, explicit.close_control_events)
            self.assertEqual(len(commands), len(other_commands))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(commands, other_commands)))
            self.assertEqual(result is None, other_result is None)
            if result is not None:
                self.assertTrue(torch.equal(result, other_result))


if __name__ == '__main__':
    unittest.main()
