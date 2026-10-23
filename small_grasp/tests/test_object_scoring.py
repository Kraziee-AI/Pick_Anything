"""Independent generic score boundaries, asset origins and missing-stage failures."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from small_grasp.object_scoring import LiftTracker, center_kinematics, contact_sample


def sample(z=.151, **overrides):
    fields = dict(center=[0., 0., z], velocity=np.zeros(3), pocket=[0., 0., z],
                  gripper_velocity=np.zeros(3), grip=.6, claw_force=3., object_force=.2)
    fields.update(overrides)
    return contact_sample(**fields)


class ObjectScoringTests(unittest.TestCase):
    def test_rotated_bottom_origin_and_center_velocity(self):
        center, velocity = center_kinematics([1, 2, 3], Rotation.from_euler('x', 90, degrees=True).as_quat(),
            [0, 0, 0], [0, 0, 2], [0, 0, .025])
        np.testing.assert_allclose(center, [1, 1.975, 3], atol=1e-10)
        np.testing.assert_allclose(velocity, [.05, 0, 0], atol=1e-10)

    def test_all_held_boundaries_are_strict(self):
        self.assertTrue(sample()['held'])
        for key, value in [('grip', .05), ('grip', 1.6), ('claw_force', 2.2), ('object_force', .1),
                           ('velocity', np.array([.15, 0, 0])), ('pocket', [0, 0, .196])]:
            with self.subTest(key=key, value=value):
                self.assertFalse(sample(**{key: value})['held'])

    def test_five_consecutive_steps_and_dropped_end_hold(self):
        tracker = LiftTracker(.07, 1 / 30)
        for step in range(1, 5):
            tracker.observe(step, sample())
        self.assertFalse(tracker.summary()['generic_grasp_lift_success'])
        tracker.observe(5, sample())
        tracker.observe(5, sample())
        tracker.observe(6, sample(claw_force=0))
        summary = tracker.summary([{'step': 5, 'phase': 'lift_hold'}, {'step': 6, 'phase': 'lift_hold'}])
        self.assertTrue(summary['generic_grasp_lift_success'])
        self.assertFalse(summary['generic_end_hold_success'])
        self.assertEqual(summary['lift_hold_samples'], 2)
        self.assertEqual(len(tracker.rows), 6)
        self.assertIsNone(summary['place_success'])
        self.assertIsNone(summary['original_task_success'])

    def test_missing_execution_or_hold_and_step_gap(self):
        tracker = LiftTracker(.07, .1)
        self.assertFalse(tracker.summary()['generic_grasp_lift_success'])
        self.assertFalse(tracker.summary()['generic_end_hold_success'])
        for step in (1, 2, 3, 4, 6):
            tracker.observe(step, sample())
        self.assertFalse(tracker.summary()['generic_grasp_lift_success'])


if __name__ == '__main__':
    unittest.main()
