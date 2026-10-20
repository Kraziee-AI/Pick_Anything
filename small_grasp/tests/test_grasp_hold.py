"""Safety bounds and robot-only signal handling of the unintegrated helper."""
import math
import unittest

from small_grasp.grasp_hold import BoundedContactHold, HoldConfig


class BoundedContactHoldTest(unittest.TestCase):
    def test_debounce_and_rate_limit(self):
        hold = BoundedContactHold(.4)
        self.assertEqual(hold.update(8, .41, 0, 1 / 30).reason, 'debounce')
        self.assertEqual(hold.update(8, .41, 0, 1 / 30).target_rad, .4)
        decision = hold.update(8, .41, 0, 1 / 30)
        self.assertEqual(decision.reason, 'bounded_closure')
        self.assertAlmostEqual(decision.target_rad, .4 - .04 / 30)

    def test_cumulative_and_joint_bounds_over_long_run(self):
        for initial in (.4, .06):
            hold = BoundedContactHold(initial)
            for _ in range(10000):
                previous = hold.target
                decision = hold.update(8, previous + .01, 0, 1 / 30)
                self.assertLessEqual(previous - decision.target_rad, .04 / 30 + 1e-12)
                self.assertLessEqual(decision.extra_closure_rad, .04 + 1e-12)
                self.assertGreaterEqual(decision.target_rad, .05)
            self.assertEqual(decision.reason, 'closure_budget_exhausted')

    def test_unsafe_or_uninformative_samples_do_not_close_and_reset(self):
        for force, joint, velocity, dt, expected in [
            (0, .41, 0, 1 / 30, 'contact_absent'),
            (30, .41, 0, 1 / 30, 'force_ceiling'),
            (50, .41, 0, 1 / 30, 'force_ceiling'),
            (16, .41, 0, 1 / 30, 'force_sufficient'),
            (8, .41, .3, 1 / 30, 'joint_moving'),
            (8, .45, 0, 1 / 30, 'command_not_settled'),
            (math.nan, .41, 0, 1 / 30, 'invalid_observation'),
            (8, .41, 0, 1, 'invalid_observation'),
            (-1, .41, 0, 1 / 30, 'invalid_observation'),
        ]:
            hold = BoundedContactHold(.4)
            hold.update(8, .41, 0, 1 / 30)
            hold.update(8, .41, 0, 1 / 30)
            decision = hold.update(force, joint, velocity, dt)
            self.assertEqual(decision.reason, expected)
            self.assertEqual(decision.target_rad, .4)
            self.assertEqual(hold.update(8, .41, 0, 1 / 30).reason, 'debounce')

    def test_configuration_rejects_invalid_values(self):
        for kwargs in ({'low_force_n': 1}, {'force_ceiling_n': 10},
                       {'closure_rate_rad_s': 0}, {'max_extra_closure_rad': math.inf},
                       {'consecutive_steps': 1.5}):
            with self.assertRaises(ValueError):
                HoldConfig(**kwargs)
        with self.assertRaises(ValueError):
            BoundedContactHold(.04)


if __name__ == '__main__':
    unittest.main()
