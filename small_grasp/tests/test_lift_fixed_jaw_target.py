"""CPU command comparisons for fixed servo lift jaws without force feedback."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from small_grasp.sim_robot import GraspExecutor, SERVO_CONFIG


class FixedLiftJawTests(unittest.TestCase):
    def executor(self, controller='servo'):
        executor = GraspExecutor.__new__(GraspExecutor)
        executor.controller, executor.grip = controller, 5
        executor.arm = list(range(5))
        executor.robot = SimpleNamespace(data=SimpleNamespace(
            joint_pos=torch.tensor([[-.3, .5, -.1, .7, -.2, .43]], dtype=torch.float64)))
        executor._grasp_hold_last_target_rad = None
        executor.motion_diagnostics, executor.commands = [], []
        executor.scorer = Mock(side_effect=AssertionError('Lift jaw interpolation must not read scoring'))
        def step(target, phase):
            executor.commands.append((phase, target.clone()))
            return True
        executor._step = step
        return executor

    def test_30_and_90_frames_keep_jaw_goal_and_exact_original_arm_interpolation(self):
        for steps in (30, 90):
            for phase in ('lift', 'lift_hold'):
                executor = self.executor()
                start = executor.robot.data.joint_pos.clone()
                goal = torch.tensor([[.1, -.1, .4, .2, .3, .39]], dtype=torch.float64)
                caller_goal = goal.clone()
                with patch.dict(SERVO_CONFIG, lift_fixed_jaw_target_enabled=True, bounded_hold_enabled=False):
                    self.assertTrue(executor._move(goal, phase, steps))
                self.assertEqual(len(executor.commands), steps)
                for i, (name, command) in enumerate(executor.commands, 1):
                    expected = torch.lerp(start, goal, min(1., i / (steps - 2)))
                    self.assertEqual(name, phase)
                    self.assertTrue(torch.equal(command[:, :5], expected[:, :5]))
                    self.assertTrue(torch.equal(command[:, 5], goal[:, 5]))
                self.assertTrue(torch.equal(goal, caller_goal))
                self.assertTrue(torch.equal(executor.robot.data.joint_pos, start))
                executor.scorer.assert_not_called()
                event = executor.motion_diagnostics[0]
                self.assertEqual(event['scope'], ['lift', 'lift_hold'])
                self.assertEqual(event['fixed_jaw_target_rad'], .39)
                self.assertAlmostEqual(event['first_actual_minus_command_rad'], .04)
                self.assertIn('unchanged caller lift goal', event['jaw_target_source'])

    def test_disabled_other_controllers_and_unrelated_phases_keep_exact_old_commands(self):
        cases = [('servo', 'lift', False), ('servo', 'lift_hold', False),
                 ('repaired', 'lift', True), ('baseline', 'lift', True)]
        cases += [('servo', phase, True) for phase in
                  ('open', 'approach', 'descend', 'close_centered', 'hold',
                   'transport', 'place', 'release', 'retreat', 'settle')]
        for controller, phase, enabled in cases:
            executor = self.executor(controller)
            start = executor.robot.data.joint_pos.clone()
            goal = torch.tensor([[.1, -.1, .4, .2, .3, .39]], dtype=torch.float64)
            with patch.dict(SERVO_CONFIG, lift_fixed_jaw_target_enabled=enabled, bounded_hold_enabled=False):
                self.assertTrue(executor._move(goal, phase, 30))
            ramp = 28 if controller == 'servo' else 20
            for i, (_, command) in enumerate(executor.commands, 1):
                self.assertTrue(torch.equal(command, torch.lerp(start, goal, min(1., i / ramp))))
            self.assertEqual(executor.motion_diagnostics, [])
            executor.scorer.assert_not_called()

    def test_explicit_caller_start_and_goal_are_not_modified(self):
        executor = self.executor()
        start = torch.tensor([[.6, .1, -.4, .5, -.8, .48]], dtype=torch.float64)
        goal = torch.tensor([[.1, -.1, .4, .2, .3, .39]], dtype=torch.float64)
        original_start, original_goal = start.clone(), goal.clone()
        with patch.dict(SERVO_CONFIG, lift_fixed_jaw_target_enabled=True, bounded_hold_enabled=False):
            self.assertTrue(executor._move(goal, 'lift', 90, start=start))
        self.assertTrue(torch.equal(start, original_start))
        self.assertTrue(torch.equal(goal, original_goal))
        self.assertEqual(executor.motion_diagnostics[0]['start_source'], 'caller_copy')
        self.assertEqual(executor.motion_diagnostics[0]['jaw_start_before_override_rad'], .48)
        for i, (_, command) in enumerate(executor.commands, 1):
            expected = torch.lerp(start, goal, min(1., i / 88))
            self.assertTrue(torch.equal(command[:, :5], expected[:, :5]))
            self.assertTrue(torch.equal(command[:, 5], goal[:, 5]))
        executor.scorer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
