"""CPU regression checks for smoothing reset and PPO warm-start semantics."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pick_anything.tasks.pick_apple.smooth_camera_cfg import (
    CameraFilteredEnvCfg,
    CameraSmoothPPOBaseRunnerCfg,
    FilteredRelativeJointPositionAction,
    SmoothCameraPPO,
)


class SmoothingTests(unittest.TestCase):
    def test_only_supported_camera_tasks_are_registered(self):
        tasks = {name for name in gym.registry if name.startswith("Pick-Apple-SO101-Staged-Camera-")}
        self.assertEqual(tasks, {
            "Pick-Apple-SO101-Staged-Camera-v8",
            "Pick-Apple-SO101-Staged-Camera-Filtered-v0",
            "Pick-Apple-SO101-Staged-Camera-Distill-v10",
        })

    def test_filter_and_resets(self):
        # Exercise actual process_actions/reset methods without constructing a simulator.
        action = FilteredRelativeJointPositionAction.__new__(FilteredRelativeJointPositionAction)
        action._raw_actions = torch.zeros(2, 1)
        action._processed_actions = torch.zeros(2, 1)
        action._filtered = torch.zeros(2, 1)
        action._scale, action._offset, action._alpha = 0.033, 0.0, 0.7
        action.cfg = SimpleNamespace(clip=None)
        action.process_actions(torch.ones(2, 1))
        torch.testing.assert_close(action.processed_actions, torch.full((2, 1), 0.0231))
        torch.testing.assert_close(action.raw_actions, torch.ones(2, 1))
        action.reset(torch.tensor([0]))
        action.process_actions(-torch.ones(2, 1))
        torch.testing.assert_close(action.processed_actions, torch.tensor([[-0.0231], [-0.01617]]))
        action.reset()
        self.assertFalse(bool(action._filtered.any()))
        action._alpha = 1.0
        action.process_actions(torch.ones(2, 1))
        torch.testing.assert_close(action.processed_actions, torch.full((2, 1), 0.033))

    def test_checkpoint_does_not_override_finetune_lr(self):
        ppo = SmoothCameraPPO.__new__(SmoothCameraPPO)
        ppo._raw_actor = torch.nn.Linear(1, 1)
        ppo._raw_critic = torch.nn.Linear(1, 1)
        params = list(ppo._raw_actor.parameters()) + list(ppo._raw_critic.parameters())
        ppo.optimizer = torch.optim.Adam(params, lr=3e-5)
        ppo.learning_rate, ppo.rnd = 3e-5, None
        old_optimizer = torch.optim.Adam(params, lr=1e-4)
        sum(p.sum() for p in params).backward()
        old_optimizer.step()
        checkpoint = {
            "actor_state_dict": ppo._raw_actor.state_dict(),
            "critic_state_dict": ppo._raw_critic.state_dict(),
            "optimizer_state_dict": old_optimizer.state_dict(),
        }
        self.assertTrue(ppo.load(checkpoint))
        self.assertEqual(ppo.learning_rate, 3e-5)
        self.assertEqual(ppo.optimizer.param_groups[0]["lr"], 3e-5)
        self.assertFalse(ppo.optimizer.state)
        ppo.load(checkpoint, {"optimizer": True})
        self.assertEqual(ppo.learning_rate, 1e-4)
        self.assertTrue(ppo.optimizer.state)

    def test_checkpoint_compatible_environment(self):
        cfg = CameraFilteredEnvCfg()
        self.assertEqual(cfg.actions.arm_action.alpha, 0.7)
        self.assertFalse(hasattr(cfg.actions.gripper_action, "alpha"))
        self.assertEqual(cfg.rewards.action_rate.weight, -0.1)
        self.assertEqual((cfg.scene.wrist_camera.width, cfg.scene.wrist_camera.height), (128, 96))
        self.assertEqual(CameraSmoothPPOBaseRunnerCfg().algorithm.learning_rate, 3e-5)


if __name__ == "__main__":
    unittest.main()
