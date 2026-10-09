"""CPU checks for the standalone v10 configuration (isaaclab_61)."""
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
import gymnasium as gym
from tensordict import TensorDict
from rsl_rl.algorithms import Distillation
from rsl_rl.models import MLPModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pick_anything.tasks.pick_apple.camera_distill_v10_cfg import (
    CameraDistillV10PickAppleEnvCfg,
    CameraDistillV10RunnerCfg,
)
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg


class DistillV10Tests(unittest.TestCase):
    def test_environment_and_runner_contract(self):
        env = CameraDistillV10PickAppleEnvCfg()
        runner = CameraDistillV10RunnerCfg().default
        self.assertIsNone(runner.clip_actions)
        self.assertEqual(runner.algorithm.class_name, "Distillation")
        self.assertEqual((env.actions.arm_action.scale, env.actions.gripper_action.scale), (0.033, 0.02))
        self.assertIsNone(env.actions.arm_action.clip)
        self.assertIsNone(env.actions.gripper_action.clip)
        self.assertEqual((env.scene.wrist_camera.width, env.scene.wrist_camera.height), (256, 192))
        self.assertEqual(env.episode_length_s, 90.0)
        task = gym.spec("Pick-Apple-SO101-Staged-Camera-Distill-v10")
        self.assertTrue(task.kwargs["env_cfg_entry_point"].endswith(
            "camera_distill_v10_cfg:CameraDistillV10PickAppleEnvCfg"))
        self.assertTrue(task.kwargs["rsl_rl_cfg_entry_point"].endswith(
            "camera_distill_v10_cfg:CameraDistillV10RunnerCfg"))
        for phase in ("early", "mid", "late"):
            self.assertEqual(env.events.reset_staged.params[f"curriculum_{phase}"], (0.125,) * 8)

    def test_smooth_teacher_checkpoint_loads_strictly(self):
        obs = TensorDict({"teacher": torch.zeros(2, 111)}, batch_size=[2])
        runner = handle_deprecated_rsl_rl_cfg(CameraDistillV10RunnerCfg().default, version("rsl-rl-lib"))
        cfg = runner.teacher.to_dict()
        cfg.pop("class_name")
        teacher = MLPModel(obs, {"teacher": ["teacher"]}, "teacher", 6, **cfg)
        checkpoint = ROOT / "logs/rsl_rl/pick_apple_staged_so101/2026-10-05_17-16-17_teacher_smooth_v1/model_2250.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        teacher.load_state_dict(payload["actor_state_dict"], strict=True)

    def test_labels_and_actions_are_unclipped(self):
        algorithm = object.__new__(Distillation)
        algorithm.student = lambda obs, stochastic_output: torch.tensor([[2.0, -1.5]])
        algorithm.teacher = lambda obs: torch.tensor([[2.3, -1.8]])
        algorithm.transition = SimpleNamespace()
        torch.testing.assert_close(algorithm.act(torch.zeros(1, 1)), torch.tensor([[2.0, -1.5]]))
        torch.testing.assert_close(algorithm.transition.privileged_actions, torch.tensor([[2.3, -1.8]]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
