"""CPU checks for the unclipped camera-distillation tasks (isaaclab_61).

Run: python scripts/test_distill_action_contract.py
Uses local teacher checkpoints; no simulator launch or training required.
"""
from pathlib import Path
from types import SimpleNamespace
from importlib.metadata import version
import sys
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import Distillation
from rsl_rl.models import MLPModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import pick_anything.tasks  # noqa: E402,F401
from pick_anything.tasks.pick_apple.agents.camera_distill_cfg import (  # noqa: E402
    CameraDistillUnclippedRunnerCfg,
    CameraDistillV9RunnerCfg,
)
from pick_anything.tasks.pick_apple.distill_env_cfg import (  # noqa: E402
    CameraDistillUnclippedPickAppleEnvCfg,
    CameraDistillV9PickAppleEnvCfg,
)
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402


class TestActionContract(unittest.TestCase):
    def test_new_tasks_preserve_control_scales_and_do_not_clip(self):
        cfg = CameraDistillUnclippedRunnerCfg().default
        self.assertIsNone(cfg.clip_actions)
        self.assertEqual(cfg.algorithm.class_name, "Distillation")
        for task, cls, scales in (
            ("Pick-Apple-SO101-Staged-Camera-Distill-Unclipped-v0",
             CameraDistillUnclippedPickAppleEnvCfg, (0.06, 0.10)),
            ("Pick-Apple-SO101-Staged-Camera-Distill-v10",
             CameraDistillV9PickAppleEnvCfg, (0.033, 0.02)),
        ):
            with self.subTest(task=task):
                env = cls()
                self.assertEqual((env.actions.arm_action.scale, env.actions.gripper_action.scale), scales)
                self.assertIsNone(env.actions.arm_action.clip)
                self.assertIsNone(env.actions.gripper_action.clip)
                self.assertEqual((env.scene.wrist_camera.width, env.scene.wrist_camera.height), (256, 192))
                self.assertTrue(gym.spec(task).kwargs["rsl_rl_cfg_entry_point"].endswith(
                    ":CameraDistillUnclippedRunnerCfg"))
        old = CameraDistillV9RunnerCfg().default
        self.assertEqual(old.clip_actions, 1.0)
        self.assertTrue(old.algorithm.class_name.endswith(":BoundedTeacherDistillation"))

    def test_both_teacher_checkpoints_load_strictly(self):
        obs = TensorDict({"teacher": torch.zeros(2, 111)}, batch_size=[2])
        runner = handle_deprecated_rsl_rl_cfg(CameraDistillUnclippedRunnerCfg().default, version("rsl-rl-lib"))
        cfg = runner.teacher.to_dict()
        cfg.pop("class_name")
        for relative in (
            "2026-09-26_16-19-15_retreat_shaping_v2/model_12999.pt",
            "2026-10-05_17-16-17_teacher_smooth_v1/model_2250.pt",
        ):
            with self.subTest(checkpoint=relative):
                teacher = MLPModel(obs, {"teacher": ["teacher"]}, "teacher", 6, **cfg)
                payload = torch.load(ROOT / "logs/rsl_rl/pick_apple_staged_so101" / relative,
                                     map_location="cpu", weights_only=False)
                teacher.load_state_dict(payload["actor_state_dict"], strict=True)

    def test_raw_teacher_labels_and_student_actions_survive_act(self):
        # A saturated teacher and student expose either accidental clamp.
        algorithm = object.__new__(Distillation)
        algorithm.student = lambda obs, stochastic_output: torch.tensor([[2.0, -1.5]]).expand(3, -1)
        algorithm.teacher = lambda obs: torch.tensor([[2.3, -1.8]]).expand(3, -1)
        algorithm.transition = SimpleNamespace()
        actions = algorithm.act(torch.zeros(3, 1))
        self.assertTrue(torch.equal(actions, torch.tensor([[2.0, -1.5]]).expand(3, -1)))
        self.assertTrue(torch.equal(algorithm.transition.privileged_actions,
                                    torch.tensor([[2.3, -1.8]]).expand(3, -1)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
