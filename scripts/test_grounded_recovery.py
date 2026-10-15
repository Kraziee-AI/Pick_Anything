"""Regression cases for GT-mask PPO's sustained grasp and HOME reset recovery."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from grounded_pick_cli import configure_caches
configure_caches()

import torch
from isaaclab.envs import ManagerBasedRLEnv
from pick_anything.grounded_pick.env import GroundedPickEnv
from pick_anything.grounded_pick.progress import GroundedStageProgress
from pick_anything.tasks.pick_apple.stages import STAGE_NAMES, PickStage


class GroundedRecoveryTests(unittest.TestCase):
    @staticmethod
    def conditions(n):
        return {key: torch.zeros(n, dtype=torch.bool) for key in STAGE_NAMES}

    def test_sustained_contact_can_handoff_without_reopening(self):
        state = GroundedStageProgress(1, "cpu")
        state.stage[:] = PickStage.PREGRASP
        state.first_step[:] = False
        cond = self.conditions(1)
        cond["GRASP"][:] = True
        for step in range(1, 6):
            state.update(torch.tensor([step]), cond, {"GRASP": 3})
            self.assertEqual(state.stage.item(), 1 if step < 3 else 2 if step < 5 else 3)
            self.assertEqual(state.grasped.item(), step == 5)
        self.assertTrue(state.just_completed[0, PickStage.GRASP])
        self.assertFalse(cond["PREGRASP"].item())

    def test_duplicate_calls_and_partial_reset_preserve_other_rows(self):
        state = GroundedStageProgress(2, "cpu")
        state.stage[:] = PickStage.PREGRASP
        state.first_step[:] = False
        cond = self.conditions(2)
        cond["GRASP"][:] = True
        for step in range(1, 3):
            state.update(torch.full((2,), step), cond, {"GRASP": 3})
            before = {name: value.clone() for name, value in vars(state).items() if isinstance(value, torch.Tensor)}
            state.update(torch.full((2,), step), cond, {"GRASP": 3})
            for name, value in before.items():
                torch.testing.assert_close(getattr(state, name), value)
        state.reset(torch.tensor([0]), stage_seed=torch.tensor([0]))
        state.update(torch.tensor([0, 2]), cond, {"GRASP": 3})
        self.assertEqual(state.stage.tolist(), [0, 1])
        self.assertEqual(state.pregrasp_contact_count.tolist(), [0, 2])
        self.assertFalse(state.grasped.any())

    def test_contact_interruption_and_first_step_cannot_unlock(self):
        state = GroundedStageProgress(1, "cpu")
        state.stage[:] = PickStage.PREGRASP
        cond = self.conditions(1)
        for step, contact in enumerate([True, True, True, False, True, True]):
            cond["GRASP"][:] = contact
            state.update(torch.tensor([step]), cond, {"GRASP": 3})
        self.assertEqual(state.stage.item(), PickStage.PREGRASP)
        self.assertFalse(state.grasped.item())

    def test_same_step_does_not_count_twice_in_grasp_stage(self):
        state = GroundedStageProgress(1, "cpu")
        state.stage[:] = PickStage.GRASP
        state.first_step[:] = False
        cond = self.conditions(1)
        cond["GRASP"][:] = True
        state.update(torch.tensor([1]), cond, {"GRASP": 3})
        before = state.confirm_count.clone()
        state.update(torch.tensor([1]), cond, {"GRASP": 3})
        torch.testing.assert_close(state.confirm_count, before)
        self.assertFalse(state.grasped.item())

    def test_home_reset_restores_position_velocity_target_only_for_home(self):
        calls = []
        robot = SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.tensor([[0., 0.2555], [0., 0.2555]])))
        for method in ("write_joint_position_to_sim_index", "write_joint_velocity_to_sim_index", "set_joint_position_target_index"):
            setattr(robot, method, lambda _method=method, **kwargs: calls.append((_method, kwargs)))
        class Scene(dict):
            num_envs = 2
        env = object.__new__(GroundedPickEnv)
        env.__dict__.update(scene=Scene(robot=robot), sim=SimpleNamespace(device="cpu"),
                            cfg=SimpleNamespace(evaluate_first_episode=False),
                            staged_pick_state=SimpleNamespace(birth_stage=torch.tensor([0, 4]), gripper_id=1))
        with patch.object(ManagerBasedRLEnv, "_reset_idx", lambda self, ids: None):
            env._reset_idx(torch.tensor([0, 1]))
        self.assertEqual(len(calls), 3)
        for name, kwargs in calls:
            self.assertEqual(kwargs["env_ids"].tolist(), [0])
            self.assertEqual(kwargs["joint_ids"], [1])
            value = kwargs.get("position", kwargs.get("velocity", kwargs.get("target")))
            self.assertAlmostEqual(value.item(), 0 if "velocity" in name else 0.2555, places=5)

    def test_opening_progress_is_bounded_symmetric_and_stops_before_closure(self):
        from pick_anything.grounded_pick.rewards import PregraspOpeningProgress
        term = object.__new__(PregraspOpeningProgress)
        term._previous = torch.zeros(2)
        term._initialized = torch.zeros(2, dtype=torch.bool)
        state = SimpleNamespace(gripper_pos=torch.tensor([0.2, 0.2]),
                                progress=SimpleNamespace(stage=torch.tensor([1, 2]), grasped=torch.tensor([False, False])),
                                update=lambda env: None)
        env = SimpleNamespace(staged_pick_state=state)
        self.assertEqual(term(env).tolist(), [0., 0.])
        state.gripper_pos[:] = 0.8
        paid = term(env)
        torch.testing.assert_close(paid, torch.tensor([3., 0.]))
        self.assertEqual(term(env).tolist(), [0., 0.])
        state.gripper_pos[:] = 0.2
        torch.testing.assert_close(term(env) + paid, torch.zeros(2))
        state.gripper_pos[:] = 1.7
        torch.testing.assert_close(term(env), torch.tensor([5.5, 0.]))
        state.gripper_pos[:] = 1.4
        self.assertEqual(term(env).tolist(), [0., 0.])
        state.progress.stage[:] = 2
        state.gripper_pos[:] = 0.4
        self.assertEqual(term(env).tolist(), [0., 0.])
        term.reset(torch.tensor([0]))
        state.progress.stage[:] = 1
        state.gripper_pos[:] = 1.0
        self.assertEqual(term(env)[0].item(), 0.)

    def test_config_validation_and_reward_scale(self):
        from pick_anything.grounded_pick.config import GroundedPickEnvCfg
        cfg = GroundedPickEnvCfg()
        cfg.validate()
        self.assertAlmostEqual(cfg.rewards.pregrasp_open_progress.weight, 0.2)
        cfg.observations.rgb_mask.image.params["mask_dropout"] = 1.0
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
