"""CPU contracts for temporal RGB/mask observations and real PPO transfer."""

from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

from grounded_pick_cli import PROJECT_ROOT, configure_caches

configure_caches()

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from pick_anything.grounded_pick.checkpoints import (
    image_history_from_checkpoint, initialize_from_rgb, rgb_to_rgb_mask_checkpoint,
)
from pick_anything.grounded_pick.observations import GroundedRGBMask
from pick_anything.grounded_pick.config import GroundedPickRunnerCfg
from pick_anything.tasks.pick_apple.agents.camera_ppo_cfg import CameraPPOBaseRunnerCfg


SOURCE = PROJECT_ROOT / "models/camera_ppo/2026-10-12_15-10-13_camera_stage_recovery_v2/model_17596.pt"


class GroundedTransferTests(unittest.TestCase):
    def test_history_alignment_duplicate_read_and_partial_reset(self):
        term = object.__new__(GroundedRGBMask)
        term._history = None
        term._history_initialized = torch.zeros(2, dtype=torch.bool)
        term._history_frame = torch.full((2,), -1)
        term._frame = torch.full((2,), -1)
        term._boxes = torch.zeros(2, 4)
        term._valid = torch.zeros(2, dtype=torch.bool)
        term._confidence = torch.zeros(2)
        term._exposure = torch.ones(2, 1, 1, 1)
        term._white_balance = torch.ones(2, 3, 1, 1)

        def frame(value, mask):
            image = torch.full((2, 4, 2, 3), value)
            image[:, 3] = mask
            return image

        first = term._stack_image(frame(0.1, 1), torch.ones(2, dtype=torch.long), 3)
        second = term._stack_image(frame(0.2, 0), torch.full((2,), 2), 3)
        torch.testing.assert_close(second[:, :4], frame(0.1, 1))
        torch.testing.assert_close(second[:, 4:8], frame(0.1, 1))
        torch.testing.assert_close(second[:, 8:], frame(0.2, 0))
        duplicate = term._stack_image(frame(0.9, 1), torch.full((2,), 2), 3)
        torch.testing.assert_close(second, duplicate)
        # Returned previous observations must not mutate when buffers advance.
        torch.testing.assert_close(first[:, 8:], frame(0.1, 1))
        term.reset(torch.tensor([0]))
        reset = term._stack_image(frame(0.7, 1), torch.full((2,), 3), 3)
        for start in (0, 4, 8):
            torch.testing.assert_close(reset[0, start:start + 4], frame(0.7, 1)[0])
        torch.testing.assert_close(reset[1, :4], frame(0.1, 1)[1])
        torch.testing.assert_close(reset[1, 4:8], frame(0.2, 0)[1])
        self.assertTrue(set(reset[:, [3, 7, 11]].unique().tolist()).issubset({0., 1.}))
        with self.assertRaises(ValueError):
            term._stack_image(frame(0.7, 1), torch.full((2,), 3), 2)

    @staticmethod
    def algorithm(runner_cfg, obs):
        cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, version("rsl-rl-lib")).to_dict()
        cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
        cfg.update(num_steps_per_env=2, multi_gpu=None)
        return PPO.construct_algorithm(obs, SimpleNamespace(num_envs=2, num_actions=6), cfg, "cpu")

    def test_real_actor_identity_optimizer_reset_and_mask_ppo_update(self):
        torch.manual_seed(42)
        saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
        converted = rgb_to_rgb_mask_checkpoint(saved)
        self.assertEqual(converted["iter"], 0)
        self.assertFalse(converted["optimizer_state_dict"]["state"])
        self.assertEqual(saved["actor_state_dict"]["cnns.wrist_rgb.0.weight"].shape[1], 3)
        current = torch.rand(2, 3, 96, 128)
        proprio, critic = torch.rand(2, 24), torch.rand(2, 49)
        original_obs = TensorDict({"wrist_rgb": current, "proprioception": proprio, "critic": critic}, [2])
        history = torch.rand(2, 12, 96, 128)
        history[:, 8:11] = current
        masked_obs = TensorDict({"rgb_mask": history, "proprioception": proprio, "critic": critic}, [2])
        original = self.algorithm(CameraPPOBaseRunnerCfg(), original_obs)
        target = self.algorithm(GroundedPickRunnerCfg().default, masked_obs)
        original.load(saved, None, True)
        target.load(converted, None, True)
        self.assertEqual(target.learning_rate, 2e-5)
        self.assertFalse(target.optimizer.state)
        original.eval_mode()
        target.eval_mode()
        with torch.no_grad():
            torch.testing.assert_close(original.get_policy()(original_obs), target.get_policy()(masked_obs),
                                       rtol=2e-5, atol=2e-5)
        target.train_mode()
        for _ in range(2):
            target.act(masked_obs)
            target.process_env_step(masked_obs, torch.tensor([1., -1.]), torch.zeros(2, dtype=torch.bool), {})
        target.compute_returns(masked_obs)
        losses = target.update()
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all() for value in losses.values()))
        weight = target.get_policy().state_dict()["cnns.rgb_mask.0.weight"]
        self.assertGreater(float(weight[:, [3, 7, 11]].abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(value).all() for value in target.get_policy().state_dict().values()))

    def test_initializations_are_distinct_idempotent_and_infer_history(self):
        parent = PROJECT_ROOT / "outputs/grounded_pick/cache/tmp"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            one = initialize_from_rgb(SOURCE, directory, 1)
            three = initialize_from_rgb(SOURCE, directory, 3)
            self.assertNotEqual(one, three)
            self.assertEqual(image_history_from_checkpoint(one), 1)
            self.assertEqual(image_history_from_checkpoint(three), 3)
            before = three.stat().st_mtime_ns
            self.assertEqual(initialize_from_rgb(SOURCE, directory, 3), three)
            self.assertEqual(three.stat().st_mtime_ns, before)
            with self.assertRaises(ValueError):
                initialize_from_rgb(SOURCE, PROJECT_ROOT.parent / "outside_initializations")

    def test_reject_incompatible_source_and_invalid_history(self):
        source = torch.load(SOURCE, map_location="cpu", weights_only=False)
        with self.assertRaises(ValueError):
            rgb_to_rgb_mask_checkpoint(source, 2)
        with self.assertRaises(ValueError):
            rgb_to_rgb_mask_checkpoint(source, 3, 0)
        source["actor_state_dict"]["cnns.wrist_rgb.0.weight"] = torch.zeros(16, 4, 5, 5)
        with self.assertRaises(ValueError):
            rgb_to_rgb_mask_checkpoint(source)

    def test_empty_claw_cannot_confirm_a_new_grasp(self):
        from pick_anything.grounded_pick.progress import GroundedStageProgress
        from pick_anything.tasks.pick_apple.stages import STAGE_NAMES

        physical = SimpleNamespace(gripper_pos=torch.tensor([-0.16]),
                                   orange_pos=torch.zeros(1, 3), pocket_pos=torch.zeros(1, 3))
        progress = GroundedStageProgress(1, "cpu", physical)
        progress.stage[:] = 1
        progress.first_step[:] = False
        conditions = {name: torch.tensor([name == "GRASP"]) for name in STAGE_NAMES}
        for step in range(1, 7):
            progress.update(torch.tensor([step]), conditions, {"GRASP": 3})
        self.assertEqual(progress.stage.item(), 1)
        self.assertFalse(progress.grasped.item())
        physical.gripper_pos[:] = 0.4
        for step in range(7, 12):
            progress.update(torch.tensor([step]), conditions, {"GRASP": 3})
        self.assertTrue(progress.grasped.item())
        self.assertEqual(progress.stage.item(), 3)

    def test_far_table_contact_cannot_confirm_even_with_positive_angle(self):
        from pick_anything.grounded_pick.progress import GroundedStageProgress
        from pick_anything.tasks.pick_apple.stages import STAGE_NAMES

        physical = SimpleNamespace(gripper_pos=torch.tensor([0.2]), orange_pos=torch.zeros(1, 3),
                                   pocket_pos=torch.tensor([[0.06, 0., 0.]]))
        progress = GroundedStageProgress(1, "cpu", physical)
        progress.stage[:] = 1
        progress.first_step[:] = False
        conditions = {name: torch.tensor([name == "GRASP"]) for name in STAGE_NAMES}
        for step in range(1, 7):
            progress.update(torch.tensor([step]), conditions, {"GRASP": 3})
        self.assertFalse(progress.grasped.item())
        physical.pocket_pos[:, 0] = 0.026
        for step in range(7, 12):
            progress.update(torch.tensor([step]), conditions, {"GRASP": 3})
        self.assertTrue(progress.grasped.item())

    def test_approach_recovers_after_contact_and_empty_penalty_stops_after_lift(self):
        from pick_anything.grounded_pick.rewards import GroundedApproachProgress, empty_gripper_before_lift

        term = object.__new__(GroundedApproachProgress)
        term._prev_dist = torch.tensor([0.21])
        term._has_prev = torch.tensor([True])
        state = SimpleNamespace(orange_pos=torch.zeros(1, 3), pocket_pos=torch.tensor([[0.2, 0., 0.]]),
                                gripper_pos=torch.tensor([-0.16]), update=lambda env: None,
                                progress=SimpleNamespace(grasped=torch.tensor([True]), lifted=torch.tensor([False]),
                                                         stage=torch.tensor([3])))
        env = SimpleNamespace(staged_pick_state=state, cfg=SimpleNamespace(grasp_min_gripper_rad=0.0))
        self.assertAlmostEqual(term(env).item(), 1.0, places=5)
        self.assertEqual(empty_gripper_before_lift(env).item(), 1.)
        state.gripper_pos[:] = 0.4
        self.assertEqual(empty_gripper_before_lift(env).item(), 0.)
        state.progress.lifted[:] = True
        state.gripper_pos[:] = -0.16
        state.pocket_pos[:, 0] = 0.1
        self.assertEqual(term(env).item(), 0.)
        self.assertEqual(empty_gripper_before_lift(env).item(), 0.)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
