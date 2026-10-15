"""CPU contracts: camera geometry, failure handling, actual PPO update and isolation."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import sys
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounded_pick_cli import configure_caches

configure_caches()

import gymnasium as gym
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO

from pick_anything.grounded_pick import TASK_ID, register_task
from pick_anything.grounded_pick.grounding import workspace_output
from pick_anything.grounded_pick.masks import boxes_to_masks, orange_caption, select_detection, sphere_boxes
from pick_anything.tasks.pick_apple.camera_env_cfg import CameraClosureFinetuneEnvCfg
from pick_anything.tasks.pick_apple.agents.camera_ppo_cfg import CameraPPORunnerCfg
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg


class GroundedPickTests(unittest.TestCase):
    @staticmethod
    def stage_conditions(n=1):
        from pick_anything.tasks.pick_apple.stages import STAGE_NAMES

        return {name: torch.zeros(n, dtype=torch.bool) for name in STAGE_NAMES}

    def test_grounded_early_contact_keeps_approach_and_pregrasp_available(self):
        from pick_anything.grounded_pick.progress import GroundedStageProgress
        from pick_anything.tasks.pick_apple.stages import StageProgress
        from pick_anything.tasks.pick_apple.camera_mdp import CameraApproachProgress

        fixed, legacy = GroundedStageProgress(1, "cpu"), StageProgress(1, "cpu")
        for progress in (fixed, legacy):
            progress.stage[:] = 1
            progress.first_step[:] = False
            for step in range(1, 3):
                cond = self.stage_conditions()
                cond["GRASP"][:] = True  # contact before accurate open-gripper alignment
                progress.update(torch.tensor([step]), cond, {"GRASP": 3})
        self.assertFalse(fixed.grasped.item())
        self.assertTrue(legacy.grasped.item())  # shared tasks deliberately retain their original semantics

        # Real approach reward must remain available after that early contact.
        term = object.__new__(CameraApproachProgress)
        term._prev_dist, term._has_prev = torch.tensor([0.04]), torch.tensor([True])
        state = SimpleNamespace(progress=fixed, orange_pos=torch.zeros(1, 3),
                                pocket_pos=torch.tensor([[0.03, 0., 0.]]), update=lambda env: None)
        self.assertAlmostEqual(term(SimpleNamespace(staged_pick_state=state)).item(), 1.0, places=5)

        for progress in (fixed, legacy):
            cond = self.stage_conditions()
            cond["PREGRASP"] = ~progress.grasped  # geometry/opening now satisfied
            progress.update(torch.tensor([5]), cond, {"PREGRASP": 1, "GRASP": 3})
        self.assertEqual(fixed.stage.item(), 2)
        self.assertEqual(legacy.stage.item(), 1)
        self.assertTrue(fixed.just_completed[0, 1].item())

    def test_grounded_grasp_requires_consecutive_confirmation_and_pays_once(self):
        from pick_anything.grounded_pick.progress import GroundedStageProgress

        progress = GroundedStageProgress(1, "cpu")
        progress.stage[:] = 2
        progress.first_step[:] = False
        for step, contact in enumerate([True, True, False, True, True, True], 1):
            cond = self.stage_conditions()
            cond["GRASP"][:] = contact
            progress.update(torch.tensor([step]), cond, {"GRASP": 3})
            self.assertEqual(progress.grasped.item(), step == 6)
            self.assertEqual(progress.stage.item(), 3 if step == 6 else 2)
        self.assertTrue(progress.just_completed[0, 2].item())
        self.assertTrue(progress.paid[0, 2].item())
        progress.update(torch.tensor([7]), cond, {"GRASP": 3})
        self.assertTrue(progress.grasped.item())
        self.assertFalse(progress.just_completed[0, 2].item())

    def test_grounded_first_step_bank_seeds_and_partial_reset(self):
        from pick_anything.grounded_pick.progress import GroundedStageProgress

        progress = GroundedStageProgress(3, "cpu")
        progress.reset(stage_seed=torch.tensor([0, 2, 3]))
        self.assertEqual(progress.grasped.tolist(), [False, True, True])
        cond = self.stage_conditions(3)
        cond["GRASP"][:] = True
        cond["LIFT"][:] = True
        progress.update(torch.ones(3, dtype=torch.long), cond, {"GRASP": 3, "LIFT": 4})
        self.assertFalse(progress.just_completed.any().item())
        self.assertEqual(progress.stage.tolist(), [0, 2, 3])
        progress.reset(torch.tensor([1]), stage_seed=torch.tensor([0]))
        self.assertEqual(progress.grasped.tolist(), [False, False, True])
        self.assertEqual(progress.stage.tolist(), [0, 0, 3])

    def test_grounded_installs_progress_before_managers_without_shared_patch(self):
        from pick_anything.grounded_pick.env import GroundedPickEnv, StagedPickState
        from pick_anything.grounded_pick.progress import GroundedStageProgress
        from pick_anything.tasks.pick_apple.stages import StageProgress
        from isaaclab.envs import ManagerBasedRLEnv

        state = SimpleNamespace(progress=StageProgress(2, "cpu"), gripper_pos=torch.zeros(2))
        legacy = StageProgress(2, "cpu")
        instance = object.__new__(GroundedPickEnv)
        instance.__dict__.update(scene=SimpleNamespace(num_envs=2), sim=SimpleNamespace(device="cpu"),
                                 cfg=SimpleNamespace(grasp_min_gripper_rad=0.0, grasp_confirm_proximity_m=0.035))
        seen = []
        def managers(env):
            seen.append(isinstance(state.progress, GroundedStageProgress))
        with patch.object(StagedPickState, "get", return_value=state), patch.object(ManagerBasedRLEnv, "load_managers", managers):
            instance.load_managers()
        self.assertEqual(seen, [True])
        self.assertEqual(type(legacy), StageProgress)

    def test_diagnostics_distinguish_latch_from_confirmed_stage(self):
        from pick_anything.grounded_pick.diagnostics import EpisodeDiagnostics

        pos = torch.tensor([[0., 0., 0.07]])
        progress = SimpleNamespace(stage=torch.tensor([1]), confirm_count=torch.zeros(1, 8, dtype=torch.long),
                                   grasped=torch.tensor([True]), lifted=torch.tensor([False]),
                                   over_box_latched=torch.tensor([False]), in_box_latched=torch.tensor([False]))
        state = SimpleNamespace(progress=progress, orange_pos=pos, pocket_pos=pos + torch.tensor([[0.04, 0, 0]]),
                                orange_vel=torch.zeros(1, 3), claw_force=torch.tensor([3.]),
                                orange_force=torch.tensor([0.3]), gripper_pos=torch.tensor([0.4]))
        robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1, 6), joint_pos_target=torch.ones(1, 6)))
        class Scene(dict):
            env_origins = torch.zeros(1, 3)
            sensors = {"wrist_camera": SimpleNamespace(cfg=SimpleNamespace(width=16, height=12))}
        scene = Scene(robot=robot, orange=SimpleNamespace(data=SimpleNamespace(root_pos_w=pos)))
        env = SimpleNamespace(num_envs=1, device="cpu", scene=scene, staged_pick_state=state,
                              episode_length_buf=torch.tensor([1]),
                              action_manager=SimpleNamespace(action=torch.zeros(1, 6)),
                              grounding_diagnostics={"boxes": torch.tensor([[2., 3., 6., 8.]]), "detected": torch.tensor([True])},
                              termination_manager=SimpleNamespace(get_term=lambda name: torch.tensor([name == "time_out"])))
        output = Path(__file__).resolve().parents[1] / "outputs/grounded_pick"
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as directory:
            diagnostic = EpisodeDiagnostics(directory, 1, 1/30, 5)
            diagnostic.start_episodes(env, [0])
            diagnostic.capture(env)
            diagnostic.capture(env)  # duplicate callbacks must not extend runs
            for step in [2, 3]:
                env.episode_length_buf[:] = step
                diagnostic.capture(env)
            env.episode_length_buf[:] = 4
            state.claw_force[:] = 0
            diagnostic.capture(env, [0], terminal=True)
            diagnostic.end_episodes(env, [0])
            diagnostic.start_episodes(env, [0])
            diagnostic.capture(env)  # autoreset / later episodes must be ignored
            diagnostic.finish({"episodes": 1}, seed=42)
            report = json.loads((Path(directory) / "summary.json").read_text())
            ep = report["episodes"][0]
            self.assertEqual(ep["steps"], 4)
            self.assertEqual(ep["max_grasp_raw_run_steps"], 3)
            self.assertTrue(ep["grasp_continuous_3_steps"])
            self.assertTrue(ep["grasp_latched_before_grasp_stage"])
            self.assertEqual(ep["max_stage"], "PREGRASP")
            self.assertEqual(ep["height_above_lift_steps"], 0)
            self.assertEqual(ep["mask_nonempty_fraction"], 1.0)
            self.assertEqual(ep["termination"], ["time_out"])
            trace = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text().splitlines()]
            self.assertEqual(trace[-1]["step"], 4)
            self.assertEqual(trace[-1]["mask_timing"], "preceding_observation")

    def test_detector_branch_only_reads_rgb_and_caches_camera_frames(self):
        from pick_anything.grounded_pick.observations import GroundedRGBMask

        term = object.__new__(GroundedRGBMask)
        term._frame = torch.tensor([-1])
        term._boxes = torch.zeros(1, 4)
        term._valid = torch.zeros(1, dtype=torch.bool)
        term._confidence = torch.zeros(1)
        calls = []

        def predict(rgb, instruction):
            calls.append(instruction)
            return torch.tensor([[2., 3., 6., 8.]]), torch.tensor([True]), torch.tensor([0.8])

        term.detector = SimpleNamespace(predict=predict)
        camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": torch.zeros(1, 12, 16, 3, dtype=torch.uint8)}),
                                 frame=torch.tensor([1]))
        # No asset lookup, camera pose, intrinsics, stage or object coordinates exist here.
        env = SimpleNamespace(scene=SimpleNamespace(sensors={"camera": camera}),
                              cfg=SimpleNamespace(observations=SimpleNamespace(rgb_mask=SimpleNamespace(enable_corruption=False))))
        kwargs = dict(sensor_cfg=SimpleNamespace(name="camera"), backend="dino", instruction="抓橘子",
                      model_name="unused", grounding_device="cpu", cache_dir="unused", local_files_only=True,
                      box_threshold=0.25, text_threshold=0.25, radius=0.0281,
                      policy_height=12, policy_width=16, box_jitter=0, mask_dropout=0,
                      dump_dir="", dump_every=1, dump_max=1)
        first = term(env, **kwargs)
        second = term(env, **kwargs)
        self.assertEqual(calls, ["抓橘子"])
        self.assertEqual(tuple(first.shape), (1, 4, 12, 16))
        self.assertEqual(int(first[:, 3].sum()), 20)
        self.assertTrue(torch.equal(first, second))
        camera.frame += 1
        term(env, **kwargs)
        self.assertEqual(len(calls), 2)

    def test_terminal_milestones_count_before_autoreset(self):
        from pick_anything.grounded_pick.env import GroundedPickEnv
        from isaaclab.envs import ManagerBasedRLEnv

        cfg = SimpleNamespace(evaluate_first_episode=True,
                              events=SimpleNamespace(reset_staged=SimpleNamespace(params={"forced_stage": 0})),
                              observations=SimpleNamespace(rgb_mask=SimpleNamespace(image=SimpleNamespace(
                                  params={"backend": "gt", "instruction": "抓橘子"}))))
        flags = SimpleNamespace(**{name: torch.tensor([True, False]) for name in
                                   ("grasped", "lifted", "over_box_latched", "in_box_latched")})
        causes = {name: torch.tensor([name == "success", name == "time_out"]) for name in
                  ("success", "time_out", "staged_orange_dropped", "staged_box_tipped", "staged_abnormal_robot")}
        env = SimpleNamespace(cfg=cfg, scene=SimpleNamespace(num_envs=2), sim=SimpleNamespace(device="cpu"), _evaluation_seen=None,
                              _evaluation_counts=dict.fromkeys(["episodes", *causes, *vars(flags)], 0),
                              episode_length_buf=torch.ones(2), staged_pick_state=SimpleNamespace(
                                  progress=flags, birth_stage=torch.tensor([4, 4])),
                              termination_manager=SimpleNamespace(get_term=causes.__getitem__))
        instance = object.__new__(GroundedPickEnv)
        instance.__dict__.update(vars(env))
        env = instance
        def reset(_env, ids):
            for flag in vars(flags).values():
                flag[ids] = False
        with patch.object(ManagerBasedRLEnv, "_reset_idx", reset):
            GroundedPickEnv._reset_idx(env, [0])
            GroundedPickEnv._reset_idx(env, [0])  # second episode must not be counted
            with self.assertRaises(SystemExit):
                GroundedPickEnv._reset_idx(env, [1])
        self.assertEqual(env._evaluation_counts["grasped"], 1)
        self.assertEqual(env._evaluation_counts["success"], 1)
        self.assertEqual(env._evaluation_counts["time_out"], 1)

    def test_language_and_empty_detection(self):
        self.assertEqual(orange_caption("抓橘子放入盒子"), "orange fruit.")
        self.assertEqual(orange_caption("pick up the orange"), "orange fruit.")
        self.assertEqual(orange_caption("orange on the left"), "orange on the left.")
        for instruction in ("抓苹果", "pick the apple", "", "抓香蕉", "抓橘子和苹果", "抓橙色盒子", "orange box"):
            with self.assertRaises(ValueError):
                orange_caption(instruction)
        for result in (
            {"boxes": torch.empty(0, 4), "scores": torch.empty(0)},
            {"boxes": torch.tensor([[float('nan'), 0, 4, 4], [3., 2, 1, 5]]), "scores": torch.ones(2)},
        ):
            box, valid, confidence = select_detection(result, "cpu")
            self.assertFalse(valid)
            self.assertEqual(confidence, 0.0)
            self.assertEqual(boxes_to_masks(box[None], torch.tensor([valid]), 10, 20).sum(), 0)
        with self.assertRaises(ValueError):
            workspace_output("/home/user123/GCRL_UR10e/generated")

    def test_projection_and_rasterization(self):
        centers = torch.tensor([[0., 0., 1.], [0., 0., -1.], [10., 0., 1.]])
        camera = torch.zeros_like(centers)
        quat = torch.tensor([[0., 0., 0., 1.]]).repeat(3, 1)
        k = torch.tensor([[100., 0, 50], [0, 100, 40], [0, 0, 1]]).repeat(3, 1, 1)
        boxes, valid = sphere_boxes(centers, camera, quat, k, 0.1)
        # Exact perspective tangent rather than radius/depth approximation.
        half_side = 100 * 0.1 / (1 - 0.1**2)**0.5
        torch.testing.assert_close(boxes[0], torch.tensor([50-half_side, 40-half_side, 50+half_side, 40+half_side]))
        mask = boxes_to_masks(boxes, valid, 80, 100)
        self.assertEqual(set(mask.unique().tolist()), {0., 1.})
        self.assertGreater(int(mask[0].sum()), 0)
        self.assertEqual(int(mask[1:].sum()), 0)
        # Same relative pose under arbitrary world translation yields identical masks.
        shift = torch.tensor([2., -4., 6.])
        moved, moved_valid = sphere_boxes(centers + shift, camera + shift, quat, k, 0.1)
        torch.testing.assert_close(moved, boxes)
        self.assertTrue(torch.equal(moved_valid, valid))
        # xyzw rotation: local forward +Z becomes world +X for +90 deg about Y.
        q = torch.tensor([[0., 2**-0.5, 0., 2**-0.5]])
        rotated, _ = sphere_boxes(torch.tensor([[1., 0., 0.]]), camera[:1], q, k[:1], 0.1)
        torch.testing.assert_close(rotated, boxes[:1])

    def test_config_isolation_and_ppo_update(self):
        old_env = CameraClosureFinetuneEnvCfg().to_dict()
        old_runner = CameraPPORunnerCfg().to_dict()
        self.assertNotIn(TASK_ID, gym.registry)
        register_task()
        from pick_anything.grounded_pick.config import GroundedPickEnvCfg, GroundedPickRunnerCfg

        env = GroundedPickEnvCfg()
        env.validate()
        self.assertEqual((env.scene.wrist_camera.width, env.scene.wrist_camera.height), (256, 192))
        self.assertEqual(env.observations.rgb_mask.image.params["radius"], 0.0281)
        env.play_mode()
        self.assertEqual(env.events.reset_staged.params["forced_stage"], 0)
        self.assertFalse(env.observations.rgb_mask.enable_corruption)
        runner = handle_deprecated_rsl_rl_cfg(GroundedPickRunnerCfg().default, version("rsl-rl-lib"))
        self.assertEqual(runner.obs_groups["actor"], ["rgb_mask", "proprioception"])
        self.assertEqual(CameraClosureFinetuneEnvCfg().to_dict(), old_env)
        self.assertEqual(CameraPPORunnerCfg().to_dict(), old_runner)
        obs = TensorDict({"rgb_mask": torch.rand(4, 4, 96, 128),
                          "proprioception": torch.zeros(4, 24), "critic": torch.zeros(4, 49)}, batch_size=[4])
        cfg = runner.to_dict()
        cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
        cfg.update(num_steps_per_env=2, multi_gpu=None)
        algorithm = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=4, num_actions=6), cfg, "cpu")
        for _ in range(2):
            actions = algorithm.act(obs)
            self.assertEqual(tuple(actions.shape), (4, 6))
            algorithm.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
        algorithm.compute_returns(obs)
        losses = algorithm.update()
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all() for value in losses.values()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
