"""CPU contracts for open-vocabulary detection and multi-object PPO."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from grounded_pick_cli import configure_caches
configure_caches()

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from pick_anything.grounded_pick.config import GroundedPickEnvCfg
from pick_anything.grounded_pick.language import goal_caption
from pick_anything.grounded_pick.grounding import GroundingDino
from pick_anything.grounded_pick.masks import orange_caption
from pick_anything.open_vocab_pick.config import OpenVocabPickEnvCfg, OpenVocabPickRunnerCfg, OBJECT_KEYS
from pick_anything.open_vocab_pick.geometry import oriented_boxes
from pick_anything.open_vocab_pick.observations import OpenVocabRGBMask
from pick_anything.open_vocab_pick import mdp


class OpenVocabTests(unittest.TestCase):
    def test_free_captions_and_legacy_validation(self):
        for instruction, expected in [
            ("pick up the striped ceramic mug", "striped ceramic mug."),
            ("grab the blue cube on the left", "blue cube on the left."),
            ("抓小黄鸭", "yellow rubber duck."), ("抓橘子放入盒子", "orange fruit."),
            ("抓苹果", "apple."), ("pick the orange bottle", "orange bottle."),
        ]:
            self.assertEqual(goal_caption(instruction), expected)
        self.assertEqual(goal_caption("抓左侧的杯子", "white mug on the left"), "white mug on the left.")
        for text in ["", "grab the", "pick it", "apple and orange", "抓未知目标"]:
            with self.assertRaises(ValueError):
                goal_caption(text)
        with self.assertRaises(ValueError):
            orange_caption("抓小黄鸭")

    def test_real_detector_api_receives_open_caption_and_handles_empty(self):
        captions = []
        class Inputs(dict):
            input_ids = torch.tensor([[1]])
            def to(self, device):
                return self
        class Processor:
            def __call__(self, images, text, return_tensors):
                captions.append(text)
                return Inputs()
            def post_process_grounded_object_detection(self, *args, **kwargs):
                return [{"boxes": torch.empty(0, 4), "scores": torch.empty(0)}]
        detector = object.__new__(GroundingDino)
        detector.processor, detector.model = Processor(), lambda **kwargs: None
        detector.device, detector.box_threshold, detector.text_threshold = "cpu", 0.25, 0.25
        boxes, found, score = detector.predict(torch.zeros(2, 8, 12, 3, dtype=torch.uint8), "striped ceramic mug")
        self.assertEqual(captions, ["striped ceramic mug."] * 2)
        self.assertFalse(found.any())
        self.assertEqual(float(boxes.sum() + score.sum()), 0)

    def test_oriented_bounds_and_near_plane(self):
        centers = torch.tensor([[0., 0., 1.], [0., 0., -1.]])
        q = torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)
        k = torch.tensor([[100., 0, 50], [0, 100, 40], [0, 0, 1]]).repeat(2, 1, 1)
        half = torch.tensor([[0.1, 0.2, 0.1]]).repeat(2, 1)
        boxes, valid = oriented_boxes(centers, q, half, torch.zeros_like(centers), q, k)
        torch.testing.assert_close(boxes[0], torch.tensor([50-100/9, 40-200/9, 50+100/9, 40+200/9]))
        self.assertEqual(valid.tolist(), [True, False])
        rotated = q.clone()
        rotated[0] = torch.tensor([0., 0., 2**-0.5, 2**-0.5])
        box, _ = oriented_boxes(centers, rotated, half, torch.zeros_like(centers), q, k)
        self.assertGreater(float(box[0, 2] - box[0, 0]), float(boxes[0, 2] - boxes[0, 0]))

    def test_detector_observation_no_scene_truth_and_language_cache(self):
        term = object.__new__(OpenVocabRGBMask)
        term.caption = ""
        term._frame = torch.tensor([-1])
        term._boxes, term._valid, term._confidence = torch.zeros(1, 4), torch.tensor([False]), torch.zeros(1)
        calls = []
        def predict(rgb, instruction, caption):
            calls.append(instruction)
            valid = instruction != "missing cup"
            return torch.tensor([[2., 3., 6., 8.]]), torch.tensor([valid]), torch.tensor([0.8 if valid else 0.])
        term.detector = SimpleNamespace(predict=predict)
        camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": torch.zeros(1, 12, 16, 3, dtype=torch.uint8)}),
                                 frame=torch.tensor([1]))
        # No scene asset lookup, goal ID, object coordinates or stage is available.
        env = SimpleNamespace(scene=SimpleNamespace(sensors={"camera": camera}),
                              cfg=SimpleNamespace(observations=SimpleNamespace(rgb_mask=SimpleNamespace(enable_corruption=False))))
        kwargs = dict(sensor_cfg=SimpleNamespace(name="camera"), backend="dino", instruction="blue cube",
                      model_name="unused", grounding_device="cpu", cache_dir="unused", local_files_only=True,
                      box_threshold=0.25, text_threshold=0.25, radius=0.0281, policy_height=12, policy_width=16,
                      box_jitter=0, mask_dropout=0, dump_dir="", dump_every=1, dump_max=1)
        first = term(env, **kwargs)
        self.assertTrue(torch.equal(first, term(env, **kwargs)))
        self.assertEqual(len(calls), 1)
        kwargs["instruction"] = "missing cup"
        missing = term(env, **kwargs)
        self.assertEqual(len(calls), 2)
        self.assertEqual(float(missing[:, 3].sum()), 0)
        self.assertEqual(tuple(first.shape), (1, 4, 12, 16))

        # A finished scored episode must not run DINO again after autoreset.
        # This optimization is disabled for ordinary play/training.
        env.cfg.evaluate_first_episode = True
        env._seen = torch.tensor([True])
        camera.frame += 1
        term(env, **kwargs)
        self.assertEqual(len(calls), 2)
        env._seen[:] = False
        term(env, **kwargs)
        self.assertEqual(len(calls), 3)
        env._seen[:] = True
        env.cfg.evaluate_first_episode = False
        camera.frame += 1
        term(env, **kwargs)
        self.assertEqual(len(calls), 4)
        env.cfg.evaluate_first_episode = True
        env._seen = None
        env._evaluation_seen = torch.tensor([True, False])
        term._frame = torch.tensor([-1, -1])
        term._boxes = torch.zeros(2, 4)
        term._valid = torch.zeros(2, dtype=torch.bool)
        term._confidence = torch.zeros(2)
        camera.frame = torch.tensor([2, 2])
        camera.data.output["rgb"] = torch.stack([torch.zeros(12, 16, 3, dtype=torch.uint8),
                                                torch.full((12, 16, 3), 7, dtype=torch.uint8)])
        selected = []
        def predict_active(rgb, instruction, caption):
            selected.append(rgb.clone())
            return torch.tensor([[2., 3., 6., 8.]]), torch.tensor([True]), torch.tensor([0.8])
        term.detector = SimpleNamespace(predict=predict_active)
        active = term(env, **kwargs)
        self.assertEqual(tuple(selected[0].shape), (1, 12, 16, 3))
        self.assertTrue((selected[0] == 7).all())
        self.assertEqual(float(active[0, 3].sum()), 0)
        self.assertEqual(float(active[1, 3].sum()), 20)

    @staticmethod
    def fake_env():
        cfg = SimpleNamespace(object_keys=OBJECT_KEYS, target_key="blue_cube", lift_height=0.08, lift_confirm_steps=5)
        robot_data = SimpleNamespace(joint_pos=torch.tensor([[0., 0., 0., 0., 0., 1.745]]),
                                     body_pos_w=torch.zeros(1, 1, 3), body_quat_w=torch.tensor([[[0., 0., 0., 1.]]]),
                                     body_lin_vel_w=torch.zeros(1, 1, 3))
        robot = SimpleNamespace(data=robot_data, find_bodies=lambda names: ([0], names), find_joints=lambda names: ([5], names))
        assets = {"robot": robot}
        sensors = {"robot_contact": SimpleNamespace(body_names=["gripper"],
                   data=SimpleNamespace(net_forces_w=torch.tensor([[[0., 0., 3.]]])))}
        from pick_anything.tabletop.catalog import OBJECTS
        pocket = torch.tensor([[0.0408, -0.0002, -0.0709]])
        for i, key in enumerate(OBJECT_KEYS):
            obj = OBJECTS[key]
            position = pocket.clone()
            position[:, 0] += 0 if key == "blue_cube" else 0.3 + 0.1 * i
            position[:, 2] -= obj.height / 2 - obj.origin_to_bottom
            assets[key] = SimpleNamespace(data=SimpleNamespace(root_pos_w=position,
                              root_quat_w=torch.tensor([[0., 0., 0., 1.]]), root_lin_vel_w=torch.zeros(1, 3)))
            sensors[f"{key}_contact"] = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.tensor([[[0., 0., 0.3]]])))
        class Scene(dict):
            pass
        scene = Scene(assets)
        scene.sensors = sensors
        env = SimpleNamespace(cfg=cfg, scene=scene, num_envs=1, device="cpu", common_step_counter=0)
        state = mdp.TargetState.get(env)
        state.reset(env, torch.tensor([0]))
        state.spawn_center_z[:] = pocket[:, 2:3]
        return env, state

    def test_target_grasp_lift_confirmation_and_wrong_object(self):
        env, state = self.fake_env()
        state.update(env)  # align with open gripper
        self.assertTrue(state.aligned.item())
        env.scene["robot"].data.joint_pos[0, 5] = 0.9
        for step in range(1, 4):
            env.common_step_counter = step
            state.update(env)
            self.assertEqual(state.grasped.item(), step == 3)
            state.update(env)  # repeated calls cannot advance confirmation
            self.assertEqual(state.grasp_run.item(), step)
        env.scene["blue_cube"].data.root_pos_w[0, 2] += 0.09
        env.scene["robot"].data.body_pos_w[0, 0, 2] += 0.09
        for step in range(4, 9):
            env.common_step_counter = step
            state.update(env)
            self.assertEqual(state.success.item(), step == 8)
        self.assertTrue(state.just_success.item())
        env.common_step_counter = 9
        state.update(env)
        self.assertFalse(state.just_success.item())
        env.scene["yellow_duck"].data.root_pos_w[0, 2] += 0.05
        env.common_step_counter = 10
        self.assertEqual(mdp.wrong_object(env).item(), 1)
        self.assertTrue(state.wrong_lift_ever.item())
        state.reset(env, torch.tensor([0]))
        self.assertFalse(state.grasped.item() or state.success.item() or state.aligned.item())
        self.assertFalse(state.wrong_lift_ever.item())
        self.assertEqual(state.max_rise.item(), 0)

    def test_three_frame_actor_transfer_preserves_actions_and_critic(self):
        from copy import deepcopy
        from pick_anything.open_vocab_pick.checkpoints import load_evaluation_actor
        runner = handle_deprecated_rsl_rl_cfg(OpenVocabPickRunnerCfg().default, version("rsl-rl-lib"))
        cfg = runner.to_dict()
        cfg.update(num_steps_per_env=2, multi_gpu=None)
        obs = TensorDict({"rgb_mask": torch.rand(2, 12, 96, 128),
                          "proprioception": torch.rand(2, 24), "critic": torch.zeros(2, 54)}, batch_size=[2])
        alg = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=2, num_actions=6), cfg, "cpu")
        source_obs = obs.clone()
        source_obs["critic"] = torch.zeros(2, 49)
        source = PPO.construct_algorithm(source_obs, SimpleNamespace(num_envs=2, num_actions=6), deepcopy(cfg), "cpu")
        source.eval_mode()
        alg.eval_mode()
        critic = deepcopy(alg._raw_critic.state_dict())
        optimizer = deepcopy(alg.optimizer.state_dict())
        saved = source.save()
        saved.update(iter=148, infos={"test": True})
        fake_runner = SimpleNamespace(alg=alg, current_learning_iteration=0)
        with patch("torch.load", return_value=saved):
            load_evaluation_actor(fake_runner, "unused.pt")
        torch.testing.assert_close(alg.get_policy()(obs), source.get_policy()(source_obs), rtol=0, atol=0)
        for key, value in critic.items():
            torch.testing.assert_close(alg._raw_critic.state_dict()[key], value, rtol=0, atol=0)
        self.assertEqual(optimizer, alg.optimizer.state_dict())
        self.assertEqual(fake_runner.current_learning_iteration, 0)
        bad = deepcopy(saved)
        bad["actor_state_dict"]["cnns.rgb_mask.0.weight"] = torch.zeros(16, 4, 5, 5)
        with patch("torch.load", return_value=bad), self.assertRaises(RuntimeError):
            load_evaluation_actor(fake_runner, "wrong_history.pt")

    def test_config_isolation_and_actual_ppo_update(self):
        legacy = GroundedPickEnvCfg().to_dict()
        cfg = OpenVocabPickEnvCfg()
        cfg.validate()
        self.assertEqual(cfg.object_keys, OBJECT_KEYS)
        self.assertEqual(cfg.observations.rgb_mask.image.func, OpenVocabRGBMask)
        self.assertFalse(hasattr(cfg.events, "reset_staged"))
        cfg.observations.rgb_mask.image.params["backend"] = "dino"
        with self.assertRaises(ValueError):
            cfg.validate()
        cfg.target_key = "yellow_duck"
        cfg.grounding_caption = "yellow rubber duck."
        cfg.validate()
        cfg.play_mode()
        self.assertFalse(cfg.observations.rgb_mask.enable_corruption)
        self.assertEqual(GroundedPickEnvCfg().to_dict(), legacy)
        runner = handle_deprecated_rsl_rl_cfg(OpenVocabPickRunnerCfg().default, version("rsl-rl-lib"))
        obs = TensorDict({"rgb_mask": torch.rand(4, 4, 96, 128), "proprioception": torch.zeros(4, 24),
                          "critic": torch.zeros(4, 54)}, batch_size=[4])
        runner_cfg = runner.to_dict()
        runner_cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
        runner_cfg.update(num_steps_per_env=2, multi_gpu=None)
        algorithm = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=4, num_actions=6), runner_cfg, "cpu")
        for _ in range(2):
            self.assertEqual(tuple(algorithm.act(obs).shape), (4, 6))
            algorithm.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
        algorithm.compute_returns(obs)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all() for value in algorithm.update().values()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
