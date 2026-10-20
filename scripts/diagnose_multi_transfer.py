"""Isolated GT controls for reset pose, opening, PD targets and distractors."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from grounded_pick_cli import PROJECT_ROOT, configure_caches


def snapshot(env, obs):
    """Capture local simulator poses and actual actor observations before action."""
    import torch
    from pick_anything.grounded_pick.observations import tensor
    origins = tensor(env.scene.env_origins)
    keys = [key for key in ("orange", "cardboard_box", "yellow_duck", "blue_cube", "red_cylinder")
            if key in env.scene.keys()]
    rows = []
    robot = env.scene["robot"]
    for i in range(env.num_envs):
        rows.append({"env_id": i, "joint_position": tensor(robot.data.joint_pos)[i].tolist(),
                     "joint_target": tensor(robot.data.joint_pos_target)[i].tolist(),
                     "objects": {key: {"position": (tensor(env.scene[key].data.root_pos_w)[i] - origins[i]).tolist(),
                                        "quat_xyzw": tensor(env.scene[key].data.root_quat_w)[i].tolist()}
                                 for key in keys}})
    actor_obs = {key: obs[key].detach().cpu().clone() for key in ("rgb_mask", "proprioception")}
    return rows, actor_obs


def main():
    configure_caches()
    from pick_anything.grounded_pick.grounding import workspace_output
    from pick_anything.grounded_pick.checkpoints import image_history_from_checkpoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["capture-single", "capture-multi", "factorial"], required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--single-state", type=Path)
    parser.add_argument("--multi-state", type=Path)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--variants", type=int, nargs="+", choices=range(16), default=None,
                        help="默认0 1，可分批跑16个组合；bit0夹爪、bit1位姿、bit2远移干扰物、bit3 PD目标")
    parser.add_argument("--xy-control", action="store_true",
                        help="仅variants 0 2：盒子及干扰物统一远移，只改变橘子XY，保留原高度与yaw")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("replicates must be positive")
    variants = args.variants if args.variants is not None else [0, 1]
    if len(set(variants)) != len(variants):
        parser.error("duplicate variants are not allowed")
    if args.xy_control and (args.mode != "factorial" or variants != [0, 2]):
        parser.error("--xy-control requires factorial --variants 0 2")
    if args.mode == "factorial" and (not args.single_state or not args.multi_state):
        parser.error("factorial needs both captured state files")
    checkpoint = args.checkpoint.resolve(strict=True)
    history = image_history_from_checkpoint(checkpoint)
    root = workspace_output(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / "tmp").mkdir()
    os.environ["TMPDIR"] = str(root / "tmp")
    os.environ["ISAACSIM_ASSET_ROOT"] = str(PROJECT_ROOT / "assets/isaac_cache")
    metadata = {"mode": args.mode, "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "replicates": args.replicates, "variants": variants, "xy_control": args.xy_control,
                "seed": args.seed, "image_history": history,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "argv": sys.argv,
                "source_sha256": {}}
    files = [Path(__file__).resolve()]
    files += list((PROJECT_ROOT / "pick_anything/open_vocab_pick").glob("*.py"))
    files += list((PROJECT_ROOT / "pick_anything/grounded_pick").glob("*.py"))
    files += [PROJECT_ROOT / "pick_anything/tasks/pick_apple/camera_env_cfg.py",
              PROJECT_ROOT / "pick_anything/tasks/pick_apple/staged_mdp.py"]
    for path in files:
        name = path.relative_to(PROJECT_ROOT)
        data = path.read_bytes()
        metadata["source_sha256"][str(name)] = hashlib.sha256(data).hexdigest()
        dest = root / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    import torch
    from rsl_rl.runners import OnPolicyRunner
    original_policy = OnPolicyRunner.get_inference_policy
    OnPolicyRunner.export_policy_to_jit = lambda *a, **k: None
    OnPolicyRunner.export_policy_to_onnx = lambda *a, **k: None
    from pick_anything.grounded_pick.observations import tensor

    def policy_factory(runner, *a, **k):
        policy = original_policy(runner, *a, **k)
        env = runner.env.unwrapped
        first = True

        def measured(obs):
            nonlocal first
            if first:
                rows, actor_obs = snapshot(env, obs)
                (root / "initial_state.json").write_text(json.dumps(rows, indent=2) + "\n")
                torch.save(actor_obs, root / "initial_actor_obs.pt")
                first = False
                if args.mode.startswith("capture-"):
                    print(f"INITIAL_STATE_CAPTURED={root / 'initial_state.json'}", flush=True)
                    raise SystemExit(0)
            action = policy(obs)
            alternate = None
            if env.common_step_counter % 30 == 0:
                zeroed = obs.clone()
                zeroed["rgb_mask"][:, 3::4] = 0
                alternate = policy(zeroed)
            env.capture_diagnosis(action=action, zero_mask_action=alternate)
            return action

        measured.reset = policy.reset
        return measured

    OnPolicyRunner.get_inference_policy = policy_factory
    base_cli = ["--checkpoint", str(checkpoint), "--mask-backend", "gt", "--instruction", "抓橘子",
                "--num_envs", str(args.replicates), "--device", args.device, "--seed", str(args.seed),
                "--viz", "none", "--dump-dir", str(root / "frames"), "--dump-every", "30", "--dump-max", "40"]
    if args.mode == "capture-single":
        sys.argv = [sys.argv[0], *base_cli]
        from grounded_pick_cli import run
        return run("eval")
    if args.mode == "capture-multi":
        sys.argv = [sys.argv[0], *base_cli, "--actor-only"]
        from open_vocab_pick_cli import run
        return run("eval")

    from pick_anything.open_vocab_pick import mdp
    from pick_anything.open_vocab_pick.env import OpenVocabPickEnv
    from pick_anything.open_vocab_pick.config import OpenVocabPickEnvCfg, OpenVocabPickRunnerCfg
    from pick_anything.open_vocab_pick.checkpoints import load_evaluation_actor
    from pick_anything.tabletop.catalog import OBJECTS
    from pick_anything.grounded_pick.masks import boxes_to_masks
    from PIL import Image
    import gymnasium as gym

    single = json.loads(args.single_state.read_text())
    multi = json.loads(args.multi_state.read_text())
    if min(len(single), len(multi)) < args.replicates:
        parser.error("capture files have fewer rows than replicates")
    (root / "reference_single.json").write_text(json.dumps(single, indent=2) + "\n")
    (root / "reference_multi.json").write_text(json.dumps(multi, indent=2) + "\n")

    def variant(i):
        v = variants[i // args.replicates]
        return {"variant_id": v, "xy_control": args.xy_control,
                "gripper": "source_default" if v & 1 else "multi_open",
                "layout": "source_pose" if v & 2 else "multi_pose",
                "distractors": "far" if args.xy_control or v & 4 else "present",
                "box_far": args.xy_control,
                "pd_targets": "source_zero_arm" if v & 8 else "multi_position"}

    def controlled_reset(env, env_ids):
        mdp.reset_scene(env, env_ids)
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        origins = tensor(env.scene.env_origins)
        state = mdp.TargetState.get(env)
        robot = env.scene["robot"]
        joints = []
        for i in ids.tolist():
            reference = multi[i % args.replicates]
            q = list(reference["joint_position"])
            if variant(i)["gripper"] == "source_default":
                q[state.gripper_joint] = float(tensor(robot.data.default_joint_pos)[i, state.gripper_joint])
            joints.append(q)
        q = torch.tensor(joints, device=env.device)
        robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=ids)
        targets = q.clone()
        for row, i in enumerate(ids.tolist()):
            if variant(i)["pd_targets"] == "source_zero_arm":
                targets[row, :5] = targets.new_tensor(single[i % args.replicates]["joint_target"][:5])
        robot.set_joint_position_target_index(target=targets, env_ids=ids)
        for key in ("cardboard_box", *env.cfg.object_keys):
            poses = []
            for i in ids.tolist():
                ref = single if variant(i)["layout"] == "source_pose" and key in ("orange", "cardboard_box") else multi
                if args.xy_control:
                    ref = multi
                obj = ref[i % args.replicates]["objects"][key]
                pos = list(obj["position"])
                if args.xy_control and key == "orange" and variant(i)["layout"] == "source_pose":
                    pos[:2] = single[i % args.replicates]["objects"][key]["position"][:2]
                if (args.xy_control and key != "orange") or (key not in ("orange", "cardboard_box") and variant(i)["distractors"] == "far"):
                    pos[0] += 100
                    pos[1] += 100
                poses.append([*pos, *obj["quat_xyzw"]])
            pose = torch.tensor(poses, device=env.device)
            pose[:, :3] += origins[ids]
            env.scene[key].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            env.scene[key].write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids)

    class DiagnosisEnv(OpenVocabPickEnv):
        def __init__(self, cfg, **kw):
            self.diagnosis_stats = {}
            self.diagnosis_done = set()
            self.diagnosis_trace = (root / "trace.jsonl").open("w")
            super().__init__(cfg, **kw)

        def capture_diagnosis(self, action=None, zero_mask_action=None, ids=None, terminal=False):
            state = mdp.TargetState.get(self)
            state.update(self)
            indices = range(self.num_envs) if ids is None else ids.tolist()
            diag = self.grounding_diagnostics
            masks = boxes_to_masks(diag["boxes"], diag["detected"], 192, 256)
            for i in indices:
                if i in self.diagnosis_done:
                    continue
                ep = self.diagnosis_stats.setdefault(i, {"env_id": i, "replicate": i % args.replicates,
                    **variant(i), "min_distance_m": 100., "max_rise_m": 0., "mask_nonempty_steps": 0,
                    "steps": 0, "physical_lift_3cm": False, "physical_lift_8cm": False,
                    "_run3": 0, "_run8": 0, "_last": -1})
                step = int(self.episode_length_buf[i])
                if ep["_last"] == step:
                    continue
                ep["_last"] = step
                held, rise = bool(state.held_now[i]), float(state.rise[i])
                ep["steps"] = step
                ep["min_distance_m"] = min(ep["min_distance_m"], float(state.distance[i]))
                ep["max_rise_m"] = max(ep["max_rise_m"], rise)
                ep["mask_nonempty_steps"] += int(masks[i].any())
                for cm in (3, 8):
                    ep[f"_run{cm}"] = ep[f"_run{cm}"] + 1 if held and rise > cm / 100 else 0
                    ep[f"physical_lift_{cm}cm"] |= ep[f"_run{cm}"] >= 5
                if step % 5 == 0 or terminal:
                    row = {"env_id": i, "step": step, **variant(i), "center_m": state.center[i].tolist(),
                           "pocket_m": state.pocket[i].tolist(), "distance_m": float(state.distance[i]),
                           "rise_m": rise, "gripper_rad": float(state.gripper_pos[i]),
                           "aligned": bool(state.aligned[i]), "held_now": held, "grasped": bool(state.grasped[i]),
                           "mask_pixels": int(masks[i].sum()), "box": diag["boxes"][i].tolist(),
                           "joint_pos": tensor(self.scene["robot"].data.joint_pos)[i].tolist()}
                    if action is not None:
                        row["action_mean"] = action[i].tolist()
                    if zero_mask_action is not None:
                        row["zero_mask_action_mean"] = zero_mask_action[i].tolist()
                    self.diagnosis_trace.write(json.dumps(row) + "\n")
                if step % 90 == 0:
                    camera = self.scene.sensors["wrist_camera"]
                    raw = tensor(camera.data.output["rgb"])[i, ..., :3]
                    out = root / "variant_frames"
                    out.mkdir(exist_ok=True)
                    stem = f"env_{i:02d}_step_{step:04d}"
                    Image.fromarray(raw.cpu().numpy()).save(out / (stem + "_rgb.png"))
                    Image.fromarray((masks[i, 0] * 255).byte().cpu().numpy()).save(out / (stem + "_mask.png"))

        def _reset_idx(self, env_ids):
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if self._seen is not None:
                ending = ids[(self.episode_length_buf[ids] > 0) & ~self._seen[ids]]
                self.capture_diagnosis(ids=ending, terminal=True)
                for i in ending.tolist():
                    ep = self.diagnosis_stats[i]
                    ep["official_success"] = bool(self.termination_manager.get_term("success")[i])
                    ep["aligned_ever"] = bool(mdp.TargetState.get(self).aligned[i])
                    ep["causes"] = [name for name in ("success", "time_out", "dropped_or_invalid")
                                    if bool(self.termination_manager.get_term(name)[i])]
                    self.diagnosis_done.add(i)
                if len(self.diagnosis_done) == self.num_envs:
                    self.diagnosis_trace.close()
                    result = {"episodes": [{k: v for k, v in self.diagnosis_stats[i].items() if not k.startswith("_")}
                                           for i in sorted(self.diagnosis_done)],
                              "xy_control": args.xy_control,
                              "notes": ["Actor frozen, GT masks, all targets orange; original success rule unchanged.",
                                        "Physical lift excludes the open-alignment latch; held_now still requires forces, proximity and relative speed.",
                                        "Full source pose factor changes orange/box translation and yaw/height together; xy_control changes only orange XY and moves every other object far."]}
                    (root / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
                    print("MULTI_TRANSFER_DIAGNOSIS=" + json.dumps(result), flush=True)
            return super()._reset_idx(env_ids)

    def config_factory():
        cfg = OpenVocabPickEnvCfg()
        cfg.target_key = "orange"
        cfg.grounding_caption = "orange fruit."
        cfg.evaluate_first_episode = True
        cfg.events.reset_scene.func = controlled_reset
        cfg.observations.rgb_mask.image.params["image_history"] = history
        return cfg

    task = "Pick-Anything-SO101-TransferDiagnosis-v0"
    gym.register(id=task, entry_point=DiagnosisEnv, disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": config_factory, "rsl_rl_cfg_entry_point": OpenVocabPickRunnerCfg,
                         "default_agent": "rsl_rl"})
    OnPolicyRunner.load = load_evaluation_actor
    sys.argv = [sys.argv[0], "--task", task, "--rl_library", "rsl_rl", "--checkpoint", str(checkpoint),
                "--num_envs", str(len(variants) * args.replicates), "--device", args.device, "--seed", str(args.seed), "--viz", "none"]
    from isaaclab_rl.entrypoints.dispatch import run_play_cli
    return run_play_cli()


if __name__ == "__main__":
    raise SystemExit(main())
