"""Export a kitless Isaac teacher trace and actual solver parameters for sim2sim.

All outputs and runtime caches stay under Pick_Anything. Requires isaaclab_61
and GPU access. Uses the existing local asset cache without changing it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "mujoco_sim2sim/output/runtime_cache"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
for key, subdir in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg"),
                    ("CUDA_CACHE_PATH", "cuda"), ("MPLCONFIGDIR", "matplotlib")):
    path = RUNTIME / subdir
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
ASSET_CACHE = Path("/tmp/https/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1")
if ASSET_CACHE.is_dir():
    os.environ.setdefault("ISAACSIM_ASSET_ROOT", str(ASSET_CACHE))

import argparse
import json
import numpy as np
import torch
import warp as wp

wp.config.kernel_cache_dir = str(RUNTIME / "warp")


def array(value):
    if hasattr(value, "torch"):
        value = value.torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return value.numpy().copy()


def main():
    from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-episodes", type=int, default=1, help="first HOME episode per parallel environment")
    parser.add_argument("--out", type=Path, default=ROOT / "mujoco_sim2sim/reference/runtime_unclipped32")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if not args.out.is_relative_to(ROOT):
        parser.error("--out must be inside Pick_Anything")
    args.out.mkdir(parents=True, exist_ok=True)

    from isaaclab_tasks.utils.hydra import resolve_presets
    from pick_anything.tasks.pick_apple.camera_env_cfg import StagedSmoothTeacherEnvCfg
    torch.manual_seed(0)
    cfg = resolve_presets(StagedSmoothTeacherEnvCfg())
    cfg.play_mode()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.eval_episodes
    cfg.sim.device = args.device
    cfg.sim.physics.solver_cfg.save_to_mjcf = str(args.out / "isaac_model.xml")
    cfg.log_dir = str(args.out)
    cfg.events.reset_staged.params["forced_stage"] = 0
    with launch_simulation(cfg, args):
        import mujoco
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_newton.physics import NewtonManager
        from pick_anything.tasks.pick_apple.staged_mdp import StagedPickState
        from run_teacher import POLICY_DEFAULT

        env = ManagerBasedRLEnv(cfg)
        try:
            robot = env.scene["robot"]
            names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            ids, found = robot.find_joints(names, preserve_order=True)
            assert found == names
            env.seed(args.seed)
            torch.manual_seed(args.seed)
            obs, _ = env.reset()
            origins = array(env.scene.env_origins)[0]
            init = [array(robot.data.joint_pos)[0, ids], array(robot.data.joint_pos_target)[0, ids]]
            for key in ("orange", "cardboard_box"):
                init.extend([array(env.scene[key].data.root_pos_w)[0] - origins,
                             array(env.scene[key].data.root_quat_w)[0]])
            np.save(args.out / "init.npy", np.concatenate(init))
            np.savez(args.out / "initial_states.npz", q=array(robot.data.joint_pos)[:, ids],
                     target=array(robot.data.joint_pos_target)[:, ids], origins=array(env.scene.env_origins),
                     orange=array(env.scene["orange"].data.root_state_w),
                     box=array(env.scene["cardboard_box"].data.root_state_w))

            solver = NewtonManager._solver
            m = solver.mj_model
            wp_model = solver.mjw_model
            params = {}
            fields = ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia",
                      "body_invweight0",
                      "dof_armature", "dof_damping", "dof_frictionloss", "dof_solref", "dof_solimp",
                      "jnt_range", "jnt_solref", "jnt_solimp", "jnt_actfrcrange", "jnt_stiffness",
                      "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange",
                      "geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_margin", "geom_gap")
            for field in fields:
                cpu = np.asarray(getattr(m, field)).copy()
                params[field + "_compiled"] = cpu
                if hasattr(wp_model, field):
                    gpu = getattr(wp_model, field).numpy()
                    params[field + "_runtime"] = gpu
                    if gpu.shape == cpu.shape:
                        getattr(m, field)[:] = gpu
                    elif gpu.shape[1:] == cpu.shape:
                        getattr(m, field)[:] = gpu[0]
            np.savez(args.out / "solver_parameters.npz", **params)
            m.opt.timestep = NewtonManager.get_solver_dt()
            mujoco.mj_saveModel(m, str(args.out / "isaac_model.mjb"))
            metadata = {
                "seed": args.seed, "physics_handles_decimation": env._physics_handles_decimation,
                "decimation": cfg.decimation, "num_substeps": cfg.sim.physics.num_substeps,
                "physics_dt": env.physics_dt, "solver_dt": NewtonManager.get_solver_dt(),
                "joint_names": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)],
                "body_names": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)],
                "actuator_names": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)],
                "stiffness": array(robot.data.joint_stiffness)[0, ids].tolist(),
                "damping": array(robot.data.joint_damping)[0, ids].tolist(),
                "solver_options": {key: np.asarray(getattr(m.opt, key)).tolist() for key in
                    ("timestep", "iterations", "ls_iterations", "integrator", "cone", "impratio", "disableflags", "enableflags")},
            }
            (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print("ISAAC_RUNTIME", json.dumps(metadata), flush=True)
            jit = torch.jit.load(str(POLICY_DEFAULT), map_location="cpu").eval()
            torch.set_num_threads(1)
            traces = {key: [] for key in ("obs", "act", "jp", "jv", "target", "orange", "box",
                                         "body_pos", "body_quat", "claw_force", "orange_force", "stage")}
            seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            outcomes = {key: 0 for key in ("success", "time_out", "staged_orange_dropped",
                                         "staged_box_tipped", "staged_abnormal_robot")}
            for t in range(args.steps):
                ob = array(obs["policy"])[0]
                with torch.inference_mode():
                    actions = jit(torch.from_numpy(array(obs["policy"])))
                    a = actions.numpy()[0]
                state = StagedPickState.get(env)
                values = {"obs": ob, "act": a, "jp": array(robot.data.joint_pos)[0, ids],
                          "jv": array(robot.data.joint_vel)[0, ids],
                          "target": array(robot.data.joint_pos_target)[0, ids],
                          "orange": array(env.scene["orange"].data.root_state_w)[0],
                          "box": array(env.scene["cardboard_box"].data.root_state_w)[0],
                          "body_pos": array(robot.data.body_pos_w)[0],
                          "body_quat": array(robot.data.body_quat_w)[0],
                          "claw_force": array(state.claw_force)[0],
                          "orange_force": array(state.orange_force)[0],
                          "stage": array(state.progress.stage)[0]}
                for key, value in values.items():
                    traces[key].append(value)
                obs, _, terminated, truncated, _ = env.step(actions.to(env.device))
                done = (terminated | truncated) & ~seen
                for key in outcomes:
                    outcomes[key] += int((env.termination_manager.get_term(key) & done).sum())
                seen |= done
                if t in (0, 2, 99, 149):
                    contact = solver.mjw_data.contact
                    extra = {"contact_" + key: getattr(contact, key).numpy() for key in
                             ("dist", "pos", "frame", "includemargin", "friction", "solref", "solreffriction",
                              "solimp", "dim", "geom", "efc_address")}
                    for key in ("shape_material_ke", "shape_material_kd", "shape_material_kf", "shape_material_mu"):
                        extra[key] = getattr(solver.model, key).numpy()
                    extra["solref_mode"] = solver.model.mujoco.solref_mode.numpy()
                    np.savez(args.out / f"contacts_{t}.npz", **extra)
                if t == 0:
                    # Deferred model notifications (startup DR, articulation
                    # parameters) reach the solver on its first physics step.
                    for field in fields:
                        if hasattr(wp_model, field):
                            gpu = getattr(wp_model, field).numpy()
                            params[field + "_runtime"] = gpu
                            cpu = np.asarray(getattr(m, field))
                            if gpu.shape == cpu.shape:
                                cpu[:] = gpu
                            elif gpu.shape[1:] == cpu.shape:
                                cpu[:] = gpu[0]
                    np.savez(args.out / "solver_parameters.npz", **params)
                    mujoco.mj_saveModel(m, str(args.out / "isaac_model.mjb"))
                    print("POST_FIRST_STEP damping", m.dof_damping[:6], "friction", m.dof_frictionloss[:6],
                          "mass", m.body_mass, "gains", m.actuator_gainprm[:, 0], flush=True)
                if t % 50 == 0:
                    print(f"trace t={t} stage={int(values['stage'])}", flush=True)
                if bool(seen.all()):
                    print("EPISODES_DONE", t + 1, outcomes, flush=True)
                    break
            np.savez(args.out / "trace.npz", **{k: np.stack(v) for k, v in traces.items()})
            summary = {"episodes": int(seen.sum()), "requested": env.num_envs, "counts": outcomes,
                       "success_rate": outcomes["success"] / max(int(seen.sum()), 1), "steps": t + 1,
                       "seed": args.seed, "policy": str(POLICY_DEFAULT), "reset_mode": "live HOME"}
            (args.out / "eval.json").write_text(json.dumps(summary, indent=2) + "\n")
            print("saved", args.out, flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
