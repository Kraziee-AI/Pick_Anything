"""教师（特权坐标）PPO 在 MuJoCo 推理（sim2sim 无视觉线——绕开渲染域差）。

teacher_smooth_v1：默认按 TeacherFine-v1 的 HOME 窗口复位。
旧 bank 的 stage-0 行分布不同，只能通过 --reset-mode bank 显式测试。
教师动作不截断，策略 30Hz、相对位置目标 120Hz；成功判定复用
task_metrics（RELEASE 稳定计数同款）。

用法（isaaclab_61 环境，工作目录 Pick_Anything；无渲染，不需要 DISPLAY）：
  python mujoco_sim2sim/run_teacher.py --episodes 64
  python mujoco_sim2sim/run_teacher.py --episodes 256 --json mujoco_sim2sim/output/teacher_eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
import torch

from env import PickMujocoEnv
from task_metrics import EpisodeTracker, summarize
from teacher_obs import MuJoCoStageState, StageMachine, TeacherObs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DEFAULT = (PROJECT_ROOT / "logs" / "rsl_rl" / "pick_apple_staged_so101"
                  / "2026-10-05_17-16-17_teacher_smooth_v1" / "exported" / "policy.pt")
BANK_DEFAULT = PROJECT_ROOT / "reset_states" / "pick_orange_bank_v1.pt"
OUT_DIR_DEFAULT = Path(__file__).resolve().parent / "output"
MAX_STEPS = 2700  # 90s × 30Hz（教师线同款）


class TeacherPolicy:
    """MLP jit（forward((N,111))→(N,6)），教师训练配置不截断动作。"""

    def __init__(self, path: Path | str = POLICY_DEFAULT, device: str = "cpu"):
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f"教师 jit 不存在: {path}")
        self._jit = torch.jit.load(str(path), map_location=device)
        self._device = torch.device(device)
        self._jit.eval()
        with torch.inference_mode():
            out = self._jit(torch.zeros(1, 111, device=self._device))
        if tuple(out.shape) != (1, 6) or not torch.isfinite(out).all():
            raise RuntimeError(f"教师 jit 自检失败: {tuple(out.shape)}")

    def act(self, obs111: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            out = self._jit(torch.from_numpy(obs111.astype(np.float32)).to(self._device).unsqueeze(0))
        return out.squeeze(0).cpu().numpy()


def load_bank_rows(bank_path: Path) -> list[dict]:
    d = torch.load(str(bank_path), map_location="cpu", weights_only=False)
    st = d["states"]
    rows = []
    for i in range(st["joint_position"].shape[0]):
        if int(st["stage"][i]) != 0:
            continue
        rows.append({
            "q": st["joint_position"][i].numpy().astype(np.float64),
            "ctrl": st["joint_target"][i].numpy().astype(np.float64),
            "orange": st["orange_pose"][i].numpy().astype(np.float64),   # pos(3)+quat xyzw
            "box": st["box_pose"][i].numpy().astype(np.float64),          # pos(3)+quat xyzw
        })
    if not rows:
        raise RuntimeError("bank 无 stage-0 行")
    return rows


def reset_from_bank(env: PickMujocoEnv, row: dict) -> None:
    """冷恢复 bank 行：臂 qpos/ctrl、橘/盒位姿、零速度（训练同口径）。"""
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[env._qadr] = row["q"]
    env.data.ctrl[env._act_ids] = row["ctrl"]
    # bank 四元数是 (x,y,z,w)（Isaac root_quat_w 原序）→ MuJoCo qpos 要 (w,x,y,z)
    env.data.qpos[env._orange_qadr: env._orange_qadr + 7] = np.concatenate(
        [row["orange"][:3], row["orange"][3:][[3, 0, 1, 2]]])
    env.data.qpos[env._box_qadr: env._box_qadr + 7] = np.concatenate(
        [row["box"][:3], row["box"][3:][[3, 0, 1, 2]]])
    env.data.qvel[:] = 0.0
    env.reset_bookkeeping(row["ctrl"])
    mujoco.mj_forward(env.model, env.data)


def cmd_viz(args: argparse.Namespace) -> None:
    """viewer 实时看教师策略跑；R 重开，C 切换外部/腕摄视角。"""
    import time
    import mujoco.viewer

    policy = TeacherPolicy(args.policy, args.device)
    rows = load_bank_rows(Path(args.bank)) if args.reset_mode == "bank" else None
    rng = np.random.default_rng(args.seed)
    env = PickMujocoEnv(seed=args.seed, action_clip=None)
    st_state = MuJoCoStageState(env)
    machine = StageMachine()
    obs_buf = TeacherObs()
    reset_requested = {"flag": False}
    camera_toggle_requested = {"flag": False}
    wrist_cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    if wrist_cam_id < 0:
        raise RuntimeError("场景缺少 wrist_camera")

    def on_key(key: int) -> None:
        if key in (ord("R"), ord("r")):
            reset_requested["flag"] = True
        elif key in (ord("C"), ord("c")):
            camera_toggle_requested["flag"] = True

    ep_i = 0
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        wrist_view = False
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = (0.35, -0.48, 0.22)
            viewer.cam.distance = 0.55
            viewer.cam.azimuth = 155
            viewer.cam.elevation = -25
        print("视角: C 切换外部/腕摄实时画面，R 重开，关窗退出", flush=True)
        next_tick = time.time()
        while viewer.is_running() and ep_i < args.episodes:
            if rows is None:
                env.reset()
            else:
                reset_from_bank(env, rows[int(rng.integers(len(rows)))])
            machine.reset()
            obs_buf.reset()
            tracker = EpisodeTracker(max_steps=args.max_steps)
            print(f"[ep{ep_i}] 开始（R 换出生重开）", flush=True)
            while viewer.is_running() and not tracker.done:
                if camera_toggle_requested["flag"]:
                    camera_toggle_requested["flag"] = False
                    wrist_view = not wrist_view
                    with viewer.lock():
                        if wrist_view:
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                            viewer.cam.fixedcamid = wrist_cam_id
                        else:
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                            viewer.cam.lookat[:] = (0.35, -0.48, 0.22)
                            viewer.cam.distance = 0.55
                            viewer.cam.azimuth = 155
                            viewer.cam.elevation = -25
                    print(f"视角: {'腕摄' if wrist_view else '外部'}", flush=True)
                if reset_requested["flag"]:
                    reset_requested["flag"] = False
                    break
                st_state.update()
                machine.update(st_state.conditions(machine))
                obs = obs_buf.append(env, st_state, machine)
                env.step(policy.act(obs))
                tracker.update(env)
                viewer.sync()
                next_tick += 1.0 / 30.0
                sleep = next_tick - time.time()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.time()
            print(f"[ep{ep_i}] → {tracker.outcome}（{tracker.steps} 步, "
                  f"min_d={tracker.min_pocket_dist:.3f}）", flush=True)
            ep_i += 1


def run_episode(env, policy, st_state, machine, obs_buf, row, max_steps):
    if row is None:
        env.reset()
    else:
        reset_from_bank(env, row)
    machine.reset()
    obs_buf.reset()
    tracker = EpisodeTracker(max_steps=max_steps)
    t0 = time.perf_counter()
    steps0 = env.n_steps
    while not tracker.done:
        st_state.update()
        machine.update(st_state.conditions(machine))
        obs = obs_buf.append(env, st_state, machine)
        a = policy.act(obs)
        env.step(a)
        tracker.update(env)
    return tracker.finish(-1, env), time.perf_counter() - t0, env.n_steps - steps0


def main() -> None:
    parser = argparse.ArgumentParser(description="教师 PPO MuJoCo 推理评估")
    parser.add_argument("--policy", type=str, default=str(POLICY_DEFAULT))
    parser.add_argument("--bank", type=str, default=str(BANK_DEFAULT))
    parser.add_argument("--reset-mode", choices=("home", "bank"), default="home",
                        help="home matches TeacherFine-v1 training; bank tests the older broad distribution")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0, help="HOME 刷位 / bank 行抽样种子")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--viz", action="store_true", help="交互 viewer 实时看教师跑（C 切换腕摄视角）")
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("--episodes and --max-steps must be positive")
    torch.set_num_threads(1)

    if args.viz:
        cmd_viz(args)
        return

    policy = TeacherPolicy(args.policy, args.device)
    rows = load_bank_rows(Path(args.bank)) if args.reset_mode == "bank" else None
    rng = np.random.default_rng(args.seed)
    picks = (rng.choice(len(rows), size=args.episodes, replace=args.episodes > len(rows))
             if rows is not None else np.arange(args.episodes))
    print(f"教师 jit: {args.policy}")
    print(f"reset={args.reset_mode}，{args.episodes} 局（seed={args.seed}）")

    env = PickMujocoEnv(seed=args.seed, action_clip=None)
    st_state = MuJoCoStageState(env)
    machine = StageMachine()
    obs_buf = TeacherObs()
    results, s_per_ep = [], []
    t_start = time.time()
    for i, row_idx in enumerate(picks):
        stats, dt, n = run_episode(env, policy, st_state, machine, obs_buf,
                                   rows[int(row_idx)] if rows is not None else None, args.max_steps)
        stats.seed = int(row_idx)
        results.append(stats)
        s_per_ep.append(dt)
        md = stats.min_pocket_dist
        print(f"[{i+1}/{args.episodes}] {args.reset_mode}#{row_idx} → {stats.outcome} ({stats.steps} 步, "
              f"grasp={int(stats.grasp)}, min_d={'inf' if md == float('inf') else f'{md:.3f}'})",
              flush=True)
    meta = {
        "policy_path": args.policy,
        "bank": args.bank if rows is not None else None,
        "reset_mode": args.reset_mode,
        "action_clip": None,
        "sampling_seed": args.seed,
        "physics_calibration": json.loads((Path(__file__).resolve().parent / "assets/calibration_metadata.json").read_text()),
        "timing": {"s_per_episode": round(float(np.mean(s_per_ep)), 2),
                   "total_s": round(time.time() - t_start, 1)},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    summary = summarize(results, meta)
    print("\n===== 教师 sim2sim 评估汇总 =====")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes_detail"},
                     ensure_ascii=False, indent=1))
    print(f"PICK_MJ_TEACHER_RESULT={json.dumps(summary, ensure_ascii=False)}")
    out_json = Path(args.json) if args.json else (
        OUT_DIR_DEFAULT / f"teacher_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
