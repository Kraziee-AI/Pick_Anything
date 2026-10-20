"""CameraApproachProgress 奖励项支付正确性探针（cam_v5 前置验证）。

standalone 断言（全部 HOME 出生，直接驱动 term 实例）：
  ① 橘子向口袋移 5cm → 支付 = 实测 Δd/scale（>0，精确势能差）
  ② 橘子反向移 5cm → 支付 = -Δd/scale（<0，对称倒扣，telescoping）
  ③ term.reset() 后首步支付 = 0（_has_prev=False，出生白拿防护）
  ④ grasped 锁存后支付恒 0（夹持后本项永久关闭）
  ⑤ HOME 出生初始距离分布：P95 应 < d_max=0.45（势能封顶设值依据）
  ⑥ 超远橘子（>0.6m）距离被 d_max 封顶，支付有界

用法（Pick_Anything 根目录）：
    conda activate isaaclab_61
    CUDA_VISIBLE_DEVICES=2 python scripts/probe_approach_progress.py --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

NUM_ENVS = 8
MOVE_M = 0.05  # ①②④的位移量
SCALE = 0.01  # 与 CameraApproachProgress 默认 scale 一致
SETTLE_STEPS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe CameraApproachProgress reward payments.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def t(buf):
    """IsaacLab ProxyArray -> torch.Tensor"""
    return buf.torch if hasattr(buf, "torch") else buf


def settle(env, steps: int = SETTLE_STEPS) -> None:
    """写位姿后沉降：物理追上、数据缓冲刷新（standalone 三坑之二）。"""
    for _ in range(steps):
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(env.physics_dt)


def shift_orange_toward(env, state, meters: float) -> None:
    """把橘子沿「口袋-橘子」连线平移（正=向口袋，负=远离），z/quat 保持不变。"""
    orange = env.scene["orange"]
    origins = t(env.scene.env_origins)
    delta = state.pocket_pos - state.orange_pos
    delta[:, 2] = 0.0  # 只平移 xy，避免悬空/压桌
    direction = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    new_local = state.orange_pos + direction * meters
    pos = new_local + origins
    quat = t(orange.data.root_quat_w)
    root = torch.cat([pos, quat], dim=-1)
    ids = torch.arange(pos.shape[0], device=pos.device)
    orange.write_root_pose_to_sim_index(root_pose=root, env_ids=ids)


def fresh(state) -> None:
    """让 StagedPickState.update 下次调用强制重算。

    update 按 env.common_step_counter 幂等（staged_mdp.py:182-184）；probe 直接
    调 term 不走 env.step、计数器不动，手动写位姿后必须失效缓存（写位姿会
    立即刷新 asset 数据缓冲，无需沉降步——也避免沉降物理把橘子滚走）。
    """
    state._last_step = -1


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
    from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

    from pick_anything.tasks.pick_apple.camera_env_cfg import CameraFrontShapedEnvCfg  # noqa: E402
    from pick_anything.tasks.pick_apple.staged_mdp import StagedPickState  # noqa: E402

    torch.manual_seed(0)
    cfg = CameraFrontShapedEnvCfg()
    cfg = resolve_presets(cfg)
    cfg.scene.num_envs = NUM_ENVS
    cfg.events.reset_staged.params["forced_stage"] = 0  # 全 HOME 出生
    env = ManagerBasedRLEnv(cfg)

    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    try:
        env.reset()
        settle(env)

        state = StagedPickState.get(env)
        state.update(env)
        rm = env.reward_manager
        term = rm._term_cfgs[rm._term_names.index("approach_progress")].func

        # ⑤ 初始距离分布（d_max 设值依据）
        d0 = (state.orange_pos - state.pocket_pos).norm(dim=-1)
        p50, p95, dmax_obs = float(d0.median()), float(d0.quantile(0.95)), float(d0.max())
        check(
            "⑤ HOME 出生初始距离 P95 < d_max=0.45",
            p95 < 0.45,
            f"P50={p50:.3f} P95={p95:.3f} max={dmax_obs:.3f}",
        )

        # ③ reset 后首步支付 0
        term.reset()
        pay_first = term(env)
        check("③ reset 后首步支付 = 0", bool((pay_first == 0).all()), f"max|pay|={float(pay_first.abs().max()):.4f}")

        # ① 移近 5cm → 支付 = 实测 Δd/scale（>0）
        d_b1 = (state.orange_pos - state.pocket_pos).norm(dim=-1).clone()
        shift_orange_toward(env, state, +MOVE_M)
        fresh(state)
        pay_near = term(env)
        d_a1 = (state.orange_pos - state.pocket_pos).norm(dim=-1)
        exp1 = (d_b1 - d_a1) / SCALE
        check(
            "① 移近支付 = +Δd/0.01（精确势能差）",
            bool(((pay_near - exp1).abs() < 0.3).all()) and bool((pay_near > 1.0).all()),
            f"pay={pay_near.cpu().numpy().round(2)} 期望={exp1.cpu().numpy().round(2)}",
        )

        # ② 移远 5cm → 支付 = 实测 Δd/scale（<0，对称倒扣）
        d_b2 = (state.orange_pos - state.pocket_pos).norm(dim=-1).clone()
        shift_orange_toward(env, state, -MOVE_M)
        fresh(state)
        pay_far = term(env)
        d_a2 = (state.orange_pos - state.pocket_pos).norm(dim=-1)
        exp2 = (d_b2 - d_a2) / SCALE
        check(
            "② 移远支付 = -Δd/0.01（精确势能差）",
            bool(((pay_far - exp2).abs() < 0.3).all()) and bool((pay_far < -1.0).all()),
            f"pay={pay_far.cpu().numpy().round(2)} 期望={exp2.cpu().numpy().round(2)}",
        )

        # ④ grasped 锁存后支付恒 0（再移近也不付）
        state.progress.grasped[:] = True
        shift_orange_toward(env, state, +MOVE_M)
        fresh(state)
        pay_latched = term(env)
        check(
            "④ grasped 锁存后支付 = 0",
            bool((pay_latched == 0).all()),
            f"max|pay|={float(pay_latched.abs().max()):.4f}",
        )
        state.progress.grasped[:] = False

        # ⑥ d_max 封顶：橘子摆到 >0.6m 外，支付有界（不随真实距离线性增长）
        fresh(state)
        state.update(env)
        orange = env.scene["orange"]
        origins = t(env.scene.env_origins)
        far_local = state.orange_pos.clone()
        far_local[:, 0] += 0.8  # 桌外方向无所谓，只测距离封顶
        root = torch.cat([far_local + origins, t(orange.data.root_quat_w)], dim=-1)
        ids = torch.arange(NUM_ENVS, device=env.device)
        orange.write_root_pose_to_sim_index(root_pose=root, env_ids=ids)
        fresh(state)
        pay_cap = term(env)
        # 真实距离跳变 ~0.8m → 若无封顶支付 ≈ -80；封顶后 ≈ -(0.45 - d_prev)/0.01
        check(
            "⑥ d_max 封顶（0.8m 跳变支付 ≫ -80 有界）",
            bool((pay_cap > -46.0).all()),
            f"mean={float(pay_cap.mean()):.3f} min={float(pay_cap.min()):.3f}（无封顶应 ≈ -80）",
        )

    except Exception:
        # kit 卸载会吞 traceback 假 0 退出（standalone 三坑之二）：趁 stderr 活着打现场
        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()

    failed = [n for n, ok, _ in results if not ok]
    print(flush=True)
    if failed:
        print(f"== 探针失败 {len(failed)}/{len(results)}：{failed} ==", flush=True)
        sys.exit(1)
    print(f"== 探针全部通过（{len(results)}/{len(results)}）==", flush=True)


if __name__ == "__main__":
    main()
