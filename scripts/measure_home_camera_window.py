"""量出 HOME 位姿下腕摄在桌面上的「橘子可见窗口」（cam_v3 可见性修复的依据）。

诊断结论（2026-09-28 cam_v2 复盘）：HOME 出生时橘子大多在腕摄画面外（几个
像素挂边缘），actor 无目标方位特征 → HOME→PREGRASP 是盲搜，33.5 万次 HOME
出生 0 成功。修复 = HOME 出生行把橘子刷进本脚本量出的窗口，让该段变成可学
的视觉伺服。

**measure 模式**（默认，不依赖新 cfg，可单独跑）：
  forced_stage=0 全 HOME 出生 → reset 沉降后把橘子按网格摆满 fruit 矩形
  （盒子挪到远角排除遮挡）→ 读 64×48 原始传感器帧 → 红色主导像素检测
  （橘子贴图 vs 米色桌面/深灰爪指）→ 只统计中心区域的橘子像素（边缘挂
  几个像素不算「可用视野」）。多轮 reset（关节噪声/盒子重刷）取交集，输出
  env 局部系窗口常量 + 网格可见性 ASCII 图 + 全帧拼图 PNG 到
  photos/home_window/。

**validate 模式**（改完 camera_env_cfg 后用）：
  用 CameraVisibleHomeEnvCfg（真实窗口采样出生）反复 HOME 出生，统计橘子
  落在画面中心区域的比例（目标 ≥90%）并存拼图。

用法（Pick_Anything 根目录）：

    conda activate isaaclab_61
    CUDA_VISIBLE_DEVICES=1 python scripts/measure_home_camera_window.py --headless --device cuda:0
    CUDA_VISIBLE_DEVICES=1 python scripts/measure_home_camera_window.py --headless --device cuda:0 --mode validate
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

# --- 网格与检测参数（一次测量用，不进任务代码）---
GRID_X, GRID_Y = 16, 8  # fruit 矩形 0.6 x 0.3 m -> 3.75cm 网格
MEASURE_ROUNDS = 3  # 关节噪声/盒子重刷轮数，取交集保余量
CENTRAL_FRACTION = 0.10  # 画面四周各剔除 10%（边缘贴边不算；>10% 内嵌即可用）
CENTRAL_MIN_PX = 10  # 中心区域橘子像素数下限（64x48=3072 px 总量）
R_DOMINANCE = 25  # R 需超过 G/B 的量（橘子贴图 vs 米色桌面）
R_MIN = 70  # R 下限（滤暗部噪声）
BOX_PARK_XY = (0.55, -0.405)  # 测几何时盒子挪到盒区远角，排除遮挡
SETTLE_STEPS = 12  # 写位姿后沉降步（reset/写位姿不渲染，standalone 三坑）
WARMUP_STEPS = 10
VALIDATE_RESETS = 6  # validate 模式出生轮数
VALIDATE_TARGET = 0.90  # validate 模式橘子入画中心的比例目标


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the wrist-camera visible window for oranges at HOME spawn.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--mode", choices=["measure", "validate"], default="measure")
    parser.add_argument("--rounds", type=int, default=MEASURE_ROUNDS, help="measure 模式 reset 轮数")
    return parser.parse_args()


def t(buf):
    """IsaacLab ProxyArray -> torch.Tensor"""
    return buf.torch if hasattr(buf, "torch") else buf


def frames_rgb(camera) -> torch.Tensor:
    """(N, H, W, 3) uint8 原始传感器帧（不含观测项光度 DR，检测稳定）。"""
    return t(camera.data.output["rgb"])[..., :3]


def orange_central_px(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """每帧（中心区域橘子像素数, 全画面橘子像素数）：(N,), (N,)。

    红色主导启发式（米色桌面/深灰爪/纸盒都不 R 主导）；全画面计数用于看
    视野总足迹，窗口只按中心区域定。
    """
    f = frames.float()
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mask = (r > g + R_DOMINANCE) & (r > b + R_DOMINANCE) & (r > R_MIN)
    n, h, w = mask.shape
    y0, y1 = int(h * CENTRAL_FRACTION), int(h * (1 - CENTRAL_FRACTION))
    x0, x1 = int(w * CENTRAL_FRACTION), int(w * (1 - CENTRAL_FRACTION))
    return mask[:, y0:y1, x0:x1].sum(dim=(1, 2)), mask.sum(dim=(1, 2))


def settle(env) -> None:
    """写位姿后的沉降步：物理追上、传感器出图（view_spawn_camera 同款循环）。"""
    for _ in range(SETTLE_STEPS):
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(env.physics_dt)


def warmup(env) -> None:
    for _ in range(WARMUP_STEPS):
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(env.physics_dt)


def write_orange_xy(env, xy: torch.Tensor, yaw: torch.Tensor | None = None) -> None:
    """把橘子摆到 env 局部 xy（z=FRUIT_SPAWN_Z，quat xyzw 同 StagedReset）。"""
    from pick_anything.tasks.pick_apple import mdp as v0

    n = xy.shape[0]
    device = xy.device
    if yaw is None:
        yaw = torch.zeros(n, device=device)
    origins = t(env.scene.env_origins)
    quat = torch.stack(
        [torch.zeros(n, device=device), torch.zeros(n, device=device),
         torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1
    )
    z = torch.full((n,), v0.FRUIT_SPAWN_Z, device=device)
    pos = torch.stack([xy[:, 0], xy[:, 1], z], dim=-1) + origins
    root = torch.cat([pos, quat], dim=-1)
    ids = torch.arange(n, device=device)
    env.scene["orange"].write_root_pose_to_sim_index(root_pose=root, env_ids=ids)


def park_box(env) -> None:
    """measure 模式把盒子挪到盒区远角，量的是纯相机几何。"""
    from pick_anything.tasks.pick_apple import mdp as v0

    origins = t(env.scene.env_origins)
    n = origins.shape[0]
    device = origins.device
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(n, 1)
    pos = torch.tensor([*BOX_PARK_XY, v0.BOX_SPAWN_Z], device=device).repeat(n, 1) + origins
    ids = torch.arange(n, device=device)
    env.scene["cardboard_box"].write_root_pose_to_sim_index(
        root_pose=torch.cat([pos, quat], dim=-1), env_ids=ids
    )


def save_montage(frames: torch.Tensor, path: Path, cols: int, tile: int = 4) -> bool:
    """64x48 帧网格拼图存 PNG（nearest 放大保像素）。"""
    try:
        from PIL import Image

        n, h, w, _ = frames.shape
        rows = (n + cols - 1) // cols
        tw, th = w * tile, h * tile
        sheet = Image.new("RGB", (tw * cols, th * rows), (32, 32, 32))
        for i in range(n):
            img = Image.fromarray(frames[i].cpu().numpy()).resize((tw, th), Image.NEAREST)
            sheet.paste(img, ((i % cols) * tw, (i // cols) * th))
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        return True
    except Exception as exc:
        print(f"[montage] 保存失败 {path.name}: {exc}", flush=True)
        return False


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
    from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

    from pick_anything.env_cfg import (  # noqa: E402
        BOX_X_LIMITS,
        BOX_Y_LIMITS,
        TABLE_X_LIMITS,
        TABLE_Y_LIMITS,
    )

    torch.manual_seed(0)

    if args.mode == "measure":
        from pick_anything.tasks.pick_apple.camera_env_cfg import CameraStagedPickAppleEnvCfg

        cfg = CameraStagedPickAppleEnvCfg()
        cfg = resolve_presets(cfg)
        cfg.scene.num_envs = GRID_X * GRID_Y
        cfg.events.reset_staged.params["forced_stage"] = 0  # 全 HOME 出生
        env = ManagerBasedRLEnv(cfg)

        try:
            cam = env.scene.sensors["wrist_camera"]
            warmup(env)

            xs = torch.linspace(TABLE_X_LIMITS[0], TABLE_X_LIMITS[1], GRID_X + 1)[:-1]
            ys = torch.linspace(TABLE_Y_LIMITS[0], TABLE_Y_LIMITS[1], GRID_Y + 1)[:-1]
            cell = (xs[1] - xs[0], ys[1] - ys[0])
            # 行主序网格 (N,2)：env i -> 格子 (i // GRID_Y, i % GRID_Y)
            grid = torch.stack(
                [xs.repeat_interleave(GRID_Y).float(), ys.repeat(GRID_X).float()], dim=-1
            ).to(env.device)
            visible_all = torch.ones(GRID_X * GRID_Y, dtype=torch.bool, device=env.device)
            anywhere_all = torch.ones(GRID_X * GRID_Y, dtype=torch.bool, device=env.device)

            for rnd in range(args.rounds):
                env.reset()
                park_box(env)
                write_orange_xy(env, grid)
                settle(env)
                frames = frames_rgb(cam)
                px, px_any = orange_central_px(frames)
                ok = px >= CENTRAL_MIN_PX
                visible_all &= ok
                anywhere_all &= px_any >= CENTRAL_MIN_PX
                out = PROJECT_ROOT / "photos" / "home_window"
                save_montage(frames, out / f"grid_round{rnd}.png", cols=GRID_X)
                print(
                    f"[round {rnd}] 中心可见 {int(ok.sum())}/{GRID_X * GRID_Y}  "
                    f"全画面 {int(anywhere_all.sum())}/{GRID_X * GRID_Y}  "
                    f"(px 中位 {int(px.median())}，最大 {int(px.max())})",
                    flush=True,
                )

            xs_c = (xs + cell[0] / 2).cpu()
            ys_c = (ys + cell[1] / 2).cpu()
            vis = visible_all.view(GRID_X, GRID_Y).cpu()
            print(flush=True)
            for j in range(GRID_Y - 1, -1, -1):  # y 从远到近
                print("  y=%+.3f |%s|" % (ys_c[j], "".join("#" if vis[i, j] else "." for i in range(GRID_X))))
            print("           " + "".join(str(i % 10) for i in range(GRID_X)))
            print(f"  x 轴：列 i 的中心 x = {xs_c[0]:+.3f} + i*{cell[0]:.3f}", flush=True)

            if visible_all.any():
                idx = visible_all.nonzero().flatten().cpu()
                any_vis = anywhere_all.view(GRID_X, GRID_Y).cpu()
                print("  全画面足迹（参考）：", flush=True)
                for j in range(GRID_Y - 1, -1, -1):
                    print("    |%s|" % "".join("#" if any_vis[i, j] else "." for i in range(GRID_X)), flush=True)
                x_min = float(xs_c[idx // GRID_Y].min() - cell[0] / 2)
                x_max = float(xs_c[idx // GRID_Y].max() + cell[0] / 2)
                y_min = float(ys_c[idx % GRID_Y].min() - cell[1] / 2)
                y_max = float(ys_c[idx % GRID_Y].max() + cell[1] / 2)
                span_x, span_y = x_max - x_min, y_max - y_min
                print(flush=True)
                print(f"== 可见窗口（{args.rounds} 轮交集，{int(visible_all.sum())}/{GRID_X * GRID_Y} 格）==", flush=True)
                print(f"HOME_ORANGE_WINDOW = ({x_min:.3f}, {x_max:.3f}, {y_min:.3f}, {y_max:.3f})", flush=True)
                print(f"   尺寸 {span_x * 100:.1f} x {span_y * 100:.1f} cm", flush=True)
                if min(span_x, span_y) < 0.06:
                    print(
                        "   [!] 短边 <6cm：窗口过窄，HOME≈PREGRASP，建议改走「高位扫视姿态」回退方案（见计划）",
                        flush=True,
                    )
                overlap_box = not (
                    x_max < BOX_X_LIMITS[0] or x_min > BOX_X_LIMITS[1]
                    or y_max < BOX_Y_LIMITS[0] or y_min > BOX_Y_LIMITS[1]
                )
                print(
                    f"   与盒区(x∈{BOX_X_LIMITS},y∈{BOX_Y_LIMITS})重叠: "
                    f"{'是，StagedReset 用 _push_apart 兜底' if overlap_box else '否'}",
                    flush=True,
                )
            else:
                print("== 没有任何格子稳定可见！不要填常量，先查相机位姿/渲染 ==", flush=True)
        except Exception:
            import traceback

            traceback.print_exc()  # kit 卸载会吞 traceback（standalone 三坑），先打现场
            raise
        finally:
            env.close()
            simulation_app.close()

    else:  # validate
        from pick_anything.tasks.pick_apple.camera_env_cfg import CameraVisibleHomeEnvCfg

        cfg = CameraVisibleHomeEnvCfg()
        print(f"    [params pre-resolve] {sorted(cfg.events.reset_staged.params.keys())}", flush=True)
        cfg = resolve_presets(cfg)
        print(f"    [params post-resolve] {sorted(cfg.events.reset_staged.params.keys())}", flush=True)
        cfg.scene.num_envs = 16
        cfg.events.reset_staged.params["forced_stage"] = 0
        env = ManagerBasedRLEnv(cfg)

        try:
            cam = env.scene.sensors["wrist_camera"]
            state = getattr(env, "staged_pick_state", None)
            # reset 事件实际执行顺序（排查父类遗留 reset_objects 是否在后）
            try:
                order = list(env.event_manager._mode_term_names.get("reset", []))
                print(f"    [reset 事件顺序] {order}", flush=True)
                for c in env.event_manager._mode_term_cfgs.get("reset", []):
                    print(f"      term {c.func.__name__}: params={sorted(c.params.keys())}", flush=True)
                for inst in env.event_manager._mode_class_term_cfgs.get("reset", []):
                    win = getattr(inst, "home_orange_window", "<无属性>")
                    fs = getattr(inst, "forced_stage", "<无属性>")
                    print(f"      instance {type(inst).__name__}: window={win} forced={fs}", flush=True)
            except Exception as exc:
                print(f"    [reset 事件顺序] 读取失败: {exc}", flush=True)
            warmup(env)
            total = hits = 0
            for rnd in range(VALIDATE_RESETS):
                env.reset()
                intended = state.spawn_orange_xy.clone() if state is not None else None
                settle(env)
                frames = frames_rgb(cam)
                px, _ = orange_central_px(frames)
                ok = px >= CENTRAL_MIN_PX
                total += ok.numel()
                hits += int(ok.sum())
                actual = t(env.scene["orange"].data.root_pos_w)[:, :2] - t(env.scene.env_origins)[:, :2]
                if state is not None:
                    misses = (~ok).nonzero().flatten().tolist()
                    for m in misses:
                        xy = intended[m].tolist()
                        ac = actual[m].tolist()
                        print(
                            f"    [miss] env{m} 出生xy=({xy[0]:+.3f},{xy[1]:+.3f}) "
                            f"沉降后=({ac[0]:+.3f},{ac[1]:+.3f}) px={int(px[m])} "
                            f"birth={int(state.birth_stage[m])}",
                            flush=True,
                        )
                save_montage(frames, PROJECT_ROOT / "photos" / "home_window" / f"validate_round{rnd}.png", cols=8)
                print(f"[validate {rnd}] 中心可见 {int(ok.sum())}/{ok.numel()}", flush=True)
            rate = hits / max(total, 1)
            print(f"== 橘子入画中心比例 {rate:.1%}（目标 ≥{VALIDATE_TARGET:.0%}）==", flush=True)
            if rate < VALIDATE_TARGET:
                print("   [!] 未达标：检查窗口常量/中心区域阈值", flush=True)
        except Exception:
            import traceback

            traceback.print_exc()
            raise
        finally:
            env.close()
            simulation_app.close()


if __name__ == "__main__":
    main()
