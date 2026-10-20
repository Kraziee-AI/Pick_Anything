"""查看出生时刻的腕部相机画面：真实 StagedReset 抽样 + kit 视口切到相机 POV。

不加载策略、不手摆位姿——直接用 CameraStagedPickAppleEnvCfg（训练同款），
每次 env.reset() 就是训练里的一次「出生」（StagedReset 按课程抽样写关节/
橘子/纸盒位姿）。reset 后把 kit 视口相机切到 env_0 的腕摄
/Robot/gripper/wowrobo_2MP_camera，主窗口实时显示相机视角；同时读一帧
真实 warp 传感器输出（64×48，策略真正看到的口径）打印亮度统计，
--save 时放大存 PNG（注意：视口走 kit 渲染、传感器走 warp 渲染，构图一致
但像素口径不同；策略输入以传感器帧为准）。

用法（Pick_Anything 根目录）：

    conda activate isaaclab_61
    bash scripts/view_spawn_camera.sh              # 自然课程抽样出生，无限循环
    bash scripts/view_spawn_camera.sh --stage 3    # 只看 LIFT 出生
    bash scripts/view_spawn_camera.sh --save       # 额外存真实 64×48 观测帧

    # headless 只存图（不开窗口）：
    python scripts/view_spawn_camera.py --device cuda:2 --resets 8 --save
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

# 腕部相机 prim（SO101 USD 自带；env_0 为首个克隆环境，与官方
# generate_data.py 的 _set_sensor_camera_view 写法一致）
WRIST_CAMERA_PRIM = "/World/envs/env_0/Robot/gripper/wowrobo_2MP_camera"

# 预热步数：warp 网格/纹理上传，最初几帧可能未就绪（view_reset_bank 同款）
WARMUP_STEPS = 10
# reset 后沉降步数：env.reset() 只写位姿不渲染，走几步物理画面才跟上出生位姿
SPAWN_SETTLE_STEPS = 10
# --save 传感器帧放大倍数（64×48 -> 640×480，nearest 保像素块）
UPSCALE = 10

# 屏幕阶段标签的底色（kit 可视化窗口左上角，扫一眼即知当前阶段）
STAGE_COLORS = {
    0: 0xFF9E9E9E,  # HOME      灰
    1: 0xFF64B5F6,  # PREGRASP  蓝
    2: 0xFFFFB74D,  # GRASP     橙
    3: 0xFF81C784,  # LIFT      绿
    4: 0xFF4DD0E1,  # STABILIZE 青
    5: 0xFFBA68C8,  # TRANSPORT 紫
    6: 0xFFFFF176,  # PLACE     黄
    7: 0xFFE57373,  # RELEASE   红
}

# 标签控件复用（viewport frame 重建后失效，捕获异常时重建）
_label_state: dict[str, object] = {"frame": None, "label": None}


def set_stage_label(text: str, color: int) -> None:
    """在 kit 可视化窗口左上角显示当前阶段（omni.ui；非 kit 后端静默跳过）。"""
    try:
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport_window

        viewport = get_active_viewport_window()
        if viewport is None:
            return
        frame, label = _label_state["frame"], _label_state["label"]
        if frame is None or label is None or frame.destroyed:
            with viewport.frame:
                with ui.Placement(alignment=ui.Alignment.LEFT, offset_x=16, offset_y=16):
                    frame = ui.Frame()
                    with frame:
                        label = ui.Label("", word_wrap=False)
            _label_state["frame"], _label_state["label"] = frame, label
        label.text = f"  {text}  "
        frame.style = {"background_color": color, "border_radius": 8, "padding": 10}
        label.style = {"font_size": 30, "color": 0xFFFFFFFF}
    except Exception:
        pass  # headless / 非 kit 后端：控制台的 [SPAWN] 行已标注阶段


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View the wrist-camera view at spawn in the visualizer.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--stage", type=int, default=-1, help="钉死出生阶段 0..7（-1=训练同款课程抽样）")
    parser.add_argument("--resets", type=int, default=0, help="出生次数（0=无限循环，关窗口/Ctrl+C 退出）")
    parser.add_argument("--hold", type=int, default=100, help="每次出生后保持观看的物理步数（0.02s/步）")
    parser.add_argument("--save", action="store_true", help="把真实传感器帧放大存 PNG 到 photos/spawn_camera/")
    parser.add_argument("--hd", action="store_true", help="腕摄换 128×96（仅分辨率；默认 64×48）")
    parser.add_argument("--distill", choices=["v6", "v4", "v3"], default=None,
                        help="用蒸馏训练 cfg 看真实学生出生画面（v4=扫视姿态+双窗口+128×96，当前口径）")
    parser.add_argument("--no-pov", action="store_true",
                        help="不切腕摄 POV——保留第三人称外部视角看机械臂姿态本身")
    return parser.parse_args()


def t(buf):
    """IsaacLab ProxyArray -> torch.Tensor"""
    return buf.torch if hasattr(buf, "torch") else buf


def set_viewport_camera(prim_path: str) -> bool:
    """把 kit 视口切到相机 POV（IsaacLab generate_data.py 的 _set_sensor_camera_view 同款）。"""
    try:
        import omni.kit.viewport.utility as viewport_util

        viewport = viewport_util.get_active_viewport()
        if viewport is None:
            # 新版 kit API：viewport 挂在 window 下
            vw = viewport_util.get_active_viewport_window()
            viewport = vw.viewport if vw is not None else None
        if viewport is None:
            return False
        viewport.set_active_camera(prim_path)
        return True
    except Exception:
        return False  # headless / 非 kit 后端：无窗口可切


def capture_rgb(camera):
    """从腕摄读 env_0 的 RGB 帧 (H, W, 3) uint8 numpy；未就绪返回 None。"""
    buf = camera.data.output
    if not buf or "rgb" not in buf:
        return None
    rgb = t(buf["rgb"])  # (N, H, W, 4) uint8
    if rgb is None or rgb.numel() == 0:
        return None
    return rgb[0, :, :, :3].cpu().numpy()


def find_cjk_font() -> str | None:
    """找一个支持中文的系统字体（照片烧字用；找不到返回 None 用 ASCII 默认字体）。"""
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(path).exists():
            return path
    import subprocess

    try:
        out = subprocess.run(
            ["fc-list", ":lang=zh", "file"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            path = line.strip().rstrip(":").strip()
            if path and Path(path).exists():
                return path
    except Exception:
        pass
    return None


def save_frame(frame, path: Path, caption_lines: list[str]) -> bool:
    """把 64×48 传感器帧放大存 PNG，顶部烧入黑底白字说明（出生号/阶段/橘子位）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(frame).convert("RGB")
        img = img.resize((img.width * UPSCALE, img.height * UPSCALE), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        font = None
        font_path = find_cjk_font()
        if font_path is not None:
            try:
                font = ImageFont.truetype(font_path, 26)
            except Exception:
                font = None
        if font is None:
            font = ImageFont.load_default()
        line_h = 34
        box_h = line_h * len(caption_lines) + 16
        draw.rectangle((0, 0, img.width, box_h), fill=(0, 0, 0))
        for i, line in enumerate(caption_lines):
            draw.text((12, 8 + i * line_h), line, fill=(255, 255, 255), font=font)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return True
    except Exception as exc:  # 存图失败不中断巡览
        print(f"      [save] 保存失败 {path.name}: {exc}", flush=True)
        return False


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

    from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402

    from pick_anything.tasks.pick_apple.camera_env_cfg import (  # noqa: E402
        CameraStagedHdSceneCfg,
        CameraStagedPickAppleEnvCfg,
    )
    from pick_anything.tasks.pick_apple.distill_env_cfg import (  # noqa: E402
        CameraDistillV3PickAppleEnvCfg,
        CameraDistillV4PickAppleEnvCfg,
        CameraDistillV6PickAppleEnvCfg,
    )
    from pick_anything.tasks.pick_apple.stages import STAGE_NAMES  # noqa: E402

    torch.manual_seed(0)

    # --distill v4|v3|off：用蒸馏训练 cfg 看真实学生出生画面。
    # v4 = P7 扫视姿态+橘子/盒子双窗口+128×96（当前口径）；v3 = 旧姿态
    # +橘子窗口+128×96。默认 off：相机 v0 任务口径（无窗口，自然刷位带）。
    # 不调 play_mode()——那会 forced_stage=0 且钳 envs，看不到真实课程抽样。
    variant = getattr(args, "distill", "off")
    if variant == "v6":
        cfg = CameraDistillV6PickAppleEnvCfg()
    elif variant == "v4":
        cfg = CameraDistillV4PickAppleEnvCfg()
    elif variant == "v3":
        cfg = CameraDistillV3PickAppleEnvCfg()
    else:
        cfg = CameraStagedPickAppleEnvCfg()
        # --hd：换 128×96 腕摄（仅换分辨率、不挂窗口；换 scene 必须在
        # resolve_presets 之前，否则 MultiBackendRendererCfg 拿不到解析）
        if getattr(args, "hd", False):
            cfg.scene = CameraStagedHdSceneCfg()
    # 训练/播放 CLI 在 register_task 里做 preset 解析（MultiBackendRendererCfg
    # -> NewtonWarpRendererCfg）；standalone 直接构造 cfg 会跳过那步，相机
    # 拿到未解析的 PresetCfg 直接崩，这里显式走同一个公开 API（默认选
    # default 档，与训练不带 presets= 时的行为一致）。
    cfg = resolve_presets(cfg)
    cfg.scene.num_envs = min(int(getattr(args, "num_envs", 0) or 2), 8)
    if args.stage >= 0:
        cfg.events.reset_staged.params["forced_stage"] = args.stage
    env = ManagerBasedRLEnv(cfg)

    try:
        cam = env.scene.sensors["wrist_camera"]
        state = getattr(env, "staged_pick_state", None)

        env.sim.set_camera_view(eye=[1.5, -1.8, 1.2], target=[0.35, -0.4, 0.1])
        for _ in range(WARMUP_STEPS):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.physics_dt)
        pov = False if getattr(args, 'no_pov', False) else set_viewport_camera(WRIST_CAMERA_PRIM)
        print(
            f"== num_envs={cfg.scene.num_envs}  "
            f"forced_stage={'自然课程' if args.stage < 0 else args.stage}  "
            f"视口POV={'腕部相机' if pov else '未切换(headless?)'}  "
            f"关窗口/Ctrl+C 退出 ==",
            flush=True,
        )

        spawn_no = 0
        while args.resets <= 0 or spawn_no < args.resets:
            if not env.sim.is_headless_or_exist_active_visualizer():
                return  # 可视化窗口已关闭
            env.reset()  # 真实出生：StagedReset 按课程抽样写关节/橘子/纸盒位姿
            # 查看器无策略：reset 事件只写位置/速度不写 PD 目标，残留的零
            # 目标会在观看期间把臂拖去关节零位（视野甩向右方）。把目标钉在
            # 出生位姿（default≈扫描姿态 ± 复位噪声）——训练里动作每步覆盖
            # 目标，无此问题（2026-10-02 用户 UI 实测反馈修复）
            _robot = env.scene["robot"]
            _robot.set_joint_position_target(_robot.data.default_joint_pos.clone())
            for _ in range(SPAWN_SETTLE_STEPS):
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(env.physics_dt)
            if not getattr(args, 'no_pov', False):
                set_viewport_camera(WRIST_CAMERA_PRIM)  # viewport frame 可能重建，幂等重设

            frame = capture_rgb(cam)
            births = [int(x) for x in state.birth_stage.tolist()] if state is not None else []
            names = " ".join(f"env{i}:阶段{x}/{STAGE_NAMES[x]}" for i, x in enumerate(births))
            xy = state.spawn_orange_xy[0].tolist() if state is not None else [0.0, 0.0]
            stats = (
                f" 传感器帧 min={frame.min()} mean={frame.mean():.1f} max={frame.max()}"
                if frame is not None
                else " 传感器空帧"
            )
            print(
                f"[SPAWN {spawn_no}] {names} 橘子xy=({xy[0]:+.3f},{xy[1]:+.3f}){stats}",
                flush=True,
            )
            if births:
                set_stage_label(
                    f"出生#{spawn_no} 阶段{births[0]}/7 {STAGE_NAMES[births[0]]}",
                    STAGE_COLORS.get(births[0], 0xFF888888),
                )
            if args.save and frame is not None:
                k = births[0] if births else -1
                name = STAGE_NAMES[k] if 0 <= k < len(STAGE_NAMES) else "NA"
                out = (
                    PROJECT_ROOT / "photos" / "spawn_camera"
                    / f"{spawn_no:03d}_stage{k}_{name}.png"
                )
                if save_frame(
                    frame,
                    out,
                    [
                        f"出生#{spawn_no}  阶段{k}/7 {name}",
                        f"橘子xy=({xy[0]:+.3f},{xy[1]:+.3f})  envs={cfg.scene.num_envs}  "
                        f"forced_stage={'自然' if args.stage < 0 else args.stage}",
                    ],
                ):
                    print(f"      [save] {out.relative_to(PROJECT_ROOT)}", flush=True)

            # 保持观看：物理步进即渲染；PD 目标由 StagedReset 写好，write_data_to_sim 持续下发
            for _ in range(args.hold):
                if not env.sim.is_headless_or_exist_active_visualizer():
                    return  # 可视化窗口已关闭
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(env.physics_dt)
            spawn_no += 1
    except Exception:
        # kit 卸载（simulation_app.close）会吞掉未处理异常的 traceback 且进程
        # 以 0 退出，趁 stderr 还活着先把现场打出来
        import traceback

        traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
