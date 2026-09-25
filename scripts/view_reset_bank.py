"""巡览八阶段 reset 状态库：逐行恢复 + 物理保持，可视化检查每个出生阶段。

模仿 IsaacLabTutorial place_vial 的 ``view_main``：不加载策略，把 bank 的
每一行（关节位置 + 控制器目标 + 橘子/纸盒位姿）写回单环境仿真，
保持若干物理步（软舵机 PD + 真实接触），按阶段 0..7 顺序巡览。每行保持
结束后打印关键物理读数（橘子高度 / 口袋距 / 爪角 / 爪力 / 漂移 / 盒局部
坐标），用于肉眼核对「恢复后再跑物理仍然稳定」。

用法（Pick_Anything 根目录）：

    conda activate isaaclab_61
    python scripts/view_reset_bank.py --visualizer kit --device cuda:2

    # 只看指定阶段、每阶段限量、放慢保持：
    python scripts/view_reset_bank.py --viz kit --stages 2,3,7 --per_stage 2 --steps_per_pose 90

    # 拍照模式：每阶段均匀抽 20 行，恢复+保持后离屏截图存盘（warp 渲染，
    # 无需窗口；阶段信息烧进图片）
    python scripts/view_reset_bank.py --device cuda:2 --photos 20
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

# 指尖捕获口袋偏移（isaaclab_assets SO101 两指接触垫中点，开爪 1.40 rad
# 下生成器运行时实测值；与 mdp.POCKET_OFFSET_B 同源）
POCKET_OFFSET_B = (0.0408, -0.0002, -0.0709)

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
        pass  # headless / 非 kit 后端：控制台的 [RESET] 行已标注阶段


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cycle the pick-apple reset bank poses in the visualizer.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--dataset", type=str, default="reset_states/pick_orange_bank_v1.pt")
    parser.add_argument("--steps_per_pose", type=int, default=60, help="每行保持的物理步数（0.02s/步）")
    parser.add_argument("--cycles", type=int, default=1, help="整体巡览轮数")
    parser.add_argument("--per_stage", type=int, default=0, help="每阶段最多巡览的行数（0=全部）")
    parser.add_argument("--stages", type=str, default="", help="只看这些阶段，如 '0,2,7'（空=全部）")
    parser.add_argument("--photos", type=int, default=0, help="拍照模式：每阶段保存的照片数（0=关闭）")
    parser.add_argument(
        "--photo_dir", type=str, default="", help="照片输出目录（默认 photos/<数据集名>）"
    )
    return parser.parse_args()


def t(buf):
    """IsaacLab ProxyArray -> torch.Tensor"""
    return buf.torch if hasattr(buf, "torch") else buf


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


def save_photo(image, path: Path, caption_lines: list[str]) -> bool:
    """把 RGB 帧存成 PNG，顶部烧入黑底白字说明（阶段/行号/物理读数）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(image).convert("RGB")
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
    except Exception as exc:  # 拍照失败不中断巡览
        print(f"      [photo] 保存失败 {path.name}: {exc}", flush=True)
        return False


def capture_rgb(photo_cam):
    """从拍照相机读一帧 RGB（H, W, 3) uint8 numpy；未就绪返回 None。"""
    buf = photo_cam.data.output
    if not buf or "rgb" not in buf:
        return None
    rgb = t(buf["rgb"])  # (1, H, W, 4) uint8
    if rgb is None or rgb.numel() == 0:
        return None
    return rgb[0, :, :, :3].cpu().numpy()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

    from pick_anything.tasks.pick_apple.reset_bank import load_reset_dataset  # noqa: E402
    from pick_anything.tasks.pick_apple.stages import STAGE_NAMES  # noqa: E402
    from pick_anything.tasks.pick_apple.staged_env_cfg import StagedPickAppleEnvCfg  # noqa: E402

    torch.manual_seed(0)

    # staged env：Newton 物理后端（write 后驱动存活），场景与训练同源
    cfg = StagedPickAppleEnvCfg()
    cfg.scene.num_envs = 1
    # 拍照模式：挂独立离屏相机，用 Newton Warp 渲染器（纯 warp 光线追踪，
    # 不依赖 kit/omni.replicator——该模块在 IsaacSim 6.1 已移除，kit 视口
    # 抓帧 API render_rgb_array 在本机不可用），可完全 headless 运行
    if args.photos > 0:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        cfg.scene.photo_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/PhotoCam",
            update_period=0.0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                clipping_range=(0.01, 20.0),
            ),
            renderer_cfg=NewtonWarpRendererCfg(),
        )
    env = ManagerBasedRLEnv(cfg)

    try:
        dataset_path = Path(args.dataset)
        if not dataset_path.is_absolute():
            dataset_path = PROJECT_ROOT / dataset_path
        artifact = load_reset_dataset(dataset_path, device=env.device)
        states = artifact["states"]

        counts = torch.bincount(states["stage"], minlength=len(STAGE_NAMES))
        print(f"== reset bank: {dataset_path}  共 {int(counts.sum())} 行 ==")
        for k, name in enumerate(STAGE_NAMES):
            print(f"  阶段 {k} {name:>9}: {int(counts[k])} 行")
        print(f"  generator: {artifact['generator']}")
        print(f"  validation: {artifact['validation']}", flush=True)

        # 巡览顺序：阶段优先，同阶段按难度（与教程 view_main 同款排序）
        order = torch.argsort(states["stage"].float() + states["difficulty"] * 0.01, stable=True).tolist()

        wanted: set[int] | None = None
        if args.stages.strip():
            wanted = {int(x) for x in args.stages.split(",") if x.strip() != ""}
        pools: dict[int, list[int]] = {}
        for row in order:
            k = int(states["stage"][row])
            if wanted is None or k in wanted:
                pools.setdefault(k, []).append(row)

        # 行选择：拍照模式每阶段均匀抽取（覆盖难度范围），否则顺序限量
        rows: list[int] = []
        for k in sorted(pools):
            pool = pools[k]
            if args.photos > 0:
                take = min(args.photos, len(pool))
                if take < len(pool):  # 均匀抽：idx = round(i*(n-1)/(take-1))
                    span = torch.linspace(0, len(pool) - 1, take).round().long().unique().tolist()
                    rows.extend(pool[i] for i in span)
                else:
                    rows.extend(pool)
            elif args.per_stage > 0:
                rows.extend(pool[: args.per_stage])
            else:
                rows.extend(pool)
        if not rows:
            raise SystemExit("过滤后没有可巡览的行（检查 --stages / --per_stage / bank 内容）。")

        robot = env.scene["robot"]
        orange = env.scene["orange"]
        box = env.scene["cardboard_box"]
        origins = t(env.scene.env_origins)
        gripper_body = [str(x) for x in robot.data.body_names].index("gripper")
        sensor = env.scene.sensors["robot_contact"]
        claw_cols = [i for i, nm in enumerate([str(x) for x in sensor.body_names])
                     if nm in ("gripper", "moving_jaw_so101_v1")]
        gripper_jid = int(robot.find_joints(["gripper"])[0][0])
        pocket_offset = torch.tensor(POCKET_OFFSET_B, device=env.device)
        zeros = torch.zeros(1, 6, device=env.device)
        ids = torch.arange(1, device=env.device)

        env.sim.set_camera_view(eye=[1.5, -1.8, 1.2], target=[0.35, -0.4, 0.1])

        # 拍照模式准备：相机就位 + 预热（warp 网格/纹理上传，最初几帧可能未就绪）
        photo_cam = env.scene.sensors.get("photo_cam") if args.photos > 0 else None
        photo_dir: Path | None = None
        photo_counts: dict[int, int] = {}
        if photo_cam is not None:
            photo_dir = (
                Path(args.photo_dir)
                if args.photo_dir
                else PROJECT_ROOT / "photos" / dataset_path.stem
            )
            eye = torch.tensor([[1.5, -1.8, 1.2]], device=env.device)
            look = torch.tensor([[0.35, -0.4, 0.1]], device=env.device)
            photo_cam.set_world_poses_from_view(eye, look)
            # 预热：走几步物理让渲染器建立帧缓冲（结果会被逐行恢复覆盖）
            for _ in range(10):
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(env.physics_dt)
            print(f"== 拍照模式：每阶段最多 {args.photos} 张 -> {photo_dir} ==", flush=True)

        def hold_readout(row: int) -> dict[str, object]:
            """保持结束后的物理读数（env 局部系）。"""
            pos = t(robot.data.body_pos_w)[:, gripper_body]
            quat = t(robot.data.body_quat_w)[:, gripper_body]
            pocket = pos + quat_apply(quat, pocket_offset.expand(1, 3)) - origins
            orange_local = t(orange.data.root_pos_w) - origins
            history = sensor.data.net_forces_w_history
            frames = t(history) if history is not None else t(sensor.data.net_forces_w).unsqueeze(1)
            per_body = frames.norm(dim=-1).amax(dim=1)
            claw = per_body[:, claw_cols].amax(dim=-1)
            in_box = quat_apply_inverse(t(box.data.root_quat_w), t(orange.data.root_pos_w) - t(box.data.root_pos_w))
            drift = (orange_local - states["orange_pose"][row][:3].to(env.device)).norm()
            return {
                "orange_z": float(orange_local[0, 2]),
                "pocket_dist": float((orange_local - pocket).norm()),
                "gripper": float(t(robot.data.joint_pos)[0, gripper_jid]),
                "claw_n": float(claw[0]),
                "drift": float(drift),
                "in_box": in_box[0].tolist(),
            }

        prev_stage = -1
        for _ in range(args.cycles):
            for row in rows:
                k = int(states["stage"][row])
                if k != prev_stage:
                    print(f"\n======== 阶段 {k} {STAGE_NAMES[k]} ========", flush=True)
                    prev_stage = k

                joint_position = states["joint_position"][row : row + 1].clone()
                joint_target = states["joint_target"][row : row + 1].clone()
                robot.write_joint_position_to_sim_index(position=joint_position, env_ids=ids)
                robot.write_joint_velocity_to_sim_index(velocity=zeros, env_ids=ids)
                for asset, key in (
                    (orange, "orange_pose"), (box, "box_pose"),
                ):
                    pose = states[key][row : row + 1].clone()
                    pose[:, :3] += origins
                    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
                    asset.write_root_velocity_to_sim_index(root_velocity=zeros, env_ids=ids)
                env.scene.write_data_to_sim()

                print(
                    f"[RESET] row={row} stage={k} {STAGE_NAMES[k]} "
                    f"difficulty={float(states['difficulty'][row]):.3f} "
                    f"grasped={bool(states['grasped'][row])}",
                    flush=True,
                )
                set_stage_label(
                    f"阶段 {k}/7 {STAGE_NAMES[k]}  row={row}",
                    STAGE_COLORS.get(k, 0xFF888888),
                )

                for _ in range(args.steps_per_pose):
                    if not env.sim.is_headless_or_exist_active_visualizer():
                        return  # 可视化窗口已关闭
                    robot.set_joint_position_target_index(target=joint_target, env_ids=ids)
                    env.scene.write_data_to_sim()
                    env.sim.step()
                    env.scene.update(env.physics_dt)

                r = hold_readout(row)
                in_box = r["in_box"]
                print(
                    f"      [hold {STAGE_NAMES[k]}] 橘子z={r['orange_z']:.3f}m 口袋距={r['pocket_dist']:.3f}m "
                    f"爪角={r['gripper']:.2f}rad 爪力={r['claw_n']:.2f}N 漂移={r['drift'] * 1000:.0f}mm "
                    f"盒局部=({in_box[0]:+.3f},{in_box[1]:+.3f},{in_box[2]:+.3f})",
                    flush=True,
                )

                if photo_dir is not None:
                    frame = capture_rgb(photo_cam)
                    if frame is not None and frame.size > 0:
                        photo_counts[k] = photo_counts.get(k, 0) + 1
                        out = (
                            photo_dir
                            / f"stage{k}_{STAGE_NAMES[k]}"
                            / f"{photo_counts[k]:02d}_row{row:05d}.png"
                        )
                        if save_photo(
                            frame,
                            out,
                            [
                                f"阶段 {k}/7 {STAGE_NAMES[k]}  row={row}  "
                                f"difficulty={float(states['difficulty'][row]):.3f}",
                                f"橘子z={r['orange_z']:.3f}m 口袋距={r['pocket_dist']:.3f}m "
                                f"爪角={r['gripper']:.2f}rad 爪力={r['claw_n']:.2f}N",
                            ],
                        ):
                            print(f"      [photo] {out.relative_to(PROJECT_ROOT)}", flush=True)
                    else:
                        print("      [photo] 空帧，跳过（RTX 可能仍在预热）", flush=True)

        if photo_dir is not None:
            print("\n== 照片汇总 ==", flush=True)
            for k in sorted(pools):
                if wanted is not None and k not in wanted:
                    continue
                name = STAGE_NAMES[k]
                n = photo_counts.get(k, 0)
                note = "" if n else "（bank 无行，跳过）"
                print(f"  阶段 {k} {name:>9}: {n} 张{note}  -> {photo_dir / f'stage{k}_{name}'}")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
