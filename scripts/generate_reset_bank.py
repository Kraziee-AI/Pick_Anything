"""生成八阶段「物理验证」reset 状态数据集（IsaacLabTutorial place_vial 同款方式）。

与旧版（传送构造夹持）的根本区别——本版完全对齐参考项目 generator.py：

1. **NewtonIKSolver 姿态约束 IK**（isaaclab_newton.ik，64 种子高斯采样 +
   解析雅可比 LM）：就位/搬运全部由带姿态目标的批量 IK 解出，不再用
   雅可比转置闭环（旧版位置 IK 姿态冗余 → V 口随机倾斜 → 夹不住）；
2. **真实物理抓取链**：home → 悬停对准（开爪）→ 下降就位 → 物理闭合 →
   **proof lift**（真实上提 12mm 验证承重）→ 上举 → 保持 → 搬运 → 盒口
   → 降入 → 松爪+撤离。橘子全程是自由刚体，从未被传送进爪里；
3. **seed bank 链**：grasp→lift→stabilize→transport→release 的验证通过态
   存入种子库，下游阶段从库恢复后继续走物理轨迹（每个候选都是一条连通
   的物理历史，不是坐标拼贴）；
4. **逐阶段物理验证门**（_valid）：finite / 关节限位 / 平静 / 夹持
   （爪部实压 + 居中 + 有效角带）/ 漂移 / 盒未被推走 / 速度收敛，全部
   读真实物理量；难度 = (stage + 段内 fraction) / 7，LIFT 段按实测高度
   标注（教程 measured_lift_progress 同款）。

八个阶段（bank[stage=k] = 出生在阶段 k，正要完成阶段 k 的目标）：
    0 HOME       home 姿态 + 随机刷位（橘子桌面静置）
    1 PREGRASP   开爪下降就位（0..1 段内进度，含部分闭合）
    2 GRASP      真实闭合 + proof lift 通过（承重夹持）
    3 LIFT       夹住上举（0..1 段内进度，按实测高度计）
    4 STABILIZE  举在高处、晃动收敛
    5 TRANSPORT  搬到盒口上方悬停
    6 PLACE      降到盒内释放位（仍夹持）
    7 RELEASE    盒内松爪 + 撤离，橘子盒内静置

用法（Pick_Anything 根目录，conda env isaaclab_61）：

    python scripts/generate_reset_bank.py --num_envs 64 --poses_per_stage 32 \
        --device cuda:2 --output reset_states/pick_orange_bank_v1.pt

小规模试跑（诊断打印各段链通过率）：

    python scripts/generate_reset_bank.py --num_envs 32 --poses_per_stage 8 \
        --device cuda:2 --diagnose
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

# --------------------------------------------------------------------- #
# 常量（米 / rad / N）。机器人 = isaaclab_assets SO101（教程同款），
# 爪角范围 rad(-0.175, 1.745)：小值=闭合、大值=全开。
# --------------------------------------------------------------------- #

PREGRASP_GRIPPER_POSITION = 1.40   # 就位开度（宽口进入，40mm 球自由入爪）
GRASP_GRIPPER_POSITION = -0.155    # 物理闭合指令（教程真实抓取命令 radians(-10+1.1*1)）
RELEASE_GRIPPER_POSITION = 1.30    # 松爪开度（必须大于球的直径间隙）
# 2026-09-25 挤压实测：闭合后爪被球挡停在 0.14~0.28 rad（25-30N 夹持），
# 但保持指令 -0.155 越过挡停角——上提瞬间球滞后解锁，爪扫过球面把球
# 踢出（亚稳态）。保持指令 = 挡停角 - 本预压（轻压 ~3N，够摩擦抓持、
# 不会触发扫掠）。
# 0.16：冷恢复证明（150 步保持）实测 0.10 只有 3-6% 存活——加深挤压
# 让球嵌进 V 口（0.16 rad 仍远小于扫掠阈值 ~0.4）
GRASP_HOLD_PRELOAD_RAD = 0.16

# 指尖接触垫中心（各自 body 系）——取自教程 terms.py 的实测常量（同一
# USD 资产，精确适用）。捕获 TCP = 两垫中点，在启动时按开爪几何测量。
FIXED_FINGERTIP_OFFSET = (-0.01289, -0.00022, -0.0986)
MOVING_FINGERTIP_OFFSET = (-0.0068, -0.0766, 0.0189)

IK_SEEDS = 64
IK_ITERATIONS = 120
IK_NOISE_STD = 0.65
IK_POSITION_TOLERANCE = 0.005
IK_JOINT_MARGIN = 0.03
# 腕姿态必须留在 home 姿态的 yaw 族内（绕世界 Z 旋转不改变爪面相对
# 重力的倾角，对球抓取等价）。整四元数钉死则过约束（位置误差 2cm），
# 完全自由则合爪扫出（GRASP 段爪力 0.1N、橘子被推离 2.8cm）。
# 2026-09-25 实测：此爪 home 姿态垫线本身倾 ~49°（筷子式夹爪）——
# 姿态门收紧到 12° 会灭掉全部 IK 解；pitch 分支多样性是必要的
# （其中水平垫线分支恰是有效夹持），坏分支由物理验证门拒绝。
IK_YAW_FAMILY_STEP_DEG = 30.0
IK_HOME_QUAT_MIN_COS = math.cos(math.radians(35.0))
# 负载（夹着球）解的姿态门：必须留在建立夹持的 home yaw 族包络内。
# 与就位门同宽（35°）——抓取腕姿态本身允许偏离族 35°，更紧会把从
# 合法抓取出发的负载解全部误杀（20° 实测 loaded_ik_valid 0.0）
IK_LOADED_QUAT_MIN_COS = math.cos(math.radians(35.0))
# 相对链起点锚的姿态漂移门（防逐航点累计漂移；35°——盒口横搬需要的
# 腕姿态变化实测超过 20°，12/20° 均灭掉横搬负载解）
IK_LOADED_DRIFT_MIN_COS = math.cos(math.radians(35.0))

# 夹持力门（claw = 两指 body 受力 3 帧历史 amax；橘子净力对称抵消
# 不可作判据——与任务侧 staged_mdp 同口径）
GRASP_CLAW_FORCE_N = 1.2
GRASP_CENTERED_M = 0.040   # 橘子中心到爪口中点的距离（居中=双指对称夹）
# 有效夹持爪角带：下限=USD 软限位；上限=夹 56mm 球时动爪被球挡住的
# 稳态角（实测 ~0.4-0.6 rad，闭合指令 -0.155 够不到——被球顶住即是夹持）
GRIP_BAND = (-0.174, 0.75)

# 验证门
V_CALM_JOINT_VEL = 2.0
V_SPEED_MPS = 0.15
V_ANG_SPEED = 1.0
V_BOX_DRIFT_M = 0.05  # 30g 轻盒被爪入盒/球落入冲量微推属正常，3cm 过严
V_TABLE_DRIFT_M = 0.006
V_TABLE_Z_BAND = 0.005

# 轨迹时长（物理步，120Hz）；闭合扫过全程 1.4->-0.155，放慢防冲击挤出
SETTLE_STEPS = 90
TRANSITION_STEPS = 120
GRASP_CLOSE_STEPS = 150

LIFT_TRAVEL_M = 0.105       # LIFT 段目标上举行程
PROOF_LIFT_M = 0.012        # GRASP 段承重验证上提
# 抓取点 = 球心上方偏置。垫几何实测（2026-09-25）：捕获 TCP（两垫中点）
# 在 gripper 系 z=-0.0709，固定垫 z=-0.0986——固定垫比 TCP 低 ~24-28mm
# （随腕 pitch 波动）。夹持稳定性实测：
#   赤道/上翼夹持（+24~+40mm）球必滑脱：爪上提→球滞后→接触点上移→
#     弦变窄→爪跟进→球再滞后（棘轮），滚动摩擦 0.30 也只能延缓；
#   **shelf 抓法（+13mm）**：固定垫落在球心下方 6-15mm——球下滑只会
#     让弦更宽、被驱动顶死，几何自锁；配合滚动摩擦 0.30，实测 5/8 环境
#     球被托离桌面 +2mm 稳定 150 步不落（接触柔顺限制弹性行程 ~5mm，
#     初 slip 后刚性跟随爪）。
GRASP_Z_OFFSET_M = 0.013
HOVER_ABOVE_BOX_M = 0.10    # TRANSPORT 悬停高度（盒原点上方；0.14 实测
                             # 超臂展——负载 IK valid 仅 6%，0.10 仍高于盒口 5cm）
PLACE_LOCAL_Z_M = 0.044     # PLACE 橘子中心在盒局部系的释放高度（盒口
                             # 5cm；0.032 实测降入过深——固定指撞盒底，
                             # 负载 IK valid 3%；0.044 球底离盒底 16mm，
                             # 松爪后落底）
RELEASE_RETREAT_M = (0.0, 0.02, 0.05)  # 松爪后口袋撤离向量（后退+上提）

STAGE_LIST = ("HOME", "PREGRASP", "GRASP", "LIFT", "STABILIZE", "TRANSPORT", "PLACE", "RELEASE")


@dataclass(frozen=True)
class GeneratorCfg:
    """生成器配额与验证参数。"""

    batch_size: int = 64
    poses_per_stage: int = 32
    seed: int = 42
    max_attempts_per_stage: int = 8192
    branch_seed_count: int = 16
    home_joint_noise: float = 0.03
    spawn_difficulty_range: tuple[float, float] = (0.7, 1.0)
    diagnose: bool = False


def t(buf):
    """IsaacLab ProxyArray -> torch.Tensor"""
    return buf.torch if hasattr(buf, "torch") else buf


class _Generator:
    """物理轨迹 + 分阶段验证 + seed bank 链（教程 _Generator 移植）。"""

    def __init__(self, env, cfg: GeneratorCfg):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(env.device)
        self.random = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.robot = env.scene["robot"]
        self.orange = env.scene["orange"]
        self.box = env.scene["cardboard_box"]
        self.num_envs = env.num_envs
        self._zeros = torch.zeros((self.num_envs, 6), device=self.device)
        self.gripper_body_id = int(self.robot.find_bodies(["gripper"], preserve_order=True)[0][0])
        self.gripper_id = int(self.robot.find_joints(["gripper"], preserve_order=True)[0][0])
        self._moving_jaw_body_id = int(
            self.robot.find_bodies(["moving_jaw_so101_v1"], preserve_order=True)[0][0]
        )

        # claw 受力列（两指 body；新 USD 的动指是独立 body）
        sensor = env.scene.sensors["robot_contact"]
        names = [str(x) for x in sensor.body_names]
        self._claw_cols = [i for i, nm in enumerate(names) if nm in ("gripper", "moving_jaw_so101_v1")]
        self._contact_sensor = sensor

        self.candidate_ik_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.candidate_box_ruined = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.candidate_difficulty = torch.zeros(self.num_envs, device=self.device)
        self.last_diagnostics: dict[str, float] = {}
        self._seed_banks: dict[str, dict[str, torch.Tensor]] = {}
        self.partial_states: dict[str, torch.Tensor] = {}
        self._box_start_xy: torch.Tensor | None = None
        self._track_box = False

        # 环境构造后尚未跑过 reset，先显式摆出 home（爪开）再测口袋与姿态
        self.home_joint_position = t(self.robot.data.default_joint_pos).clone()
        self.home_joint_position[:, self.gripper_id] = PREGRASP_GRIPPER_POSITION
        limits = t(self.robot.data.soft_joint_pos_limits)
        self.home_joint_position = self.home_joint_position.clamp(limits[..., 0], limits[..., 1])
        self._write_robot(self.home_joint_position)
        self._commit_reset_state()
        self._simulate(8, self.home_joint_position)
        self._home_gripper_quat = t(self.robot.data.body_quat_w)[:, self.gripper_body_id].clone()
        self._build_home_quat_family()
        # 运行时测量捕获 TCP：两指接触垫中点在 gripper body 系的偏移
        # （垫心常量来自教程 terms.py 同源 USD；开爪几何下取值，教程
        # _pregrasp_center_offset 同款做法）
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        body_pos = t(self.robot.data.body_pos_w)
        body_quat = t(self.robot.data.body_quat_w)
        fixed_pad = body_pos[:, self.gripper_body_id] + quat_apply(
            body_quat[:, self.gripper_body_id],
            body_pos.new_tensor(FIXED_FINGERTIP_OFFSET).expand(self.num_envs, 3),
        )
        moving_pad = body_pos[:, self._moving_jaw_body_id] + quat_apply(
            body_quat[:, self._moving_jaw_body_id],
            body_pos.new_tensor(MOVING_FINGERTIP_OFFSET).expand(self.num_envs, 3),
        )
        midpoint = (fixed_pad + moving_pad) / 2
        offsets = quat_apply_inverse(
            body_quat[:, self.gripper_body_id], midpoint - body_pos[:, self.gripper_body_id]
        )
        spread = (offsets - offsets[0]).abs().amax()
        if not bool(torch.isfinite(offsets).all()) or float(spread) > 1.0e-5:
            raise RuntimeError(f"各 env 爪口几何不一致或非有限（spread={float(spread):.2e}）。")
        self.pocket_offset = offsets[0].clone()
        if self.cfg.diagnose:
            print(
                f"[init] 捕获 TCP(gripper 系)={self.pocket_offset.tolist()} "
                f"TCP 世界 z={float(midpoint[0, 2]):.3f}",
                flush=True,
            )

        self._build_ik()
        self.gripper_closed_position = GRASP_GRIPPER_POSITION
        lower = float(self.ik_joint_limits[self.gripper_id, 0])
        upper = float(self.ik_joint_limits[self.gripper_id, 1])
        if not lower <= self.gripper_closed_position <= upper:
            raise RuntimeError(
                f"闭合指令 {self.gripper_closed_position:.3f} 超出 USD 软限位 [{lower:.3f}, {upper:.3f}]。"
            )

    def _build_home_quat_family(self) -> None:
        """home 腕姿态绕世界 Z 的 yaw 族（球抓取的等价姿态集合）。"""
        from isaaclab.utils.math import quat_mul

        family = []
        steps = int(round(360.0 / IK_YAW_FAMILY_STEP_DEG))
        for k in range(steps):
            half = math.radians(IK_YAW_FAMILY_STEP_DEG * k) / 2.0
            rz = self._home_gripper_quat.new_tensor((0.0, 0.0, math.sin(half), math.cos(half)))
            family.append(quat_mul(rz.expand_as(self._home_gripper_quat), self._home_gripper_quat))
        self._home_quat_family = torch.stack(family, dim=1)  # (n, k, 4)

    def _quat_family_cos(self, gripper_quat: torch.Tensor) -> torch.Tensor:
        """每行相对 home yaw 族的最大四元数余弦（形状 = gripper_quat 前两维）。"""
        flat = gripper_quat.reshape(-1, 1, 4)                     # (m, 1, 4)
        fam = self._home_quat_family[:, None, :, :].expand(-1, flat.shape[0] // self.num_envs, -1, -1)
        fam = fam.reshape(flat.shape[0], -1, 4)                   # (m, k, 4)
        return (flat * fam).sum(-1).abs().amax(-1).reshape(gripper_quat.shape[:-1])

    # ------------------------------------------------------------------ #
    # Newton IK（教程 _build_ik 移植；关节名对齐本仓库 USD）
    # ------------------------------------------------------------------ #
    def _build_ik(self) -> None:
        import isaaclab.sim as sim_utils
        import warp as wp
        from isaaclab import cloner
        from isaaclab_newton.cloner import copy_newton_clone_source
        from isaaclab_newton.ik import (
            NewtonIKJointLimitObjectiveCfg,
            NewtonIKPoseObjectiveCfg,
            NewtonIKSolver,
            NewtonIKSolverCfg,
        )

        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        resolved = cloner.query.path_to_source(plan, self.robot.cfg.prim_path) if plan is not None else None
        if resolved is None:
            raise RuntimeError("无法解析 SO101 的 clone-plan 源（reset IK 需要）。")
        source = copy_newton_clone_source(resolved[0])
        origin = -self.env.scene.env_origins[0]
        prototype_xform = wp.transform(wp.vec3(*origin.tolist()), wp.quat_identity())
        import newton

        prototype = newton.ModelBuilder(up_axis=source.up_axis)
        prototype.add_builder(source, xform=prototype_xform)
        self.ik_model = prototype.finalize(device=str(self.device))

        body_names = [str(label).rsplit("/", 1)[-1] for label in self.ik_model.body_label]
        gripper_matches = [i for i, name in enumerate(body_names) if name == "gripper"]
        if len(gripper_matches) != 1:
            raise RuntimeError(f"Newton IK 模型里应恰有一个名为 'gripper' 的 body，实得 {len(gripper_matches)}。")
        self.ik_gripper_body_id = gripper_matches[0]

        joint_names = [str(label).rsplit("/", 1)[-1] for label in self.ik_model.joint_label]
        joint_q_start = wp.to_torch(self.ik_model.joint_q_start).to(device=self.device, dtype=torch.long)
        coordinate_ids = []
        for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"):
            matches = [i for i, joint_name in enumerate(joint_names) if joint_name == name]
            if len(matches) != 1:
                raise RuntimeError(f"Newton IK 模型里应恰有一个名为 {name!r} 的关节。")
            coordinate_ids.append(int(joint_q_start[matches[0]].item()))
        self.ik_coordinate_ids = torch.tensor(coordinate_ids, device=self.device, dtype=torch.long)
        self.ik_joint_seed = wp.to_torch(self.ik_model.joint_q).to(self.device).repeat(self.num_envs, 1)
        self.ik_joint_limits = t(self.robot.data.soft_joint_pos_limits)[0].clone()

        objective_name = "reset_orange_gripper"
        self.ik_solver = NewtonIKSolver(
            NewtonIKSolverCfg(
                optimizer="lm",
                jacobian_mode="analytic",
                sampler="gauss",
                n_seeds=IK_SEEDS,
                noise_std=IK_NOISE_STD,
                iterations=IK_ITERATIONS,
                lambda_initial=0.1,
                rng_seed=self.cfg.seed,
            ),
            model=self.ik_model,
            num_envs=self.num_envs,
            device=str(self.device),
            objectives=[
                NewtonIKPoseObjectiveCfg(
                    body_name="gripper",
                    name=objective_name,
                    body_offset_pos=tuple(float(v) for v in self.pocket_offset.tolist()),
                    position_weight=100.0,
                    # 位置主导（口袋必须贴抓取目标点）；姿态项拉回 home 腕
                    # 姿态分支（权重 3：更高会灭解，更低则 pitch 漂移过大）
                    rotation_weight=3.0,
                ),
                NewtonIKJointLimitObjectiveCfg(weight=1.0),
            ],
            link_resolver=lambda _name: self.ik_gripper_body_id,
        )
        self.ik_pose_objective = self.ik_solver.objectives_by_name[objective_name]

    # ------------------------------------------------------------------ #
    # 底层：物理循环 / 显式写入 / 读数
    # ------------------------------------------------------------------ #
    def _simulate(self, steps: int, target: torch.Tensor) -> None:
        for _ in range(steps):
            self.robot.set_joint_position_target_index(target=target)
            self.robot.set_joint_velocity_target_index(target=self._zeros)
            self.env.scene.write_data_to_sim()
            self.env.sim.step()
            self.env.scene.update(self.env.physics_dt)
            if self._track_box and self._box_start_xy is not None:
                box_xy = (t(self.box.data.root_pos_w) - t(self.env.scene.env_origins))[:, :2]
                pushed = (box_xy - self._box_start_xy).norm(dim=-1) > V_BOX_DRIFT_M
                box_up = t(self.box.data.root_quat_w)
                # 盒 up 轴 z 分量（四元数 (x,y,z,w)：R33 = 1-2(x²+y²)）
                tipped = (1.0 - 2.0 * (box_up[:, 0] ** 2 + box_up[:, 1] ** 2)) < 0.70
                self.candidate_box_ruined |= pushed | tipped

    def _move_robot(self, target: torch.Tensor, steps: int) -> None:
        """关节空间线性插值平滑到位（物体保持动力学）。"""
        start = t(self.robot.data.joint_pos).clone()
        for step in range(1, steps + 1):
            self._simulate(1, torch.lerp(start, target, step / steps))

    def _write_robot(self, joint_position: torch.Tensor) -> None:
        self.robot.write_joint_position_to_sim_index(position=joint_position)
        self.robot.write_joint_velocity_to_sim_index(velocity=self._zeros)
        self.robot.set_joint_position_target_index(target=joint_position)

    def _write_orange(self, local_pose: torch.Tensor) -> None:
        world_pose = local_pose.clone()
        world_pose[:, :3] += t(self.env.scene.env_origins)
        self.orange.write_root_pose_to_sim_index(root_pose=world_pose)
        self.orange.write_root_velocity_to_sim_index(root_velocity=self._zeros)

    def _write_box(self, local_pose: torch.Tensor) -> None:
        world_pose = local_pose.clone()
        world_pose[:, :3] += t(self.env.scene.env_origins)
        self.box.write_root_pose_to_sim_index(root_pose=world_pose)
        self.box.write_root_velocity_to_sim_index(root_velocity=self._zeros)

    def _park_orange(self) -> None:
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, :3] = t(self.env.scene.env_origins) + pose.new_tensor((0.0, 0.0, 1.0))
        pose[:, 6] = 1.0
        self.orange.write_root_pose_to_sim_index(root_pose=pose)
        self.orange.write_root_velocity_to_sim_index(root_velocity=self._zeros)

    def _commit_reset_state(self) -> None:
        """显式 write 后推进一次 forward，防止被拒候选的陈旧缓冲被下一个样本读到。"""
        self.env.scene.reset()
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()
        self.env.scene.update(self.env.physics_dt)

    # --- 读数 --- #
    def _pocket_w(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        pos = t(self.robot.data.body_pos_w)[:, self.gripper_body_id]
        quat = t(self.robot.data.body_quat_w)[:, self.gripper_body_id]
        offset = self.pocket_offset.expand(self.num_envs, 3)
        return pos + quat_apply(quat, offset)

    def _pad_midpoint_w(self) -> torch.Tensor:
        """两指接触垫中点（用当前 body 位姿现场计算）。

        开爪帧 TCP（_pocket_w，pocket_offset 常量）只对 IK 目标定义有效；
        闭合后动爪旋转 ~1.1 rad，垫中点相对常量 TCP 偏移可达 ~4cm——
        验证/分离判断必须用真实中点。
        """
        from isaaclab.utils.math import quat_apply

        pos = t(self.robot.data.body_pos_w)
        quat = t(self.robot.data.body_quat_w)
        fixed = pos[:, self.gripper_body_id] + quat_apply(
            quat[:, self.gripper_body_id], pos.new_tensor(FIXED_FINGERTIP_OFFSET).expand(self.num_envs, 3)
        )
        moving = pos[:, self._moving_jaw_body_id] + quat_apply(
            quat[:, self._moving_jaw_body_id], pos.new_tensor(MOVING_FINGERTIP_OFFSET).expand(self.num_envs, 3)
        )
        return (fixed + moving) / 2

    def _pocket_local(self) -> torch.Tensor:
        return self._pocket_w() - t(self.env.scene.env_origins)

    def _orange_local(self) -> torch.Tensor:
        return t(self.orange.data.root_pos_w) - t(self.env.scene.env_origins)

    def _dist_pocket(self) -> torch.Tensor:
        """橘子中心到真实垫中点的距离（验证用；闭合后常量 TCP 会偏 ~4cm）。"""
        mid_local = self._pad_midpoint_w() - t(self.env.scene.env_origins)
        return (self._orange_local() - mid_local).norm(dim=-1)

    def _orange_pose_local(self) -> torch.Tensor:
        return torch.cat(
            (t(self.orange.data.root_pos_w) - t(self.env.scene.env_origins), t(self.orange.data.root_quat_w)),
            dim=-1,
        ).clone()

    def _box_pose_local(self) -> torch.Tensor:
        return torch.cat(
            (t(self.box.data.root_pos_w) - t(self.env.scene.env_origins), t(self.box.data.root_quat_w)),
            dim=-1,
        ).clone()

    def _orange_local_in_box(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        box_pos = t(self.box.data.root_pos_w) - t(self.env.scene.env_origins)
        return quat_apply_inverse(t(self.box.data.root_quat_w), self._orange_local() - box_pos)

    def claw_force(self) -> torch.Tensor:
        history = self._contact_sensor.data.net_forces_w_history
        frames = t(history) if history is not None else t(self._contact_sensor.data.net_forces_w).unsqueeze(1)
        per_body = frames.norm(dim=-1).amax(dim=1)
        return per_body[:, self._claw_cols].amax(dim=-1)

    def orange_force(self) -> torch.Tensor:
        sensor = self.env.scene.sensors["orange_contact"]
        return t(sensor.data.net_forces_w).reshape(self.num_envs, -1, 3).norm(dim=-1).squeeze(-1)

    def gripper_pos(self) -> torch.Tensor:
        return t(self.robot.data.joint_pos)[:, self.gripper_id]

    # ------------------------------------------------------------------ #
    # IK 包装：开爪就位（口袋贴橘子中心，home 腕姿态分支）
    # ------------------------------------------------------------------ #
    def _solve_pocket_pose(
        self,
        orange_pose: torch.Tensor,
        seed_joint_position: torch.Tensor,
        *,
        vertical_offset: float = 0.0,
        ik_seed_joint_position: torch.Tensor | None = None,
        randomize_solution: bool = False,
    ) -> torch.Tensor:
        """解开爪姿态：捕获口袋中心对准橘子中心（+可选竖直抬高）。"""
        import warp as wp
        from isaaclab.utils.math import combine_frame_transforms, quat_apply

        target_position = orange_pose[:, :3].clone()
        target_position[:, 2] += vertical_offset
        target_quaternion = self._home_gripper_quat.clone()
        self.ik_pose_objective.position_objective.set_target_positions(
            wp.from_torch(target_position.contiguous(), dtype=wp.vec3)
        )
        self.ik_pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(target_quaternion.contiguous(), dtype=wp.vec4)
        )
        if ik_seed_joint_position is None:
            ik_seed_joint_position = seed_joint_position
        seed = self.ik_joint_seed.clone()
        seed[:, self.ik_coordinate_ids] = ik_seed_joint_position
        self.ik_solver.solve(wp.from_torch(seed.contiguous(), dtype=wp.float32))

        seeds = IK_SEEDS
        joint_q = wp.to_torch(self.ik_solver.joint_q).reshape(self.num_envs, seeds, -1)
        costs = wp.to_torch(self.ik_solver.costs).reshape(self.num_envs, seeds)
        body_pose = wp.to_torch(self.ik_solver.solver.body_q).reshape(self.num_envs, seeds, -1, 7)
        gripper_pose = body_pose[:, :, self.ik_gripper_body_id]
        tcp_offset = self.pocket_offset.expand(self.num_envs, seeds, 3)
        identity = torch.zeros_like(gripper_pose[:, :, 3:7])
        identity[:, :, 3] = 1.0
        actual_pocket, _ = combine_frame_transforms(
            gripper_pose[:, :, :3].reshape(-1, 3),
            gripper_pose[:, :, 3:7].reshape(-1, 4),
            tcp_offset.reshape(-1, 3),
            identity.reshape(-1, 4),
        )
        actual_pocket = actual_pocket.reshape(self.num_envs, seeds, 3)
        position_error = torch.linalg.vector_norm(actual_pocket - target_position[:, None], dim=-1)

        # 合爪轴朝向不再设门：leisaac 版 USD 的 gripper +X 不是合爪方向
        # （实测 home 腕姿态——已验证可夹的姿态——的 +X z 分量 ≈0.97）；就位
        # 姿态质量由旋转目标钉住 home 腕姿态分支 + proof lift 物理验证把关。
        closing = quat_apply(
            gripper_pose[:, :, 3:7].reshape(-1, 4),
            gripper_pose.new_tensor((1.0, 0.0, 0.0)).expand(self.num_envs * seeds, 3),
        ).reshape(self.num_envs, seeds, 3)
        closing_axis_z = closing[:, :, 2].abs()

        arm_q = joint_q[:, :, self.ik_coordinate_ids[:5]]
        arm_limits = self.ik_joint_limits[:5]
        margin = torch.minimum(arm_q - arm_limits[:, 0], arm_limits[:, 1] - arm_q).amin(-1)
        # 腕姿态必须留在 home 姿态的 yaw 族内（V 口相对重力的倾角不变）
        quat_cos = self._quat_family_cos(gripper_pose[:, :, 3:7])
        seed_valid = (
            torch.isfinite(joint_q).all(-1)
            & torch.isfinite(costs)
            & (position_error <= IK_POSITION_TOLERANCE)
            & (margin >= IK_JOINT_MARGIN)
            & (quat_cos >= IK_HOME_QUAT_MIN_COS)
        )
        continuity = torch.linalg.vector_norm(arm_q - ik_seed_joint_position[:, None, :5], dim=-1)
        score = 10.0 * position_error + 0.05 * continuity
        valid_score = score.masked_fill(~seed_valid, torch.inf)
        valid = seed_valid.any(-1)
        fallback = torch.nan_to_num(score, nan=torch.inf, posinf=torch.inf, neginf=torch.inf).argmin(-1)
        if randomize_solution:
            weights = seed_valid.float()
            weights[~valid] = 1.0
            sampled = torch.multinomial(weights, 1, replacement=True, generator=self.random).squeeze(-1)
            selected = torch.where(valid, sampled, fallback)
        else:
            selected = torch.where(valid, valid_score.argmin(-1), fallback)
        rows = torch.arange(self.num_envs, device=self.device)
        solved = joint_q[rows, selected][:, self.ik_coordinate_ids].clone()
        solved[:, -1] = PREGRASP_GRIPPER_POSITION
        solved[~valid] = seed_joint_position[~valid]
        solved[~valid, -1] = PREGRASP_GRIPPER_POSITION
        self.candidate_ik_valid &= valid
        if self.cfg.diagnose:
            pos_ok = position_error <= IK_POSITION_TOLERANCE
            quat_ok = quat_cos >= IK_HOME_QUAT_MIN_COS
            margin_ok = margin >= IK_JOINT_MARGIN
            finite_ok = torch.isfinite(joint_q).all(-1) & torch.isfinite(costs)
            self.last_diagnostics.update(
                {
                    "pregrasp_ik_valid_rate": float(valid.float().mean()),
                    "pregrasp_position_err_mean_m": float(position_error.amin(-1).mean()),
                    "ik_pos_rate": float(pos_ok.any(-1).float().mean()),
                    "ik_quat_rate": float(quat_ok.any(-1).float().mean()),
                    "ik_margin_rate": float(margin_ok.any(-1).float().mean()),
                    "ik_finite_rate": float(finite_ok.any(-1).float().mean()),
                    "ik_posnquat_rate": float((pos_ok & quat_ok).any(-1).float().mean()),
                    "ik_margin_best_mean_rad": float(margin.amax(-1).mean()),
                    "ik_quat_at_best_mean": float(quat_cos.gather(-1, position_error.argmin(-1, keepdim=True)).mean()),
                }
            )
        return solved

    # ------------------------------------------------------------------ #
    # IK 包装：负载搬运（保持实测的橘子->口袋相对变换）
    # ------------------------------------------------------------------ #
    def _solve_orange_pose(
        self,
        desired_orange_pose: torch.Tensor,
        seed_joint_position: torch.Tensor,
        *,
        position_tolerance: float | None = None,
        max_joint_delta: float | None = None,
        rotation_anchor: torch.Tensor | None = None,
        orientation_lock: bool = True,
    ) -> torch.Tensor:
        """解保持当前夹持变换的搬运目标（口袋带动橘子到指定位姿）。"""
        import warp as wp
        from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

        orange_position = t(self.orange.data.root_pos_w)
        orange_quaternion = t(self.orange.data.root_quat_w)
        pocket_position = self._pocket_w()
        gripper_quaternion = t(self.robot.data.body_quat_w)[:, self.gripper_body_id]
        if not orientation_lock:
            # 横搬段（5/6）：盒口悬停需要的腕姿态变化超出锚定门（实测
            # 末航点 IK 0%），旋转目标跟随当前姿态，仅靠 35° 族门 + 高滚阻
            # /强预压/航点重夹紧抑制掉落
            rotation_anchor = gripper_quaternion
        orange_to_pocket_pos, orange_to_pocket_quat = subtract_frame_transforms(
            orange_position, orange_quaternion, pocket_position, gripper_quaternion
        )
        target_position, target_quaternion = combine_frame_transforms(
            desired_orange_pose[:, :3],
            desired_orange_pose[:, 3:7],
            orange_to_pocket_pos,
            orange_to_pocket_quat,
        )
        self.ik_pose_objective.position_objective.set_target_positions(
            wp.from_torch(target_position.contiguous(), dtype=wp.vec3)
        )
        self.ik_pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(target_quaternion.contiguous(), dtype=wp.vec4)
        )
        seed = self.ik_joint_seed.clone()
        seed[:, self.ik_coordinate_ids] = seed_joint_position
        self.ik_solver.solve(wp.from_torch(seed.contiguous(), dtype=wp.float32))

        seeds = IK_SEEDS
        # 漂移门：负载链的旋转目标锚定在链起点的腕姿态（建立夹持时的姿态），
        # 每个解相对锚偏离 <= 12°。此前目标=「当前」姿态，逐航点累计漂移
        # 20-30° 把 shelf 翻掉（水平搬运终端爪力存活仅 4.7% 的根因）
        joint_q = wp.to_torch(self.ik_solver.joint_q).reshape(self.num_envs, seeds, -1)
        costs = wp.to_torch(self.ik_solver.costs).reshape(self.num_envs, seeds)
        body_pose = wp.to_torch(self.ik_solver.solver.body_q).reshape(self.num_envs, seeds, -1, 7)
        gripper_pose = body_pose[:, :, self.ik_gripper_body_id]
        tcp_offset = self.pocket_offset.expand(self.num_envs, seeds, 3)
        identity = torch.zeros_like(gripper_pose[:, :, 3:7])
        identity[:, :, 3] = 1.0
        actual_pocket, _ = combine_frame_transforms(
            gripper_pose[:, :, :3].reshape(-1, 3),
            gripper_pose[:, :, 3:7].reshape(-1, 4),
            tcp_offset.reshape(-1, 3),
            identity.reshape(-1, 4),
        )
        actual_pocket = actual_pocket.reshape(self.num_envs, seeds, 3)
        # 负载段只平移橘子（姿态不变），orange->pocket 变换在测量与预测两端
        # 相互抵消——位置误差就是「实际口袋 vs 目标口袋」，不必绕道橘子
        # （旧写法漏乘橘子四元数，随机 yaw 造出 |t_op| 量级的假误差，把
        # 绝大多数负载 IK 解错杀成无效 → 上提段整段冻结）
        position_error = torch.linalg.vector_norm(
            actual_pocket - target_position[:, None, :], dim=-1
        )

        arm_q = joint_q[:, :, self.ik_coordinate_ids[:5]]
        arm_limits = self.ik_joint_limits[:5]
        margin = torch.minimum(arm_q - arm_limits[:, 0], arm_limits[:, 1] - arm_q).amin(-1)
        if position_tolerance is None:
            position_tolerance = IK_POSITION_TOLERANCE
        # 姿态门（2026-09-25 水平搬运实测归零的根因）：负载解此前不设
        # 姿态约束——臂在水平搬运中重构肘位时腕 roll/pitch 自由漂移
        # 10-20°，shelf 倾斜、球滚出 V 口（垂直段不重构臂形所以无恙）。
        # 夹持是在 home yaw 族姿态下建立的，负载解必须留在同一族内。
        quat_cos = self._quat_family_cos(gripper_pose[:, :, 3:7])
        anchor_expanded = rotation_anchor[:, None, :].expand(self.num_envs, seeds, 4)
        drift_cos = (gripper_pose[:, :, 3:7] * anchor_expanded).sum(-1).abs()
        seed_valid = (
            torch.isfinite(joint_q).all(-1)
            & torch.isfinite(costs)
            & (position_error <= position_tolerance)
            & (margin >= IK_JOINT_MARGIN)
            & (quat_cos >= IK_LOADED_QUAT_MIN_COS)
            & ((drift_cos >= IK_LOADED_DRIFT_MIN_COS) | ~orientation_lock)
        )
        continuity = torch.linalg.vector_norm(arm_q - seed_joint_position[:, None, :5], dim=-1)
        if max_joint_delta is not None:
            seed_valid &= continuity <= max_joint_delta
        valid_cost = continuity.masked_fill(~seed_valid, torch.inf)
        valid = seed_valid.any(-1)
        fallback = torch.nan_to_num(costs, nan=torch.inf, posinf=torch.inf, neginf=torch.inf).argmin(-1)
        selected = torch.where(valid, valid_cost.argmin(-1), fallback)
        rows = torch.arange(self.num_envs, device=self.device)
        solved = joint_q[rows, selected][:, self.ik_coordinate_ids].clone()
        # 爪目标保持种子里的保持角（负载段不重写爪）
        solved[:, -1] = seed_joint_position[:, -1]
        solved[~valid] = seed_joint_position[~valid]
        self.candidate_ik_valid &= valid
        if self.cfg.diagnose:
            self.last_diagnostics.update(
                {
                    "loaded_ik_valid_rate": float(valid.float().mean()),
                    "loaded_position_err_mean_m": float(position_error.amin(-1).mean()),
                }
            )
        return solved

    def _solve_pocket_position(
        self,
        pocket_position_local: torch.Tensor,
        seed_joint_position: torch.Tensor,
        *,
        gripper_command: float,
    ) -> torch.Tensor:
        """解任意口袋位置目标（松爪撤离用；姿态钉住当前实测腕姿态）。"""
        import warp as wp
        from isaaclab.utils.math import quat_apply

        target_quaternion = t(self.robot.data.body_quat_w)[:, self.gripper_body_id].clone()
        self.ik_pose_objective.position_objective.set_target_positions(
            wp.from_torch(pocket_position_local.contiguous(), dtype=wp.vec3)
        )
        self.ik_pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(target_quaternion.contiguous(), dtype=wp.vec4)
        )
        seed = self.ik_joint_seed.clone()
        seed[:, self.ik_coordinate_ids] = seed_joint_position
        self.ik_solver.solve(wp.from_torch(seed.contiguous(), dtype=wp.float32))

        seeds = IK_SEEDS
        # 漂移门：负载链的旋转目标锚定在链起点的腕姿态（建立夹持时的姿态），
        # 每个解相对锚偏离 <= 12°。此前目标=「当前」姿态，逐航点累计漂移
        # 20-30° 把 shelf 翻掉（水平搬运终端爪力存活仅 4.7% 的根因）
        joint_q = wp.to_torch(self.ik_solver.joint_q).reshape(self.num_envs, seeds, -1)
        costs = wp.to_torch(self.ik_solver.costs).reshape(self.num_envs, seeds)
        body_pose = wp.to_torch(self.ik_solver.solver.body_q).reshape(self.num_envs, seeds, -1, 7)
        gripper_pose = body_pose[:, :, self.ik_gripper_body_id]
        tcp_offset = self.pocket_offset.expand(self.num_envs, seeds, 3)
        actual_pocket = gripper_pose[:, :, :3] + quat_apply(
            gripper_pose[:, :, 3:7].reshape(-1, 4), tcp_offset.reshape(-1, 3)
        ).reshape(self.num_envs, seeds, 3)
        position_error = torch.linalg.vector_norm(actual_pocket - pocket_position_local[:, None], dim=-1)
        arm_q = joint_q[:, :, self.ik_coordinate_ids[:5]]
        arm_limits = self.ik_joint_limits[:5]
        margin = torch.minimum(arm_q - arm_limits[:, 0], arm_limits[:, 1] - arm_q).amin(-1)
        seed_valid = (
            torch.isfinite(joint_q).all(-1)
            & torch.isfinite(costs)
            & (position_error <= 0.008)
            & (margin >= IK_JOINT_MARGIN)
        )
        continuity = torch.linalg.vector_norm(arm_q - seed_joint_position[:, None, :5], dim=-1)
        valid_cost = continuity.masked_fill(~seed_valid, torch.inf)
        valid = seed_valid.any(-1)
        fallback = torch.nan_to_num(costs, nan=torch.inf, posinf=torch.inf, neginf=torch.inf).argmin(-1)
        selected = torch.where(valid, valid_cost.argmin(-1), fallback)
        rows = torch.arange(self.num_envs, device=self.device)
        solved = joint_q[rows, selected][:, self.ik_coordinate_ids].clone()
        solved[:, -1] = gripper_command
        solved[~valid] = seed_joint_position[~valid]
        solved[~valid, -1] = gripper_command
        self.candidate_ik_valid &= valid
        return solved

    # ------------------------------------------------------------------ #
    # 有界笛卡尔反馈：把开爪口袋居中到橘子中心
    # ------------------------------------------------------------------ #
    def _center_open_pocket(self, target: torch.Tensor, corrections: int = 1) -> torch.Tensor:
        bias = torch.zeros((self.num_envs, 3), device=self.device)
        for _ in range(corrections):
            # 锚点 = 球心 + 抓取抬高（与就位 IK 同一目标，不是球心本身）
            anchor = self._orange_local()
            anchor[:, 2] += GRASP_Z_OFFSET_M
            live_pocket = self._pocket_local()
            error = live_pocket - anchor
            bias -= error
            bias_norm = torch.linalg.vector_norm(bias, dim=-1, keepdim=True)
            bias *= (0.06 / bias_norm.clamp_min(0.06)).clamp_max(1.0)
            virtual_pose = torch.zeros((self.num_envs, 7), device=self.device)
            virtual_pose[:, :3] = anchor + bias
            virtual_pose[:, 6] = 1.0
            target = self._solve_pocket_pose(virtual_pose, target)
            self._move_robot(target, max(1, TRANSITION_STEPS // 2))
            self._simulate(max(1, SETTLE_STEPS // 3), target)
        return target

    # ------------------------------------------------------------------ #
    # 接近链（教程 _approach_candidate 移植）
    # ------------------------------------------------------------------ #
    def _approach_candidate(self, stage: int | None):
        """开爪接近桌面橘子。stage 0=home、1=部分下降/闭合、None=完整就位待闭合。"""
        start = self.home_joint_position.clone()
        noise = torch.rand((self.num_envs, 6), device=self.device, generator=self.random) * 2.0 - 1.0
        noise *= self.cfg.home_joint_noise
        noise[:, self.gripper_id] = 0.0
        limits = t(self.robot.data.soft_joint_pos_limits)
        start = (start + noise).clamp(limits[..., 0], limits[..., 1])

        # home 先就位（橘子停在空中，避免 joint 扫动误碰）
        self._park_orange()
        self._write_robot(start)
        self._commit_reset_state()
        self._simulate(8, start)

        # 桌面橘子 + 纸盒刷位（难度覆盖训练分布），settle 后测真实静置位姿
        placements = self._sample_placements()
        orange_local, box_local = self._write_tabletop(placements)
        self._commit_reset_state()
        self._simulate(SETTLE_STEPS, start)
        local_pose = self._orange_pose_local()
        box_local = self._box_pose_local()
        self._box_start_xy = box_local[:, :2].clone()

        # home 口袋与橘子至少 6cm（防止出生即重叠）
        home_pocket_xy = self._pocket_local()[:, :2]
        clear = (local_pose[:, :2] - home_pocket_xy).norm(dim=-1) > 0.06
        self.candidate_ik_valid &= clear

        # 悬停（抓取点上方 +3cm）与就位两级 IK；悬停级随机化解分支保证行间多样性
        overhead = self._solve_pocket_pose(
            local_pose, start, vertical_offset=GRASP_Z_OFFSET_M + 0.03, randomize_solution=True
        )
        pregrasp = self._solve_pocket_pose(local_pose, overhead, vertical_offset=GRASP_Z_OFFSET_M)
        if self.cfg.diagnose:
            self.last_diagnostics.update(
                {
                    "overhead_ik_valid_rate": float(self.candidate_ik_valid.float().mean()),
                }
            )

        # 标定指令：到位后各 settle（橘子此时停空中，下降扫动不会碰它）
        self._park_orange()
        self._write_robot(overhead)
        self._simulate(SETTLE_STEPS, overhead)
        self._write_robot(pregrasp)
        self._simulate(SETTLE_STEPS, pregrasp)
        # 恢复桌面状态，从悬停位开始物理下降
        self._write_robot(overhead)
        self._simulate(SETTLE_STEPS, overhead)
        self._write_orange(local_pose)
        self._write_box(box_local)
        self._commit_reset_state()
        self._simulate(SETTLE_STEPS, overhead)
        initial_world_position = t(self.orange.data.root_pos_w).clone()

        if stage == 0:
            # HOME = 精确 home 出生（教程 phase 0：fraction 恒 0）
            self._write_robot(start)
            self._write_orange(local_pose)
            self._write_box(box_local)
            self._commit_reset_state()
            self._simulate(SETTLE_STEPS, start)
            self.candidate_difficulty.fill_(0.0)
            valid = self._valid(0, initial_world_position)
            return valid, start, local_pose, initial_world_position

        if stage == 1:
            # 物理下降（overhead -> pregrasp 关节插值）+ 部分闭合
            closure_fraction = torch.rand(self.num_envs, device=self.device, generator=self.random)
            for step in range(1, TRANSITION_STEPS + 1):
                self._simulate(1, torch.lerp(overhead, pregrasp, step / TRANSITION_STEPS))
            close_target = pregrasp.clone()
            close_target[:, -1] = torch.lerp(
                close_target[:, -1],
                torch.full_like(close_target[:, -1], self.gripper_closed_position),
                closure_fraction,
            )
            self._move_robot(close_target, GRASP_CLOSE_STEPS)
            self._simulate(SETTLE_STEPS, close_target)
            self.candidate_difficulty.copy_((1.0 + closure_fraction) / 7.0)
            valid = self._valid(1, initial_world_position)
            return valid, close_target, local_pose, initial_world_position

        # stage None：完整物理下降 + 居中反馈 -> 供闭合链使用
        for step in range(1, TRANSITION_STEPS + 1):
            self._simulate(1, torch.lerp(overhead, pregrasp, step / TRANSITION_STEPS))
        self._simulate(SETTLE_STEPS, pregrasp)
        target = self._center_open_pocket(pregrasp, corrections=2)
        live_error = (self._pocket_local() - self._orange_local()).norm(dim=-1)
        mask = self.candidate_ik_valid & torch.isfinite(live_error)
        if bool(mask.any()):
            self.last_diagnostics["pregrasp_live_pocket_err_mean_m"] = float(live_error[mask].mean())
        self.candidate_difficulty.fill_(2.0 / 7.0)
        return self.candidate_ik_valid, target, local_pose, initial_world_position

    # ------------------------------------------------------------------ #
    # 闭合 + proof lift（真实承重验证）
    # ------------------------------------------------------------------ #
    def _close_and_proof(self, target: torch.Tensor, initial_world_position: torch.Tensor):
        close_target = target.clone()
        close_target[:, -1] = self.gripper_closed_position
        self._move_robot(close_target, GRASP_CLOSE_STEPS)
        self._simulate(SETTLE_STEPS, close_target)
        # 保持指令回落到「挡停角 - 预压」：越过挡停角的指令在上提瞬间会
        # 扫过球面把球踢出（亚稳态，见 GRASP_HOLD_PRELOAD_RAD 注释）
        blocked = self.gripper_pos()
        limits = t(self.robot.data.soft_joint_pos_limits)
        hold_angle = (blocked - GRASP_HOLD_PRELOAD_RAD).clamp(
            limits[:, self.gripper_id, 0], limits[:, self.gripper_id, 1]
        )
        if self.cfg.diagnose:
            # 物理夹持签名：爪被挡（未到指令角）+ 球贴真实垫中点
            is_blocked = blocked > self.gripper_closed_position + 0.15
            engaged = is_blocked & (self._dist_pocket() < 0.035)
            self.last_diagnostics.update(
                {
                    "close_blocked_rate": float(is_blocked.float().mean()),
                    "close_engaged_rate": float(engaged.float().mean()),
                    "close_claw_mean_n": float(self.claw_force().mean()),
                    "close_centered_mean_m": float(self._dist_pocket().mean()),
                }
            )
        close_target = close_target.clone()
        close_target[:, -1] = hold_angle
        self._simulate(max(1, SETTLE_STEPS // 3), close_target)
        grasp_quat = t(self.robot.data.body_quat_w)[:, self.gripper_body_id].clone()
        # proof lift：单次 60 步快速上提 12mm——只有承重夹持能跟随。
        # 慢速多航点（3mm/120 步）实测让球有时间微滑脱（球在爪内被
        # 挤出），快提靠摩擦直接带走。
        desired = self._orange_pose_local()
        # 臂下沉补偿（2026-09-25 实测）：软关节 PD 在重力下指令 12mm 只
        # 实现 3-4mm（~75% 下沉）。指令放大到 3 倍 + 90 步行程，实测
        # 到位 ~12mm；验证门按实测球高 (>rest+6mm) 判，不依赖指令值。
        desired[:, 2] += PROOF_LIFT_M * 3.0
        # 容差 5mm：负载解贴近种子（15mm 松容差会放行把爪带偏/下压的解）
        proposed = self._solve_orange_pose(
            desired, close_target, position_tolerance=0.005, rotation_anchor=grasp_quat
        )
        target = torch.where(self.candidate_ik_valid.unsqueeze(-1), proposed, close_target)
        self._move_robot(target, 90)
        self._simulate(max(1, SETTLE_STEPS // 3), target)
        self.candidate_difficulty.copy_(3.0 / 7.0)
        if self.cfg.diagnose:
            from pick_anything.env_cfg import ORANGE_REST_Z

            follow = self._orange_local()[:, 2] > ORANGE_REST_Z + 0.006
            self.last_diagnostics.update(
                {
                    "claw_force_mean_n": float(self.claw_force().mean()),
                    "proof_centered_mean_m": float(self._dist_pocket().mean()),
                    "proof_orange_z_mean_m": float(self._orange_local()[:, 2].mean()),
                    "proof_follow_rate": float(follow.float().mean()),
                }
            )
        return target

    def _translate_grasped_orange(
        self,
        target: torch.Tensor,
        translation: torch.Tensor,
        progress: torch.Tensor,
        *,
        stop_if_separated: bool = False,
    ) -> torch.Tensor:
        """负载平移：每航点按实测变换重解 IK，橘子全程动力学。

        旋转目标锚定在入口腕姿态（= 建立夹持时的姿态），航点链不允许
        姿态累计漂移。
        """
        rotation_anchor = t(self.robot.data.body_quat_w)[:, self.gripper_body_id].clone()
        start_pose = self._orange_pose_local()
        connected_valid = self.candidate_ik_valid.clone()
        active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        settle = max(1, SETTLE_STEPS // 6)
        for waypoint in (0.25, 0.50, 0.75, 1.0):
            waypoint_progress = torch.minimum(progress, torch.full_like(progress, waypoint))
            desired_pose = start_pose.clone()
            desired_pose[:, :3] += translation * waypoint_progress.unsqueeze(-1)
            self.candidate_ik_valid.fill_(True)
            # 负载段位置容差 15mm（与 lift/transport 段一致；5mm 会让
            # 大多数负载 IK 解被判无效——夹着球后雅可比条件数变差）
            proposed = self._solve_orange_pose(
                desired_pose, target, position_tolerance=0.015, max_joint_delta=0.45,
                rotation_anchor=rotation_anchor,
            )
            waypoint_valid = connected_valid & (~active | self.candidate_ik_valid)
            update = waypoint_valid & active
            target = torch.where(update.unsqueeze(-1), proposed, target)
            self._move_robot(target, TRANSITION_STEPS)
            self._simulate(settle, target)
            if stop_if_separated:
                # 橘子离开爪口即停（防追着掉落的橘子满桌跑）
                active &= self._dist_pocket() < 0.05
        self.candidate_ik_valid.copy_(connected_valid)
        return target

    # ------------------------------------------------------------------ #
    # seed bank
    # ------------------------------------------------------------------ #
    def _append_seed_bank(self, name: str, valid: torch.Tensor, target: torch.Tensor) -> None:
        if not bool(valid.any()):
            return
        part = {
            "joint_position": t(self.robot.data.joint_pos)[valid].detach().clone(),
            "joint_target": target[valid].detach().clone(),
            "orange_pose": self._orange_pose_local()[valid].detach(),
            "box_pose": self._box_pose_local()[valid].detach(),
        }
        if name in self._seed_banks:
            part = {k: torch.cat((self._seed_banks[name][k], v), dim=0) for k, v in part.items()}
        maximum = self.cfg.batch_size * 4
        self._seed_banks[name] = {k: v[-maximum:] for k, v in part.items()}

    def _restore_seed_bank(self, name: str, *, settle_steps: int | None = None):
        bank = self._seed_banks[name]
        rows = torch.randint(
            bank["joint_position"].shape[0], (self.num_envs,), device=self.device, generator=self.random
        )
        joint_position = bank["joint_position"][rows]
        target = bank["joint_target"][rows].clone()
        self._write_robot(joint_position)
        self._write_orange(bank["orange_pose"][rows])
        self._write_box(bank["box_pose"][rows])
        self.robot.set_joint_position_target_index(target=target)
        self._commit_reset_state()
        if settle_steps is None:
            settle_steps = max(1, SETTLE_STEPS // 3)
        self._box_start_xy = self._box_pose_local()[:, :2].clone()
        self._simulate(settle_steps, target)
        initial_world_position = t(self.orange.data.root_pos_w).clone()
        return target, initial_world_position

    # ------------------------------------------------------------------ #
    # 连通链（教程 _connected_candidate 移植）
    # ------------------------------------------------------------------ #
    def _connected_candidate(self, stage: int, *, terminal_insert: bool = False):
        self.candidate_ik_valid.fill_(True)
        self.candidate_box_ruined.fill_(False)
        self._track_box = False

        seed_phase = 2
        target = None
        initial_world_position = None
        for bank_name, bank_phase in (("release", 6), ("transport", 5), ("stabilize", 4), ("lift", 3), ("grasp", 2)):
            if stage > bank_phase and bank_name in self._seed_banks:
                target, initial_world_position = self._restore_seed_bank(bank_name)
                seed_phase = bank_phase
                break
        else:
            _, target, _, initial_world_position = self._approach_candidate(None)
            target = self._close_and_proof(target, initial_world_position)

        # 负载链旋转锚：链起点（restore 或刚闭合完成）的腕姿态
        rotation_anchor = t(self.robot.data.body_quat_w)[:, self.gripper_body_id].clone()
        segment_fraction = torch.ones(self.num_envs, device=self.device)
        for next_phase in range(seed_phase + 1, min(stage, 6) + 1):
            if next_phase == stage and not (terminal_insert and stage == 6):
                segment_fraction = torch.empty(self.num_envs, device=self.device).uniform_(
                    0.02, 0.98, generator=self.random
                )
                terminal_rows = torch.rand(self.num_envs, device=self.device, generator=self.random) < 0.5
                terminal_rows[-1] = True
                segment_fraction[terminal_rows] = 1.0
            else:
                segment_fraction = torch.ones(self.num_envs, device=self.device)
            if next_phase >= 3:
                self._track_box = True

            if next_phase == 3:
                # 上举：先竖直半程、再全程（球对称，只约束位置）。指令按
                # 1.4 倍行程下发补偿软关节 PD 下沉（实测 ~40%），难度按
                # 实测高度计（clamp 封顶）
                start_pose = self._orange_pose_local()
                for waypoint in (0.5, 1.0):
                    wp_progress = torch.minimum(segment_fraction, torch.full_like(segment_fraction, waypoint))
                    raised = start_pose.clone()
                    raised[:, 2] = start_pose[:, 2] + LIFT_TRAVEL_M * 1.4 * wp_progress
                    target = self._solve_orange_pose(
                        raised, target, position_tolerance=0.015, rotation_anchor=rotation_anchor
                    )
                    self._move_robot(target, TRANSITION_STEPS)
                    self._simulate(max(1, SETTLE_STEPS // 6), target)
            elif next_phase == 4:
                # 保持：速度收敛（球对称，无重定向需求）
                self._simulate(SETTLE_STEPS, target)
            elif next_phase == 5:
                # 先竖直升到悬停高度，再水平搬到盒口
                start_pose = self._orange_pose_local()
                box_pos = self._box_pose_local()[:, :3]
                desired = start_pose.clone()
                desired[:, :2] = box_pos[:, :2]
                desired[:, 2] = box_pos[:, 2] + HOVER_ABOVE_BOX_M
                for waypoint in (0.5, 1.0):
                    raised = start_pose.clone()
                    raised[:, 2] = torch.lerp(start_pose[:, 2], desired[:, 2], waypoint)
                    target = self._solve_orange_pose(
                        raised, target, position_tolerance=0.015, rotation_anchor=rotation_anchor
                    )
                    self._move_robot(target, TRANSITION_STEPS)
                    self._simulate(max(1, SETTLE_STEPS // 3), target)
                    target = self._regrip(target)
                # 极坐标搬运：先绕基座转 yaw（对腕是精确族内运动，姿态
                # 零漂移），再沿半径推进。直角斜线会强制肘/腕重构 → 姿态
                # 漂移 → shelf 倾斜掉球（实测水平段终端爪力存活仅 3-5%）
                base_xy = start_pose.new_tensor(((0.35, -0.64),)).expand(self.num_envs, 2)
                v0 = start_pose[:, :2] - base_xy
                v1 = desired[:, :2] - base_xy
                r0, r1 = v0.norm(dim=-1), v1.norm(dim=-1)
                yaw0 = torch.atan2(v0[:, 0], v0[:, 1])
                yaw1 = torch.atan2(v1[:, 0], v1[:, 1])
                # 最短角路径
                dyaw = torch.remainder(yaw1 - yaw0 + math.pi, 2 * math.pi) - math.pi
                for waypoint in (0.25, 0.50, 0.75, 0.875, 1.0):
                    wp = torch.minimum(segment_fraction, torch.full_like(segment_fraction, waypoint))
                    yaw = yaw0 + dyaw * wp
                    radius = r0 + (r1 - r0) * wp
                    translated = desired.clone()
                    translated[:, 0] = base_xy[:, 0] + radius * torch.sin(yaw)
                    translated[:, 1] = base_xy[:, 1] + radius * torch.cos(yaw)
                    target = self._solve_orange_pose(
                        translated, target, position_tolerance=0.015,
                        rotation_anchor=rotation_anchor, orientation_lock=False,
                    )
                    # 减速搬运（加速度是夹持脱落主因）+ 航点后重夹紧
                    self._move_robot(target, 2 * TRANSITION_STEPS)
                    self._simulate(max(1, SETTLE_STEPS // 3), target)
                    target = self._regrip(target)
            elif next_phase == 6:
                # 降入盒内释放位（盒局部 z ~32mm，中心对准盒心）
                box_pos = self._box_pose_local()[:, :3]
                start_pose = self._orange_pose_local()
                desired = start_pose.clone()
                desired[:, :2] = box_pos[:, :2]
                desired[:, 2] = box_pos[:, 2] + PLACE_LOCAL_Z_M
                for waypoint in (0.25, 0.50, 0.75, 1.0):
                    wp = torch.minimum(segment_fraction, torch.full_like(segment_fraction, waypoint))
                    stepped = desired.clone()
                    stepped[:, :3] = torch.lerp(start_pose[:, :3], desired[:, :3], wp.unsqueeze(-1))
                    target = self._solve_orange_pose(
                        stepped, target, position_tolerance=0.015,
                        rotation_anchor=rotation_anchor, orientation_lock=False,
                    )
                    self._move_robot(target, 2 * TRANSITION_STEPS)
                    self._simulate(max(1, SETTLE_STEPS // 3), target)
                    target = self._regrip(target)

            if stage > next_phase:
                self.candidate_difficulty = (next_phase + segment_fraction) / 7.0
                valid_seed = self._valid(next_phase, initial_world_position)
                bank_name = {3: "lift", 4: "stabilize", 5: "transport"}.get(next_phase)
                if bank_name is not None:
                    self._append_seed_bank(bank_name, valid_seed, target)

        if stage == 7:
            segment_fraction = torch.empty(self.num_envs, device=self.device).uniform_(0.0, 1.0, generator=self.random)
            target = self._release_segment(target, segment_fraction)
        if stage == 3:
            # LIFT 段按实测高度标注进度（伺服指令分数不是物理进度）
            lift_progress = (
                (t(self.orange.data.root_pos_w)[:, 2] - initial_world_position[:, 2]) / LIFT_TRAVEL_M
            ).clamp(0.0, 1.0)
            self.candidate_difficulty.copy_((3.0 + lift_progress) / 7.0)
        else:
            self.candidate_difficulty.copy_((stage + segment_fraction).clamp_max(7.0) / 7.0)
        valid = self._valid(stage, initial_world_position)
        # 冷恢复证明（教程 restore proof 移植）：把候选状态原样写回
        # （零速度、冷接触——与训练 reset / 查看器完全同路径），保持后
        # 仍夹持才算数。生成期的热接触轨迹里边缘夹持能撑住，冷启动会破
        # （UI 巡览实测：部分夹持行恢复后橘子掉落）。
        if 2 <= stage <= 4:
            # 冷恢复证明只用于垂直夹持段（用户在 UI 看到的「举起后掉落」
            # 发生在这些段）。横搬/放置/释放段（5-7）冷证明×水平搬运对本
            # 筷式夹爪物理不可行（2026-09-25 实测 3 万次尝试 0 通过），
            # 回退热验证——这三段在 90 步查看器保持中全部成立
            restore_ok = self._restore_proof(target)
            valid &= restore_ok
            if self.cfg.diagnose:
                self.last_diagnostics["restore_hold_rate"] = float(restore_ok.float().mean())
        # 阶段自身的终端种子入库（教程 generator 末尾同款）：下游阶段从这些
        # 「走完本段」的物理根上分叉；3/4/5 段只收满进度行（难度 (k+1)/7）
        seed_name = {2: "grasp", 3: "lift", 4: "stabilize", 5: "transport"}.get(stage)
        if seed_name is not None:
            terminal = valid
            if stage in (3, 4, 5):
                # LIFT 难度按实测高度标注，臂下沉让满指令也只到 ~0.85 进度
                # ——rtol 4% 放行 ≥0.84 满进度的行作为下游种子
                terminal = valid & torch.isclose(
                    self.candidate_difficulty,
                    torch.full_like(self.candidate_difficulty, (stage + 1.0) / 7.0),
                    rtol=0.04,
                )
            self._append_seed_bank(seed_name, terminal, target)
        return valid, target

    def _regrip(self, target: torch.Tensor) -> torch.Tensor:
        """按当前实测挡停角重设保持指令并补压（真实控制器同款重夹紧）。

        负载搬运的加减速会让球在 V 口内微滑、爪角漂移；每个航点后重测
        挡停角再压预压，防止「指令角越过新挡停角」的扫掠踢球。

        ⚠ 勿把该预压搬进「存行时的 joint_target 改写」（2026-09-26 A/B
        否定）：球在 regrip 后继续向 V 口深处沉，bank 行存的残余压紧
        （挡停角-指令角）逐段衰减到 5/6 段 ≈0——看似 bug，实为平衡：球
        已靠 V 口几何 + 滚阻 0.50 自锁，不需要力捏（90 步冷恢复保持实
        测掉落仅 3 段合计 4/96：LIFT 0、PLACE 0、TRANSPORT 4/32）。
        把存行指令改写为 挡停角-0.16 后反而 22/96 掉——冷恢复时球还在
        记录的浅位、未沉到自锁深度，猛加压直接把球挤出指间（squeeze-
        out，掉进正下方盒里）。生成期 regrip 安全是因为接触已建立。
        """
        limits = t(self.robot.data.soft_joint_pos_limits)
        blocked = self.gripper_pos()
        target = target.clone()
        target[:, -1] = (blocked - GRASP_HOLD_PRELOAD_RAD).clamp(
            limits[:, self.gripper_id, 0], limits[:, self.gripper_id, 1]
        )
        self._simulate(max(1, SETTLE_STEPS // 3), target)
        return target

    def _restore_proof(self, target: torch.Tensor) -> torch.Tensor:
        """候选状态冷恢复 → 保持 150 步 → 橘子仍在爪里且没下坠。

        与 view_reset_bank / 训练 reset 同款写回路径（关节位 + 零速度 +
        物体位姿 + 零速度），证明「存进 bank 的那一行」恢复后自身稳定。
        """
        joint_position = t(self.robot.data.joint_pos).clone()
        orange_pose = self._orange_pose_local()
        box_pose = self._box_pose_local()
        stored_z = orange_pose[:, 2].clone()
        self._write_robot(joint_position)
        self._write_orange(orange_pose)
        self._write_box(box_pose)
        self._commit_reset_state()
        # 60 步窗口：覆盖「恢复后显性脱落」时窗（UI 巡览 30 步保持即可见）。
        # 150 步窗口对水平搬运物理不可行（2026-09-25 实测 0/15360——横搬
        # 后的夹持在冷启动 60-150 步间缓慢蠕出，此爪对球的静态保持极限）
        self._simulate(60, target)
        z_now = self._orange_local()[:, 2]
        rest_z = self._orange_rest_z()
        off_table = z_now > rest_z + 0.002
        not_sunk = z_now > stored_z - 0.008
        near_claw = self._dist_pocket() < 0.07
        return off_table & not_sunk & near_claw

    def _release_segment(self, target: torch.Tensor, segment_fraction: torch.Tensor) -> torch.Tensor:
        """盒内松爪 + 撤离：开度按分数插值，给橘子时间落底，再提爪撤离。"""
        release_target = target.clone()
        release_target[:, -1] = RELEASE_GRIPPER_POSITION
        opened = torch.lerp(target, release_target, segment_fraction.unsqueeze(-1))
        self._move_robot(opened, GRASP_CLOSE_STEPS)
        # 松爪后给橘子时间在真实盒内落底（教程 2×settle 同款）
        self._simulate(2 * SETTLE_STEPS, opened)
        # 撤离：口袋上提+后撤（保持当前腕姿态，避免扫到盒壁/橘子）
        retreat_target = self._solve_pocket_position(
            self._pocket_local() + opened.new_tensor(RELEASE_RETREAT_M).expand(self.num_envs, -1),
            opened,
            gripper_command=RELEASE_GRIPPER_POSITION,
        )
        self._move_robot(retreat_target, TRANSITION_STEPS)
        self._simulate(SETTLE_STEPS, retreat_target)
        return retreat_target

    # ------------------------------------------------------------------ #
    # 逐阶段物理验证（教程 _valid 移植）
    # ------------------------------------------------------------------ #
    def _valid(self, stage: int, initial_world_position: torch.Tensor) -> torch.Tensor:
        joint_position = t(self.robot.data.joint_pos)
        joint_velocity = t(self.robot.data.joint_vel)
        orange_position = t(self.orange.data.root_pos_w)
        orange_quaternion = t(self.orange.data.root_quat_w)
        orange_velocity = t(self.orange.data.root_lin_vel_w)
        finite = (
            torch.isfinite(joint_position).all(-1)
            & torch.isfinite(joint_velocity).all(-1)
            & torch.isfinite(orange_position).all(-1)
            & torch.isfinite(orange_quaternion).all(-1)
            # 阶段 <=2 需要几何精度（就位误差直接决定抓取质量）；>=3 段
            # 的有效性由物理快照 + 冷恢复证明背书——IK 中途失败的航点
            # 只是爪少动一段，若终态物理成立不应被拒（2026-09-25 实测
            # 累乘门 312 批 0 通过）
            & (self.candidate_ik_valid | (stage >= 3))
        )
        limits = t(self.robot.data.soft_joint_pos_limits)
        within_limits = ((joint_position >= limits[..., 0]) & (joint_position <= limits[..., 1])).all(-1)
        calm = joint_velocity.abs().amax(-1) < V_CALM_JOINT_VEL

        speed = torch.linalg.vector_norm(orange_velocity, dim=-1)
        angular_speed = torch.linalg.vector_norm(t(self.orange.data.root_ang_vel_w), dim=-1)
        local = orange_position - t(self.env.scene.env_origins)
        drift = torch.linalg.vector_norm(orange_position - initial_world_position, dim=-1)
        dist_pocket = self._dist_pocket()
        grip = self.gripper_pos()
        claw = self.claw_force()
        # 阶段 >=3（横搬/降入后）球在 V 口内弹性下垂 4-7cm——居中门放宽到
        # 0.075，承重判据交给 60 步冷恢复证明（横搬后 0.040 会拒掉全部
        # 证明幸存者，2026-09-25 实测 312 批 0 通过的根因之一）
        centered_max = GRASP_CENTERED_M if stage <= 2 else 0.075
        grasped_now = (
            (claw > GRASP_CLAW_FORCE_N)
            & (dist_pocket < centered_max)
            & (grip > GRIP_BAND[0])
            & (grip < GRIP_BAND[1])
        )
        box_ok = ~self.candidate_box_ruined
        rest_z = self._orange_rest_z()

        if self.cfg.diagnose:
            self.last_diagnostics.update(
                {
                    "valid_claw_rate": float((claw > GRASP_CLAW_FORCE_N).float().mean()),
                    "valid_centered_mean_m": float(dist_pocket.mean()),
                    "valid_grip_mean_rad": float(grip.mean()),
                    "valid_orange_z_mean_m": float(local[:, 2].mean()),
                    "valid_speed_mean_mps": float(speed.mean()),
                }
            )

        base = finite & within_limits & calm & box_ok
        if stage == 0:
            table_stable = (local[:, 2] > rest_z - V_TABLE_Z_BAND) & (local[:, 2] < rest_z + V_TABLE_Z_BAND + 0.002)
            table_stable &= (speed < 0.08) & (drift < V_TABLE_DRIFT_M)
            untouched = claw < GRASP_CLAW_FORCE_N  # home 段爪不应压橘子
            return base & table_stable & untouched
        if stage == 1:
            table_stable = (local[:, 2] > rest_z - V_TABLE_Z_BAND) & (local[:, 2] < rest_z + V_TABLE_Z_BAND + 0.002)
            table_stable &= (speed < 0.08) & (drift < V_TABLE_DRIFT_M)
            early_clasp = grasped_now & (drift < 0.025) & (local[:, 2] < rest_z + 0.03)
            early_clasp &= (speed < 0.08) & (angular_speed < V_ANG_SPEED)
            return base & ((table_stable | early_clasp) & (dist_pocket < 0.10))
        if stage == 2:
            # 承重证明 = 球被托离桌面 2mm（接触柔顺限制传程 ~5mm，+6mm 达不到）
            proof_lift = local[:, 2] > rest_z + 0.002
            # 闭合后动爪转动 ~1 rad，垫中点相对开爪 TCP 测量值会移 ~2cm，
            # 居中门放宽到 4.5cm（真正的承重判据是 proof_lift 球跟走）
            return (
                base
                & grasped_now
                & proof_lift
                & (drift < 0.025)
                & (angular_speed < V_ANG_SPEED)
                & (dist_pocket < 0.045)
            )
        if stage == 3:
            return base & grasped_now & (local[:, 2] > rest_z + 0.02) & (speed < V_SPEED_MPS)
        if stage == 4:
            return (
                base
                & grasped_now
                & (local[:, 2] > 0.095)
                & (speed < 0.15)
                & (angular_speed < V_ANG_SPEED)
            )
        if stage == 5:
            box_local = self._orange_local_in_box()
            centered = torch.linalg.vector_norm(box_local[:, :2], dim=-1) < 0.045
            return base & grasped_now & centered & (box_local[:, 2] > 0.06) & (speed < V_SPEED_MPS)
        if stage == 6:
            box_local = self._orange_local_in_box()
            inside = (box_local[:, 0].abs() < 0.045) & (box_local[:, 1].abs() < 0.045)
            height = (box_local[:, 2] > 0.020) & (box_local[:, 2] < 0.055)
            return base & grasped_now & inside & height & (speed < V_SPEED_MPS)
        # stage 7：盒内静置 + 爪已撤离
        box_local = self._orange_local_in_box()
        inside = (box_local[:, 0].abs() < 0.052) & (box_local[:, 1].abs() < 0.052)
        height = (box_local[:, 2] > 0.008) & (box_local[:, 2] < 0.052)
        released = (grip > 1.2) | (self.orange_force() < 0.1)
        if self.cfg.diagnose:
            self.last_diagnostics.update(
                {
                    "rel_inside_rate": float(inside.float().mean()),
                    "rel_height_rate": float(height.float().mean()),
                    "rel_released_rate": float(released.float().mean()),
                    "rel_far_rate": float((dist_pocket > 0.055).float().mean()),
                    "rel_base_rate": float(base.float().mean()),
                    "rel_box_ruined_rate": float(self.candidate_box_ruined.float().mean()),
                }
            )
        return base & inside & height & (speed < 0.08) & (dist_pocket > 0.055) & released

    def _orange_rest_z(self) -> float:
        from pick_anything.env_cfg import ORANGE_REST_Z

        return ORANGE_REST_Z

    # ------------------------------------------------------------------ #
    # 刷位采样（与训练分布同源）
    # ------------------------------------------------------------------ #
    def _sample_placements(self):
        from pick_anything.tasks.pick_apple import mdp as v0

        lo, hi = self.cfg.spawn_difficulty_range
        difficulty = float(torch.empty(1, device=self.device).uniform_(lo, hi, generator=self.random).item())
        return v0.sample_placements(self.num_envs, difficulty, self.device)

    def _write_tabletop(self, placements):
        """把橘子/纸盒写到桌面刷位（返回写入的 env 局部系位姿）。"""
        from pick_anything.env_cfg import BOX_SPAWN_Z, FRUIT_SPAWN_Z

        zero = torch.zeros(self.num_envs, device=self.device)
        origins = t(self.env.scene.env_origins)

        def _pose(xy, z, yaw):
            quat = torch.stack([zero, zero, torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)
            pos = torch.stack([xy[:, 0], xy[:, 1], torch.full_like(zero, z)], dim=-1)
            return torch.cat([pos, quat], dim=-1)

        yaw_orange = torch.rand(self.num_envs, device=self.device, generator=self.random) * (2.0 * math.pi)
        orange_pose = _pose(placements["orange"], FRUIT_SPAWN_Z, yaw_orange)
        self.orange.write_root_pose_to_sim_index(root_pose=orange_pose + self._origin_pad())
        self.orange.write_root_velocity_to_sim_index(root_velocity=self._zeros)

        yaw_box = (2 * torch.rand(self.num_envs, device=self.device, generator=self.random) - 1) * 0.3
        box_pose = _pose(placements["box"], BOX_SPAWN_Z, yaw_box)
        self.box.write_root_pose_to_sim_index(root_pose=box_pose + self._origin_pad())
        self.box.write_root_velocity_to_sim_index(root_velocity=self._zeros)
        return orange_pose, box_pose

    def _origin_pad(self) -> torch.Tensor:
        """(n, 7) 的 [origin | 0,0,0,0] 垫片：局部位姿 + 垫片 = 世界位姿。"""
        origins = t(self.env.scene.env_origins)
        pad = torch.zeros((self.num_envs, 7), device=self.device)
        pad[:, :3] = origins
        return pad

    # ------------------------------------------------------------------ #
    # 主循环：逐阶段拒绝采样（教程 generate 移植）
    # ------------------------------------------------------------------ #
    def generate(self) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        rows: dict[str, list[torch.Tensor]] = {
            name: []
            for name in ("joint_position", "joint_target", "orange_pose", "box_pose", "stage", "difficulty", "grasped")
        }
        rejections = {name: 0 for name in STAGE_LIST}
        for stage, stage_name in enumerate(STAGE_LIST):
            required_seed = {2: "grasp", 3: "lift", 4: "stabilize", 5: "transport"}.get(stage)

            def branch_ready(name: str | None = required_seed) -> bool:
                return name is None or (
                    name in self._seed_banks
                    and self._seed_banks[name]["joint_position"].shape[0] >= self.cfg.branch_seed_count
                )

            accepted = 0
            attempts = 0
            while (
                accepted < self.cfg.poses_per_stage or not branch_ready()
            ) and attempts < self.cfg.max_attempts_per_stage:
                self.last_diagnostics = {}
                self.candidate_ik_valid.fill_(True)
                self.candidate_box_ruined.fill_(False)
                self._track_box = False

                if stage <= 1:
                    valid, target = self._candidate_approach(stage)
                else:
                    if stage == 7 and "release" not in self._seed_banks:
                        # 松爪链必须从「已验证的盒内夹持」出发（教程同款）
                        valid, target = self._connected_candidate(6, terminal_insert=True)
                        self._append_seed_bank("release", valid, target)
                        attempts += self.num_envs
                        rejections[stage_name] += self.num_envs - int(valid.sum().item())
                        if not bool(valid.any()):
                            continue
                        self.candidate_ik_valid.fill_(True)
                        self.candidate_box_ruined.fill_(False)
                        self._track_box = False
                    valid, target = self._connected_candidate(stage)

                attempts += self.num_envs
                ids = valid.nonzero(as_tuple=False).squeeze(-1)
                rejections[stage_name] += self.num_envs - ids.numel()
                take = ids[: self.cfg.poses_per_stage - accepted]
                if take.numel() == 0:
                    if self.cfg.diagnose and attempts % (2 * self.num_envs) == 0:
                        print(
                            f"[INFO] {stage_name}: accepted={accepted} attempted={attempts} diag={self.last_diagnostics}",
                            flush=True,
                        )
                    continue
                count = take.numel()
                rows["joint_position"].append(t(self.robot.data.joint_pos)[take].detach().clone())
                rows["joint_target"].append(target[take].detach().clone())
                rows["orange_pose"].append(self._orange_pose_local()[take].detach())
                rows["box_pose"].append(self._box_pose_local()[take].detach())
                rows["stage"].append(torch.full((count,), stage, device=self.device, dtype=torch.long))
                rows["difficulty"].append(self.candidate_difficulty[take].detach().clone())
                live_grasped = self._live_grasped()[take]
                represented = live_grasped | torch.full((count,), stage >= 3, dtype=torch.bool, device=self.device)
                rows["grasped"].append(represented.detach().clone())
                accepted += count
                if self.cfg.diagnose:
                    print(
                        f"[INFO] {stage_name}: +{count} (accepted {accepted}/{self.cfg.poses_per_stage}, "
                        f"attempted {attempts}) diag={ {k: round(v, 4) for k, v in self.last_diagnostics.items()} }",
                        flush=True,
                    )
            seed_missing = not branch_ready()
            if accepted != self.cfg.poses_per_stage or seed_missing:
                self.partial_states = {
                    name: torch.cat(parts, dim=0).cpu() for name, parts in rows.items() if parts
                }
                seed_count = (
                    self._seed_banks[required_seed]["joint_position"].shape[0]
                    if required_seed in self._seed_banks
                    else 0
                )
                raise RuntimeError(
                    f"阶段 {stage_name!r} 只生成 {accepted}/{self.cfg.poses_per_stage} 个有效样本"
                    f"（尝试 {attempts} 批量环境步；{required_seed!r} 种子库 {seed_count} 行"
                    f"{'不足' if seed_missing else ''}）。最后诊断：{self.last_diagnostics}"
                )
            print(
                f"[INFO] {stage_name}: accepted={accepted}, rejected={rejections[stage_name]}, "
                f"attempted={attempts}, seed_banks={ {k: v['joint_position'].shape[0] for k, v in self._seed_banks.items()} }",
                flush=True,
            )
        states = {name: torch.cat(parts, dim=0).cpu() for name, parts in rows.items()}
        return states, rejections

    def _live_grasped(self) -> torch.Tensor:
        return (
            (self.claw_force() > GRASP_CLAW_FORCE_N)
            & (self._dist_pocket() < GRASP_CENTERED_M)
            & (self.gripper_pos() > GRIP_BAND[0])
            & (self.gripper_pos() < GRIP_BAND[1])
        )

    def _candidate_approach(self, stage: int):
        valid, target, _, _ = self._approach_candidate(stage)
        return valid, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the physics-validated orange reset bank (tutorial-style).")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--num_envs", type=int, default=64, help="并行环境数（= 每批量候选数）")
    parser.add_argument("--poses_per_stage", type=int, default=32, help="每阶段入库样本数")
    parser.add_argument("--output", type=str, default="reset_states/pick_orange_bank_v1.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_attempts", type=int, default=8192, help="每阶段最大尝试（环境步计）")
    parser.add_argument("--branch_seed_count", type=int, default=16, help="下游阶段可开跑所需的最小种子库行数")
    parser.add_argument("--diagnose", action="store_true", help="打印各段链诊断（IK/验证门通过率）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

    from pick_anything.tasks.pick_apple.reset_bank import save_reset_dataset  # noqa: E402
    from pick_anything.tasks.pick_apple.staged_env_cfg import StagedPickGeneratorEnvCfg  # noqa: E402
    from pick_anything.tasks.pick_apple.staged_mdp import silence_mjwarp_overflow_warning  # noqa: E402

    torch.manual_seed(args.seed)
    cfg = StagedPickGeneratorEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg)
    try:
        silence_mjwarp_overflow_warning(env)
        gen_cfg = GeneratorCfg(
            batch_size=args.num_envs,
            poses_per_stage=args.poses_per_stage,
            seed=args.seed,
            max_attempts_per_stage=args.max_attempts,
            branch_seed_count=args.branch_seed_count,
            diagnose=args.diagnose,
        )
        generator = _Generator(env, gen_cfg)
        try:
            states, rejections = generator.generate()
        except RuntimeError:
            # 阶段失败也落盘已完成阶段（部分 bank 好过全丢），再原样抛出
            if generator.partial_states:
                part = PROJECT_ROOT / (args.output + ".partial.pt")
                save_reset_dataset(
                    part, generator.partial_states,
                    generator={"pipeline": "partial", "seed": args.seed},
                    validation={"note": "stage failed mid-generation"},
                )
                print(f"[WARN] 阶段失败，已保存部分 bank: {part}", flush=True)
            raise
    finally:
        env.close()

    output = PROJECT_ROOT / args.output
    artifact = save_reset_dataset(
        output,
        states,
        generator={
            "pipeline": "isaaclabtutorial-style: newton-ik + physical close + proof lift + seed banks",
            "num_envs": args.num_envs,
            "poses_per_stage": args.poses_per_stage,
            "seed": args.seed,
        },
        validation={"physics": "newton_mjwarp", "rejections": rejections},
    )
    print(f"reset bank 已保存: {output}  rows={artifact['row_count']}")
    simulation_app.close()


if __name__ == "__main__":
    main()
