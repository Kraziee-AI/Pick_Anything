"""八阶段（staged）抓橘子入盒任务的 MDP 项：观测、奖励、终止、事件、课程。

与 v0 的 ``mdp.py``（三级物理阶段 + 进度奖励）并存：本模块实现任务书要求
的八阶段显式状态机版本，注册为独立 gym id ``Pick-Apple-SO101-Staged-v0``
（历史名保留），不改动 v0 任务的行为。

2026-09-25：场景只保留橘子（用户定版：去掉苹果/红色立方体替身）——原
「苹果=目标、橘子=干扰物」的双果结构改为「橘子=唯一目标」，误抓橘子惩罚
（wrong_object）随干扰物一并删除。

核心对象 ``StagedPickState``：
* 几何缓存沿用 v0 ``PickState`` 的模式（env 局部系、每控制步幂等刷新）；
* 阶段推进由 :class:`StageProgress`（stages.py）驱动，条件全部来自物理
  状态（接触力 / 位置 / 相对速度），阶段编号只作锁存记录；
* 出生阶段来自 reset 状态库行（``StagedReset``）或 home 随机刷位，出生
  即有的进度不发奖（``paid`` 掩码）。

重要实测约束（2026-09-23，见 memory pick-apple-gripper-platform-bug）：
* PhysX 平台 bug：``write_joint_position`` 之后 gripper 关节的隐式驱动目标
  永久失效——staged 任务已整体切 Newton 后端（make_newton_physics），
  write 后 set 精确执行、无间歇失效。
* 爪角方向：0 rad=全闭、1.745 rad=全开（USD 实测；v0 旧注释相反）。
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from pick_anything.tasks.pick_apple import mdp as v0
from pick_anything.tasks.pick_apple.reset_bank import MIN_SAMPLES_PER_STAGE, load_reset_dataset
from pick_anything.tasks.pick_apple.stages import (
    GRASP_CONFIRM_STEPS,
    GRASP_MAX_GRIPPER_RAD,
    GRASP_MIN_GRIPPER_RAD,
    GRASP_PROXIMITY_M,
    LIFT_CONFIRM_STEPS,
    LIFT_Z_M,
    NUM_STAGES,
    PLACE_APPROACH_MIN_Z_M,
    PLACE_INSIDE_XY_M,
    PLACE_INSIDE_Z_MAX_M,
    PLACE_INSIDE_Z_MIN_M,
    PLACE_JAW_NEAR_M,
    PLACE_OVER_BOX_XY_M,
    PREGRASP_POCKET_M,
    RELEASE_CLEARANCE_M,
    RELEASE_SETTLED_SPEED_MPS,
    RETREAT_REWARD_TARGET_M,
    STABILIZE_CONFIRM_STEPS,
    STABILIZE_MIN_Z_M,
    STABILIZE_REL_VEL_MPS,
    STAGE_NAMES,
    TRANSPORT_MIN_Z_M,
    TRANSPORT_REL_VEL_MPS,
    PickStage,
    StageProgress,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 爪角语义（USD 实测）：0=全闭 .. 1.745=全开。开度分数 = 爪角/行程。
GRIPPER_TRAVEL_RAD = 1.745
# 夹持力载体 body（2026-09-24 实测：两根指的碰撞体都挂在 gripper body 下，
# jaw body 受力恒 0；夹持反作用力读数在 gripper 列 3-5N）
CLAW_BODY_NAMES = ("gripper", "moving_jaw_so101_v1")
# 夹持力门（claw body 受力）：高于桌蹭基线（~2N），低于实测夹持（3-5N）；
# 橘子净力对称抵消（≈mg 恒定）不可作夹持判据
GRASP_CLAW_FORCE_N = 2.2  # 2.5→2.2：bank 恢复实测 claw 在 2.4-3.2N 波动，边缘抖动会断 3 步确认
# HOME 阶段粗接近门：口袋到橘子距离（进入预抓取区域即完成 HOME）
HOME_APPROACH_M = 0.12
# 悬停点（盒口上方，env 局部系）
HOVER_ABOVE_BOX_M = 0.14
# PREGRASP 就位需要的最小开度（接近全开；爪角 1.745=全开）
PREGRASP_MIN_GRIPPER_RAD = 1.30
# lift_rise 高度势能饱和线 = LIFT 完成线 + 此裕度（2026-09-27：抬过完成
# 线后继续上升零支付，防无界刷高；裕度覆盖 LIFT 4 步确认期间的继续上抬）
LIFT_RISE_CAP_MARGIN_M = 0.02

# --- 课程：出生阶段采样权重的三档表（按累计 reset 轮数线性过渡）---
# early: home 为主 + 中段辅助（PPO 先学接近与就位）
CURRICULUM_EARLY = (0.40, 0.20, 0.15, 0.08, 0.04, 0.04, 0.05, 0.04)
# mid: 覆盖抓取-抬起-搬运（中段密集练习）
CURRICULUM_MID = (0.20, 0.12, 0.12, 0.12, 0.10, 0.10, 0.12, 0.12)
# late: home 为主（收敛完整任务），后期阶段保底非零
CURRICULUM_LATE = (0.50, 0.16, 0.10, 0.06, 0.04, 0.04, 0.06, 0.04)
CURRICULUM_MID_RESETS = 20_000  # 累计 reset 次数达到该值时进入 mid
CURRICULUM_LATE_RESETS = 120_000


def _t(buf):
    return buf.torch if hasattr(buf, "torch") else buf


class StagedPickState:
    """八阶段共享状态：几何缓存 + 阶段条件 + 出生/成功统计。"""

    def __init__(self, env: ManagerBasedRLEnv):
        self.device = env.device
        n = env.num_envs
        robot = env.scene["robot"]
        # isaaclab_assets SO101：动指 body 名是 moving_jaw_so101_v1（旧
        # leisaac 叫 jaw）；body id 与 joint id 分开存（曾被混用）
        self.jaw_id = int(robot.find_bodies(["moving_jaw_so101_v1"])[0][0])
        self.gripper_body_id = int(robot.find_bodies(["gripper"])[0][0])
        self.gripper_id = int(robot.find_joints(["gripper"])[0][0])

        self.progress = StageProgress(n, self.device)
        self._last_step = -1

        # 几何缓存（env 局部系）
        self.jaw_pos = torch.zeros(n, 3, device=self.device)
        self.orange_pos = torch.zeros(n, 3, device=self.device)
        self.orange_vel = torch.zeros(n, 3, device=self.device)
        self.jaw_vel = torch.zeros(n, 3, device=self.device)
        self.box_pos = torch.zeros(n, 3, device=self.device)
        self.box_up_z = torch.ones(n, device=self.device)
        self.orange_local = torch.zeros(n, 3, device=self.device)
        self.pocket_pos = torch.zeros(n, 3, device=self.device)
        self.hover_pos = torch.zeros(n, 3, device=self.device)

        # 接触与夹爪
        self.orange_force = torch.zeros(n, device=self.device)
        self.gripper_pos = torch.zeros(n, device=self.device)
        self.arm_force = torch.zeros(n, device=self.device)
        self.ee_force = torch.zeros(n, device=self.device)
        self.claw_force = torch.zeros(n, device=self.device)
        self._contact_cols = None

        # 扰动参考（spawn 位）
        self.spawn_box_xy = torch.zeros(n, 2, device=self.device)
        self.spawn_orange_xy = torch.zeros(n, 2, device=self.device)
        self.box_disp = torch.zeros(n, device=self.device)
        self.orange_disp = torch.zeros(n, device=self.device)

        # 出生/成功统计（用户核心指标：按出生阶段分别统计成功率）
        self.birth_stage = torch.zeros(n, dtype=torch.long, device=self.device)
        self.max_stage = torch.zeros(n, dtype=torch.long, device=self.device)
        self._birth_stats = torch.zeros(NUM_STAGES, 2, device=self.device)  # [tries, wins]
        self._stage_reach = torch.zeros(NUM_STAGES, device=self.device)
        self.reset_count = 0
        self.difficulty = 0.6
        # RELEASE 成功门：训练/部署统一 0.055（2026-09-27 用户定版双轨口径
        # 之一；撤退奖励的饱和线另挂 RETREAT_REWARD_TARGET_M=0.07，见
        # reward_release_retreat——奖励推均值到 7cm、成功判 5.5cm，1.5cm
        # 差值吸收 σ0.15 噪声）
        self.release_clearance_m = RELEASE_CLEARANCE_M

    # ------------------------------------------------------------------ #
    @classmethod
    def get(cls, env: ManagerBasedRLEnv) -> StagedPickState:
        state = getattr(env, "staged_pick_state", None)
        if state is None:
            state = cls(env)
            env.staged_pick_state = state
        return state

    def _resolve_contact_cols(self, env: ManagerBasedRLEnv):
        if self._contact_cols is None:
            sensor = env.scene.sensors["robot_contact"]
            names = [str(name) for name in sensor.body_names]
            arm_cols = [i for i, name in enumerate(names) if name in v0.ARM_LINK_NAMES]
            ee_cols = [i for i, name in enumerate(names) if name in v0.EE_BODY_NAMES]
            claw_cols = [i for i, name in enumerate(names) if name in CLAW_BODY_NAMES]
            if not arm_cols or not claw_cols:
                raise RuntimeError(f"robot_contact bodies {names} lack {v0.ARM_LINK_NAMES}/{CLAW_BODY_NAMES}")
            self._contact_cols = (arm_cols, ee_cols, claw_cols)
        return self._contact_cols

    def update(self, env: ManagerBasedRLEnv) -> None:
        """每控制步幂等刷新：几何 -> 阶段条件 -> StageProgress 推进。"""
        if self._last_step == env.common_step_counter:
            return
        self._last_step = env.common_step_counter

        origins = _t(env.scene.env_origins)
        robot = env.scene["robot"]
        orange = env.scene["orange"]
        box = env.scene["cardboard_box"]

        self.jaw_pos = _t(robot.data.body_pos_w)[:, self.jaw_id] - origins
        self.jaw_vel = _t(robot.data.body_lin_vel_w)[:, self.jaw_id]
        self.orange_pos = _t(orange.data.root_pos_w) - origins
        self.orange_vel = _t(orange.data.root_lin_vel_w)
        box_pos_w = _t(box.data.root_pos_w)
        self.box_pos = box_pos_w - origins
        box_quat = _t(box.data.root_quat_w)
        e_z = torch.zeros_like(self.box_pos)
        e_z[:, 2] = 1.0
        self.box_up_z = quat_apply_inverse(box_quat, e_z)[:, 2]
        self.orange_local = quat_apply_inverse(box_quat, self.orange_pos - self.box_pos)
        self.hover_pos = self.box_pos.clone()
        self.hover_pos[:, 2] += HOVER_ABOVE_BOX_M

        # 接触力（net = 法向净力；对称夹持会抵消，只做辅助信号）
        orange_sensor = env.scene.sensors["orange_contact"]
        self.orange_force = _t(orange_sensor.data.net_forces_w).reshape(env.num_envs, -1, 3).norm(dim=-1).squeeze(-1)

        self.gripper_pos = _t(robot.data.joint_pos)[:, self.gripper_id]
        robot_sensor = env.scene.sensors["robot_contact"]
        history = robot_sensor.data.net_forces_w_history
        frames = _t(history) if history is not None else _t(robot_sensor.data.net_forces_w).unsqueeze(1)
        per_body = frames.norm(dim=-1).amax(dim=1)
        arm_cols, ee_cols, claw_cols = self._resolve_contact_cols(env)
        self.arm_force = per_body[:, arm_cols].amax(dim=-1)
        self.ee_force = per_body[:, ee_cols].amax(dim=-1)
        # 爪部受力（gripper+jaw）：真实夹持力的载体（jaw 单列恒 0）
        self.claw_force = per_body[:, claw_cols].amax(dim=-1)

        # 捕获口袋（gripper body 系固定偏移，指尖 pinch 点）
        gripper_pos = _t(robot.data.body_pos_w)[:, self.gripper_body_id] - origins
        gripper_quat = _t(robot.data.body_quat_w)[:, self.gripper_body_id]
        offset = torch.tensor(v0.POCKET_OFFSET_B, device=self.device).expand(env.num_envs, 3)
        self.pocket_pos = gripper_pos + quat_apply(gripper_quat, offset)

        # ---- 八阶段物理条件（全部由真实物理量决定）----
        dist_pocket = (self.orange_pos - self.pocket_pos).norm(dim=-1)
        rel_vel = (self.orange_vel - self.jaw_vel).norm(dim=-1)
        orange_low = self.orange_pos[:, 2] < v0.GRASP_LIFT_Z
        # 真实夹持：爪部实压 + 橘子在爪口 + 有效夹持角带。
        # 近距门 dist_pocket（jaw body 离口袋几何上就有 ~9-12cm）；力门
        # claw 受力（橘子净力对称夹持时抵消，恒 ≈mg）。
        grasped_now = (
            (self.claw_force > GRASP_CLAW_FORCE_N)
            & (self.orange_force > 0.15)
            & (dist_pocket < GRASP_PROXIMITY_M)
            & (self.gripper_pos > GRASP_MIN_GRIPPER_RAD)
            & (self.gripper_pos < GRASP_MAX_GRIPPER_RAD)
        )
        cond = {
            "HOME": (dist_pocket < HOME_APPROACH_M) & orange_low & (~self.progress.grasped),
            "PREGRASP": (dist_pocket < PREGRASP_POCKET_M)
            & (self.gripper_pos > PREGRASP_MIN_GRIPPER_RAD)
            & orange_low
            & (~self.progress.grasped),
            "GRASP": grasped_now,
            "LIFT": (self.orange_pos[:, 2] > LIFT_Z_M)
            & (dist_pocket < GRASP_PROXIMITY_M + 0.01)
            & (self.orange_force > 0.2),
            "STABILIZE": (self.orange_pos[:, 2] > STABILIZE_MIN_Z_M)
            & (rel_vel < STABILIZE_REL_VEL_MPS)
            & grasped_now,
            "TRANSPORT": (self.orange_pos[:, 2] > TRANSPORT_MIN_Z_M)
            & (rel_vel < TRANSPORT_REL_VEL_MPS)
            & (self.orange_local[:, :2].norm(dim=-1) < PLACE_OVER_BOX_XY_M)
            & (self.orange_local[:, 2] > PLACE_APPROACH_MIN_Z_M)
            & grasped_now,
            "PLACE": (self.orange_local[:, 0].abs() < PLACE_INSIDE_XY_M)
            & (self.orange_local[:, 1].abs() < PLACE_INSIDE_XY_M)
            & (self.orange_local[:, 2] > PLACE_INSIDE_Z_MIN_M)
            & (self.orange_local[:, 2] < PLACE_INSIDE_Z_MAX_M)
            & (dist_pocket < PLACE_JAW_NEAR_M),
            "RELEASE": (self.orange_local[:, 0].abs() < PLACE_INSIDE_XY_M)
            & (self.orange_local[:, 1].abs() < PLACE_INSIDE_XY_M)
            & (self.orange_local[:, 2] > PLACE_INSIDE_Z_MIN_M)
            & (self.orange_local[:, 2] < PLACE_INSIDE_Z_MAX_M)
            & (dist_pocket > self.release_clearance_m)
            & (self.orange_vel.norm(dim=-1) < RELEASE_SETTLED_SPEED_MPS),
        }
        confirm = {
            "HOME": 1,
            "PREGRASP": 1,
            "GRASP": GRASP_CONFIRM_STEPS,
            "LIFT": LIFT_CONFIRM_STEPS,
            "STABILIZE": STABILIZE_CONFIRM_STEPS,
            "TRANSPORT": 1,
            "PLACE": 1,
        }
        self.progress.update(env.episode_length_buf, cond, confirm)
        self.max_stage = torch.maximum(self.max_stage, self.progress.stage)

        # 扰动
        self.box_disp = (self.box_pos[:, :2] - self.spawn_box_xy).norm(dim=-1)
        self.orange_disp = (self.orange_pos[:, :2] - self.spawn_orange_xy).norm(dim=-1)

    # ------------------------------------------------------------------ #
    def on_env_reset(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
        """episode 结束时折叠出生-成功统计，然后清 per-episode 状态。"""
        if env_ids.numel() > 0:
            for k in range(NUM_STAGES):
                born = env_ids[self.birth_stage[env_ids] == k]
                if born.numel():
                    self._birth_stats[k, 0] += born.numel()
                    self._birth_stats[k, 1] += int(self.progress.succeeded_episode[born].sum())
            self._stage_reach += torch.bincount(
                self.max_stage[env_ids].clamp(max=NUM_STAGES - 1), minlength=NUM_STAGES
            ).float()
            self.reset_count += int(env_ids.numel())
        self.progress.reset(env_ids)

    def log_metrics(self, env: ManagerBasedRLEnv) -> None:
        """把分出生阶段的成功率写入训练日志（用户核心指标）。"""
        log = env.extras.setdefault("log", {})
        tries = self._birth_stats[:, 0].clamp_min(1.0)
        rates = self._birth_stats[:, 1] / tries
        for k, name in enumerate(STAGE_NAMES):
            log[f"Staged/birth_{name}_success"] = rates[k]
            log[f"Staged/birth_{name}_count"] = self._birth_stats[k, 0]
        log["Staged/home_born_success"] = rates[0]  # 完整任务成功率代理指标
        reach_total = self._stage_reach.sum().clamp_min(1.0)
        for k, name in enumerate(STAGE_NAMES):
            log[f"Staged/reach_{name}"] = self._stage_reach[k] / reach_total
        log["Staged/mean_stage"] = self.progress.stage.float().mean()


# ---------------------------------------------------------------------- #
# Reset 事件：阶段课程采样 + 状态库恢复 / home 随机刷位
# ---------------------------------------------------------------------- #


def curriculum_weights(reset_count: int) -> torch.Tensor:
    """三档出生权重按累计 reset 次数线性插值（home 全程非零）。"""
    early = torch.tensor(CURRICULUM_EARLY, dtype=torch.float32)
    mid = torch.tensor(CURRICULUM_MID, dtype=torch.float32)
    late = torch.tensor(CURRICULUM_LATE, dtype=torch.float32)
    if reset_count >= CURRICULUM_LATE_RESETS:
        return late
    if reset_count >= CURRICULUM_MID_RESETS:
        t = (reset_count - CURRICULUM_MID_RESETS) / (CURRICULUM_LATE_RESETS - CURRICULUM_MID_RESETS)
        return (1 - t) * mid + t * late
    t = min(reset_count / CURRICULUM_MID_RESETS, 1.0)
    return (1 - t) * early + t * mid


def silence_mjwarp_overflow_warning(env: ManagerBasedRLEnv) -> None:
    """关闭 mujoco_warp 求解器内核的 overflow printf（linesearch/contact 溢出刷屏）。

    mujoco_warp 在 ``io.put_model`` 中硬编码 ``opt.warn_overflow = True``，
    且 isaaclab_newton 未暴露该开关；内核在每次 launch 时读取
    ``bool(m.opt.warn_overflow)``，因此运行期翻转即可生效。overflow 计数
    （atomic_or 到 overflow_out）照常记录，仅不再打印。每次 reset 幂等
    重断言，对求解器重建（NewtonManager._solver 置 None 后重建）免疫。
    """
    manager = getattr(env.sim, "physics_manager", None)
    solver = getattr(manager, "_solver", None)
    opt = getattr(getattr(solver, "mjw_model", None), "opt", None)
    if opt is not None and hasattr(opt, "warn_overflow"):
        opt.warn_overflow = False


class StagedReset(ManagerTermBase):
    """八阶段出生：按课程从 reset 状态库采样，缺失阶段报警并安全回退。

    * 阶段 0（HOME）：v0 同款随机刷位（难度课程保留）。
    * 阶段 >=1：从 reset 状态库恢复「关节位置+控制器目标+橘子/纸盒位姿」，
      并按行 stage 种子化阶段锁存（出生即有的进度不发奖）。
    * 任何中段阶段样本数 < MIN_SAMPLES_PER_STAGE 时打印 WARNING 并把该阶
      段出生权重清零（绝不静默使用不足数据）；全部中段缺失时回退
      home-only。Newton 后端下 write 后驱动存活，出生夹持状态的可信度
      由生成器的物理保持验证背书。
    * RELEASE 成功门 0.055 已统一训练/部署口径（2026-09-27，见
      StagedPickState.release_clearance_m 与 stages.py 双轨注释），不再
      需要 reset 时切换注入。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.dataset_path = str(cfg.params.get("dataset_path", "") or "")
        self.forced_stage = cfg.params.get("forced_stage", None)
        self.bank = None
        if self.dataset_path:
            artifact = load_reset_dataset(self.dataset_path, device=env.device)
            states = artifact["states"]
            self.bank = states
            counts = torch.bincount(states["stage"], minlength=NUM_STAGES)
            missing = [(STAGE_NAMES[k], int(counts[k])) for k in range(1, NUM_STAGES)
                       if counts[k] < MIN_SAMPLES_PER_STAGE]
            if missing:
                print(
                    f"[StagedReset] WARNING: reset bank 样本不足的阶段（< {MIN_SAMPLES_PER_STAGE} 行）: {missing}；"
                    f"对应阶段出生已回退到 home。请用 scripts/generate_reset_bank.py 重新生成完整 bank。"
                )
            self.counts = counts.to(env.device)
        else:
            print("[StagedReset] WARNING: 未配置 reset 数据集路径，八阶段出生退化为 home-only。")
            self.counts = torch.zeros(NUM_STAGES, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor,
                 dataset_path: str = "", forced_stage: int | None = None) -> None:
        del dataset_path, forced_stage
        silence_mjwarp_overflow_warning(env)  # 幂等；每次 reset 重断言防求解器重建后复活
        state = StagedPickState.get(env)
        state.on_env_reset(env, env_ids)
        n = env_ids.numel()
        device = env.device

        # 出生阶段采样（评估模式强制指定阶段）
        if self.forced_stage is not None:
            stage_pick = torch.full((n,), int(self.forced_stage), dtype=torch.long, device=device)
        else:
            weights = curriculum_weights(state.reset_count).to(device)
            usable = weights.clone()
            usable[1:][self.counts[1:] < MIN_SAMPLES_PER_STAGE] = 0.0
            if float(usable.sum()) <= 0.0:
                usable = torch.zeros_like(usable)
                usable[0] = 1.0
            stage_pick = torch.multinomial(usable, n, replacement=True)

        # home 行：v0 随机刷位
        placements = v0.sample_placements(n, state.difficulty, device)
        yaw_fruit = torch.rand(n, device=device) * (2.0 * math.pi)
        yaw_box = (2 * torch.rand(n, device=device) - 1) * 0.3 * max(state.difficulty, 0.3)
        zero = torch.zeros(n, device=device)
        origins = _t(env.scene.env_origins)[env_ids]

        def pose(xy, z, yaw):
            quat = torch.stack([zero, zero, torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)
            pos = torch.stack([xy[:, 0], xy[:, 1], torch.full_like(xy[:, 0], z)], dim=-1) + origins
            return torch.cat([pos, quat], dim=-1)

        # bank 行：按阶段分桶均匀取行
        bank_rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.bank is not None:
            covered = torch.zeros(n, dtype=torch.bool, device=device)
            for k in range(1, NUM_STAGES):
                sel = torch.nonzero(stage_pick == k, as_tuple=False).flatten()
                if sel.numel() == 0 or self.counts[k] < MIN_SAMPLES_PER_STAGE:
                    continue
                pool = torch.nonzero(self.bank["stage"] == k, as_tuple=False).flatten()
                pick = pool[torch.randint(int(pool.numel()), (int(sel.numel()),), device=device)]
                bank_rows.append((sel, pick))
                covered[sel] = True
            stage_pick = torch.where(covered, stage_pick, torch.zeros_like(stage_pick))

        # 机器人：bank 行恢复关节位置/速度/目标
        robot = env.scene["robot"]
        for sel, pick in bank_rows:
            jp = self.bank["joint_position"][pick].to(device)
            jt = self.bank["joint_target"][pick].to(device)
            robot.write_joint_position_to_sim_index(position=jp, env_ids=env_ids[sel])
            robot.write_joint_velocity_to_sim_index(
                velocity=torch.zeros(int(sel.numel()), 6, device=device), env_ids=env_ids[sel]
            )
            robot.set_joint_position_target_index(target=jt, env_ids=env_ids[sel])

        # 物体：home 行随机刷位，bank 行恢复库位姿（env 局部 -> 世界）
        # 位移惩罚参考点必须跟随实际写入位姿：bank 行用 bank 位姿的 env 局部 xy，
        # 否则 *_disturbed 相对于一个从未生效的随机刷位计算、恒定饱和（2026-09-24 训练复盘）
        bank_keys = {"cardboard_box": "box_pose", "orange": "orange_pose"}
        spawn_xy = {
            "cardboard_box": placements["box"].clone(),
            "orange": placements["orange"].clone(),
        }
        for name, asset, xy, z, yaw in (
            ("cardboard_box", env.scene["cardboard_box"], placements["box"], v0.BOX_SPAWN_Z, yaw_box),
            ("orange", env.scene["orange"], placements["orange"], v0.FRUIT_SPAWN_Z, yaw_fruit),
        ):
            root = pose(xy, z, yaw)
            vel = torch.zeros(n, 6, device=device)
            if bank_rows:
                for sel, pick in bank_rows:
                    p = self.bank[bank_keys[name]][pick].to(device).clone()
                    spawn_xy[name][sel] = p[:, :2]  # bank 位姿存的是 env 局部系
                    p[:, :3] += origins[sel]
                    root = root.index_copy(0, sel, p)
            asset.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)

        # 出生阶段种子（paid 掩码使出生即有的进度不发奖）
        state.progress.reset(env_ids, stage_seed=stage_pick)
        state.birth_stage[env_ids] = stage_pick
        state.max_stage[env_ids] = stage_pick
        state.spawn_box_xy[env_ids] = spawn_xy["cardboard_box"]
        state.spawn_orange_xy[env_ids] = spawn_xy["orange"]
        # v0 的 PickState（env.pick_state）被 safe_*_disturbed 奖励项读取，
        # 其 spawn 参考若不随 reset 更新则恒为零向量，位移=物体到 env 原点
        # 距离（~0.5m），扰动惩罚永久饱和（2026-09-24 训练复盘实锤）
        v0state = v0.PickState.get(env)
        v0state.spawn_box_xy[env_ids] = spawn_xy["cardboard_box"]
        v0state.spawn_orange_xy[env_ids] = spawn_xy["orange"]
        state._last_step = -1


# ---------------------------------------------------------------------- #
# 观测
# ---------------------------------------------------------------------- #


def staged_policy_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Actor 观测（全部相对量 + 物理里程碑；评估语义与训练一致）。"""
    state = StagedPickState.get(env)
    state.update(env)
    feats = torch.cat(
        [
            state.orange_pos - state.pocket_pos,    # 橘子相对口袋
            state.hover_pos - state.pocket_pos,     # 盒口悬停点相对口袋
            state.orange_vel,                       # 橘子速度（env 系）
            state.orange_local,                     # 盒局部系放置几何
            (state.gripper_pos / GRIPPER_TRAVEL_RAD).clamp(0, 1).unsqueeze(-1),  # 开度分数
            state.progress.progress_flags(),        # 物理里程碑（Markov 化）
        ],
        dim=-1,
    )
    # 物理发散（NaN/inf）时先于终止被观测读走会中止训练；钳到安全值并让
    # staged_abnormal_robot（含 finite 检查）在同一步把该环境终止重置
    return torch.nan_to_num(feats, nan=0.0, posinf=2.0, neginf=-2.0)


def staged_critic_extra(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Critic 特权观测：接触力、臂受力、盒姿态、阶段 one-hot。"""
    state = StagedPickState.get(env)
    state.update(env)
    one_hot = torch.nn.functional.one_hot(state.progress.stage, NUM_STAGES).float()
    return torch.cat(
        [
            state.orange_force.clamp(0, 20).unsqueeze(-1),
            state.arm_force.clamp(0, 20).unsqueeze(-1),
            state.ee_force.clamp(0, 20).unsqueeze(-1),
            state.box_up_z.unsqueeze(-1),
            one_hot,
        ],
        dim=-1,
    )


# ---------------------------------------------------------------------- #
# 奖励：八阶段稠密项（按当前阶段门控）+ 一次性阶段完成 + 终局成功
# ---------------------------------------------------------------------- #


def _gate(env: ManagerBasedRLEnv, stage: int) -> torch.Tensor:
    state = StagedPickState.get(env)
    state.update(env)
    return (state.progress.stage == stage).float()


def reward_home_approach(env: ManagerBasedRLEnv, std: float = 0.12) -> torch.Tensor:
    """阶段 0：口袋接近橘子的存在型塑形（小权重防停留刷分）。"""
    state = StagedPickState.get(env)
    state.update(env)
    d = (state.orange_pos - state.pocket_pos).norm(dim=-1)
    return (1.0 - torch.tanh(d / std)) * _gate(env, int(PickStage.HOME))


def reward_pregrasp_distance(env: ManagerBasedRLEnv, std: float = 0.05) -> torch.Tensor:
    """阶段 1：精就位 + 保持开爪。"""
    state = StagedPickState.get(env)
    state.update(env)
    d = (state.orange_pos - state.pocket_pos).norm(dim=-1)
    open_frac = (state.gripper_pos / GRIPPER_TRAVEL_RAD).clamp(0, 1)
    return (1.0 - torch.tanh(d / std)) * open_frac * _gate(env, int(PickStage.PREGRASP))


def reward_grasp_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """阶段 2：真实夹持接触（爪部压力 + 有效夹持角带）。

    压力读爪部（gripper/jaw body）接触力——橘子净力对称抵消（≈mg 恒定），
    夹得越紧爪部读数越大，梯度真实。硬门橘子在爪口（<7.5cm）防蹭桌刷分。
    """
    state = StagedPickState.get(env)
    state.update(env)
    grip_ok = (
        (state.gripper_pos > GRASP_MIN_GRIPPER_RAD) & (state.gripper_pos < GRASP_MAX_GRIPPER_RAD)
    ).float()
    at_orange = ((state.orange_pos - state.pocket_pos).norm(dim=-1) < GRASP_PROXIMITY_M).float()
    return (state.claw_force.clamp(0, 5) / 5) * grip_ok * at_orange * _gate(env, int(PickStage.GRASP))


class reward_grasp_close(ManagerTermBase):
    """阶段 2：势能差塑形——就位后引导合爪与下压（IsaacLabTutorial ApproachProgress 同思路）。

    2026-09-24 训练复盘：GRASP 阶段此前只有「真实夹持力」一个稀疏源，
    策略收敛到开爪悬停局部最优（grasp_contact 趋零、99.98% 超时）。本项
    按势能差付费：爪角向闭合每变化 ``angle_scale`` 弧度、口袋每贴近橘子
    ``dist_scale`` 米各付一单位。悬停不付、倒退对称倒扣——开合震荡的净
    收益为零，无法刷分（difference-of-potentials 无悬停激励）。仅在爪口
    仍在橘子附近时生效，把橘子挤离后断供，与 orange_disturbed 惩罚共同
    抑制莽撞闭合。

    2026-09-27 去 clamp：旧版对每步增量 ``clamp(-1, 1)`` 截断——「慢速
    合爪（每步增量 < scale，足额付分）+ 快速张开（增量超界被截断，只罚
    一点）」每循环净赚，det_align_v1 崩塌策略实测靠此驻留 GRASP 段。去掉
    clamp 后支付严格等于势能差（telescoping），任意速度剖面的往复净收益
    恒为零；每步量被物理速度界住（|Δgrip|/0.2 ≲ 0.35、|Δd|/0.008 ≲ 1.3）。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_grip = torch.zeros(self.num_envs, device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)
        self._has_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._has_prev[ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        angle_scale: float = 0.2,
        dist_scale: float = 0.008,
    ) -> torch.Tensor:
        state = StagedPickState.get(env)
        state.update(env)
        dist_pocket = (state.orange_pos - state.pocket_pos).norm(dim=-1)
        gate = (state.progress.stage == int(PickStage.GRASP)) & (dist_pocket < 0.075)
        angle_prog = (self._prev_grip - state.gripper_pos) / angle_scale
        dist_prog = (self._prev_dist - dist_pocket) / dist_scale
        pay = torch.where(self._has_prev & gate, angle_prog + dist_prog, torch.zeros_like(angle_prog))
        self._prev_grip.copy_(state.gripper_pos)
        self._prev_dist.copy_(dist_pocket)
        self._has_prev.fill_(True)
        return pay


class SafeGraspClose(reward_grasp_close):
    """reward_grasp_close 的 NaN 防护版（与 SafeStageComplete 同模式）。"""

    def __call__(self, env):
        return torch.nan_to_num(super().__call__(env), nan=0.0, posinf=1.0, neginf=-1.0)


def reward_lift_height(env: ManagerBasedRLEnv, std: float = 0.08) -> torch.Tensor:
    """阶段 3：橘子离桌高度（tanh bump，仅作高度指示，主要引导见 lift_rise）。"""
    state = StagedPickState.get(env)
    state.update(env)
    h = (state.orange_pos[:, 2] - _TABLE_TOP_Z).clamp_min(0)
    return torch.tanh(h / std) * _gate(env, int(PickStage.LIFT))


class reward_lift_rise(ManagerTermBase):
    """阶段 3：势能差塑形——付「橘子离桌高度增量」（grasp_close 同款防刷分结构）。

    2026-09-25 晨检：LIFT 段隔夜 4200 迭代零进展（reach_LIFT 2.4%→1.8%，
    STABILIZE 恒 0）——现有 tanh bump 付「绝对高度」，夹住后吊在低处也能
    持续刷分，没有上升梯度（教程经验：bump 教策略停住不动）。本项按势能
    差付费：橘子每升高 ``rise_scale`` 付一单位，下落对称倒扣，悬停不付。
    门控：LIFT 段 + 已夹持 + 橘子仍在爪口（口袋距 < GRASP_PROXIMITY_M）。

    2026-09-27 两处加固：(1) 去 ``clamp(-1, 1)``——截断造成「慢升足额付、
    快降只罚一点」的每循环净赚（det_align_v1 崩塌模式之一，与 grasp_close
    同根）；(2) 高度势能在 LIFT 完成线 + ``LIFT_RISE_CAP_MARGIN_M`` 饱和——
    升到该线后继续抬升零支付，总支付被 (上限-起始高) 界住，且不鼓励无谓
    高抛。改后任意速度剖面往复净收益恒为零（严格 telescoping）。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_h = torch.zeros(self.num_envs, device=self.device)
        self._has_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._has_prev[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, rise_scale: float = 0.01) -> torch.Tensor:
        state = StagedPickState.get(env)
        state.update(env)
        h = (state.orange_pos[:, 2] - _TABLE_TOP_Z).clamp_min(0)
        potential = h.clamp(max=LIFT_Z_M - _TABLE_TOP_Z + LIFT_RISE_CAP_MARGIN_M)
        dist_pocket = (state.orange_pos - state.pocket_pos).norm(dim=-1)
        gate = (
            (state.progress.stage == int(PickStage.LIFT))
            & state.progress.grasped
            & (dist_pocket < GRASP_PROXIMITY_M)
        )
        rise = (potential - self._prev_h) / rise_scale
        pay = torch.where(self._has_prev & gate, rise, torch.zeros_like(rise))
        self._prev_h.copy_(potential)
        self._has_prev.fill_(True)
        return pay


class SafeLiftRise(reward_lift_rise):
    """reward_lift_rise 的 NaN 防护版（与 SafeStageComplete 同模式）。"""

    def __call__(self, env):
        return torch.nan_to_num(super().__call__(env), nan=0.0, posinf=1.0, neginf=-1.0)


class reward_release_retreat(ManagerTermBase):
    """阶段 7：势能差塑形——付「爪-橘子距离增量」（lift_rise 同款防刷分结构）。

    2026-09-26 推理诊断：9250 迭代策略 100% 卡在 RELEASE「爪撤离」——开爪
    后悬停在 3-5cm（place_descent 的终点姿态），继续后撤零收益，够不着成
    功门；而 stage-7 出生态自带 ~10.45cm 撤退位白拿 final_success，撤退
    动作全程零梯度。本项把撤退变成有钱赚的动作：爪每退远 ``retreat_scale``
    付一单位。门控：RELEASE 段 + 橘子仍在盒内（xy）。

    2026-09-27 重写防刷分结构（det_align_v1 崩塌实证：崩塌策略在 RELEASE
    段把爪在 1.2cm↔8.5cm 间往复戳、橘子早已静置，release_retreat 单项
    +6.9/局，0/32 成功）。旧版两处漏洞：(a) ``clamp(-1,1)`` 截断——慢退
    足额付、快戳截断只罚一点，每循环净赚；(b) 「成功线+3cm」距离窗硬门
    ——窗外断供造成窗口边缘不对称跳跃。修法：势能 φ = min(爪距,
    RETREAT_REWARD_TARGET_M=0.07) 饱和，支付 = Δφ/scale 无 clamp——任意
    速度剖面往复严格 telescoping 净零，每局总支付被 (0.07-起始距) 界住。

    双轨口径（2026-09-27 用户定版）：**撤退奖励峰值 0.07m、成功门 0.055m**
    ——奖励把爪均值推到 7cm（饱和点），成功判定在 5.5cm，1.5cm 差值吸收
    σ0.15 采样噪声的抖动。此前"门=饱和线=0.07"的零裕度结构实测让 10 步
    确认（断一步即清零）成为不可能事件、成功率钉 1%；退到 0.07 后继续
    后退零支付，想再赚只能让成功结算 +20（一次性终局奖，与撤退动作完全
    对齐，无刷分空间）。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)
        self._has_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._has_prev[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, retreat_scale: float = 0.005) -> torch.Tensor:
        state = StagedPickState.get(env)
        state.update(env)
        dist_pocket = (state.orange_pos - state.pocket_pos).norm(dim=-1)
        potential = dist_pocket.clamp(max=RETREAT_REWARD_TARGET_M)
        gate = (
            (state.progress.stage == int(PickStage.RELEASE))
            & (state.orange_local[:, 0].abs() < PLACE_INSIDE_XY_M)
            & (state.orange_local[:, 1].abs() < PLACE_INSIDE_XY_M)
        )
        retreat = (potential - self._prev_dist) / retreat_scale
        pay = torch.where(self._has_prev & gate, retreat, torch.zeros_like(retreat))
        self._prev_dist.copy_(potential)
        self._has_prev.fill_(True)
        return pay


class SafeReleaseRetreat(reward_release_retreat):
    """reward_release_retreat 的 NaN 防护版（与 SafeLiftRise 同模式）。"""

    def __call__(self, env):
        return torch.nan_to_num(super().__call__(env), nan=0.0, posinf=1.0, neginf=-1.0)


def reward_stabilize_calm(env: ManagerBasedRLEnv, vstd: float = 0.2) -> torch.Tensor:
    """阶段 4：相对晃动收敛 + 高度带保持。"""
    state = StagedPickState.get(env)
    state.update(env)
    rel = (state.orange_vel - state.jaw_vel).norm(dim=-1)
    in_band = (state.orange_pos[:, 2] > STABILIZE_MIN_Z_M).float()
    return (1.0 - torch.tanh(rel / vstd)) * in_band * _gate(env, int(PickStage.STABILIZE))


def reward_transport_progress(env: ManagerBasedRLEnv, std: float = 0.15) -> torch.Tensor:
    """阶段 5：橘子到盒口上方悬停点。"""
    state = StagedPickState.get(env)
    state.update(env)
    d = (state.orange_pos - state.hover_pos).norm(dim=-1)
    return (1.0 - torch.tanh(d / std)) * _gate(env, int(PickStage.TRANSPORT))


def reward_place_descent(env: ManagerBasedRLEnv) -> torch.Tensor:
    """阶段 6：盒局部系内接近释放位 + 盒内空间指示。

    深度峰值 2026-09-26 由 0.03（盒内正中）抬到 0.042：盒口 0.05，峰值停在
    口下方 8mm——爪不必跟到盒底，松爪后橘子短距落入即可，直接降低爪/腕蹭
    盒壁的几何压力（用户实测放橘子常碰盒）。
    """
    state = StagedPickState.get(env)
    state.update(env)
    inside_xy = (
        (state.orange_local[:, 0].abs() < PLACE_INSIDE_XY_M) & (state.orange_local[:, 1].abs() < PLACE_INSIDE_XY_M)
    ).float()
    depth = 1.0 - torch.tanh((state.orange_local[:, 2] - 0.042).abs() / 0.05)
    return inside_xy * depth * _gate(env, int(PickStage.PLACE))


class reward_release_open(ManagerTermBase):
    """阶段 7：势能差塑形——付「爪开度增量」（lift_rise 同款防刷分结构）。

    旧版付 open_frac×calm 的状态积：RELEASE 段开着爪悬在盒口每步持续领分
    （实测 +0.65/局），与撤退形成竞争吸引子——策略学会停在爪距 5.4cm 悬停
    领津贴而不是撤离过成功门（2026-09-27 实测 model_2000：400 采样爪距钉
    5.1-5.5cm、撤退分仅 +0.04、0/32 成功）。改势能差：开度每增加
    ``open_scale`` 付一单位，合拢对称倒扣，悬停不付、每局总支付被开度
    行程界住——只教「松爪」这一个动作，不补贴驻留。门控：RELEASE 段 +
    橘子在盒内（xy）（PLACE 已完成，此时松爪必为正确动作，不再需要 calm
    门）。防抖同款：无 clamp 严格 telescoping，往复净零。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_open = torch.zeros(self.num_envs, device=self.device)
        self._has_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._has_prev[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, open_scale: float = 0.1) -> torch.Tensor:
        state = StagedPickState.get(env)
        state.update(env)
        open_frac = (state.gripper_pos / GRIPPER_TRAVEL_RAD).clamp(0, 1)
        gate = (
            (state.progress.stage == int(PickStage.RELEASE))
            & (state.orange_local[:, 0].abs() < PLACE_INSIDE_XY_M)
            & (state.orange_local[:, 1].abs() < PLACE_INSIDE_XY_M)
        )
        opening = (open_frac - self._prev_open) / open_scale
        pay = torch.where(self._has_prev & gate, opening, torch.zeros_like(opening))
        self._prev_open.copy_(open_frac)
        self._has_prev.fill_(True)
        return pay


class SafeReleaseOpen(reward_release_open):
    """reward_release_open 的 NaN 防护版（与 SafeLiftRise 同模式）。"""

    def __call__(self, env, open_scale: float = 0.1):
        return torch.nan_to_num(
            super().__call__(env, open_scale=open_scale), nan=0.0, posinf=1.0, neginf=-1.0
        )


class reward_stage_complete(ManagerTermBase):
    """八个一次性阶段完成奖励（just_completed 每控制步至多付一次）。

    bonus 按阶段递增 (3,4,6,8,8,10,14,30)：后期完成信息量更高，但与稠密
    项量级平衡。出生即有的阶段由 paid 掩码豁免（不刷出生奖）。
    """

    STAGE_BONUS = (3.0, 4.0, 6.0, 8.0, 8.0, 10.0, 14.0, 30.0)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        state = StagedPickState.get(env)
        state.update(env)
        bonus = torch.as_tensor(self.STAGE_BONUS, device=env.device)
        return (state.progress.just_completed.float() * bonus).sum(dim=-1)


def reward_final_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """终局成功一次性大奖励（RELEASE 稳定确认，与阶段 7 完成奖叠加）。"""
    state = StagedPickState.get(env)
    state.update(env)
    return state.progress.just_completed[:, int(PickStage.RELEASE)].float() * 20.0


# ---- 惩罚（阶段感知的额外罚；v0 的扰动/碰撞项在 env cfg 复用）----


def penalty_premature_release(env: ManagerBasedRLEnv) -> torch.Tensor:
    """已夹持后、未到盒内释放位就松开（夹持消失 + 开度大 + 不在盒内）。"""
    state = StagedPickState.get(env)
    state.update(env)
    in_release = (
        (state.orange_local[:, 0].abs() < PLACE_INSIDE_XY_M)
        & (state.orange_local[:, 1].abs() < PLACE_INSIDE_XY_M)
        & (state.orange_local[:, 2] < PLACE_INSIDE_Z_MAX_M)
    )
    lost = state.progress.grasped & (state.orange_force < 0.15) & (state.gripper_pos > 1.2)
    gate = (state.progress.stage >= int(PickStage.GRASP)).float()
    return (lost & ~in_release).float() * gate


def penalty_ee_bad_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 2.5,
    clear_of_orange: float = 0.075,
    orange_rest_force: float = 0.2,
) -> torch.Tensor:
    """爪压非橘子物体惩罚（staged 版，替换 v0 的 ee_bad_contact）。

    v0 版两处失真：①「非橘子」判定用 dist_jaw>7cm——jaw body 到口袋
    几何上就有 ~9-12cm，真实夹持被当成坏接触持续罚款；②读 jaw 单列
    受力——该 USD 两指碰撞体都在 gripper body 下，jaw 列恒 0（罚从未
    触发）。本版：爪部受力（claw）高于桌蹭基线（2.5N）且（橘子不在
    爪口 或 橘子未被实压）才罚——真实夹持不罚，压桌/压盒罚。
    """
    state = StagedPickState.get(env)
    state.update(env)
    dist_pocket = (state.orange_pos - state.pocket_pos).norm(dim=-1)
    not_orange = (dist_pocket > clear_of_orange) | (state.orange_force < orange_rest_force)
    return ((state.claw_force > threshold) & not_orange).float()


class penalty_box_contact(ManagerTermBase):
    """搬运开始后盒体受力惩罚（爪/腕压盒壁的直接信号，连续 ``confirm_steps`` 步才罚）。

    2026-09-26 用户实测放橘子常蹭盒。此前爪-盒接触零成本：ee_collision 对
    「爪中有橘子」豁免（压壁时橘子正在爪里，永远命中豁免）、box_disturbed
    只罚位移、box_tipped 只罚翻倒。本项直读盒体接触传感器：TRANSPORT 起
    盒体净力 > threshold 连续 confirm_steps 步才罚。传感器由
    PickRLSceneCfg.box_contact 提供。

    2026-09-27 探针实测（16 envs × 60 步）改为持续确认制：net_forces_w 是
    全部接触的合力，含桌面支撑——空盒静置中位 1.46N（=盒重 0.14-0.16kg）、
    橘子入盒后仍在同量级（Newton 求解器对盒-橘静接触读数偏低），但求解器
    瞬态尖峰可到 5.4N（>2N 步占比 0.7-2.3%）。旧的瞬时二值判罚把这些
    1-2 步尖峰也当压壁罚，给价值函数注入噪声；爪/腕真正压壁是持续多步
    >2N——连续 3 步确认即可区分（橘子落入盒的冲击同样只有 1-2 步）。
    旧版 -4.0/步的量级也顺带下修到 -2.0（与 ee_collision 同档）。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._streak[ids] = 0

    def __call__(
        self, env: ManagerBasedRLEnv, threshold: float = 2.0, confirm_steps: int = 3
    ) -> torch.Tensor:
        state = StagedPickState.get(env)
        state.update(env)
        sensor = env.scene.sensors["box_contact"]
        force = _t(sensor.data.net_forces_w).reshape(env.num_envs, -1, 3).norm(dim=-1).squeeze(-1)
        gate = state.progress.stage >= int(PickStage.TRANSPORT)
        hot = (force > threshold) & gate
        self._streak = torch.where(hot, self._streak + 1, torch.zeros_like(self._streak))
        return (self._streak >= confirm_steps).float()


def penalty_workspace(env: ManagerBasedRLEnv) -> torch.Tensor:
    """口袋离开安全工作空间（桌面矩形上方带）。"""
    state = StagedPickState.get(env)
    state.update(env)
    (x0, x1), (y0, y1) = v0._RECTS["fruit"]
    inside = (
        (state.pocket_pos[:, 0] > x0 - 0.05)
        & (state.pocket_pos[:, 0] < x1 + 0.05)
        & (state.pocket_pos[:, 1] > y0 - 0.05)
        & (state.pocket_pos[:, 1] < y1 + 0.05)
        & (state.pocket_pos[:, 2] > _TABLE_TOP_Z - 0.02)
        & (state.pocket_pos[:, 2] < 0.45)
    )
    return (~inside).float()


# ---------------------------------------------------------------------- #
# 终止（独立可记录的终止原因）
# ---------------------------------------------------------------------- #


def _debug_release(env: ManagerBasedRLEnv, state: StagedPickState) -> None:
    """PICK_DEBUG_RELEASE=1 时按 1Hz 打印每环境 RELEASE 未满足项。

    阈值与上方条件表完全一致，只读状态不参与仿真逻辑；默认关闭零开销。
    用于 play 可视化时回答「为什么还不算成功」。
    """
    if os.environ.get("PICK_DEBUG_RELEASE", "") != "1":
        return
    now = time.monotonic()
    if now - _debug_release.last_t < 1.0:
        return
    _debug_release.last_t = now

    dist_pocket = (state.orange_pos - state.pocket_pos).norm(dim=-1)
    checks = (
        ("盒内xy", state.orange_local[:, :2].abs().amax(dim=-1) < PLACE_INSIDE_XY_M),
        (
            "盒内z",
            (state.orange_local[:, 2] > PLACE_INSIDE_Z_MIN_M)
            & (state.orange_local[:, 2] < PLACE_INSIDE_Z_MAX_M),
        ),
        ("爪撤离", dist_pocket > state.release_clearance_m),
        ("静置", state.orange_vel.norm(dim=-1) < RELEASE_SETTLED_SPEED_MPS),
    )
    parts = []
    for i in range(env.num_envs):
        stage = STAGE_NAMES[int(state.progress.stage[i])]
        missing = [name for name, ok in checks if not bool(ok[i])]
        stable = int(state.progress.stable_count[i])
        suffix = f"缺:{'|'.join(missing)}" if missing else "全满足"
        parts.append(f"env{i}[{stage}] 确认{stable}/10 爪距{dist_pocket[i] * 100:.1f}cm {suffix}")
    print("[RELEASE诊断] " + " | ".join(parts), flush=True)


_debug_release.last_t = 0.0


def staged_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """成功 = RELEASE 稳定确认（StageProgress 已含连续稳定计数）。"""
    state = StagedPickState.get(env)
    state.update(env)
    state.log_metrics(env)
    _debug_release(env, state)
    return state.progress.success


def staged_orange_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """橘子掉落：已夹持过后掉出桌面 / 低空。"""
    state = StagedPickState.get(env)
    state.update(env)
    (x0, x1), (y0, y1) = v0._RECTS["fruit"]
    off = (
        (state.orange_pos[:, 0] < x0 - 0.08)
        | (state.orange_pos[:, 0] > x1 + 0.08)
        | (state.orange_pos[:, 1] < y0 - 0.08)
        | (state.orange_pos[:, 1] > y1 + 0.08)
        | (state.orange_pos[:, 2] < 0.02)
    )
    return off & state.progress.grasped


def staged_box_tipped(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = StagedPickState.get(env)
    state.update(env)
    (x0, x1), (y0, y1) = v0._RECTS["fruit"]
    off = (
        (state.box_pos[:, 0] < x0 - 0.15)
        | (state.box_pos[:, 0] > x1 + 0.15)
        | (state.box_pos[:, 1] < y0 - 0.15)
        | (state.box_pos[:, 1] > y1 + 0.15)
    )
    return (state.box_up_z < 0.70) | off


def staged_abnormal_robot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """关节/物体状态非有限或速度爆掉（软舵机振铃阈值 25 rad/s）即终止。"""
    robot = env.scene["robot"]
    vel = _t(robot.data.joint_vel)
    finite = torch.isfinite(vel).all(dim=-1) & torch.isfinite(_t(robot.data.joint_pos)).all(dim=-1)
    for name in ("orange", "cardboard_box"):
        asset = env.scene[name]
        finite &= torch.isfinite(_t(asset.data.root_pos_w)).all(dim=-1)
    return (~finite) | (vel.abs().amax(dim=-1) > 25.0)


# 桌面高度（v0 常量转发，避免魔法数散落）
_TABLE_TOP_Z = v0.TABLE_TOP_Z


def safe_joint_pos_rel(env):
    """joint_pos_rel 的防 NaN 包装（发散环境被终止前的一步不污染 buffer）。"""
    from isaaclab.envs.mdp import joint_pos_rel

    return torch.nan_to_num(joint_pos_rel(env), nan=0.0, posinf=5.0, neginf=-5.0)


def safe_joint_vel_rel(env):
    from isaaclab.envs.mdp import joint_vel_rel

    return torch.nan_to_num(joint_vel_rel(env), nan=0.0, posinf=5.0, neginf=-5.0)


def safe_last_action(env):
    from isaaclab.envs.mdp import last_action

    return torch.nan_to_num(last_action(env), nan=0.0, posinf=1.0, neginf=-1.0)


# ---------------------------------------------------------------------------
# 奖励输出级 NaN 防护：发散环境在被终止重置的那一帧，奖励计算含 NaN
# （位置/位移 NaN），rsl-rl 的 check_nan 会中止训练。统一钳到安全值——
# 终止条件（staged_abnormal_robot 的 finite 检查）同一步会把该环境重置。
# ---------------------------------------------------------------------------


def _nan_safe(func):
    import functools

    @functools.wraps(func)  # 保留原签名：Manager 按签名校验奖励项参数
    def wrapped(env, *args, **kwargs):
        return torch.nan_to_num(func(env, *args, **kwargs), nan=0.0, posinf=10.0, neginf=-10.0)

    return wrapped


safe_reward_home_approach = _nan_safe(reward_home_approach)
safe_reward_pregrasp_distance = _nan_safe(reward_pregrasp_distance)
safe_reward_grasp_contact = _nan_safe(reward_grasp_contact)
safe_reward_lift_height = _nan_safe(reward_lift_height)
safe_reward_stabilize_calm = _nan_safe(reward_stabilize_calm)
safe_reward_transport_progress = _nan_safe(reward_transport_progress)
safe_reward_place_descent = _nan_safe(reward_place_descent)
# reward_release_open 已改为带状态的 ManagerTermBase 类，见上方
# SafeReleaseOpen（不能用 _nan_safe 函数包装）
safe_reward_final_success = _nan_safe(reward_final_success)
safe_penalty_premature_release = _nan_safe(penalty_premature_release)
safe_penalty_workspace = _nan_safe(penalty_workspace)
safe_box_disturbed = _nan_safe(v0.box_disturbed)
safe_box_disturbed_late = _nan_safe(v0.box_disturbed_late)
safe_orange_disturbed = _nan_safe(v0.orange_disturbed)
safe_arm_collision = _nan_safe(v0.arm_collision)
safe_ee_bad_contact = _nan_safe(v0.ee_bad_contact)
safe_penalty_ee_bad_contact = _nan_safe(penalty_ee_bad_contact)


class SafeBoxContact(penalty_box_contact):
    """penalty_box_contact 的 NaN 防护版（与 SafeLiftRise 同模式）。

    penalty_box_contact 已改为带连续确认状态的 ManagerTermBase，不能用
    _nan_safe 函数包装，改用子类遮蔽 __call__。签名与父类保持一致（显式
    threshold/confirm_steps），满足 Manager 按签名校验奖励项参数的约定。
    """

    def __call__(self, env, threshold: float = 2.0, confirm_steps: int = 3):
        return torch.nan_to_num(
            super().__call__(env, threshold=threshold, confirm_steps=confirm_steps),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )


class SafeStageComplete(reward_stage_complete):
    """reward_stage_complete 的 NaN 防护版（一次性奖励同样可能读到 NaN）。"""

    def __call__(self, env):
        return torch.nan_to_num(super().__call__(env), nan=0.0, posinf=10.0, neginf=-10.0)
