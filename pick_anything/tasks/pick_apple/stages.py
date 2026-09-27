"""八任务阶段的定义、物理完成阈值与阶段状态机。

任务书要求的八个阶段（同一 PPO 策略的不同进度/出生起点，不是八个模型）：

    0 HOME       安全初始姿态，橘子在桌面可抓取位
    1 PREGRASP   张开的夹爪就位：橘子中心处于捕获口袋内且开度足够
    2 GRASP      夹爪真实夹住橘子（接触力 + 贴爪 + 部分闭合，连续若干步）
    3 LIFT       橘子被抬离桌面到安全高度并保持
    4 STABILIZE  调整到稳定搬运姿态：高度足以越过盒壁、相对晃动小
    5 TRANSPORT  搬运到纸盒开口上方悬停点（盒局部坐标系）
    6 PLACE      橘子降到盒内有效空间的安全释放位置
    7 RELEASE    松爪撤离，橘子在盒内连续稳定 -> 最终成功

设计原则：

* 阶段推进条件全部来自物理状态（接触力 / 位置 / 速度），阶段编号本身
  只是锁存进度的记录，永远不单独作为发奖依据。
* ``StageProgress`` 参考 IsaacLabTutorial 的 ``PlacementProgress`` 模式：
  里程碑锁存 + 连续稳定计数 + 出生日志种子恢复。从阶段 k 出生时，里程碑
  种子推进到 k 级，且 <= k 的阶段完成奖励标记为已付——出生即有的进度
  不发奖，防止 PPO 靠"出生刷阶段奖"。
* 奖励项读 ``just_completed``（本控制步新达成的阶段集合，一次性）；
  观测读 ``progress_flags``（物理里程碑布尔，Markov 化阶段奖励，且评估
  时语义与训练完全一致——里程碑是物理事实，不是人为阶段标签）。
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import torch


class PickStage(IntEnum):
    """任务阶段编号（严格递增，值即 reset 数据集里的 stage 字段）。"""

    HOME = 0
    PREGRASP = 1
    GRASP = 2
    LIFT = 3
    STABILIZE = 4
    TRANSPORT = 5
    PLACE = 6
    RELEASE = 7


STAGE_NAMES: tuple[str, ...] = tuple(stage.name for stage in PickStage)
NUM_STAGES = len(STAGE_NAMES)

# --------------------------------------------------------------------- #
# 各阶段完成条件阈值（物理量，单位 m / m/s / N / rad）
# --------------------------------------------------------------------- #

# 阶段 0 HOME 完成（即达成 PREGRASP 就位条件）：橘子中心在捕获口袋
# POCKET_OFFSET_B 半径内、夹爪开度足够（USD 实测爪角 0 rad=全闭、
# 1.745 rad=全开；staged 侧用 PREGRASP_MIN_GRIPPER_RAD=1.30 近全开门）、
# 橘子仍在桌面未被意外举离或撞跑。
PREGRASP_POCKET_M = 0.035

# 阶段 1 PREGRASP 完成 -> GRASP：真实夹持。净接触力（橘子静置桌面时自重
# 支撑约 0.25 N，被两指夹压显著更高）+ 贴爪 + 开口处于"夹住 41 mm 橘子"
# 的中间档（全闭 1.745 rad 会把橘子挤飞），连续 GRASP_CONFIRM_STEPS 步
# 确认，不许凭瞬时距离判定。
GRASP_FORCE_N = 1.2
GRASP_PROXIMITY_M = 0.075
# 新 SO101（isaaclab_assets，2026-09-25 起）爪角 rad(-0.175, 1.745)、小值=
# 闭合。夹 56mm 球时动爪被球挡停在 ~0.2-0.5 rad（40N 夹持）——被挡住
# 即是有效夹持，带上限取 0.75（与生成器 GRIP_BAND 同口径；全闭 -0.155
# 反而是夹空）。
GRASP_MIN_GRIPPER_RAD = -0.174
GRASP_MAX_GRIPPER_RAD = 0.75
GRASP_CONFIRM_STEPS = 3

# 阶段 2 GRASP 完成 -> LIFT：橘子中心抬到安全高度并持续保持（桌面
# 41 mm + 橘子半径 21 mm + 爪体离桌余量）。
LIFT_Z_M = 0.105
LIFT_CONFIRM_STEPS = 4

# 阶段 3 LIFT 完成 -> STABILIZE：搬运稳定性。橘子与夹爪的相对速度小
# （晃动收敛）且保持在安全高度带内，连续若干步。
STABILIZE_MIN_Z_M = 0.105
STABILIZE_REL_VEL_MPS = 0.18
STABILIZE_CONFIRM_STEPS = 6

# 阶段 4 STABILIZE 完成 -> TRANSPORT：高度足以让橘子（含半径）越过盒壁
# （盒口 z = 桌面 0.041 + 壁高 0.05 = 0.091；橘子中心 >= 盒口 + 半径 +
# 余量）且夹持稳定。
TRANSPORT_MIN_Z_M = 0.118
TRANSPORT_REL_VEL_MPS = 0.22

# 阶段 5 TRANSPORT 完成 -> PLACE：橘子到达盒口上方悬停区（盒局部 XY 接近
# 中心、局部 Z 在盒口之上一定余量）且仍被夹持。
PLACE_OVER_BOX_XY_M = 0.055
PLACE_APPROACH_MIN_Z_M = 0.10  # 盒局部系：盒底为 0，盒口 0.05，此值即口上方 5 cm

# 阶段 6 PLACE 完成 -> RELEASE：橘子中心已在盒内有效空间（内壁半宽之内、
# 介于盒底上方 8 mm 与盒口下沿之间），夹爪仍在附近（携带或刚释放）。
PLACE_INSIDE_XY_M = 0.052
PLACE_INSIDE_Z_MIN_M = 0.008
PLACE_INSIDE_Z_MAX_M = 0.052
PLACE_JAW_NEAR_M = 0.10

# 阶段 7 RELEASE 完成 = 最终成功：橘子在盒内 + 夹爪已离开 + 速度收敛，
# 确认计数净攒满 RELEASE_STABLE_STEPS 步（防抖见 StageProgress.update）。
# 双轨口径（2026-09-27 用户定版）：成功门训练/部署统一 5.5cm；撤退奖励
# 的势能饱和线 7cm——奖励把爪均值推到 7cm，成功判定在 5.5cm，中间
# 1.5cm 吸收 σ0.15 采样噪声（此前门=饱和线=0.07 的零裕度结构实测把
# 10 步确认变成不可能事件，成功率钉 1%）。
RELEASE_CLEARANCE_M = 0.055
RETREAT_REWARD_TARGET_M = 0.07
RELEASE_SETTLED_SPEED_MPS = 0.15
RELEASE_STABLE_STEPS = 10


class StageProgress:
    """每环境的阶段锁存状态机：推进、一次性完成标志、成功确认。

    所有张量按环境批量维护，partial reset 只清 ``env_ids`` 对应行。
    ``update`` 每控制步调用一次，由 PickState 统一驱动（同旧实现的幂等
    刷新模式），奖励/终止/观测项共享同一份结果。
    """

    def __init__(self, num_envs: int, device: torch.device | str):
        self.device = torch.device(device)
        n = num_envs
        self.stage = torch.zeros(n, dtype=torch.long, device=self.device)
        # 阶段完成奖励是否已付（一次性；出生种子把 <= 出生阶段的都置 True）
        self.paid = torch.zeros(n, NUM_STAGES, dtype=torch.bool, device=self.device)
        # 本控制步新达成的阶段（奖励项的消费接口，每步刷新）
        self.just_completed = torch.zeros(n, NUM_STAGES, dtype=torch.bool, device=self.device)
        # 里程碑锁存（观测/终止用，物理事实）
        self.pregrasp_latched = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.grasped = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.lifted = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.stabilized = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.over_box_latched = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.in_box_latched = torch.zeros(n, dtype=torch.bool, device=self.device)
        # 连续确认计数（每阶段独立一列，跨步累积、条件断则清零）与最终稳定计数
        self.confirm_count = torch.zeros(n, NUM_STAGES, dtype=torch.long, device=self.device)
        self.stable_count = torch.zeros(n, dtype=torch.long, device=self.device)
        # 最终成功
        self.success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.succeeded_episode = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.time_to_success = torch.full((n,), -1, dtype=torch.long, device=self.device)
        # 出生后第一次 update 不付奖也不推进（参考 PhysicalMilestoneReward 的
        # _initialized 屏蔽：reset 沉降瞬间满足的条件不是策略功劳）
        self.first_step = torch.ones(n, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ #
    def reset(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor | None = None,
        *,
        stage_seed: torch.Tensor | None = None,
    ) -> None:
        """清 episode 状态；出生阶段为 k 时里程碑种子推进到 k 级且 <=k 不付奖。"""
        ids = slice(None) if env_ids is None else env_ids
        self.stage[ids] = 0
        self.paid[ids] = False
        self.just_completed[ids] = False
        self.pregrasp_latched[ids] = False
        self.grasped[ids] = False
        self.lifted[ids] = False
        self.stabilized[ids] = False
        self.over_box_latched[ids] = False
        self.in_box_latched[ids] = False
        self.confirm_count[ids] = 0
        self.stable_count[ids] = 0
        self.success[ids] = False
        self.succeeded_episode[ids] = False
        self.time_to_success[ids] = -1
        self.first_step[ids] = True
        if stage_seed is not None:
            self.seed_from_stage(ids, stage_seed)

    def seed_from_stage(self, ids, stage: torch.Tensor) -> None:
        """按出生阶段种子化里程碑（真实 reset 状态库行的 stage 字段）。

        出生在阶段 k = 阶段 0..k-1 已经完成（完成奖标记已付，不再发），
        阶段 k 本身进行中——其完成奖仍由策略达成后正常发放。
        """
        stage = stage.to(device=self.device, dtype=torch.long).clamp(0, NUM_STAGES - 1)
        self.stage[ids] = stage
        paid_mask = torch.arange(NUM_STAGES, device=self.device).unsqueeze(0) < stage.unsqueeze(1)
        self.paid[ids] = self.paid[ids] | paid_mask
        self.pregrasp_latched[ids] = stage >= int(PickStage.PREGRASP)
        self.grasped[ids] = stage >= int(PickStage.GRASP)
        self.lifted[ids] = stage >= int(PickStage.LIFT)
        self.stabilized[ids] = stage >= int(PickStage.STABILIZE)
        self.over_box_latched[ids] = stage >= int(PickStage.TRANSPORT)
        self.in_box_latched[ids] = stage >= int(PickStage.PLACE)

    # ------------------------------------------------------------------ #
    def update(
        self,
        step: torch.Tensor,
        cond: dict[str, torch.Tensor],
        confirm: dict[str, int],
    ) -> torch.Tensor:
        """按物理条件推进阶段，返回锁存的成功标志。

        ``cond[name]`` 是该阶段目标已达成的物理条件（瞬时布尔），
        ``confirm[name]`` 是需要连续满足的控制步数（1 = 立即）。
        顺序检查阶段 1..7：当前阶段条件连续达成足够步数则推进到下一阶段
        （推进即完成，一次性付奖），同一步允许连续推进多个阶段。
        """
        self.just_completed.zero_()
        first = self.first_step.clone()
        self.first_step.fill_(False)
        # 出生后第一步：只锁存里程碑，不推进、不付奖（reset 沉降不是策略功劳）
        gate = ~first
        names = [PickStage(k).name for k in range(0, NUM_STAGES - 1)]  # HOME..PLACE
        zeros = torch.zeros_like(self.stage)
        for name in names:
            k = int(PickStage[name])
            need = confirm.get(name, 1)
            # 只从当前阶段推进（stage==k），阶段编号单调不回退。2026-09-24
            # 实测：旧条件 stage<RELEASE 下，合爪到 <1.30 rad 时 PREGRASP
            # 条件断而 HOME 条件（口袋距<12cm、橘子未离桌）仍成立，阶段被
            # 2→1 拖回，GRASP 段奖励在「开始合爪」瞬间全部静默——抓取结构
            # 性学不会。循环内 stage 原地更新，级联推进（一步跨多阶）保留。
            met_now = gate & (self.stage == k) & cond.get(name, zeros)
            col = self.confirm_count[:, k]
            # 该阶段独立计数：条件断则清零（贴边抖动无法累积确认）
            col = torch.where(cond.get(name, zeros), col + 1, torch.zeros_like(col))
            col = torch.where(gate, col, torch.zeros_like(col))
            self.confirm_count[:, k] = col
            confirmed = met_now & (col >= need)
            if bool(confirmed.any()):
                newly = confirmed & ~self.paid[:, k]
                self.just_completed[:, k] |= newly
                self.paid[:, k] = self.paid[:, k] | newly
                self.stage = torch.where(confirmed, torch.full_like(self.stage, k + 1), self.stage)
            # 里程碑锁存保持物理事实（观测用，与阶段推进/付费独立）
            raw = cond.get(name, zeros)
            if name == "PREGRASP":
                self.pregrasp_latched |= raw
            elif name == "GRASP":
                self.grasped |= raw
            elif name == "LIFT":
                self.lifted |= raw
            elif name == "STABILIZE":
                self.stabilized |= raw
            elif name == "TRANSPORT":
                self.over_box_latched |= raw
            elif name == "PLACE":
                self.in_box_latched |= raw

        # RELEASE 阶段完成 = 最终成功：确认计数净攒满 RELEASE_STABLE_STEPS 步。
        # 2026-09-27 防抖：条件断开一步由清零改为 -2（下限 0）——净攒满要求
        # 在线率 >2/3。σ0.15 采样噪声下爪距围绕均值 ±1-2cm 抖动，旧版"断一
        # 步即清零"让 10 连几乎不可能（成功率钉 1%）；防抖后偶发回踩只回退
        # 两步，而悬停/戳橘子（在线率≪2/3）依旧净减永远攒不满，不放松对
        # "实质驻留门外+静置"的要求，也不打开刷分口子（成功是一次性终局奖）。
        release_ready = gate & cond.get("RELEASE", torch.zeros_like(self.success))
        self.stable_count.copy_(
            torch.where(
                release_ready,
                self.stable_count + 1,
                (self.stable_count - 2).clamp_min(0),
            )
        )
        newly_successful = (~self.success) & (self.stable_count >= RELEASE_STABLE_STEPS)
        self.time_to_success = torch.where(newly_successful, step, self.time_to_success)
        self.success |= newly_successful
        self.succeeded_episode |= newly_successful
        newly_paid = newly_successful & ~self.paid[:, int(PickStage.RELEASE)]
        self.just_completed[:, int(PickStage.RELEASE)] = newly_paid
        self.paid[:, int(PickStage.RELEASE)] |= newly_paid
        return self.success.clone()

    def progress_flags(self) -> torch.Tensor:
        """物理里程碑布尔（观测用）：与阶段编号不同，评估时语义一致。"""
        return torch.stack(
            [
                self.pregrasp_latched.float(),
                self.grasped.float(),
                self.lifted.float(),
                self.over_box_latched.float(),
                self.in_box_latched.float(),
                self.success.float(),
            ],
            dim=-1,
        )
