"""八阶段抓橘子入盒任务的环境配置（gym id: Pick-Apple-SO101-Staged-v0，历史名保留）。

与 v0（PickAppleEnvCfg）的区别：
* 观测非对称：actor 只看相对量 + 物理里程碑；critic 额外看接触力/阶段
  one-hot 等特权信息（obs_groups={"actor": ["policy"], "critic": ["critic"]}）；
* reset 事件换成 ``StagedReset``：按课程从物理验证 reset 状态库采样出生
  阶段，缺失阶段报警并安全回退 home-only；
* 奖励为八阶段稠密项 + 一次性阶段完成 + 终局成功；惩罚复用 v0 的扰动/
  碰撞项并新增过早松爪 / 工作空间违规（干扰物已删，误抓橘子惩罚随之移除）；
* 动作尺度对齐 lift 验证值（0.06/0.10）——v0 的 0.15/0.35 在该 USD 软舵
  机上会引起 PD 振铃（实测爪速 ±8 rad/s）；
* 夹爪 reset 范围按修正后的爪角语义（0=全闭）取 (0.5, 1.6)：出生横跨
  「半闭兜住带」之上到全开，开/闭都是可学的动作。

另含 ``StagedPickGeneratorEnvCfg``：reset 生成器专用的 raw 桌面环境
（无奖励/终止、raw 刷位事件），结构对齐 IsaacLabTutorial 的
SO101VialGeneratorEnvCfg。
"""

from __future__ import annotations

from isaaclab.envs.mdp import (
    RelativeJointPositionActionCfg,
    action_rate_l2,
    is_terminated_term,
    joint_pos_rel,
    time_out,
    randomize_actuator_gains,
    randomize_joint_parameters,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_joints_by_offset,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonManager

from pick_anything.tasks.pick_apple import mdp as v0
from pick_anything.tasks.pick_apple import staged_mdp
from pick_anything.tasks.pick_apple.pick_env_cfg import ARM_JOINT_NAMES, PickAppleEnvCfg


def make_newton_physics() -> NewtonCfg:
    """Newton/MJWarp 物理配置（参数取自 IsaacLabTutorial 同类抓取任务的
    验证值）。切换动机：PhysX 后端实测存在 write_joint_position 杀死爪
    隐式驱动 + 臂动期间 set 间歇失效两个缺陷；参考项目的 reset-恢复 +
    训练整套在此后端跑通。"""
    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            cone="elliptic",
            impratio=10.0,
            update_data_interval=2,
            iterations=100,
            ls_iterations=50,  # cfg 默认值；低于此值随机策略下 linesearch 不收敛频发
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        num_substeps=2,
        debug_mode=False,
    )


# --------------------------------------------------------------------- #
# Newton 接触模型（移植自 IsaacLabTutorial env_cfg 的 workshop 验证值）
# --------------------------------------------------------------------- #

_CONTACT_STIFFNESS = 1.57e5
_CONTACT_DAMPING = 1.12e3
_FRICTION = 0.9
# 滚动摩擦是球形目标夹持的关键参数（2026-09-25 实测）：教程的 0.05 对
# 药瓶（圆柱沿轴滚动不改变夹持）无害，但球靠「滚出夹口」逃逸——滑动
# 要克服 μN（40N 夹持 + μ0.9 ≈ 36N，逃不掉），滚动几乎无阻力。0.30
# 让球在 40N 双垫夹持下静态锁死（真实橘子皮的滚阻同样很高）。
_ROLLING_FRICTION = 0.50  # 0.30 冷恢复存活仅 3-6%（球仍能缓慢滚出），0.50 锁死
_TORSIONAL_FRICTION = 0.10
_SOLIMP = (0.7, 0.95, 0.0001, 0.5, 2.0)
_SOLREF = (0.002, 1.5)
_contact_model_registered = False


def _initialize_contacts(_event) -> None:
    """把 workshop 验证过的接触模型应用到每个 Newton shape。

    参考项目整套抓取（含 reset 恢复后夹持稳定）在这组参数上验证；滚动
    摩擦 0.05 对球形目标尤其关键——没有它橘子会在 V 口内滚向宽端滑出。
    """
    import newton

    builder = NewtonManager._builder
    if builder is None:
        return

    num_shapes = len(builder.shape_body)
    for shape_index in range(num_shapes):
        builder.shape_material_ke[shape_index] = _CONTACT_STIFFNESS
        builder.shape_material_kd[shape_index] = _CONTACT_DAMPING
        builder.shape_material_mu[shape_index] = _FRICTION
        builder.shape_material_mu_rolling[shape_index] = _ROLLING_FRICTION
        builder.shape_material_mu_torsional[shape_index] = _TORSIONAL_FRICTION

    # Prototype builders register these attributes, but Newton's cloner does
    # not currently carry that registration to the main builder.
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    for name, value in (("mujoco:geom_solimp", _SOLIMP), ("mujoco:geom_solref", _SOLREF)):
        attribute = builder.custom_attributes.get(name)
        if attribute is None:
            continue
        if attribute.values is None:
            attribute.values = {}
        for shape_index in range(num_shapes):
            attribute.values[shape_index] = value


def _register_contact_model() -> None:
    """每进程注册一次接触模型初始化回调（仅 Newton 后端触发）。"""
    global _contact_model_registered
    if _contact_model_registered:
        return
    from isaaclab.physics import PhysicsEvent

    NewtonManager.register_callback(
        _initialize_contacts,
        PhysicsEvent.MODEL_INIT,
        name="pick_anything_contact_model",
    )
    _contact_model_registered = True


_register_contact_model()


# 夹爪关节 reset 抽样范围（0=全闭 .. 1.745=全开；中段出生由 bank 提供）
STAGED_GRIPPER_RESET_RANGE = (0.5, 1.6)


@configclass
class StagedActionsCfg:
    """相对关节位置动作；尺度对齐 lift 实测稳定值（防 PD 振铃）。"""

    arm_action: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=0.06,  # lift 验证 0.033~0.04；v0 的 0.15 会激励振铃
    )
    gripper_action: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=0.10,
    )


@configclass
class StagedObservationsCfg:
    """非对称 Actor-Critic 观测。

    policy（actor）：本体感受 + 相对目标（橘子/盒口悬停点相对口袋）
    + 盒局部放置几何 + 夹爪开度 + 物理里程碑标志（与阶段编号不同，评估
    时语义一致，无标签泄漏）。
    critic：policy 全部内容 + 特权（接触力/臂受力/盒姿态/阶段 one-hot）。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=staged_mdp.safe_joint_pos_rel)
        joint_vel = ObsTerm(func=staged_mdp.safe_joint_vel_rel, clip=(-5.0, 5.0))
        last_action = ObsTerm(func=staged_mdp.safe_last_action)
        features = ObsTerm(func=staged_mdp.staged_policy_features, clip=(-2.0, 2.0))

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 3

    @configclass
    class CriticCfg(PolicyCfg):
        privileged = ObsTerm(func=staged_mdp.staged_critic_extra)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 1  # 特权组不做历史堆叠

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class StagedEventsCfg:
    """startup 域随机化沿用 v0 的条目；reset 换成八阶段课程采样。

    事件顺序：关节复位在前（home 部分生效），StagedReset 在后（bank 行
    覆盖关节与物体 + 阶段种子），与参考项目一致。
    """

    # --- startup 域随机化（与 v0 相同的条目定义，此处显式复制以隔离改动）---
    orange_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("orange"),
            "static_friction_range": [1.0, 1.4],
            "dynamic_friction_range": [0.8, 1.1],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 64,
        },
    )
    box_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cardboard_box"),
            "static_friction_range": [0.5, 0.8],
            "dynamic_friction_range": [0.4, 0.6],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 64,
        },
    )
    box_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cardboard_box"),
            "mass_distribution_params": [0.14, 0.16],
            "operation": "abs",
        },
    )
    claw_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper", "moving_jaw_so101_v1"]),
            "static_friction_range": [0.8, 1.1],
            "dynamic_friction_range": [0.6, 0.9],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 64,
        },
    )
    joint_gains = EventTerm(
        func=randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": [0.9, 1.1],
            "damping_distribution_params": [0.9, 1.1],
            "operation": "scale",
        },
    )
    joint_friction = EventTerm(
        func=randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": [0.01, 0.05],
            "operation": "abs",
        },
    )

    # --- reset：关节复位（home 部分）在前，八阶段出生在后 ---
    reset_arm_joints = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": list((-0.10, 0.10)),
            "velocity_range": [0.0, 0.0],
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
        },
    )
    # 2026-09-23 关键修复：任何 write_joint_position（包括标准
    # reset_joints_by_offset）都会使 gripper 关节的隐式驱动在 PhysX GPU
    # 管线下永久失效。staged 任务已切 Newton 后端（write 后驱动存活），
    # 但仍保持不写爪关节的配置，使行为与历史 bank 口径一致；bank 出生由
    # StagedReset 写爪角并靠生成器的物理保持验证背书。
    reset_gripper = None
    reset_staged = EventTerm(
        func=staged_mdp.StagedReset,
        mode="reset",
        params={
            "dataset_path": "",
            "forced_stage": None,
        },
    )


@configclass
class PhysicsDREventsCfg(StagedEventsCfg):
    """Plan B 物理域随机化（TeacherFine-v2 / Camera-v10 共用）：执行器带逐局随机。

    参照 vs 现状 vs 差异理由（2026-10-08 sim2sim 混沌墙定案后立项，见
    mujoco_sim2sim/README.md「门窗口诊断轮」）：
    - 参照：IsaacLab lift_env_cfg.EventCfg 执行器增益 scale [0.5,2.0]；
      现状（StagedEventsCfg）gains/friction 仅 startup 期随机——每进程一采、
      所有局共享同一物理，等于没有执行器 DR。
    - 差异 1（startup→reset）：teacher_smooth_v1 对 Isaac 物理常数过拟合
      （爪峰过门余量仅 +0.0041 rad），换求解器即混沌分岔；逐局随机才能训出
      容忍带。真机迁移同理（真机动力学与仿真亦差 2-5%）。
    - 差异 2（区间加宽）：gains [0.9,1.1]→[0.75,1.25]（下界覆盖实测跨求解器
      kd 偏差 ×0.73~0.75）；friction [0.01,0.05]→[0.005,0.08]（abs）。首版
      保守（官方 lift 用 ±100%），razor-thin 任务先保成功率再逐步加宽。
    - 不动：材料/质量 DR 仍 startup（Newton 接触回调 MODEL_INIT 固定 ke/kd/mu
      与材料 DR 的覆盖顺序未验证，单变量纪律）；不加 push/armature（对
      锁存时序是二阶效应，第二版再议）。
    """

    joint_gains = EventTerm(
        func=randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": [0.75, 1.25],
            "damping_distribution_params": [0.75, 1.25],
            "operation": "scale",
        },
    )
    joint_friction = EventTerm(
        func=randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": [0.005, 0.08],
            "operation": "abs",
        },
    )


@configclass
class StagedRewardsCfg:
    """八阶段奖励表（weight × dt = 有效值；一次性项按 bonus 计）。

    稠密项每阶段一个（门控于当前阶段，有效值 ~0.2-1.0/步）；一次性阶段
    完成 bonus=(3,4,6,8,8,10,14,30)（weight×dt=1 → bonus 即有效值）与终
    局成功 +20；惩罚复用 v0 的扰动/碰撞项并新增过早松爪/工作空间违规/
    盒体受力（TRANSPORT 起，爪压盒壁的直接信号）。
    """

    # --- 八阶段稠密 ---
    home_approach = RewTerm(func=staged_mdp.safe_reward_home_approach, weight=0.5)
    pregrasp_distance = RewTerm(func=staged_mdp.safe_reward_pregrasp_distance, weight=0.6)
    grasp_contact = RewTerm(func=staged_mdp.safe_reward_grasp_contact, weight=0.8)
    # GRASP 段势能差塑形（教程同款）：合爪 0.2 rad/单位 + 贴近 8 mm/单位，
    # 一次完整闭合约 +2.2，拉动策略离开「开爪悬停」局部最优
    grasp_close = RewTerm(func=staged_mdp.SafeGraspClose, weight=8.0)
    # LIFT 段势能差塑形：每升高 1cm 付一单位（weight 6 → 一次完整举起
    # 10.5cm ≈ +2.1 总支付），悬停不付、下落倒扣；bump 降权为高度指示
    lift_rise = RewTerm(func=staged_mdp.SafeLiftRise, weight=6.0)
    lift_height = RewTerm(func=staged_mdp.safe_reward_lift_height, weight=0.2)
    stabilize_calm = RewTerm(func=staged_mdp.safe_reward_stabilize_calm, weight=0.6)
    transport_progress = RewTerm(func=staged_mdp.safe_reward_transport_progress, weight=0.6)
    place_descent = RewTerm(func=staged_mdp.safe_reward_place_descent, weight=0.6)
    # 松爪塑形（2026-09-27 改势能差）：旧状态积版在 RELEASE 段开爪悬停
    # 每步领分（实测 +0.65/局），与撤退竞争吸引子、策略钉在爪距 5.4cm 不
    # 撤离；新版只为「张开动作本身」付一次分，悬停零收入
    release_open = RewTerm(func=staged_mdp.SafeReleaseOpen, weight=0.8)
    # RELEASE 段撤退塑形（饱和势能差）：势能在 0.07（RETREAT_REWARD_TARGET_M）
    # 饱和且无 clamp——退到 0.07m 奖励拿满、之后零支付，往复严格
    # telescoping 净零；成功门另挂 0.055（stages.py），1.5cm 差值吸收 σ0.15
    # 噪声。权重 10→30（2026-09-27）：从放置姿态 5.4cm 撤到 7cm 原只值
    # ~+1 有效分拉不动策略（实测 400 采样钉 5.1-5.5cm 不撤）；×3 后 ≈+3.2，
    # 对标阶段 bonus 量级（3-30），每局上限仍有界（~14 units×30×dt）
    release_retreat = RewTerm(func=staged_mdp.SafeReleaseRetreat, weight=30.0)

    # --- 一次性：八阶段完成 + 终局成功 ---
    stage_complete = RewTerm(func=staged_mdp.SafeStageComplete, weight=30.0)
    final_success = RewTerm(func=staged_mdp.safe_reward_final_success, weight=600.0)  # +20 有效

    # --- 惩罚 ---
    box_disturbed = RewTerm(func=staged_mdp.safe_box_disturbed, weight=-0.25)
    box_disturbed_late = RewTerm(func=staged_mdp.safe_box_disturbed_late, weight=-0.1)
    orange_disturbed = RewTerm(func=staged_mdp.safe_orange_disturbed, weight=-0.5)
    arm_collision = RewTerm(func=staged_mdp.safe_arm_collision, weight=-3.0)
    # staged 版：近距门用口袋距（v0 的 dist_jaw 几何恒 9-12cm，真实夹持会被误罚）
    ee_collision = RewTerm(func=staged_mdp.safe_penalty_ee_bad_contact, weight=-2.0)
    # 爪/腕压盒壁（2026-09-26 引入，2026-09-27 改持续确认制）：TRANSPORT 起
    # 盒体净力 >2N 连续 3 步才罚——瞬时判罚会把求解器尖峰（探针实测静置
    # 中位 1.46N、尖峰 5.4N、>2N 占比 0.7-2.3%）和橘子落盒冲击当压壁。
    # 权重 -4→-2：-4/步量级过猛（det_align_v1 崩塌诱因之一），与
    # ee_collision 同档，真实压壁 10 步 ≈ -0.7 有效
    box_contact = RewTerm(func=staged_mdp.SafeBoxContact, weight=-2.0)
    premature_release = RewTerm(func=staged_mdp.safe_penalty_premature_release, weight=-1.0)
    workspace = RewTerm(func=staged_mdp.safe_penalty_workspace, weight=-0.5)
    # 抖动惩罚加倍（2026-09-26）：std 上界降到 0.15 后该项才真正开始约束
    # 策略输出的高频振荡（此前被 σ0.3 采样噪声淹没，确定性推理抖动现形）
    action_rate = RewTerm(func=action_rate_l2, weight=-0.01)

    orange_dropped = RewTerm(
        func=is_terminated_term, weight=-600.0, params={"term_keys": "staged_orange_dropped"}
    )
    box_ruined = RewTerm(
        func=is_terminated_term, weight=-360.0, params={"term_keys": "staged_box_tipped"}
    )
    abnormal_robot = RewTerm(
        func=is_terminated_term, weight=-360.0, params={"term_keys": "staged_abnormal_robot"}
    )


@configclass
class StagedTerminationsCfg:
    """独立可记录的终止原因：成功 / 掉落 / 盒毁 / 异常 / 超时。"""

    success = DoneTerm(func=staged_mdp.staged_success)
    staged_orange_dropped = DoneTerm(func=staged_mdp.staged_orange_dropped)
    staged_box_tipped = DoneTerm(func=staged_mdp.staged_box_tipped)
    staged_abnormal_robot = DoneTerm(func=staged_mdp.staged_abnormal_robot)
    time_out = DoneTerm(func=time_out, time_out=True)


@configclass
class StagedPickAppleEnvCfg(PickAppleEnvCfg):
    """八阶段版任务配置（继承 v0 的场景/DR/sim 参数）。"""

    actions: StagedActionsCfg = StagedActionsCfg()
    observations: StagedObservationsCfg = StagedObservationsCfg()
    events: StagedEventsCfg = StagedEventsCfg()
    rewards: StagedRewardsCfg = StagedRewardsCfg()
    terminations: StagedTerminationsCfg = StagedTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 物理后端切到 Newton/MJWarp（见 make_newton_physics 注释）
        self.sim.physics = make_newton_physics()
        # 默认挂 reset bank（文件缺失时 StagedReset 打 WARNING 回退 home-only）；
        # 生成器子类换了无 reset 事件的事件表，跳过挂载
        if hasattr(self.events, "reset_staged"):
            try:
                from pick_anything.paths import PICK_ORANGE_RESET_BANK

                self.events.reset_staged.params["dataset_path"] = str(PICK_ORANGE_RESET_BANK)
            except ImportError:
                pass

    def play_mode(self):
        """可视化/评估模式：强制全部环境从 HOME 出生（完整任务）。"""
        super().play_mode()
        # 完整任务评估的出生策略（用户核心指标）：home-only，不使用中段出生
        self.events.reset_staged.params["forced_stage"] = 0
        # RELEASE 成功门 0.055 已统一训练/部署口径（2026-09-27 双轨定版：
        # 成功门 0.055 + 撤退奖励饱和线 0.07，见 stages.py），无需再覆盖


@configclass
class GeneratorEventsCfg:
    """reset 生成器专用事件：仅 startup 域随机化，无 reset 事件。

    生成器完全通过显式 write 驱动场景（_write_robot/_write_orange/
    _commit_reset_state，对齐 IsaacLabTutorial 生成器），不经过
    ManagerBasedRLEnv 的 reset 事件链——避免 StagedReset 从旧 bank 采样
    污染生成分布。
    """

    orange_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("orange"),
            "static_friction_range": [1.0, 1.4],
            "dynamic_friction_range": [0.8, 1.1],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 64,
        },
    )
    box_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cardboard_box"),
            "static_friction_range": [0.5, 0.8],
            "dynamic_friction_range": [0.4, 0.6],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 64,
        },
    )
    box_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cardboard_box"),
            "mass_distribution_params": [0.14, 0.16],
            "operation": "abs",
        },
    )
    claw_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper", "moving_jaw_so101_v1"]),
            "static_friction_range": [0.8, 1.1],
            "dynamic_friction_range": [0.6, 0.9],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 64,
        },
    )


@configclass
class GeneratorObservationsCfg:
    """生成器最小观测（env 构造需要一个 obs 组；生成循环不消费它）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=joint_pos_rel)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class StagedPickGeneratorEnvCfg(StagedPickAppleEnvCfg):
    """reset 生成器专用 raw 环境（对齐教程 SO101VialGeneratorEnvCfg）。

    * 事件只保留 startup 域随机化（材质/质量桶），无 reset 事件——生成器
      显式 write 机器人/橘子/纸盒；
    * 无奖励与终止；
    * 观测退化为单个关节位置项（构造需要，不被消费）；
    * 场景/物理（Newton + 接触模型回调）与 staged 训练任务完全同源。
    """

    observations: GeneratorObservationsCfg = GeneratorObservationsCfg()
    events: GeneratorEventsCfg = GeneratorEventsCfg()
    rewards = None
    terminations = None

    def __post_init__(self):
        super().__post_init__()
        # 生成器不需要 episode 计时
        self.episode_length_s = 60.0
