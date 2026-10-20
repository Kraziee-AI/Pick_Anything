"""蒸馏版第三版环境配置（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v0）。

teacher-student 蒸馏（rsl_rl ``DistillationRunner``，IsaacLab 教程 anymal_d
``rsl_rl_distillation_cfg`` 同款机制）：**student 在环境里执行动作，加载自
状态版 checkpoint 的 teacher 对每个到访状态给出动作标签**（在线 DAgger，
``rsl_rl/algorithms/distillation.py`` 的 ``act()`` 语义）。reset bank 课程
照常从 8 个阶段出生，student 早期乱走也不丢中后段状态覆盖。

观测组与 camera 版的差异仅一处：critic 组换成 teacher 组——蒸馏无 critic，
teacher 直接消费 staged 任务的 policy 组（111 维含 history=3，逐字复用，
与状态版训练分布完全一致，strict 加载的 MLP teacher 专用）。阶段状态机/
reset 课程/奖励/终止/腕摄 scene（近裁剪面覆盖）全部继承 camera 版。
"""

from __future__ import annotations

from isaaclab.utils import configclass

from pick_anything.tasks.pick_apple.camera_env_cfg import (
    CameraClosureHdSceneCfg,
    CameraProprioceptionCfg,
    CameraProprioceptionV2Cfg,
    CameraStagedHdSceneCfg,
    CameraStagedPickAppleEnvCfg,
    HOME_BOX_WINDOW,
    HOME_ORANGE_WINDOW,
    SCAN_ORANGE_WINDOW,
    WristImageCfg,
    scan_home_closure_robot_cfg,
    scan_home_robot_cfg,
)
from pick_anything.tasks.pick_apple.staged_env_cfg import StagedObservationsCfg


@configclass
class DistillObservationsCfg:
    """student（腕部 RGB + 本体感受）与 teacher（staged policy 组）两套输入。"""

    wrist_rgb: WristImageCfg = WristImageCfg()
    # (N, 3, 48, 64)，episode 一致光度 DR 照旧；组名保持 wrist_rgb——
    # camera_mdp.CameraImage 按该组名查 enable_corruption
    proprioception: CameraProprioceptionCfg = CameraProprioceptionCfg()
    teacher: StagedObservationsCfg.PolicyCfg = StagedObservationsCfg.PolicyCfg()
    # 逐字复用：joint_pos/joint_vel/last_action/features + history=3 → 111 维，
    # 与 Pick-Apple-SO101-Staged-v0 的 actor 输入完全同构（teacher strict load 前提）


@configclass
class CameraDistillPickAppleEnvCfg(CameraStagedPickAppleEnvCfg):
    """蒸馏版：继承相机版（scene/课程 late 档/play_mode/物理），仅换观测组。"""

    observations: DistillObservationsCfg = DistillObservationsCfg()


@configclass
class DistillV1ObservationsCfg:
    """v1 蒸馏观测：student 本体补 joint_target（教程四项口径，24 维）。"""

    wrist_rgb: WristImageCfg = WristImageCfg()
    proprioception: CameraProprioceptionV2Cfg = CameraProprioceptionV2Cfg()
    # (N, 24)：pos/vel/target/last_action——软伺服下 pos 恒滞后 target，
    # 夹持时策略只有看到 PD 目标才知道自己命令的闭合度（教程
    # ProprioceptionCfg 四项；v0 缺 target）
    teacher: StagedObservationsCfg.PolicyCfg = StagedObservationsCfg.PolicyCfg()
    # teacher 组不变：111 维含 history=3，与状态版 actor 输入同构（strict load）


@configclass
class CameraDistillV1PickAppleEnvCfg(CameraStagedPickAppleEnvCfg):
    """蒸馏 v1：v0 + HOME 可见窗口（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v1）。

    2026-09-30 distill_v0 复盘（17249 轮平台 Loss≈0.17 / success≈0.51 /
    home_born_success≈0.17）：HOME 出生橘子落在腕摄视野外的自然刷位带，
    student 前段盲搜是结构性信息瓶颈。挂 home_orange_window（camera v1
    同款，cam_v3 实测可见带）把 HOME→PREGRASP 变成可学的视觉伺服。

    动作粒度（0.06/0.10）、episode 60s、late 课程全部保持 v0——teacher
    （状态线 model_1750）在这些语义下训练，不能动；故**不**继承
    CameraVisibleHomeEnvCfg（它连带 0.033/0.02 + 90s）。teacher 对窗口带
    （y∈[-0.34,-0.302]，略近于自然刷位）属轻微外推，状态策略直接读橘子
    位姿，风险低。
    """

    observations: DistillV1ObservationsCfg = DistillV1ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_staged.params["home_orange_window"] = HOME_ORANGE_WINDOW


@configclass
class CameraDistillV2PickAppleEnvCfg(CameraDistillV1PickAppleEnvCfg):
    """蒸馏 v2：细粒度教师兼容语义 + 教程相位均匀 reset（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v2）。

    2026-10-01 distill_v1 平台期复盘（9250 轮 Loss≈0.047 冻结 / success≈0.50 /
    home_born≈0.019 / PREGRASP≈0.18）：agent 侧五项已与 IsaacLabTutorial
    逐字对齐仍平台，瓶颈是教师 model_1750 的粗动作粒度（臂 0.06/夹爪 0.10，
    夹爪 ~5 步全闭≈开关）在 64×48 像素输入下不可模仿，且窗口带在旧教师
    训练分布之外（标签 OOD）。cam_v5（细粒度 0.033/0.02 相机 PPO）同期
    home_born 0.37 且陡峭爬升证明该语义下视觉可学。故 v2 配套新细粒度
    教师（StagedFineTeacherEnvCfg，--checkpoint 必须指向其 model，旧
    model_1750 动作尺度不兼容）：

    * 动作粒度 0.033/0.02、episode 90s：与相机 v1/v2 线及新教师完全一致；
    * 相位均匀 reset（各 1/8，教程蒸馏核心设计）：birth 即覆盖全任务、
      教师标签处处有效；v1 的 CAMERA_CURRICULUM_LATE home 0.30 把 30%
      预算泡在学不动的盲搜段。早/中/晚三档全置均匀——90s episode 下
      reset 计数涨得慢，只改 late 档则近万轮泡在 early/mid 的 home 加权里。

    student 观测/DR/奖励表全部继承不动（蒸馏是行为克隆，奖励不参与）。
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.scale = 0.033
        self.actions.gripper_action.scale = 0.02
        self.episode_length_s = 90.0
        uniform = (0.125,) * 8
        self.events.reset_staged.params["curriculum_early"] = uniform
        self.events.reset_staged.params["curriculum_mid"] = uniform
        self.events.reset_staged.params["curriculum_late"] = uniform


@configclass
class CameraDistillV3PickAppleEnvCfg(CameraDistillV2PickAppleEnvCfg):
    """蒸馏 v3：v2 + 腕摄 128×96（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v3）。

    2026-10-02 distill_v2 终局复盘（教师 100% / Loss 与教程同水位 0.05 /
    晚段出生 1.0 / **HOME 18909 局 0 成功、确定性 0/256**）→ 逐项对照
    IsaacLabTutorial place_vial 定因，教程成功的条件是任务信息几何而非
    蒸馏配方（相机/动作/超参我们已逐字对齐）：

    * 教程物体：vial 躺平细长圆柱 ~2.5cm×10cm → 64×48 里几十像素长条，
      位置+朝向都可读；HOME 出生仅 5×8cm 小框（reset_poses.pt 实测
      x∈[0.201,0.254] y∈[-0.056,0.023]）；episode 20s（600 步）。
    * 我们物体：45mm 球，HOME→橘子 P50 22cm → 64×48 里 ~8 像素圆斑；
      出生散布 22×3.8cm 均匀带；episode 90s（2700 步）。

    → 同分辨率下学生「看不清」，每步 ~0.007 rad 的 BC 误差在 500+ 步
    接近+87 步合爪的长链里无法闭环修正（cam_v5 RL 能学是"往大方向凑"
    不需要精度，BC 学的是教师的精确伺服）。v3 唯一改动 = 分辨率 ×2
    （橘子面积 ×4），单变量归因；CNN/MLP 维度由 rsl_rl 按输入自适应。
    代价：渲染像素 ×4，采集变慢（预计 2-3×）。
    """

    scene: CameraStagedHdSceneCfg = CameraStagedHdSceneCfg(num_envs=1024)


@configclass
class CameraDistillV4PickAppleEnvCfg(CameraDistillV3PickAppleEnvCfg):
    """蒸馏 v4：v3 + 高位扫视 HOME 姿态 + 盒子可见窗口（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v4）。

    2026-10-02 用户看出生画面定方向：「初始画面要有橘子**和盒子**（盒子可
    部分）」——教程 place_vial 出生即见 vial+rack，而我们旧 HOME 只见桌沿
    窄带、盒子完全不可见（学生开局不知道盒子方向，搬运段只能盲猜）。v4
    在 v3（128×96）之上两件套：

    * HOME 姿态换高位扫视（SCAN_HOME_ARM_JOINTS；2026-10-02 第二轮从 P7
      换成 workshop roll=-0.804 族 O 姿态——实测 P7 roll=0 俯角 47° 近平视、
      盒带正在相机正下方只露墙线；教程 HOME 几何=24cm 高+俯角 29°，
      probe_camera_geometry 网格实测选型，photos/cam_geometry/）；
    * HOME 出生盒子刷进 HOME_BOX_WINDOW（相机正下方之外的深区、桌沿合法、
      与橘子窗口 13.5cm 间隙由 x 向间隔 + StagedReset 单调 y-推保证）。

    **--checkpoint 必须指向新姿态下重训的教师**（TeacherFine-v0 已同步换
    姿态/窗口，teacher_fine_v5）——旧 model_4250 是旧 HOME 姿态分布，动作
    标签不匹配新出生状态。bank 中后段行、奖励表、课程、其余全部继承 v3。
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = scan_home_robot_cfg()
        # 扫视姿态配套：橘子窗口收右（旧左缘在该姿态出上框）+ 盒子右侧窗口
        self.events.reset_staged.params["home_orange_window"] = SCAN_ORANGE_WINDOW
        self.events.reset_staged.params["home_box_window"] = HOME_BOX_WINDOW
        # HOME 关节噪声 ±0.1→±0.05：±0.1 的 pan 抖动=视野横移 ±3cm，把贴缘
        # 盒子摇出去一半出生（官方出生=固定舞台无关节噪声，这是它「每次
        # 都露一半」的结构性原因之一）。中后段 bank 行不受影响
        self.events.reset_arm_joints.params["position_range"] = [-0.05, 0.05]


# --------------------------------------------------------------------- #
# v5：episode 90s→20s（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v5）。
# 2026-10-02 distill_v4 中途复盘（3172 轮）：Loss/behavior 0.031 已低于
# IsaacLabTutorial 蒸馏终值 0.046（算法侧逐项核对全同），但 HOME 出生恒 0、
# GRASP-born 19% 不涨——与教程未对齐的最后一项信息几何是 episode 长度。
# 教程三条线（teacher/蒸馏/直训）全部 20s，收敛局均值 128 步；我们的细粒度
# 教师完成任务均值仅 121 步（HOME 链约 300-400 步），失败局却在 90s/2700 步
# 帽下游荡到死，烧掉绝大部分采样预算。砍到 20s（600 步）：失败局代价 ÷4.5，
# 同墙钟 DAgger 出生状态覆盖 ×4.5，掐掉长尾复合误差链。teacher 沿用
# model_4250（O 姿态实测 96.1%，v4 docstring「必须重训教师」已被实测推翻）。
# 阶段机/确认步数全是控制步、与局长无关（StageProgress.update 的 step 仅记
# time_to_success）；课程三档均匀 1/8 不变。后手：time_out 异常压 success → 30s。
# --------------------------------------------------------------------- #


@configclass
class CameraDistillV5PickAppleEnvCfg(CameraDistillV4PickAppleEnvCfg):
    """蒸馏 v5：v4 + episode 90s→20s（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v5）。

    单变量 = episode 长度；其余（128×96 分辨率、O 姿态、双窗口、均匀课程）
    全部继承 v4。
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0


# --------------------------------------------------------------------- #
# v6：合爪盲区修复相机（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v6）。
# 2026-10-03 根因定案（详见 camera_env_cfg.CameraClosureHdSceneCfg docstring）：
# 合爪瞬间橘子出画（PREGRASP 时钉底缘 2/6 不可见、GRASP 右下双缘裁切）——
# 全部 run「PREGRASP→GRASP ≤18%」的结构性死点，episode 长度/分辨率/算法超参
# 均治不了（v5 与 v4 完全同轨、GRASP-born 19% 纹丝不动）。v6 = v5 + 俯角
# -12°+FOV×1.5 相机（教师读状态不受影响，仍用 model_4250）；其余全继承。
# --------------------------------------------------------------------- #


@configclass
class CameraDistillV6PickAppleEnvCfg(CameraDistillV5PickAppleEnvCfg):
    """蒸馏 v6：v5 + 合爪盲区修复相机（gym id: Pick-Apple-SO101-Staged-Camera-Distill-v6）。"""

    scene: CameraClosureHdSceneCfg = CameraClosureHdSceneCfg(num_envs=1024)

    def __post_init__(self):
        super().__post_init__()
        # super 链（V4.__post_init__）会把 robot 换成旧相机的
        # scan_home_robot_cfg()——必须在 super 之后换成 closure 相机版
        self.scene.robot = scan_home_closure_robot_cfg()


# --------------------------------------------------------------------- #
# v7：混合蒸馏口径 episode 20s→90s（gym id
# Pick-Apple-SO101-Staged-Camera-Distill-v7）。student 改为 cam_v9
# （Camera-v7 任务）热启动（agents/camera_distill_cfg 的
# CameraDistillV7RunnerCfg）：其原生 episode=90s，HOME 链 300-400 步 +
# 细粒度合爪 ~87 步在 20s/600 步帽下大量 timeout——直训线 Camera-v6→v7
# 已实测该帽勒死学习者（time_out 72→88%、连 LIFT 出生都 0）；v5 的 20s
# 实验「与 v4 无差异」只说明从零 BC 当时什么都学不动，不构成 20s 无害的
# 证据。教师完成均值 121 步不受影响；闭包相机/O 姿态/双窗口/均匀课程/
# 128×96/0.033/0.02 全部继承 v6 不动。
# --------------------------------------------------------------------- #


@configclass
class CameraDistillV7PickAppleEnvCfg(CameraDistillV6PickAppleEnvCfg):
    """蒸馏 v7：v6 + 热启动 student 的原生 episode 口径 90s（gym id: ...-v7）。"""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 90.0


# --------------------------------------------------------------------- #
# v8：HOME 加权固定课程（gym id Pick-Apple-SO101-Staged-Camera-Distill-v8）。
# 2026-10-06 终局复盘后的「对照官方找差异」迭代：官方蒸馏均匀 8 相位的
# 前提是学生全段都弱；我们学生中后段出生成功率已 0.99（GRASP）/0.91（LIFT），
# 唯一短板是 HOME 全链（随机口径 0.115 卡死 = DAgger 全链标签稀疏的不动点）。
# 对官方「按需分配采样」精神的忠实适配 = 课程三档钉死 home 0.40（表取
# CAMERA_CURRICULUM_LATE_V2 同款，cam 线 v8 已验证固定表消除课程重爬瞬态），
# HOME 链 DAgger 数据 ×3.2。其余（90s/闭包相机/128×96/均匀动作语义）继承 v7。
# --------------------------------------------------------------------- #


@configclass
class CameraDistillV8PickAppleEnvCfg(CameraDistillV7PickAppleEnvCfg):
    """蒸馏 v8：v7 + HOME 加权固定课程（gym id: ...-v8）。"""

    def __post_init__(self):
        super().__post_init__()
        heavy = (0.40, 0.18, 0.16, 0.10, 0.04, 0.04, 0.04, 0.04)
        self.events.reset_staged.params["curriculum_early"] = heavy
        self.events.reset_staged.params["curriculum_mid"] = heavy
        self.events.reset_staged.params["curriculum_late"] = heavy


# --------------------------------------------------------------------- #
# v9：纯蒸馏·官方式从零 + 分辨率 256×192（gym id
# Pick-Apple-SO101-Staged-Camera-Distill-v9）。2026-10-07 用户指令「把纯
# 蒸馏线训练出来，对比官方」——官方差异终表剩余可行动项 = 感知误差 vs
# 任务容差：128×96 下橘子位置误差 ~2-3mm/px ≈ 合爪容差本身（PREGRASP→
# GRASP 0.386 ≈ cam_v9 的 0.357 = 图像信息地板的铁证）；256×192 把误差
# 压到 ~1-1.5mm/px，低于容差——让「学生能大致执行教师动作」（官方成功的
# 前提）成立的最直接手段。教师与分辨率无关（状态 MLP，平滑版 model_2250
# 确定性 100% 已就绪）；**从零初始化**（官方口径，无热启动=无漂移问题，
# agent 侧用 v1 的 BoundedTeacherDistillation 而非 WarmStart 变体）。
# 若 v9 仍卡死 → 预登记下一杠杆：噪声特权教师（教师训练时橘子位姿加噪，
# 学出"可见变量可执行"的宽容策略——官方教师天然如此）。其余继承 v7
# （90s/闭包相机/均匀 8 相位/0.033/0.02）。
# --------------------------------------------------------------------- #


@configclass
class CameraDistillV9PickAppleEnvCfg(CameraDistillV7PickAppleEnvCfg):
    """蒸馏 v9：v7 语义 + 腕摄 256×192（gym id: ...-v9）。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.wrist_camera.width = 256
        self.scene.wrist_camera.height = 192


@configclass
class CameraDistillUnclippedPickAppleEnvCfg(CameraDistillPickAppleEnvCfg):
    """Coarse state teacher: original 0.06/0.10 control, camera + 24-D proprio.

    Use with retreat_shaping_v2/model_12999.pt and the unclipped runner.
    Preserve base HOME/reset geometry; do not introduce the fine teacher's
    0.033/0.02 action scales or scan HOME windows.
    """

    observations: DistillV1ObservationsCfg = DistillV1ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.wrist_camera.width = 256
        self.scene.wrist_camera.height = 192
