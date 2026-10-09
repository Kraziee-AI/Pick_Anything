"""腕部相机版八阶段任务的环境配置（gym id: Pick-Apple-SO101-Staged-Camera-v0）。

与 ``StagedPickAppleEnvCfg``（Pick-Apple-SO101-Staged-v0）的唯一区别是
actor 的输入：腕部相机 RGB + 本体感受替换「相对目标特征 + 里程碑」。
八阶段状态机、reset 课程（StagedReset + bank）、奖励表、终止条件、
Newton 物理全部原样继承——阶段推进是环境侧逻辑，不依赖 actor 观测。
critic 保留全部特权观测（非对称 AC，结构对齐 IsaacLabTutorial 的
``SO101VialCameraEnvCfg``）。

相机 prim 是 SO101 USD 自带的 ``gripper/wowrobo_2MP_camera``（实机同款
2MP 腕摄）；spawn 时把 clippingRange 压到 (0.001, 5.0)——默认近裁剪面
会把爪下的橘子裁出画面（教程实测必改项）。64×48 分辨率 / 30Hz 更新
（控制 30Hz，每步一张新图），与教程视觉任务完全同参。
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from pick_anything.robots import SO101_FOLLOWER_CFG
from pick_anything.tasks.pick_apple import camera_mdp, staged_mdp
from pick_anything.tasks.pick_apple.pick_env_cfg import PickRLSceneCfg
from pick_anything.tasks.pick_apple.staged_env_cfg import (
    PhysicsDREventsCfg,
    StagedObservationsCfg,
    StagedPickAppleEnvCfg,
    StagedRewardsCfg,
)


# --------------------------------------------------------------------- #
# 带相机裁剪面覆盖的 SO101 spawn（教程 _spawn_so101_with_camera_overrides 同款）
# --------------------------------------------------------------------- #


def _apply_camera_clipping_range(
    stage: Any,
    robot_prim_path: str,
    pitch_deg: float = 0.0,
    fov_scale: float = 1.0,
) -> None:
    """把腕部相机近裁剪面压到 1mm，爪下目标不被默认裁剪面裁掉。

    可选外科参数（默认关闭 = 历史版本逐字一致）：

    * ``pitch_deg``：相机局部 X 轴俯角（度）。相机 prim 用 ``xformOp:orient``
      四元数——``XformCommonAPI.SetRotate`` 与 orient 不兼容（实测返回 False
      静默失败），须把俯角四元数右乘进现有 orient（局部系旋转）。USD 相机
      朝局部 -Z、绕 +X 正角 = 视轴上抬，故「往下看」取负角。
    * ``fov_scale``：FOV 放大倍数（focalLength 等比缩小）。

    spawn 时写 USD 属性，先于 sensor init 缓存内参（clippingRange 同机制，
    实测有效；运行时改无效）。定版值见 CLOSURE_VIEW_* 常量。
    """
    import math

    from pxr import Gf, UsdGeom  # 延迟导入：模块加载不依赖 pxr
    from pxr.UsdGeom import Camera as UsdGeomCamera

    camera = stage.GetPrimAtPath(f"{robot_prim_path}/gripper/wowrobo_2MP_camera")
    camera.GetAttribute("clippingRange").Set(Gf.Vec2f(0.001, 5.0))

    if pitch_deg != 0.0:
        pitch_q = Gf.Quatf(
            math.cos(math.radians(pitch_deg) / 2.0),
            Gf.Vec3f(math.sin(math.radians(pitch_deg) / 2.0), 0.0, 0.0),
        )
        for op in UsdGeom.Xformable(camera).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set((op.Get() * pitch_q).Normalize())
                break
    if fov_scale != 1.0:
        usd_cam = UsdGeomCamera(camera)
        focal = usd_cam.GetFocalLengthAttr().Get()
        if focal:
            usd_cam.GetFocalLengthAttr().Set(focal / fov_scale)
    if pitch_deg != 0.0 or fov_scale != 1.0:
        print(f"[CAM-CLOSURE] {camera.GetPath()} pitch={pitch_deg}deg fov_scale={fov_scale}", flush=True)


# 合爪盲区修复定版参数（2026-10-03，渲染帧实测选型，详见
# CameraClosureHdSceneCfg docstring）
CLOSURE_VIEW_PITCH_DEG = -12.0
CLOSURE_VIEW_FOV_SCALE = 1.5


@clone
def _spawn_so101_with_closure_camera(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    prim = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    _apply_camera_clipping_range(
        prim.GetStage(), prim_path,
        pitch_deg=CLOSURE_VIEW_PITCH_DEG, fov_scale=CLOSURE_VIEW_FOV_SCALE,
    )
    return prim


CLOSURE_CAMERA_SO101_CFG = SO101_FOLLOWER_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=SO101_FOLLOWER_CFG.spawn.replace(func=_spawn_so101_with_closure_camera),
)


def scan_home_closure_robot_cfg():
    """O 姿态 HOME 关节角 + 合爪盲区修复相机的机器人 cfg 副本。"""
    return CLOSURE_CAMERA_SO101_CFG.replace(
        init_state=CLOSURE_CAMERA_SO101_CFG.init_state.replace(joint_pos=dict(SCAN_HOME_ARM_JOINTS))
    )


@clone
def _spawn_so101_with_camera_overrides(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    prim = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    _apply_camera_clipping_range(prim.GetStage(), prim_path)
    return prim


CAMERA_SO101_CFG = SO101_FOLLOWER_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=SO101_FOLLOWER_CFG.spawn.replace(func=_spawn_so101_with_camera_overrides),
)


@configclass
class CameraStagedSceneCfg(PickRLSceneCfg):
    """staged 场景 + SO101 USD 自带的腕部相机。

    num_envs 默认降到 1024（渲染开销；教程同规模），CLI ``--num_envs``
    仍可覆盖。物理复制/env 间距沿用 staged 口径。
    """

    robot = CAMERA_SO101_CFG

    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera",
        spawn=None,  # 相机 prim 由机器人 USD 自带，只读不生成
        data_types=["rgb"],
        width=64,
        height=48,
        update_period=1.0 / 30.0,
        update_latest_camera_pose=True,
        renderer_cfg=MultiBackendRendererCfg(),
    )


@configclass
class CameraStagedHdSceneCfg(CameraStagedSceneCfg):
    """v3 蒸馏专用：腕摄 128×96（HD）。

    2026-10-02 distill_v2 失败的逐项教程对照结论（详见
    CameraDistillV3PickAppleEnvCfg docstring）：教程 place_vial 的 vial
    躺平 ~10cm 长（视觉签名几十像素）且 HOME 出生仅 5×8cm 小框、20s 短
    局；我们的橘子 45mm 球在 22cm 外只有 ~8 像素、出生散布 22×3.8cm、
    90s 长局——同款 64×48 下学生看不清，长链模仿误差无法闭环修正。
    分辨率 ×2 是「物体更小更远更散」的按比例补偿，其余与教程逐字一致
    （单变量归因）。CNN/MLP 维度由 rsl_rl 按输入自适应，无需改网络配置。
    """

    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera",
        spawn=None,
        data_types=["rgb"],
        width=128,
        height=96,
        update_period=1.0 / 30.0,
        update_latest_camera_pose=True,
        renderer_cfg=MultiBackendRendererCfg(),
    )


@configclass
class CameraClosureHdSceneCfg(CameraStagedSceneCfg):
    """合爪盲区修复相机（俯角 -12° + FOV ×1.5）+ 腕摄 128×96。

    2026-10-03 根因定案：渲染帧实测（view_spawn_camera --stage 0/1/2 + HSV
    橙色像素定量 + verify_box_diff 差分）发现**合爪瞬间橘子出画**——原装
    相机朝前看，橘子越近在画面里越往下沉，PREGRASP（口袋距橘子 3.5cm）时
    橘子钉在画面底缘（v≈0.97，2/6 出生完全不可见）、GRASP 时右下角双缘
    裁切。这正是全部 13 个 run「PREGRASP→GRASP 出生成功率 ≤18%」共同死点
    的结构性原因（教师 96-100% 因读坐标不靠眼）。教程 place_vial 同一颗
    相机不瞎是因 vial 为 10cm 长条、近端出画远端仍在。

    修复（spawn 时 USD 覆盖，机制同 clippingRange）：俯角 -12°（orient
    四元数右乘）+ FOV ×1.5（focalLength 13.6→9.067，HFOV 75°→~99°）。
    验证（128×96 定量）：HOME 橘子 v≈0.14（131-176px，顶部裕量 8px）+
    盒子差分 8/8 可见均值 645px（盒心正中，比修复前 498px 更好）；PREGRASP
    4/4 可见 v≈0.76-0.81（底缘裕量 5-13px）；GRASP 完整在画内居中偏右
    （u≈0.68/v≈0.67，339-489px）。FOV 放大的信息代价：HOME 橘子面积
    280-360→131-176px，分辨率必须 128×96 起（64×48 下只剩 ~6px 直径）。
    """

    robot = CLOSURE_CAMERA_SO101_CFG

    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera",
        spawn=None,
        data_types=["rgb"],
        width=128,
        height=96,
        update_period=1.0 / 30.0,
        update_latest_camera_pose=True,
        renderer_cfg=MultiBackendRendererCfg(),
    )


@configclass
class WristImageCfg(ObsGroup):
    """腕部 RGB 观测（episode 一致光度域随机化 + 传感器噪声）。

    单 term、不拼接：观测管理器输出 (N, 3, H, W)，rsl_rl CNNModel 按
    维度识别为 2D 组路由进卷积编码器。
    """

    image = ObsTerm(
        func=camera_mdp.CameraImage,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "exposure_range": (0.75, 1.25),
            "contrast_range": (0.85, 1.15),
            "white_balance_range": (0.90, 1.10),
            "brightness_range": (-0.05, 0.05),
        },
        noise=UniformNoiseCfg(n_min=-0.025, n_max=0.025),
    )

    def __post_init__(self):
        self.enable_corruption = True


@configclass
class CameraProprioceptionCfg(ObsGroup):
    """实机可用的本体感受（复用 staged_mdp 的防 NaN 包装）。

    噪声幅度取教程实机口径（编码器 ±0.01 rad / 测速 ±0.02）；不含
    相对目标特征与里程碑标志——那些在实机上不可得，特权信息全部
    留给 critic。
    """

    joint_pos = ObsTerm(
        func=staged_mdp.safe_joint_pos_rel,
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=staged_mdp.safe_joint_vel_rel,
        clip=(-5.0, 5.0),
        noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
    )
    last_action = ObsTerm(func=staged_mdp.safe_last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CameraObservationsCfg:
    """可部署 actor 输入（图像 + 本体感受）+ 特权 critic 状态。"""

    wrist_rgb: WristImageCfg = WristImageCfg()
    proprioception: CameraProprioceptionCfg = CameraProprioceptionCfg()
    critic: StagedObservationsCfg.CriticCfg = StagedObservationsCfg.CriticCfg()


# 相机版 late 档课程（2026-09-28 cam_v1 it6445 复盘：birth_HOME_success 恒
# 0% 而 home 出生占 50% reset 纯耗算力；PREGRASP→GRASP 条件通过率 ~11% 是
# 最大断层。home 降到 0.30，权重移给 PREGRASP/GRASP/LIFT 与两个薄弱收尾
# 段。仅覆盖 late 档，early/mid 沿用 staged_mdp 共享表；非相机 staged 任务
# 不受影响）
CAMERA_CURRICULUM_LATE = (0.30, 0.18, 0.18, 0.10, 0.04, 0.04, 0.08, 0.08)


@configclass
class CameraStagedPickAppleEnvCfg(StagedPickAppleEnvCfg):
    """腕部相机版八阶段任务（阶段/奖励/reset 课程/物理全部继承 staged）。"""

    scene: CameraStagedSceneCfg = CameraStagedSceneCfg(num_envs=1024)
    observations: CameraObservationsCfg = CameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # late 档断层带加权仅作用于相机版（staged_mdp.curriculum_weights 的
        # 任务级覆盖入口；forced_stage/评估口径不受影响）
        self.events.reset_staged.params["curriculum_late"] = CAMERA_CURRICULUM_LATE

    def play_mode(self):
        """可视化/评估模式：沿用 staged 的 home-only 完整任务口径。"""
        super().play_mode()
        # 图像策略逐环境渲染，评估/查看压到 8 个环境（教程同款）
        self.scene.num_envs = min(self.scene.num_envs, 8)


# HOME 行橘子刷位窗口（cam_v3 可见性修复）。2026-09-28 探针实测
# （scripts/measure_home_camera_window.py，3 轮关节噪声交集的稳定可见带，
# 拼图存 photos/home_window/）：HOME 位姿下腕摄只在桌面远端看到这条
# ~22 x 3.8 cm 横带（相机位低、足迹小；自然刷位 y∈[-0.52,-0.34] 恰好在带
# 外——这就是 cam_v2 33.5 万次 HOME 出生 0 成功的几何根源）。x_hi 由 0.49
# 收到 0.46：实测 0.46-0.48 处橘子只挂中心区边缘（px 7-8），砍掉保稳定。
# 盒带顶 y=-0.39 距窗口底 5cm < 间隙 13.5cm，StagedReset 里推盒子保橘子。
HOME_ORANGE_WINDOW = (0.240, 0.460, -0.340, -0.302)

# 蒸馏 v4 的 HOME 高位扫视姿态 + 盒子可见窗口（2026-10-02 用户定方向：
# 出生画面橘子+盒子同框，对齐教程「出生即见任务目标」）。
#
# 2026-10-02 第二轮重做（用户反馈 P7 视角「离桌子太近、视野小」，
# scripts/probe_camera_geometry.py 网格实测，photos/cam_geometry/）：
# * 参照（教程 HOME = workshop 姿态）：相机 24.1cm 高、俯角 29°——腕 roll
#   -0.804 把 USD 相机挂载的 -35° 偏转收成陡峭俯视；FOV 双方同一 USD
#   相机 prim（HFOV 75.2°/VFOV 58.7°），无差异可抄。
# * 现状（P7：roll=0）：相机 22.0cm、俯角 47° 近乎平视；且盒带
#   y∈[-0.50,-0.44] 正在相机（y≈-0.45）正下方 → 盒子永远贴底缘、掠射角
#   只剩几条墙线（差分铁证），橘盒同列但盒子视觉上「只露一角」。
# * 差异理由：roll=0 时「相机高」与「看得陡」是运动学矛盾（实测前沿：
#   最陡 36° 时相机仅 15.8cm），roll=-0.804 族轻松 23cm+24°——即教程
#   HOME 几何。故 HOME 回到 workshop 臂形（roll=-0.804），pan 收到
#   -0.05、wf 提到 1.58 把光轴对准橘盒中点 (0.38,-0.40)；窗口同步挪
#   出相机正下方（盒窗往深处/右侧），13.5cm 间隙由 x 向间隔保证
#   （dx≥0.115 → dy_need≤0.07，窗内恒可解，不再依赖单调推深的钳位）。
SCAN_HOME_ARM_JOINTS = {
    "shoulder_pan": -0.05,
    "shoulder_lift": -0.907,
    "elbow_flex": 0.190,
    "wrist_flex": 1.580,
    "wrist_roll": -0.804,
    "gripper": 0.2555162024919699,
}
# 扫视姿态下的橘子窗口（左收为盒窗让出 x 向间隙：x_hi 0.38→0.32；O 姿态
# 实拍窗口四角均在画面上中部，u≈50-70 v≈20-35）。仅 v4/教师用，不动共享
# 的 HOME_ORANGE_WINDOW（相机 v1/v2 线的旧姿态口径）。
SCAN_ORANGE_WINDOW = (0.290, 0.320, -0.380, -0.340)
# 盒子中心窗口（x_lo, x_hi, y_lo, y_hi）。2026-10-02 第二轮：从相机正下方
# (0.40-0.48, -0.50..-0.44) 挪到深区 (0.44-0.49, -0.45..-0.40)——实测深区
# 盒体以实体面块出镜（差分 662px@(0.44,-0.40)），浅区只剩掠射墙线。
# 间隙可解性：橘窗 x≤0.32 vs 盒窗 x≥0.44 → dx≥0.115 → dy_need≤0.070，
# 橘窗底 -0.38 - 0.07 = -0.45 = 盒窗 y_lo，单调推深恒在窗内不收钳。
# 硬约束：v0 盒矩形 x[0.14,0.56] y[-0.51,-0.39]（fail-fast 校验求交）。
HOME_BOX_WINDOW = (0.440, 0.490, -0.450, -0.400)


def scan_home_robot_cfg():
    """返回换 P7 扫视 HOME 姿态的相机机器人 cfg 副本（不改共享实例）。"""
    return CAMERA_SO101_CFG.replace(
        init_state=CAMERA_SO101_CFG.init_state.replace(joint_pos=dict(SCAN_HOME_ARM_JOINTS))
    )


@configclass
class CameraProprioceptionV2Cfg(CameraProprioceptionCfg):
    """v1 任务专用本体组：补第 4 项 joint_target（教程口径）。

    软伺服 + delta 动作下，接触反力使 pos 恒滞后/偏离 target——夹持时
    策略只有看到 PD 目标才知道自己命令的闭合度（教程 ProprioceptionCfg
    四项：pos/vel/target/last_action；我们移植时丢了 target，cam_v4 补齐，
    噪声 ±0.005 同教程）。不动 CameraProprioceptionCfg——蒸馏 student 的
    输入维度保持不变。
    """

    joint_target = ObsTerm(
        func=staged_mdp.safe_joint_target,
        noise=UniformNoiseCfg(n_min=-0.005, n_max=0.005),
    )


@configclass
class CameraVisibleHomeEnvCfg(CameraStagedPickAppleEnvCfg):
    """v1：HOME 出生橘子刷进腕摄可见窗口（gym id: Pick-Apple-SO101-Staged-Camera-v1）。

    仅比 v0 多一个 reset 参数（home_orange_window，StagedReset 侧消费），
    HOME→PREGRASP 从盲搜变成可学的视觉伺服；late 课程 home 0.30 权重不动
    （home 出生的采样预算现在有产出）。v0 类零改动——蒸馏线继承 v0，出生
    分布与其教师（状态线）训练分布保持一致。play/评估的 home-only 口径
    自动带窗口，与实机部署条件一致（开工前目标入镜）。窗口变宽的后续
    手段是「高位扫视姿态」（reset_arm_joints 偏置），见训练复盘。

    cam_v4 教程口径移植（2026-09-29 对照 IsaacLabTutorial place_vial，同款
    SO101+腕摄视觉 PPO 确定性审计 99% 的差异定位）：
    * **动作粒度**：臂 0.15→0.033、夹爪 0.35→0.02 rad/步。原口径夹爪 5 步
      全闭≈开关、臂步长是教程 4.5×——视觉闭环每步位移噪声放大 5~17×，
      这是 PREGRASP→GRASP ~9% 断层（合爪需毫米级）且一夜超参修不动的主因；
      教程注释「std 上界须保 approach 可学」的前提正是小 scale（0.3×0.02）。
    * **第 4 项本体 joint_target**（CameraProprioceptionV2Cfg）。
    * episode 60s→90s：动作慢 4.5×，防慢策略被预算封顶。
    三项均只挂 v1；v0（蒸馏）与状态线零影响。actor 输入维度变化（18→24）
    ——cam_v4 起从头训练，不加载 cam_v3 checkpoint。
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_staged.params["home_orange_window"] = HOME_ORANGE_WINDOW
        self.observations.proprioception = CameraProprioceptionV2Cfg()
        self.actions.arm_action.scale = 0.033
        self.actions.gripper_action.scale = 0.02
        self.episode_length_s = 90.0


# --------------------------------------------------------------------- #
# v2：前段教程式势能差塑形（cam_v5）
# --------------------------------------------------------------------- #


@configclass
class CameraFrontRewardsCfg(StagedRewardsCfg):
    """v2 奖励表：前段 tanh 存在项置零，换教程式 approach 势能差塑形。

    cam_v4 复盘（16907 迭代平台期）：home_born_success 全程 0、PREGRASP
    出生仅 7.9%。奖励侧根因：home_approach/pregrasp_distance 是 tanh 存在
    包，~2 倍 std（24cm/10cm）外梯度近零，而橘子刷位距爪 15-35cm 正落
    盲区；且存在项是驻留津贴（C4）。IsaacLabTutorial 纯摄像头 PPO 同款
    任务达 99% 的核心差异就是 ApproachProgressReward：势能差、每接近
    1cm 付一单位、active 直到 grasped——25cm 和 2cm 处梯度相同。

    两个旧项置零但保留（weight=0 仍计算/记录，便于与 v1 对账诊断）。
    后段奖励表（grasp_close 8.0 起）全部原样继承。

    量级对账（有效 = weight×raw×dt, dt=1/30）：典型初始距离 0.15-0.35m
    → raw 15-35 单位 → 有效 3~7（封顶 9），与前段阶段 bonus 3+4+6=13 同
    量级不主导（E4）；近场与 grasp_close 叠加伺服梯度 ≈0.53/cm。
    """

    home_approach = RewTerm(func=staged_mdp.safe_reward_home_approach, weight=0.0)
    pregrasp_distance = RewTerm(func=staged_mdp.safe_reward_pregrasp_distance, weight=0.0)
    approach_progress = RewTerm(func=camera_mdp.SafeCameraApproachProgress, weight=6.0)


# v2 late 档课程：home 0.30→0.40（出生有梯度后每个 home 出生都有产出，
# 直接服务 home_born_success 目标指标）；PREGRASP/GRASP 断层带保持 0.34；
# 已饱和的 RELEASE/PLACE（cam_v4 出生成功率 100%/99.7%）各 0.08→0.04。
# 仅 v2 使用；CAMERA_CURRICULUM_LATE 与 early/mid 共享表不动。
CAMERA_CURRICULUM_LATE_V2 = (0.40, 0.18, 0.16, 0.10, 0.04, 0.04, 0.04, 0.04)


@configclass
class CameraFrontShapedEnvCfg(CameraVisibleHomeEnvCfg):
    """v2：前段教程式势能差塑形 + home 加权课程（gym id: Pick-Apple-SO101-Staged-Camera-v2）。

    在 v1（可见窗口 + 教程动作粒度 + joint_target 本体）基础上只改两项：
    奖励表换 CameraFrontRewardsCfg、late 课程换 CAMERA_CURRICULUM_LATE_V2。
    观测/动作/物理/状态机/bank 全部继承，v0/v1 与蒸馏线零影响。
    """

    rewards: CameraFrontRewardsCfg = CameraFrontRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_staged.params["curriculum_late"] = CAMERA_CURRICULUM_LATE_V2


# --------------------------------------------------------------------- #
# v3：奖励全表缩放到教程量级（cam_v6；PPO 配套见 camera_ppo_cfg 的
# CameraTutorialPPORunnerCfg）
# --------------------------------------------------------------------- #

CAMERA_REWARD_SCALE = 1.0 / 30.0
"""v3 全表奖励缩放：我们每局回报 ~200-250 → 教程 ~7（比例不变仅缩尺度）。"""


@configclass
class CameraTutorialEnvCfg(CameraFrontShapedEnvCfg):
    """v3：v2 语义 + 全表奖励 ×CAMERA_REWARD_SCALE（gym id: Pick-Apple-SO101-Staged-Camera-v3）。

    cam_v5 复盘（2026-10-01，17002 迭代：峰值 87.6%@15125 后回落 81%）。
    与 IsaacLabTutorial vision run（2026-09-24_00-13-08）逐项对照后排除了
    两个"嫌疑"——它们在教程日志里同样存在，不是病根：
    * **std 钉死 0.3 上界**：教程全程同样钉死（entropy 1.28979 = 6 维 σ0.3
      理论值 1.2896），照样 5000 迭代 93% 随机 / 99% 确定性；
    * **固定 LR 1e-4**：教程 adaptive 实际运行区间 3e-5~1.7e-4（均值 ≈1e-4），
      与我们固定值等效。
    真正的断层在 value 函数拟合：**Loss/value 教程全程 ≈0.1，我们 500→1000
    且随训练增长**——每局回报 ~200-250（stage bonus 合计 83 + 终局 20 +
    稠密项累积）vs 教程 ~7，价值目标尺度大 30 倍，critic 拟合不动 →
    advantage 噪声大 → 策略在 87% 附近震荡回落（教程曲线单调到 93%）。

    修法：v2 任务语义全保留（窗口/细动作/90s/课程/奖励结构比例），全表
    奖励统一 ×1/30 对齐教程量级；优化器同步回教程配方（entropy 0.005
    只有在缩放后才回到教程的 entropy:reward 相对压力）。缩放在
    __post_init__ 里对 dataclass fields 遍历进行——新增奖励项自动覆盖，
    weight=0 的对账项保持 0。
    """

    def __post_init__(self):
        super().__post_init__()
        for field_info in fields(self.rewards):
            term = getattr(self.rewards, field_info.name, None)
            if term is not None and hasattr(term, "weight"):
                term.weight = round(term.weight * CAMERA_REWARD_SCALE, 6)


# --------------------------------------------------------------------- #
# v5：HOME 出生几何升级——高位扫视 O 姿态 + 盒子可见窗口（与蒸馏 v4 / 教师
# v5 完全同款几何；PPO 配套沿用 v4 的 CameraTutorialFixedLRPPORunnerCfg）
# --------------------------------------------------------------------- #


@configclass
class CameraScanHomeEnvCfg(CameraTutorialEnvCfg):
    """v5：v3 环境 + HOME 出生几何升级（gym id: Pick-Apple-SO101-Staged-Camera-v5）。

    2026-10-02 用户看出生画面定方向（「离桌子太近、视野小」，橘子+盒子要
    同框），蒸馏 v4 / 教师线已换 O 姿态（workshop roll=-0.804 族，相机
    23cm 高/倾 26°，probe_camera_geometry 网格实测选型，photos/
    cam_geometry/）+ SCAN_ORANGE_WINDOW/HOME_BOX_WINDOW 双窗口；同日实测
    旧教师 model_4250 在新几何下确定性 246/256=96.1%——新几何任务可解。
    相机直训线同步该几何，与蒸馏 v4 学生**同出生分布**，两条线可直接对照。

    单变量原则：相对 Camera-v4（cam_v6b 配方）只改 HOME 出生几何三件套
    （姿态/双窗口/关节噪声 ±0.1→±0.05），64×48 分辨率、奖励缩放、课程、
    90s episode 全部不动——128×96 是蒸馏线的独立变量，不混入本实验。
    新 gym id 不动 CameraTutorialEnvCfg：cam_v6b 在跑，v3/v4 注册保可复现。
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = scan_home_robot_cfg()
        self.events.reset_staged.params["home_orange_window"] = SCAN_ORANGE_WINDOW
        self.events.reset_staged.params["home_box_window"] = HOME_BOX_WINDOW
        # 与蒸馏 v4 同款降噪：±0.1 的 pan 抖动=视野横移 ±3cm，会把贴缘盒子
        # 摇出去一半出生（官方出生=固定舞台无关节噪声）
        self.events.reset_arm_joints.params["position_range"] = [-0.05, 0.05]


@configclass
class CameraScanHomeShortEnvCfg(CameraScanHomeEnvCfg):
    """v6：v5 + episode 90s→20s（gym id: Pick-Apple-SO101-Staged-Camera-v6）。

    2026-10-02 distill_v4 中途复盘 + 教程日志实测（三条线全部 20s、收敛局
    128-155 步、time_out 仅 3.8-6%），而我们的细粒度教师完成任务均值 121
    步——失败局在 90s/2700 步帽下游荡是采样预算黑洞。单变量 = episode
    长度；64×48 分辨率、奖励缩放、PPO 配方全部继承 v5。课程三档表不动：
    reset 快 ~4.5× → 更早进入 home 加权 late 档（符合意图）。阶段机/确认
    步数全是控制步、与局长无关。后手：time_out >0.75 且 success 被压 → 30s。
    与蒸馏 v5 同日同变量启动，两条线可对照（学生同出生分布）。
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0


@configclass
class CameraClosureScanHomeEnvCfg(CameraScanHomeEnvCfg):
    """v7：v5 + 合爪盲区修复相机（gym id: Pick-Apple-SO101-Staged-Camera-v7）。

    单变量主线 = 相机（俯角 -12° + FOV ×1.5，见 CameraClosureHdSceneCfg
    docstring 的根因与验证数据）；配套两处连带：

    * 分辨率 64×48→128×96：FOV 放大使 HOME 橘子只剩 ~6px 直径，必须补像素
      （蒸馏线 v3 起已是 128×96，两线从此同口径，可互相对照）；
    * episode 回 90s（v6 的 20s 实测勒死学习者：time_out 72→88%、连 LIFT
      出生都 0——学习者每局需要 >600 步，600 步帽下成功奖励发不出来，
      PPO 失去完成信号梯度）。

    其余（O 姿态 HOME 几何、双窗口、噪声 ±0.05、缩放奖励、固定 LR、课程）
    全部继承 v5。相机直训线与蒸馏 v6 学生同出生分布+同相机，直接对照。
    """

    scene: CameraClosureHdSceneCfg = CameraClosureHdSceneCfg(num_envs=1024)

    def __post_init__(self):
        super().__post_init__()
        # super 链（ScanHome.__post_init__）会把 robot 换成旧相机的
        # scan_home_robot_cfg()——必须在 super 之后换成 closure 相机版
        self.scene.robot = scan_home_closure_robot_cfg()


@configclass
class CameraClosureFinetuneEnvCfg(CameraClosureScanHomeEnvCfg):
    """v8：v7 + 课程三档钉死为固定表（gym id: Pick-Apple-SO101-Staged-Camera-v8）。

    2026-10-04 cam_v9_cont 复盘定因：--checkpoint 续训时课程 reset_count 归零、
    从 early 重爬,叠加梯度被 HOME 链占据,bank 的 LIFT/STABILIZE「悬空冷恢复」
    出生态发生遗忘性退化（LIFT-born 51%→22%、STABILIZE 43%→7%）;HOME 则在
    σ0.09 下随机口径平台 0.75（确定性 91%）。**IsaacLabTutorial 全程固定任务
    分布（无课程,每局完整任务）**——续训/精修的正确口径是固定出生分布。

    修复：三档课程全部钉死为 staged 默认 LATE 表（home 0.50,其余阶段保底
    非零防遗忘）——从 checkpoint 续训时无 early 瞬态、bank 中段占比恒定。
    其余（闭包相机/O 姿态/双窗口/90s/奖励/固定 LR/init_at_random_ep_len
    =框架默认 True）全部继承 v7。用于从强 checkpoint 起的精修段。
    """

    def __post_init__(self):
        super().__post_init__()
        fixed = tuple(staged_mdp.CURRICULUM_LATE)
        self.events.reset_staged.params["curriculum_early"] = fixed
        self.events.reset_staged.params["curriculum_mid"] = fixed
        self.events.reset_staged.params["curriculum_late"] = fixed


@configclass
class CameraFineTuneHdEnvCfg(CameraClosureFinetuneEnvCfg):
    """相机 v9：v8 + 腕摄 256×192——蒸馏学生 PPO 收尾口径（gym id: Pick-Apple-SO101-Staged-Camera-v9）。

    2026-10-07 路线 A（用户批准，见 agents/hybrid_ppo.py）：拿 v9 蒸馏学生
    （distill_v9_hd256/model_500）做 PPO 初始化继续训练——**CNN 输入必须是
    256×192**（与该学生的训练分辨率一致，分辨率不匹配=strict load 直接报错）。
    与 distill v9 相同的覆写方式（scene.wrist_camera.width/height 在
    __post_init__ 末尾覆盖，已验证生效）。其余全部继承 v8（闭包相机/O 姿态/
    双窗口/固定 home-0.50 课程/90s/0.033/0.02/奖励 ×1/30）——cam_v12 已实证
    该环境语义 + 低 LR + 守护收割 = 92.19%。
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.wrist_camera.width = 256
        self.scene.wrist_camera.height = 192


# --------------------------------------------------------------------- #
# 蒸馏 v2 的细粒度特权教师（gym id: Pick-Apple-SO101-Staged-TeacherFine-v0）
# --------------------------------------------------------------------- #


@configclass
class StagedFineTeacherEnvCfg(StagedPickAppleEnvCfg):
    """蒸馏 v2 的特权教师：相机 v2 线语义（细粒度 + 窗口 + 90s + 前段势能塑形）+ 状态观测。

    2026-10-01 distill_v1 平台期定因：旧教师 model_1750 是粗动作粒度
    （0.06/0.10）训练的——粗步长在 64×48 像素输入下不可模仿（cam_v4 复盘
    定论），且窗口带在其训练分布之外（标签 OOD）。本教师在与蒸馏 v2
    完全一致的环境语义下重新训练：动作粒度 0.033/0.02、HOME 窗口、
    episode 90s（cam_v5 同期 home_born 0.37 爬升证明该语义视觉可学）。

    **奖励必须是 CameraFrontRewardsCfg**（2026-10-01 teacher_fine_v1 失败
    教训：首版教师保守沿用 staged 默认奖励表，10000 轮后 home_born≈0.002
    全程为零——细粒度下爪每步只挪 0.033 rad，home_approach/
    pregrasp_distance 的 tanh 存在包在橘子距爪 15-35cm 处零梯度（cam_v4
    复盘同一结论），HOME→PREGRASP 对 PPO 不可学；旧教师能容忍是因为
    粗步长几步就跨过盲区。approach_progress 势能差在远场也有梯度，
    cam_v5 已实证）。课程同步换 CAMERA_CURRICULUM_LATE_V2（home 0.40，
    与 cam_v5 完全一致——本教师 = CameraFrontShapedEnvCfg 语义 + 特权观测）。

    观测保持 StagedObservationsCfg（policy 111 维含 history=3）——教师
    MLP 结构与 STAGED_ACTOR_CFG 完全一致，蒸馏时 strict load 不受影响；
    scene 继承 staged（无腕摄，无渲染开销，可上 4096 envs）。定义于本
    文件仅因 HOME_ORANGE_WINDOW / CameraFrontRewardsCfg 常量在此（放
    staged_env_cfg 会循环 import）。

    **奖励全表 ×CAMERA_REWARD_SCALE**（2026-10-01 teacher_fine_v3 复盘：
    2974 轮 home_born≡0、PREGRASP-born≡0、success 0.32→0.11 下滑、std 钉
    0.3——与 cam_v5 同病：每局回报 ~200+ vs 教程 ~7，Loss/value 500-1000
    vs 0.1，critic 拟合不动 → advantage 噪声。教师与 cam_v6
    （CameraTutorialEnvCfg）同步 ×1/30；优化器在 agent 侧同步回教程配方
    ——见 staged_ppo_cfg.StagedFineTeacherPPORunnerCfg）。

    **2026-10-02 v5 语义升级：HOME 换高位扫视姿态 + 盒子可见窗口**
    （用户定方向：出生画面橘子+盒子同框，对齐教程「出生即见任务目标」；
    当日第二轮从 P7 换 workshop roll=-0.804 族 O 姿态，实拍选型见
    photos/cam_geometry/ 与 camera_env_cfg 的 SCAN_HOME_ARM_JOINTS 注释）。
    旧教师 model_4250 是旧 HOME 姿态训练的，新姿态下需重训
    （teacher_fine_v5）；bank 中后段行不受影响。
    """

    rewards: CameraFrontRewardsCfg = CameraFrontRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = scan_home_robot_cfg()
        self.events.reset_staged.params["home_orange_window"] = SCAN_ORANGE_WINDOW
        self.events.reset_staged.params["home_box_window"] = HOME_BOX_WINDOW
        # HOME 关节噪声收敛（同蒸馏 v4：±0.1 pan 抖动会把贴缘盒子摇出画面）
        self.events.reset_arm_joints.params["position_range"] = [-0.05, 0.05]
        self.events.reset_staged.params["curriculum_late"] = CAMERA_CURRICULUM_LATE_V2
        self.actions.arm_action.scale = 0.033
        self.actions.gripper_action.scale = 0.02
        self.episode_length_s = 90.0
        for field_info in fields(self.rewards):
            term = getattr(self.rewards, field_info.name, None)
            if term is not None and hasattr(term, "weight"):
                term.weight = round(term.weight * CAMERA_REWARD_SCALE, 6)


@configclass
class StagedSmoothTeacherEnvCfg(StagedFineTeacherEnvCfg):
    """平滑教师（gym id: Pick-Apple-SO101-Staged-TeacherFine-v1）：教师奖励表唯一变量
    = action_rate 惩罚加大到能压平 bang-bang。

    2026-10-05 蒸馏线九版证伪终局（logs/night_watch_2026-10-04.md）定因：teacher 是
    ±1 饱和的 bang-bang 控制器（相对位置动作空间的最优形态；旧教师实测
    E[ΣΔa²]≈0.49/步 = 单维每 ~8 步满摆一次），从带一步滞后的 30Hz 腕部像素复刻
    「何时切换」的时序精度超出图像推断误差地板——从零 BC 学不会（v0-v6）、cam_v9
    热启动也保不住（位级 82.4%、0.009 权重位移即 0%）。要让标签可模仿，必须让教师
    本身输出连续平滑动作。

    **定标**：action_rate_l2 = Σ(Δa)²（isaaclab.envs.mdp），有效支付 = raw×w×dt。
    旧有效权重 -0.01/30=-0.000333 → 单维满摆（Δ=2, raw=4）只罚 5.3e-5，bang-bang
    近似免费。新有效权重 **-4.0**（super 的 ×1/30 循环之后直接覆盖）：满摆单步罚
    0.53、bang-bang 全局代价 ≈7.8（≈40% 终局奖 20）——而平滑策略（E[raw]≤0.05，
    RMS Δa/维 ≈0.09/步）全程只付 ≈0.5，时间预算极宽松（教师 121 步完成 vs 2700 步
    帽）。平方核天然惩罚大跳、宽容小步。**监控**：Episode_Reward/action_rate 日志值
    = -4.0×E[raw]，目标 ≥-0.2（即 E[raw]≤0.05，为旧教师 0.49 的 ≤1/10）；
    若训练后教师仍 bang-bang → 权重 ×2 重训。

    其余逐字继承 StagedFineTeacherEnvCfg（观测 111 维/history=3、细粒度 0.033/0.02、
    O 姿态+双窗口、90s、×1/30 奖励缩放、CAMERA_CURRICULUM_LATE_V2）——教师 MLP
    仍是 STAGED_ACTOR_CFG 结构，蒸馏 v7 的 strict load 不受影响。PPO 配方沿用
    StagedFineTeacherPPORunnerCfg（teacher_fine_v4 已验证：adaptive 3e-4 +
    entropy 0.005 + ×1/30 后 home_born 0.95+ 收割）。风险预注册：若 home_born 爬升
    显著慢于 v4（iter 3000 时 <0.5）= 惩罚压垮可学性 → 降 -1.0 重训。
    """

    def __post_init__(self):
        super().__post_init__()
        # super 链末尾已把全表 ×CAMERA_REWARD_SCALE（action_rate 变 -0.000333）；
        # 在其后直接覆盖为有效值 -4.0（定标见类 docstring）
        self.rewards.action_rate.weight = -4.0


@configclass
class CameraClosureDRFinetuneEnvCfg(CameraClosureFinetuneEnvCfg):
    """v10：v8 + 执行器带逐局随机（gym id: Pick-Apple-SO101-Staged-Camera-v10）。

    相机线 Plan B 落点：在 cam_v12 冠军口径（v8 固定课程 LATE）上唯一加
    PhysicsDREventsCfg（执行器带逐局随机）。训练配方照抄 cam_v12 夜巡：
    从强 checkpoint 续训 + LR 3e-5 注入 + 512 envs + 每 1000 iter 确定性
    guard-eval（两连 <0.848 即停，留最优）。v8 注册不动可复现。
    """

    events: PhysicsDREventsCfg = PhysicsDREventsCfg()
