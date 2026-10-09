"""MuJoCo 版 Pick_Anything 最小环境（sim2sim 第一阶段：只物理 + 复位，不接策略）。

对齐 Isaac 训练侧的复刻契约（常量逐项标注来源，改动前先对照原文件）：
* 关节顺序与名字 = robots.py:15-22（6 关节，gripper 0=全闭/1.745=全开）。
* HOME 姿态 = SCAN_HOME_ARM_JOINTS（camera_env_cfg.py:239-246，O 姿态
  roll=-0.804）。
* 出生窗口 = SCAN_ORANGE_WINDOW / HOME_BOX_WINDOW（camera_env_cfg.py:250,257，
  v4/v5/TeacherFine 线口径）；盒 yaw ±0.3*difficulty（staged_mdp home 行）。
* 动作 = RelativeJointPositionAction：target = qpos + action*scale，
  scale 0.033/0.02（camera_env_cfg.py:496-497，teacher_fine/相机线同款）。
* 时序 = 控制 30Hz、每控制步 8 个 1/240 物理步（Isaac decimation=4 ×
  MJWarp num_substeps=2）。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SCENE_XML = ASSETS_DIR / "scene_pick.xml"

# 关节顺序 = Isaac 侧 ARM_JOINT_NAMES + gripper（robots.py:15-22）
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# SCAN_HOME_ARM_JOINTS（camera_env_cfg.py:239-246），顺序同 JOINT_NAMES
SCAN_HOME_ARM_JOINTS = np.array(
    [-0.05, -0.907, 0.190, 1.580, -0.804, 0.2555162024919699]
)

# 出生窗口 (x_lo, x_hi, y_lo, y_hi)（camera_env_cfg.py:250,257）
ORANGE_WINDOW = (0.290, 0.320, -0.380, -0.340)
BOX_WINDOW = (0.440, 0.490, -0.450, -0.400)

ORANGE_SPAWN_Z = 0.09  # FRUIT_SPAWN_Z（mdp）
BOX_SPAWN_Z = 0.046    # TABLE_TOP_Z(0.041)+0.005
BOX_YAW_RANGE = 0.18   # ±0.3 rad * difficulty 0.6（staged_mdp home 行）

# 动作 scale（camera_env_cfg.py:496-497）：臂 5 关节 0.033、爪 0.02
ACTION_SCALE = np.array([0.033] * 5 + [0.02])

# Isaac rsl_rl wrapper 对原始动作的 clip（camera_ppo_cfg.py clip_actions=1.0）
ACTION_CLIP = 1.0

CONTROL_DT = 1.0 / 30.0
PHYSICS_DT = 1.0 / 240.0
SUBSTEPS_PER_ACTION = 8  # 8 * 1/240 = 1/30
SOLVER_SUBSTEPS_PER_TARGET = 2  # apply_actions runs at 120 Hz, not at policy 30 Hz

# 出生关节噪声 ±0.05（camera_env_cfg.py:438/494：±0.1 pan 抖动会摇出画面）
RESET_JOINT_NOISE = 0.05


def _free_quat(yaw: float) -> np.ndarray:
    """绕 z 的单位四元数 (w, x, y, z)。"""
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


class PickMujocoEnv:
    """单环境 MuJoCo 复刻：reset（HOME+窗口刷位）/ step（delta 位置目标）。

    只做物理。obs 组包与策略推理在 policy_wrapper.py / run_policy.py
    （相机 PPO sim2sim 第二阶段），本类提供关节/物体原始量读取。
    """

    def __init__(self, xml_path: str | Path = SCENE_XML, seed: int | None = None,
                 action_clip: float | None = ACTION_CLIP):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._rng = np.random.default_rng(seed)
        self.action_clip = action_clip

        self._joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        }
        missing = [n for n, i in self._joint_ids.items() if i < 0]
        if missing:
            raise ValueError(f"scene 缺关节（与 Isaac 契约不符）: {missing}")
        self._qadr = np.array(
            [self.model.jnt_qposadr[self._joint_ids[n]] for n in JOINT_NAMES]
        )
        self._vadr = np.array(
            [self.model.jnt_dofadr[self._joint_ids[n]] for n in JOINT_NAMES]
        )
        self._act_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in JOINT_NAMES
            ]
        )
        if any(i < 0 for i in self._act_ids):
            raise ValueError("scene 缺执行器（与 Isaac 契约不符）")
        self._velocity_act_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_velocity")
            for n in JOINT_NAMES
        ])

        self._orange_qadr = int(
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "orange_root")
            ]
        )
        self._box_qadr = int(
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_root")
            ]
        )
        self._orange_vadr = int(
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "orange_root")
            ]
        )

        self.last_action = np.zeros(len(JOINT_NAMES))
        self._joint_target_raw = SCAN_HOME_ARM_JOINTS.copy()
        self._n_steps = 0
        self._n_ctrl_clipped = 0
        self.contact_forces = np.zeros((self.model.nbody, 3))
        self.contact_force_history = np.zeros((3, self.model.nbody, 3))

    def reset_bookkeeping(self, target: np.ndarray) -> None:
        """Clear per-episode state after either random reset or bank restoration."""
        self.last_action[:] = 0.0
        self._joint_target_raw = np.asarray(target, dtype=float).copy()
        self._n_steps = 0
        self._n_ctrl_clipped = 0
        self.contact_forces[:] = 0.0
        self.contact_force_history[:] = 0.0
        ids = self._velocity_act_ids[self._velocity_act_ids >= 0]
        self.data.ctrl[ids] = 0.0

    def _capture_contacts(self) -> None:
        # The contact table and solver forces refer to the same pre-integration
        # state here. Capture before refreshing kinematics for observations.
        self.contact_forces[:] = 0.0
        force = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            mujoco.mj_contactForce(self.model, self.data, i, force)
            world = contact.frame.reshape(3, 3).T @ force[:3]
            b1, b2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            self.contact_forces[b1] -= world
            self.contact_forces[b2] += world
        self.contact_force_history[1:] = self.contact_force_history[:-1].copy()
        self.contact_force_history[0] = self.contact_forces

    def refresh_kinematics(self) -> None:
        """Refresh current body poses and COM velocities without re-solving contacts."""
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_comVel(self.model, self.data)

    def reseed(self, seed: int) -> None:
        """换出生随机源（每局可复现：seed 唯一决定轨迹，策略确定性）。"""
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------- reset
    def reset(self, joint_noise: float = RESET_JOINT_NOISE) -> dict[str, np.ndarray]:
        """HOME 出生：关节=SCAN_HOME+噪声，橘/盒在双窗口均匀刷位。

        返回本局出生位形（关节/橘/盒位姿），便于记录与对拍。
        """
        mujoco.mj_resetData(self.model, self.data)

        q = SCAN_HOME_ARM_JOINTS.copy()
        if joint_noise > 0.0:
            q[:5] += self._rng.uniform(-joint_noise, joint_noise, size=5)
        # Isaac's HOME event writes only the five arm joints. The cold Newton
        # state starts with gripper=0; default_joint_pos remains the obs offset.
        q[5] = 0.0
        self.data.qpos[self._qadr] = q
        self.data.ctrl[self._act_ids] = q

        ox, oy = self._rng.uniform(ORANGE_WINDOW[0::2], ORANGE_WINDOW[1::2])
        orange_quat = _free_quat(self._rng.uniform(0.0, 2.0 * np.pi))
        self.data.qpos[self._orange_qadr : self._orange_qadr + 7] = np.concatenate(
            [[ox, oy, ORANGE_SPAWN_Z], orange_quat]
        )

        bx, by = self._rng.uniform(BOX_WINDOW[0::2], BOX_WINDOW[1::2])
        # 橘-盒最小分隔 = Isaac 源码常量 BOX_FRUIT_MIN_SEPARATION_M=0.13
        # （env_cfg.py:62；原 0.135 为生成时手写值，对齐源码）
        dy_need = np.sqrt(max(0.0, 0.13**2 - (bx - ox)**2))
        by = np.clip(min(by, oy - dy_need), BOX_WINDOW[2], BOX_WINDOW[3])
        box_quat = _free_quat(self._rng.uniform(-BOX_YAW_RANGE, BOX_YAW_RANGE))
        self.data.qpos[self._box_qadr : self._box_qadr + 7] = np.concatenate(
            [[bx, by, BOX_SPAWN_Z], box_quat]
        )

        self.data.qvel[:] = 0.0
        self.reset_bookkeeping(q)
        mujoco.mj_forward(self.model, self.data)
        return {
            "joints": q,
            "orange_pos": np.array([ox, oy, ORANGE_SPAWN_Z]),
            "box_pos": np.array([bx, by, BOX_SPAWN_Z]),
        }

    # ------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> None:
        """RelativeJointPositionAction：target = qpos + a*scale。
        action_clip 由对应策略配置决定；教师 None，相机策略 1.0。

        同一个策略动作保持 1/30 秒，每 1/120 秒用最新 q 重算目标，
        每个目标执行两个 1/240 秒求解步。目标不做 ctrlrange 裁剪。
        """
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != len(JOINT_NAMES):
            raise ValueError(f"action 需要 {len(JOINT_NAMES)} 维，得到 {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("action contains NaN or infinity")
        a = action.copy() if self.action_clip is None else np.clip(action, -self.action_clip, self.action_clip)
        self.last_action = a.copy()
        for _ in range(SUBSTEPS_PER_ACTION // SOLVER_SUBSTEPS_PER_TARGET):
            target = self.joint_pos + a * ACTION_SCALE
            self._joint_target_raw = target.copy()
            self.data.ctrl[self._act_ids] = target
            for _ in range(SOLVER_SUBSTEPS_PER_TARGET):
                mujoco.mj_step(self.model, self.data)
        # The training scene updates sensors lazily: history is sampled when
        # observations read it, once per 30 Hz policy step (three such frames).
        self._capture_contacts()
        # mj_step 返回时 xpos/cam_xpos 是积分前运动学（≤1/240s 陈旧）；
        # 渲染、pocket 与相对速度需要当前 qpos/qvel 的运动学。
        self.refresh_kinematics()
        self._n_steps += 1

    def hold(self, n_control_steps: int, target: np.ndarray | None = None) -> None:
        """位置目标钉在 target（默认 HOME），跑 n 个控制步——悬停/下沉诊断用。"""
        tgt = SCAN_HOME_ARM_JOINTS.copy() if target is None else np.asarray(target, float)
        self.data.ctrl[self._act_ids] = tgt
        self._joint_target_raw = tgt.copy()
        for _ in range(n_control_steps):
            for _ in range(SUBSTEPS_PER_ACTION):
                mujoco.mj_step(self.model, self.data)
            self._capture_contacts()
            self.refresh_kinematics()

    # ---------------------------------------------------------- 读取
    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self._qadr].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self._vadr].copy()

    @property
    def joint_target(self) -> np.ndarray:
        return self.data.ctrl[self._act_ids].copy()

    @property
    def joint_target_raw(self) -> np.ndarray:
        """最近一次 120 Hz 更新写入的 PD 位置目标。"""
        return self._joint_target_raw.copy()

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def n_ctrl_clipped(self) -> int:
        return self._n_ctrl_clipped

    @property
    def orange_pose(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.data.qpos[self._orange_qadr : self._orange_qadr + 7]
        return q[:3].copy(), q[3:7].copy()  # pos, quat (w,x,y,z)

    @property
    def orange_vel(self) -> np.ndarray:
        """橘子线速度（自由关节前 3 个 dof）。"""
        return self.data.qvel[self._orange_vadr : self._orange_vadr + 3].copy()

    @property
    def box_pose(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.data.qpos[self._box_qadr : self._box_qadr + 7]
        return q[:3].copy(), q[3:7].copy()

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """body 世界位姿（xpos/xquat），用于 gripper/moving_jaw 等诊断。"""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"body 不存在: {name}")
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()
