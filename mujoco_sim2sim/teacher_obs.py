"""教师（特权坐标观测）PPO 的 MuJoCo obs 组包 + 八阶段状态机移植。

契约（全部逐字对照 Isaac 源码，teacher_smooth_v1 = 100% 确定性教师）：
* obs 111 维 = 4 个 term × 3 步历史（旧→新，IsaacLab CircularBuffer.buffer
  语义：最旧在开头、最新在末尾，逐 term 展平后按 cfg 顺序拼接）：
  [joint_pos_rel(6), joint_vel(6,clip±5), last_action(6), features(19,clip±2)]
* features(19) = orange−pocket(3) + hover−pocket(3) + orange_vel(3) +
  orange_local(3) + 开度分数(1) + 里程碑布尔(6)（staged_mdp.staged_policy_features）
* 里程碑由八阶段物理门推进（stages.py StageProgress）：爪力取两个
  夹爪刚体最近 3 个策略帧的净接触力最大值，橘子取当前帧。
* joint_pos_rel 相对 SCAN_HOME（教师 env.yaml 实测 init_state 即此值）。
* reset 清除历史，首次 append 用当前观测填满；状态机首步只锁存，
  不推进阶段（first_step gate）。
"""

from __future__ import annotations

import numpy as np
import mujoco

from env import SCAN_HOME_ARM_JOINTS
from task_metrics import POCKET_OFFSET_B, quat_apply, quat_to_mat

# ---- 门常量（stages.py / staged_mdp.py 逐字，行号注于侧）----
GRIPPER_TRAVEL_RAD = 1.745            # staged_mdp.py:58
GRASP_CLAW_FORCE_N = 2.2              # staged_mdp.py:67
HOME_APPROACH_M = 0.12                # staged_mdp.py:70
HOVER_ABOVE_BOX_M = 0.14              # staged_mdp.py:82（mdp.py:72 同值）
PREGRASP_MIN_GRIPPER_RAD = 1.30       # staged_mdp.py:76
PREGRASP_POCKET_M = 0.035             # stages.py:59
GRASP_PROXIMITY_M = 0.075             # stages.py:66
GRASP_MIN_GRIPPER_RAD = -0.174        # stages.py:71
GRASP_MAX_GRIPPER_RAD = 0.75          # stages.py:72
GRASP_LIFT_Z = 0.09                   # mdp.py:75（orange_low 门）
LIFT_Z_M = 0.105                      # stages.py:77
STABILIZE_MIN_Z_M = 0.105             # stages.py:82
STABILIZE_REL_VEL_MPS = 0.18          # stages.py:83
STABILIZE_CONFIRM_STEPS = 6           # stages.py:84
TRANSPORT_MIN_Z_M = 0.118             # stages.py:89
TRANSPORT_REL_VEL_MPS = 0.22          # stages.py:90
PLACE_OVER_BOX_XY_M = 0.055           # stages.py:94
PLACE_APPROACH_MIN_Z_M = 0.10         # stages.py:95
PLACE_INSIDE_XY_M = 0.052             # stages.py:99
PLACE_INSIDE_Z_MIN_M = 0.008          # stages.py:100
PLACE_INSIDE_Z_MAX_M = 0.052          # stages.py:101
PLACE_JAW_NEAR_M = 0.10               # stages.py:102
RELEASE_CLEARANCE_M = 0.055           # stages.py:110
RELEASE_SETTLED_SPEED_MPS = 0.15      # stages.py:112
RELEASE_STABLE_STEPS = 10             # stages.py:113
GRASP_CONFIRM_STEPS = 3
LIFT_CONFIRM_STEPS = 4
CONFIRM = {"HOME": 1, "PREGRASP": 1, "GRASP": GRASP_CONFIRM_STEPS,
           "LIFT": LIFT_CONFIRM_STEPS, "STABILIZE": STABILIZE_CONFIRM_STEPS,
           "TRANSPORT": 1, "PLACE": 1}
STAGE_NAMES = ("HOME", "PREGRASP", "GRASP", "LIFT", "STABILIZE", "TRANSPORT", "PLACE", "RELEASE")
OBS_DIM = 111  # 37 × 3 历史


class MuJoCoStageState:
    """每控制步从 MuJoCo 读状态（staged_mdp.StagedPickState.update 的移植）。"""

    def __init__(self, env):
        self.env = env
        self._grip_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._jaw_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        self._orange_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "orange")

    def update(self):
        env = self.env
        self.jaw_pos = env.data.xpos[self._jaw_body].copy()
        res = np.zeros(6)
        mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self._jaw_body, res, 0)
        self.jaw_vel = res[3:].copy()          # 世界系线速度
        self.orange_pos, _ = env.orange_pose
        self.orange_vel = env.orange_vel
        self.box_pos, self.box_quat = env.box_pose
        R_box = quat_to_mat(self.box_quat)
        self.box_up_z = R_box[2, 2]
        self.orange_local = R_box.T @ (self.orange_pos - self.box_pos)
        self.hover_pos = self.box_pos + np.array([0.0, 0.0, HOVER_ABOVE_BOX_M])

        self.orange_force = float(np.linalg.norm(env.contact_forces[self._orange_body]))
        self.claw_force = float(np.linalg.norm(
            env.contact_force_history[:, [self._grip_body, self._jaw_body]], axis=-1).max())
        self.gripper_pos = float(env.joint_pos[5])
        gp, gq = env.body_pose("gripper")
        self.pocket_pos = gp + quat_apply(gq, POCKET_OFFSET_B)
        self.dist_pocket = float(np.linalg.norm(self.orange_pos - self.pocket_pos))
        self.rel_vel = float(np.linalg.norm(self.orange_vel - self.jaw_vel))

    def conditions(self, m: "StageMachine") -> dict[str, bool]:
        """八阶段瞬时布尔（staged_mdp.py:243-269 逐条）。"""
        orange_low = self.orange_pos[2] < GRASP_LIFT_Z
        grasped_now = (
            self.claw_force > GRASP_CLAW_FORCE_N
            and self.orange_force > 0.15
            and self.dist_pocket < GRASP_PROXIMITY_M
            and self.gripper_pos > GRASP_MIN_GRIPPER_RAD
            and self.gripper_pos < GRASP_MAX_GRIPPER_RAD
        )
        ol = self.orange_local
        return {
            "HOME": self.dist_pocket < HOME_APPROACH_M and orange_low and not m.grasped,
            "PREGRASP": (self.dist_pocket < PREGRASP_POCKET_M
                         and self.gripper_pos > PREGRASP_MIN_GRIPPER_RAD
                         and orange_low and not m.grasped),
            "GRASP": grasped_now,
            "LIFT": (self.orange_pos[2] > LIFT_Z_M
                     and self.dist_pocket < GRASP_PROXIMITY_M + 0.01
                     and self.orange_force > 0.2),
            "STABILIZE": (self.orange_pos[2] > STABILIZE_MIN_Z_M
                          and self.rel_vel < STABILIZE_REL_VEL_MPS and grasped_now),
            "TRANSPORT": (self.orange_pos[2] > TRANSPORT_MIN_Z_M
                          and self.rel_vel < TRANSPORT_REL_VEL_MPS
                          and float(np.linalg.norm(ol[:2])) < PLACE_OVER_BOX_XY_M
                          and ol[2] > PLACE_APPROACH_MIN_Z_M and grasped_now),
            "PLACE": (abs(ol[0]) < PLACE_INSIDE_XY_M and abs(ol[1]) < PLACE_INSIDE_XY_M
                      and ol[2] > PLACE_INSIDE_Z_MIN_M and ol[2] < PLACE_INSIDE_Z_MAX_M
                      and self.dist_pocket < PLACE_JAW_NEAR_M),
            "RELEASE": (abs(ol[0]) < PLACE_INSIDE_XY_M and abs(ol[1]) < PLACE_INSIDE_XY_M
                        and ol[2] > PLACE_INSIDE_Z_MIN_M and ol[2] < PLACE_INSIDE_Z_MAX_M
                        and self.dist_pocket > RELEASE_CLEARANCE_M
                        and float(np.linalg.norm(self.orange_vel)) < RELEASE_SETTLED_SPEED_MPS),
        }


class StageMachine:
    """StageProgress 移植：单调推进 + 确认计数 + 里程碑锁存 + RELEASE 防抖。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.stage = 0
        self.pregrasp_latched = False
        self.grasped = False
        self.lifted = False
        self.stabilized = False
        self.over_box_latched = False
        self.in_box_latched = False
        self.confirm_count = [0] * 8
        self.stable = 0
        self.success = False
        self.first_step = True

    def update(self, cond: dict[str, bool]) -> None:
        gate = not self.first_step
        self.first_step = False
        for k, name in enumerate(STAGE_NAMES[:-1]):          # HOME..PLACE
            met = gate and self.stage == k and cond[name]
            if cond[name]:
                self.confirm_count[k] += 1
            else:
                self.confirm_count[k] = 0
            if not gate:
                self.confirm_count[k] = 0
            if met and self.confirm_count[k] >= CONFIRM[name]:
                self.stage = k + 1
            raw = cond[name]
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
        # RELEASE：稳定计数 +1 / 断一步 −2，攒满 10 = 成功（stages.py:253-267）
        if gate and cond["RELEASE"]:
            self.stable += 1
        else:
            self.stable = max(0, self.stable - 2)
        if not self.success and self.stable >= RELEASE_STABLE_STEPS:
            self.success = True

    def flags(self) -> np.ndarray:
        return np.array([self.pregrasp_latched, self.grasped, self.lifted,
                         self.over_box_latched, self.in_box_latched, self.success],
                        dtype=np.float32)


class TeacherObs:
    """111 维 obs（4 term × 3 历史，旧→新）。

    reset 后的第一次 append：历史用当前值填满 3 格（Isaac 实测——出生 obs
    的 obs[0:6]==obs[6:12]==obs[12:18]，CircularBuffer 非零填充语义）。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._h = {k: [np.zeros(n, dtype=np.float32) for _ in range(3)]
                   for k, n in [("jp", 6), ("jv", 6), ("la", 6), ("ft", 19)]}
        self._empty = True

    def append(self, env, st: MuJoCoStageState, m: StageMachine) -> np.ndarray:
        jp = (env.joint_pos - SCAN_HOME_ARM_JOINTS).astype(np.float32)
        jv = np.clip(env.joint_vel, -5.0, 5.0).astype(np.float32)
        la = env.last_action.astype(np.float32)
        feats = np.concatenate([
            st.orange_pos - st.pocket_pos,
            st.hover_pos - st.pocket_pos,
            st.orange_vel,
            st.orange_local,
            [max(0.0, min(1.0, st.gripper_pos / GRIPPER_TRAVEL_RAD))],
            m.flags(),
        ]).astype(np.float32)
        feats = np.clip(np.nan_to_num(feats, nan=0.0, posinf=2.0, neginf=-2.0), -2.0, 2.0)
        for key, val in [("jp", jp), ("jv", jv), ("la", la), ("ft", feats)]:
            if self._empty:
                self._h[key] = [val.copy() for _ in range(3)]
            else:
                self._h[key].pop(0)
                self._h[key].append(val)
        self._empty = False
        return np.concatenate([np.concatenate(self._h[k]) for k in ("jp", "jv", "la", "ft")])
