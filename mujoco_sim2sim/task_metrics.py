"""sim2sim 评估判据：成功/失败/里程碑（纯 numpy，无 torch/mujoco 依赖）。

全部阈值逐项对照 Isaac 侧（staged_mdp.py / stages.py / mdp.py，行号注于常量）。
相对 Isaac 完整版状态机的简化：
* 不复刻八阶段推进与力门（课程记账用，eval 只需终局判据）；
* success = RELEASE 稳定条件直接逐步判定（物理条件蕴含此前各段完成）；
* dropped 的「曾夹持」放宽为「曾举起」（ever_lift，z>0.105 过）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---- 常量（Isaac 来源标注）----
FRUIT_RECT = ((0.05, 0.65), (-0.60, -0.30))   # env_cfg.py FRUIT_RECT
# The calibrated gripper now uses the actual USD/solver body frame. A local
# offset fitted at HOME cannot compensate for incorrect joint anchors.
POCKET_OFFSET_B = np.array([0.0408, -0.0002, -0.0709])
GRASP_LIFT_Z = 0.105                            # staged_mdp LIFT_Z_M
IN_BOX_XY = 0.052                               # PLACE/RELEASE 盒局部 |x|,|y| 门
IN_BOX_Z = (0.008, 0.052)                       # 盒局部 z 带（staged_mdp PLACE）
OVER_BOX_XY = 0.055                             # TRANSPORT 盒局部 XY 门
RELEASE_CLEAR_M = 0.055                         # RELEASE_CLEARANCE_M（成功门）
SETTLE_SPEED_MPS = 0.15                         # RELEASE_SETTLED_SPEED_MPS
STABLE_TARGET = 10                              # RELEASE_STABLE_STEPS
STABLE_DECAY = 2                                # 断一步 −2（stages.py:257-263）
MAX_STEPS_DEFAULT = 2700                        # 90s × 30Hz
BOX_UP_Z_MIN = 0.70                             # staged_box_tipped 的 up_z 门
FRUIT_RECT_PAD = 0.08                           # orange_dropped 外扩
BOX_RECT_PAD = 0.15                             # box 出桌矩形外扩
QVEL_MAX = 25.0                                 # staged_abnormal_robot

OUTCOMES = ("success", "timeout", "dropped", "box_tipped", "abnormal")


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(w,x,y,z) 四元数旋转向量：v + 2w·(qv×v) + 2·qv×(qv×v)。

    （2026-10-08 修 bug：旧式写成 v + 2·qv×(v + qv×v) + 2w·(qv×v)，
    多出 2(qv×v) 项——口袋点一直被错旋 ~49mm，教师 obs 对拍抓的现行。）
    """
    w, x, y, z = q
    qv = np.array([x, y, z])
    return v + 2.0 * w * np.cross(qv, v) + 2.0 * np.cross(qv, np.cross(qv, v))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pocket_pos(grip_pos: np.ndarray, grip_quat: np.ndarray) -> np.ndarray:
    """捕获口袋 = gripper body 位姿 + 本体系偏移（指尖 pinch 点）。"""
    return grip_pos + quat_apply(grip_quat, POCKET_OFFSET_B)


def body_local_in_box(p: np.ndarray, box_pos: np.ndarray,
                      box_quat: np.ndarray) -> np.ndarray:
    return quat_to_mat(box_quat).T @ (p - box_pos)


def box_up_z(box_quat: np.ndarray) -> float:
    return float(quat_to_mat(box_quat)[2, 2])  # 盒局部 z 轴 · 世界 z


@dataclass
class EpStats:
    seed: int = -1
    outcome: str = "timeout"
    steps: int = 0
    n_ctrl_clipped: int = 0
    grasp: bool = False            # 诊断（宽口径）：曾举起 且口袋曾贴近
    lift: bool = False             # 橘 z 曾 > GRASP_LIFT_Z
    over_box: bool = False         # 橘曾到盒口上方（宽口径）
    max_orange_z: float = 0.0
    min_pocket_dist: float = math.inf
    stable_count: int = 0
    time_to_success: int = -1
    extra: dict = field(default_factory=dict)


class EpisodeTracker:
    """每控制步 update(env)（物理 step 之后）；done/outcome 给出终局。"""

    def __init__(self, max_steps: int = MAX_STEPS_DEFAULT):
        self.max_steps = max_steps
        self._reset_state()

    def _reset_state(self) -> None:
        self.steps = 0
        self.stable = 0
        self.success = False
        self.ever_lift = False
        self.over_box = False
        self.max_orange_z = 0.0
        self.min_pocket_dist = math.inf
        self._fail: str | None = None

    def reset(self) -> None:
        self._reset_state()

    def update(self, env) -> None:
        self.steps += 1
        jp = env.joint_pos
        jv = env.joint_vel
        o_pos, _ = env.orange_pose
        b_pos, b_quat = env.box_pose
        o_vel = env.orange_vel

        if not (np.isfinite(jp).all() and np.isfinite(jv).all()
                and np.isfinite(o_pos).all() and np.isfinite(b_pos).all()):
            self._fail = "abnormal"
            return
        if float(np.abs(jv).max()) > QVEL_MAX:
            self._fail = "abnormal"
            return

        gp, gq = env.body_pose("gripper")
        pk = pocket_pos(gp, gq)
        d_pk = float(np.linalg.norm(o_pos - pk))
        self.min_pocket_dist = min(self.min_pocket_dist, d_pk)
        self.max_orange_z = max(self.max_orange_z, float(o_pos[2]))

        o_local = body_local_in_box(o_pos, b_pos, b_quat)
        in_box = (abs(o_local[0]) < IN_BOX_XY and abs(o_local[1]) < IN_BOX_XY
                  and IN_BOX_Z[0] < o_local[2] < IN_BOX_Z[1])
        settled = float(np.linalg.norm(o_vel)) < SETTLE_SPEED_MPS
        release_ok = in_box and d_pk > RELEASE_CLEAR_M and settled
        self.stable = self.stable + 1 if release_ok else max(0, self.stable - STABLE_DECAY)
        if not self.success and self.stable >= STABLE_TARGET:
            self.success = True

        self.ever_lift = self.ever_lift or float(o_pos[2]) > GRASP_LIFT_Z
        self.over_box = self.over_box or (
            abs(o_local[0]) < OVER_BOX_XY and abs(o_local[1]) < OVER_BOX_XY
            and o_local[2] > 0.10)

        # 失败（Isaac 终止项的简化口径）
        if self._fail is None:
            out_fruit = not (FRUIT_RECT[0][0] - FRUIT_RECT_PAD <= o_pos[0] <= FRUIT_RECT[0][1] + FRUIT_RECT_PAD
                             and FRUIT_RECT[1][0] - FRUIT_RECT_PAD <= o_pos[1] <= FRUIT_RECT[1][1] + FRUIT_RECT_PAD)
            if out_fruit and self.ever_lift:
                self._fail = "dropped"
            elif float(o_pos[2]) < 0.02 and self.ever_lift and not in_box:
                self._fail = "dropped"
            elif (box_up_z(b_quat) < BOX_UP_Z_MIN
                  or not (FRUIT_RECT[0][0] - BOX_RECT_PAD <= b_pos[0] <= FRUIT_RECT[0][1] + BOX_RECT_PAD
                          and FRUIT_RECT[1][0] - BOX_RECT_PAD <= b_pos[1] <= FRUIT_RECT[1][1] + BOX_RECT_PAD)):
                self._fail = "box_tipped"

    @property
    def done(self) -> bool:
        return self.success or self._fail is not None or self.steps >= self.max_steps

    @property
    def outcome(self) -> str:
        if self._fail is not None:
            return self._fail
        if self.success:
            return "success"
        return "timeout"

    def finish(self, seed: int, env=None) -> EpStats:
        return EpStats(
            seed=seed,
            outcome=self.outcome,
            steps=self.steps,
            n_ctrl_clipped=env.n_ctrl_clipped if env is not None else 0,
            grasp=bool(self.ever_lift and self.min_pocket_dist < 0.08),
            lift=bool(self.ever_lift),
            over_box=bool(self.over_box),
            max_orange_z=float(self.max_orange_z),
            min_pocket_dist=float(self.min_pocket_dist),
            stable_count=self.stable,
            time_to_success=self.steps if self.success else -1,
        )


def wilson_ci95(k: int, n: int) -> tuple[float, float]:
    """Wilson 区间（成功率 CI）。"""
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(results: list[EpStats], meta: dict | None = None) -> dict:
    n = len(results)
    counts = {o: sum(1 for r in results if r.outcome == o) for o in OUTCOMES}
    k = counts["success"]
    steps = [r.steps for r in results]
    clip_frac = (sum(r.n_ctrl_clipped for r in results)
                 / max(sum(r.steps for r in results), 1))
    out = {
        "episodes": n,
        "success_rate": k / n if n else 0.0,
        "success_rate_ci95": list(wilson_ci95(k, n)),
        "counts": counts,
        "milestones": {
            "grasp_rate": sum(r.grasp for r in results) / n if n else 0.0,
            "lift_rate": sum(r.lift for r in results) / n if n else 0.0,
            "over_box_rate": sum(r.over_box for r in results) / n if n else 0.0,
        },
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "p50_steps": float(np.percentile(steps, 50)) if steps else 0.0,
        "p90_steps": float(np.percentile(steps, 90)) if steps else 0.0,
        "mean_ctrl_clip_frac": clip_frac,
    }
    if meta:
        out.update(meta)
    out["episodes_detail"] = [
        {"seed": r.seed, "outcome": r.outcome, "steps": r.steps,
         "grasp": r.grasp, "lift": r.lift, "over_box": r.over_box,
         "min_pocket_dist": (None if math.isinf(r.min_pocket_dist)
                             else round(r.min_pocket_dist, 4))}
        for r in results
    ]
    return out
