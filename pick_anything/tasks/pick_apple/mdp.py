"""Custom MDP terms for the orange-picking-into-box task.

The task is a small phase machine driven by one shared per-env state object
(:class:`PickState`), cached on the environment and refreshed at most once per
control step (guarded by ``common_step_counter``), so every reward /
termination / observation term sees identical numbers regardless of the order
the managers evaluate them in.

2026-09-25: the scene carries a single fruit — the orange (user decision:
remove the apple / its red-cube RL stand-in). All former apple-as-target code
now targets the orange; the distractor-orange code is gone.

Phases (derived from physics, not a hand-switched schedule):

1. *approach* - claw near the orange, orange still on the table;
2. *pre-grasp (state A)* - orange center inside the claw's capture pocket
   (fixed offset in the gripper frame) and claw open wide enough that a pure
   close captures the fruit; one-shot stage bonus;
3. *grasped*  - orange lifted above ``GRASP_LIFT_Z`` while close to the jaw and
   pressed by it (net contact force on the orange), sticky per episode;
4. *carry*    - orange carried toward the hover point above the box;
5. *placed*   - orange released inside the box interior (box-local frame),
   settled, and the claw pulled away -> success termination.

The stage ladder is strictly increasing (state A +8 < grasp +15 < place +45
effective), so the policy always profits by advancing a stage instead of
farming an earlier one. Reaching state A requires no orange contact - the
open claw simply hovers around the fruit - and contact before the close is
not rewarded anywhere (knocking the orange away stays penalized via
``orange_disturbed``).

Penalties follow the task spec: disturbing the box before the orange is
grasped is punished harder than afterwards. Box disturbance is measured as
displacement from the spawn position, which is attribution-free (no reliance
on PhysX contact filter pairing) and immune to the constant table-resting
contact force. Robot collision penalties use a contact sensor on the arm
links (base and claw excluded): any contact on a link is undesired, and claw
contact far from the orange can only be table / tray, so no pair attribution
is needed there either.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from pick_anything.env_cfg import (
    BOX_SPAWN_Z,
    BOX_X_LIMITS,
    BOX_Y_LIMITS,
    FRUIT_SPAWN_Z,
    ROBOT_BASE_XY,
    TABLE_X_LIMITS,
    TABLE_Y_LIMITS,
    TABLE_TOP_Z,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --- asset geometry (see scripts/make_cardboard_box.py and env_cfg.py) ---
# RL 场景的抓取目标是 ~40 mm 橘子球（scale 0.55，USD 自带 23g 质量）
BOX_HEIGHT_M = 0.05  # tray wall height
IN_BOX_XY_LIMIT_M = 0.052  # box-local |x|,|y| bound for the orange center
IN_BOX_Z_MIN_M = 0.008
IN_BOX_Z_MAX_M = 0.052
HOVER_ABOVE_BOX_M = 0.14  # carry target height above the box origin

# --- phase thresholds ---
GRASP_LIFT_Z = 0.09  # orange center height that counts as "lifted"
GRASP_PROXIMITY_M = 0.085  # jaw-to-orange distance while lifted
GRASP_FORCE_N = 0.12  # net contact force on the orange while lifted
RELEASE_CLEARANCE_M = 0.055  # jaw must back off this far to score the place
SETTLED_SPEED_MPS = 0.20  # orange speed that counts as "settled in the box"

# --- stage A: capture-ready pre-grasp (claw placed so a pure close captures) ---
# Capture pocket center, fixed offset in the gripper body frame (the USD
# finger direction), proven on this USD lineage by Isaac-Lift-SO101: closing
# the jaw on any object centered inside the pocket traps it.
# 2026-09-23 修正：旧值 (-0.057,0,0) 落在指中段 V 口宽区，球/块都会
# 滑出（实测 0 留存）。lift 工作区同款爪的实测捕获点在指尖区：
# gripper body 系 (-0.0875, 0.015, 0)，捕获半径 0.038。
# 捕获口袋 = 两指接触垫中点在 gripper body 系的偏移（isaaclab_assets
# SO101，开爪 1.40 rad 下运行时实测；垫心常量取自教程 terms.py 同源
# USD）。注意固定垫比本 TCP 低 ~28mm、动垫高 ~27mm（筷子式错位垫），
# 抓球用 shelf 几何（固定垫托球心下 6-15mm）——见生成器注释。
POCKET_OFFSET_B = (0.0408, -0.0002, -0.0709)
PREGRASP_POCKET_M = 0.038  # orange center inside this radius of the pocket
# gripper travel: 0 rad = fully closed .. 1.745 rad = fully open (USD 实测；
# v0 旧注释方向相反). The ~40 mm orange needs a wide entry gap, so "open or
# half-open but still capture-ready" means joint >= 1.30 rad (near full open).
GRIPPER_TRAVEL_RAD = 1.745
PREGRASP_MIN_GRIPPER_RAD = 1.30

# --- robot collision penalties ---
# arm links that must touch nothing (base is fixed/mounted, claw is exempt so
# it may touch the orange); with self-collision off, any force here means the
# link hit the table / tray / the fruit / the ground. The claw penalty watches
# only the moving jaw: at the neutral reset pose the claw's fixed finger
# (gripper body) grazes the table with a constant ~2 N, while the jaw reads a
# clean zero baseline - and the jaw is the finger that actually reaches the
# table / tray first.
ARM_LINK_NAMES = ("shoulder", "upper_arm", "lower_arm", "wrist")
# 实测 body 列表：base/shoulder/upper_arm/lower_arm/wrist/gripper/
# moving_jaw_so101_v1（isaaclab_assets SO101；旧 leisaac 叫 jaw）
EE_BODY_NAMES = ("moving_jaw_so101_v1",)
ARM_CONTACT_FORCE_N = 1.0
EE_CONTACT_FORCE_N = 1.0
# claw contact is undesired only when it is NOT orange contact: either the jaw
# is far from the orange, or the orange is not being pressed (resting on the
# table the 23 g orange reads ~0.23 N, a squeeze reads well above that)
EE_CLEAR_OF_ORANGE_M = 0.07
ORANGE_REST_FORCE_N = 0.45

# --- failure thresholds ---
BOX_UPRIGHT_MIN = 0.70  # box up-axis z component; below -> tipped over
ORANGE_GONE_Z = 0.02  # orange below this has fallen off the table
TABLE_FALL_MARGIN_M = 0.08  # xy slack past the table edges before "off table"
BOX_GONE_MARGIN_M = 0.15

# --- disturbance penalties (displacement deadband / full-penalty span) ---
BOX_DEADBAND_M = 0.012
BOX_SPAN_M = 0.09
ORANGE_DEADBAND_M = 0.03
ORANGE_SPAN_M = 0.12

# --- spawn-region shaping (same distribution as env_cfg.sample_object_positions) ---
_SECTOR_CENTERS = {"box": 0.21, "fruit": 0.21}
_SECTOR_HALVES = {"box": 0.05, "fruit": 0.09}
_SECTOR_ANGLES_DEG = {"box": 30.0, "fruit": 55.0}
_RECTS = {
    "box": (BOX_X_LIMITS, BOX_Y_LIMITS),
    "fruit": (TABLE_X_LIMITS, TABLE_Y_LIMITS),
}

_MIN_DIFFICULTY = 0.2
_EMA_DECAY = 0.97
_REJECTION_ROUNDS = 48
_MIN_IMPROVEMENT_M = 0.005  # progress payments need this much new ground

# --- tray placement (see sample_placements) ---
# tray half-diagonal 0.113 + orange radius 0.0203: an orange center this far
# from the box center (Euclidean) is outside the footprint at any box yaw
BOX_FRUIT_CLEAR_M = 0.135
# candidate slots stay inside the arm's proven placement envelope: the polar
# box sector tops out at r = 0.26 m, so the fallback must not exceed it either
BOX_MAX_RADIUS_M = 0.26
_CANDIDATE_RADII = tuple(round(0.16 + 0.02 * i, 3) for i in range(6))
_CANDIDATE_ANGLES_DEG = tuple(range(-60, 61, 5))


def _t(buf):
    """Normalize an IsaacLab 3.0 ProxyArray buffer to torch.Tensor."""
    return buf.torch if hasattr(buf, "torch") else buf


def _polar_sector(n: int, kind: str, difficulty: float, device) -> torch.Tensor:
    """Sample n (x, y) points in the reach annulus sector, clipped to the rect.

    Radii shrink toward the sector center and angles tighten as difficulty
    drops, so early training deals with a compact, easy workspace.
    """
    r = _SECTOR_CENTERS[kind] + (2 * torch.rand(n, device=device) - 1) * _SECTOR_HALVES[kind] * difficulty
    half_angle = math.radians(_SECTOR_ANGLES_DEG[kind]) * max(difficulty, 0.3)
    theta = (2 * torch.rand(n, device=device) - 1) * half_angle
    x = ROBOT_BASE_XY[0] + r * torch.sin(theta)
    y = ROBOT_BASE_XY[1] + r * torch.cos(theta)
    return torch.stack([x, y], dim=-1)


def _in_rect(xy: torch.Tensor, kind: str) -> torch.Tensor:
    (x0, x1), (y0, y1) = _RECTS[kind]
    return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)


def _box_candidates(device) -> torch.Tensor:
    """Fixed env-local xy slots for the tray, filtered to the box rect."""
    pts = [
        (
            ROBOT_BASE_XY[0] + r * math.sin(math.radians(a)),
            ROBOT_BASE_XY[1] + r * math.cos(math.radians(a)),
        )
        for r in _CANDIDATE_RADII
        for a in _CANDIDATE_ANGLES_DEG
    ]
    cand = torch.tensor(pts, dtype=torch.float32, device=device)
    return cand[_in_rect(cand, "box")]


def _push_apart(mover: torch.Tensor, anchor: torch.Tensor, min_sep: float, kind: str = "fruit") -> torch.Tensor:
    """Guarantee pairwise separation by pushing movers out along their line.

    Rect-clamped so the push can never shove an object off its allowed rect.
    If the clamp re-violates (edge cases), aim the leftover at the rect center,
    which always has ``min_sep`` of room in the 0.6 x 0.3 m fruit rect.
    """

    (x0, x1), (y0, y1) = _RECTS[kind]
    delta = mover - anchor
    d = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    target = anchor + delta / d * (min_sep + 0.005)
    target[:, 0] = target[:, 0].clamp(x0, x1)
    target[:, 1] = target[:, 1].clamp(y0, y1)
    viol = (d.squeeze(-1) < min_sep).unsqueeze(-1)
    out = torch.where(viol, target, mover)
    center = torch.tensor([(x0 + x1) / 2, (y0 + y1) / 2], device=mover.device, dtype=mover.dtype)
    to_center = center - anchor
    to_center = to_center / to_center.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    target2 = anchor + to_center * (min_sep + 0.005)
    target2[:, 0] = target2[:, 0].clamp(x0, x1)
    target2[:, 1] = target2[:, 1].clamp(y0, y1)
    still = ((out - anchor).norm(dim=-1) < min_sep).unsqueeze(-1)
    return torch.where(still, target2, out)


def _snap_box_clear(box: torch.Tensor, orange: torch.Tensor) -> torch.Tensor:
    """Replace tray spots that overlap the orange (or the rect) with the clearest candidate slot.

    The difficulty-scaled polar sectors are near-concentric - at low difficulty
    both collapse onto the same ~0.21 m radius, where the fruit-to-tray
    separation constraint is geometrically unsatisfiable and rejection sampling
    exhausts its rounds. Snapping invalid tray draws to the candidate slot
    farthest from the fruit makes "the orange never spawns in the tray" a hard
    guarantee instead of a probabilistic one.
    """

    dist = (box - orange).norm(dim=-1)
    clear = _in_rect(box, "box") & (dist >= BOX_FRUIT_CLEAR_M)
    cand = _box_candidates(box.device)  # (m, 2)
    scores = (cand.unsqueeze(0) - orange.unsqueeze(1)).norm(dim=-1)  # (n, m)
    best = cand[scores.argmax(dim=-1)]
    out = torch.where(clear.unsqueeze(-1), box, best)
    # final radial nudge for slots still grazing the orange corner: pushes
    # only violators, rect-clamped, so a graze can never become an overlap
    out = _push_apart(out, orange, BOX_FRUIT_CLEAR_M, kind="box")
    # keep the tray inside the arm's placement envelope (push clamps along the
    # rect can overshoot the reachable annulus in the rect corners)
    base = torch.tensor(ROBOT_BASE_XY, device=out.device, dtype=out.dtype)
    rel = out - base
    r = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return base + rel / r * torch.minimum(r, torch.full_like(r, BOX_MAX_RADIUS_M))


def sample_placements(n: int, difficulty: float, device) -> dict[str, torch.Tensor]:
    """Vectorized twin of ``env_cfg.sample_object_positions`` (box/orange).

    Joint rejection-samples the separation constraint (the near-concentric
    sectors at low difficulty make it partly unsatisfiable, so the rounds can
    exhaust), then applies deterministic fallbacks so validity is a hard
    guarantee rather than a probabilistic one: straggler trays snap to the
    candidate slot farthest from the orange, and a straggler orange is pushed
    clear of the tray.
    """
    box = _polar_sector(n, "box", difficulty, device)
    orange = _polar_sector(n, "fruit", difficulty, device)

    def constraints() -> tuple[torch.Tensor, torch.Tensor]:
        # recomputed fresh from the CURRENT draws every round, so an accepted
        # object can never stay valid against a stale partner position
        ok_box = _in_rect(box, "box")
        ok_orange = _in_rect(orange, "fruit") & ((orange - box).norm(dim=-1) >= BOX_FRUIT_CLEAR_M)
        return ok_box, ok_orange

    ok_box, ok_orange = constraints()
    for _ in range(_REJECTION_ROUNDS):
        for xy, ok, kind in ((box, ok_box, "box"), (orange, ok_orange, "fruit")):
            resample = ~ok
            if resample.any():
                xy[resample] = _polar_sector(int(resample.sum()), kind, difficulty, device)
        ok_box, ok_orange = constraints()
        if ok_box.all() and ok_orange.all():
            break
    box = _snap_box_clear(box, orange)
    # last resort: the fruit rect is far roomier than the narrow box band, so
    # the escaper is the orange, not the tray
    orange = _push_apart(orange, box, BOX_FRUIT_CLEAR_M)
    return {"box": box, "orange": orange}


class PickState:
    """Shared per-env task state: phases, geometry caches and the curriculum knob.

    Created lazily and pinned on the env object (``env.pick_state``) so all
    manager terms share one instance. :meth:`update` runs once per control
    step regardless of which term calls it first.
    """

    def __init__(self, env: ManagerBasedRLEnv):
        self.device = env.device
        n = env.num_envs
        robot = env.scene["robot"]
        # 动指 body 名：isaaclab_assets SO101 为 moving_jaw_so101_v1（旧
        # leisaac USD 叫 jaw，2026-09-25 换资产后此处漏改导致首次 reset 崩溃）
        self.jaw_id = int(robot.find_bodies(["moving_jaw_so101_v1"])[0][0])
        self.gripper_id = int(robot.find_joints(["gripper"])[0][0])

        # geometry caches (env-local frame, i.e. world minus env origin)
        self.jaw_pos = torch.zeros(n, 3, device=self.device)
        self.orange_pos = torch.zeros(n, 3, device=self.device)
        self.orange_vel = torch.zeros(n, 3, device=self.device)
        self.box_pos = torch.zeros(n, 3, device=self.device)
        self.box_up_z = torch.ones(n, device=self.device)
        self.orange_local = torch.zeros(n, 3, device=self.device)
        self.pocket_pos = torch.zeros(n, 3, device=self.device)

        # phases
        self.held_now = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.just_grasped = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.has_grasped = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.pregrasp_now = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.just_pregrasp = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.has_pregrasp = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.orange_in_box = torch.zeros(n, dtype=torch.bool, device=self.device)  # position-only
        self.over_box = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(n, dtype=torch.bool, device=self.device)

        # claw state and robot collision forces
        self.gripper_pos = torch.zeros(n, device=self.device)
        self.gripper_vel = torch.zeros(n, device=self.device)
        self.arm_force = torch.zeros(n, device=self.device)
        self.ee_force = torch.zeros(n, device=self.device)
        self.orange_force = torch.zeros(n, device=self.device)
        self._contact_cols: tuple[list[int], list[int]] | None = None

        # derived quantities reused by several terms
        self.dist_jaw_orange = torch.zeros(n, device=self.device)
        self.dist_orange_pocket = torch.zeros(n, device=self.device)
        self.dist_orange_hover = torch.zeros(n, device=self.device)
        self.box_disp = torch.zeros(n, device=self.device)
        self.orange_disp = torch.zeros(n, device=self.device)

        # progress-based shaping (best-so-far + per-step payment); each meter
        # of ground is paid once, so no dense term can be farmed by idling
        self.best_approach = torch.full((n,), 1.0, device=self.device)
        self.best_hover = torch.full((n,), 1.0, device=self.device)
        self.best_height = torch.full((n,), 0.0, device=self.device)
        self.pay_approach = torch.zeros(n, device=self.device)
        self.pay_hover = torch.zeros(n, device=self.device)
        self.pay_height = torch.zeros(n, device=self.device)

        # spawn references for disturbance penalties
        self.spawn_box_xy = torch.zeros(n, 2, device=self.device)
        self.spawn_orange_xy = torch.zeros(n, 2, device=self.device)

        # per-episode bookkeeping + curriculum
        self.succeeded_episode = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.success_ema = 0.0
        self.difficulty = _MIN_DIFFICULTY
        self._last_step = -1

    # ------------------------------------------------------------------ #
    @classmethod
    def get(cls, env: ManagerBasedRLEnv) -> PickState:
        state = getattr(env, "pick_state", None)
        if state is None:
            state = cls(env)
            env.pick_state = state
        return state

    def _resolve_contact_cols(self, env: ManagerBasedRLEnv) -> tuple[list[int], list[int]]:
        """Map arm-link / claw bodies to their columns in the robot contact sensor.

        The sensor reports one force column per matching rigid body, in the
        sensor view's own order; matching by name (via ``sensor.body_names``)
        keeps this correct regardless of that order.
        """

        if self._contact_cols is None:
            sensor = env.scene.sensors["robot_contact"]
            names = [str(name) for name in sensor.body_names]
            arm_cols = [i for i, name in enumerate(names) if name in ARM_LINK_NAMES]
            ee_cols = [i for i, name in enumerate(names) if name in EE_BODY_NAMES]
            if not arm_cols or not ee_cols:
                raise RuntimeError(
                    f"robot_contact sensor bodies {names} lack the expected arm links "
                    f"{ARM_LINK_NAMES} / claw bodies {EE_BODY_NAMES}"
                )
            self._contact_cols = (arm_cols, ee_cols)
        return self._contact_cols

    def update(self, env: ManagerBasedRLEnv) -> None:
        if self._last_step == env.common_step_counter:
            return
        self._last_step = env.common_step_counter

        origins = _t(env.scene.env_origins)  # (n, 3)
        robot = env.scene["robot"]
        orange = env.scene["orange"]
        box = env.scene["cardboard_box"]

        self.jaw_pos = _t(robot.data.body_pos_w)[:, self.jaw_id] - origins
        self.orange_pos = _t(orange.data.root_pos_w) - origins
        self.orange_vel = _t(orange.data.root_lin_vel_w)
        box_pos_w = _t(box.data.root_pos_w)
        self.box_pos = box_pos_w - origins
        box_quat = _t(box.data.root_quat_w)

        # box up-axis z component: e_z expressed in the box frame, z entry
        e_z = torch.zeros_like(self.box_pos)
        e_z[:, 2] = 1.0
        self.box_up_z = quat_apply_inverse(box_quat, e_z)[:, 2]

        # orange in box-local frame
        orange_local = quat_apply_inverse(box_quat, self.orange_pos - self.box_pos)
        self.orange_local = orange_local
        self.orange_in_box = (
            (orange_local[:, 0].abs() < IN_BOX_XY_LIMIT_M)
            & (orange_local[:, 1].abs() < IN_BOX_XY_LIMIT_M)
            & (orange_local[:, 2] > IN_BOX_Z_MIN_M)
            & (orange_local[:, 2] < IN_BOX_Z_MAX_M)
        )
        self.over_box = (orange_local[:, :2].norm(dim=-1) < IN_BOX_XY_LIMIT_M + 0.03) & (
            self.orange_pos[:, 2] > self.box_pos[:, 2]
        )

        # grasp phase
        sensor = env.scene.sensors["orange_contact"]
        orange_force = _t(sensor.data.net_forces_w).reshape(env.num_envs, -1, 3).norm(dim=-1).squeeze(-1)
        self.orange_force = orange_force
        prev = self.has_grasped.clone()
        self.held_now = (
            (self.orange_pos[:, 2] > GRASP_LIFT_Z)
            & ((self.jaw_pos - self.orange_pos).norm(dim=-1) < GRASP_PROXIMITY_M)
            & (orange_force > GRASP_FORCE_N)
        )
        self.just_grasped = self.held_now & ~prev
        self.has_grasped = self.has_grasped | self.held_now

        # carry / release
        hover = self.box_pos.clone()
        hover[:, 2] += HOVER_ABOVE_BOX_M
        self.dist_orange_hover = (self.orange_pos - hover).norm(dim=-1)
        self.dist_jaw_orange = (self.jaw_pos - self.orange_pos).norm(dim=-1)
        speed = self.orange_vel.norm(dim=-1)
        released = self.dist_jaw_orange > RELEASE_CLEARANCE_M
        self.success = self.orange_in_box & released & (speed < SETTLED_SPEED_MPS) & self.has_grasped
        self.succeeded_episode = self.succeeded_episode | self.success

        # claw opening and robot collision forces (net force per rigid body,
        # max over the recent physics substeps so a one-substep tap between
        # control steps cannot slip through the decimation gap)
        self.gripper_pos = _t(robot.data.joint_pos)[:, self.gripper_id]
        self.gripper_vel = _t(robot.data.joint_vel)[:, self.gripper_id]
        robot_sensor = env.scene.sensors["robot_contact"]
        history = robot_sensor.data.net_forces_w_history
        frames = _t(history) if history is not None else _t(robot_sensor.data.net_forces_w).unsqueeze(1)
        per_body_force = frames.norm(dim=-1).amax(dim=1)  # (n, num_sensor_bodies)
        arm_cols, ee_cols = self._resolve_contact_cols(env)
        self.arm_force = per_body_force[:, arm_cols].amax(dim=-1)
        self.ee_force = per_body_force[:, ee_cols].amax(dim=-1)

        # capture pocket: where the orange center must sit for a pure jaw
        # close to trap it (fixed offset in the gripper body frame)
        gripper_pos = _t(robot.data.body_pos_w)[:, self.gripper_id] - origins
        gripper_quat = _t(robot.data.body_quat_w)[:, self.gripper_id]
        offset = torch.tensor(POCKET_OFFSET_B, device=self.device, dtype=torch.float32).expand(env.num_envs, 3)
        self.pocket_pos = gripper_pos + quat_apply(gripper_quat, offset)
        self.dist_orange_pocket = (self.orange_pos - self.pocket_pos).norm(dim=-1)

        # stage A: capture-ready pre-grasp - orange inside the capture pocket
        # and claw open enough that closing captures it; one-shot per
        # episode, paid by pregrasp_onset
        orange_down = (~self.has_grasped) & (self.orange_pos[:, 2] < GRASP_LIFT_Z)
        claw_capture_ready = self.gripper_pos >= PREGRASP_MIN_GRIPPER_RAD
        self.pregrasp_now = orange_down & (self.dist_orange_pocket < PREGRASP_POCKET_M) & claw_capture_ready
        self.just_pregrasp = self.pregrasp_now & ~self.has_pregrasp
        self.has_pregrasp = self.has_pregrasp | self.pregrasp_now

        # --- progress shaping (pay once per meter of improvement) ---
        # approach: claw-to-orange distance, only while the orange is low/down
        gate_approach = ~self.held_now & (self.orange_pos[:, 2] < GRASP_LIFT_Z)
        improved = gate_approach & (self.dist_jaw_orange < self.best_approach - _MIN_IMPROVEMENT_M)
        self.pay_approach = torch.where(improved, self.best_approach - self.dist_jaw_orange, torch.zeros_like(improved, dtype=torch.float32))
        self.best_approach = torch.where(improved, self.dist_jaw_orange, self.best_approach)

        # carry height: orange climb while grasped and not yet over the box
        gate_height = self.has_grasped & ~self.over_box
        improved = gate_height & (self.orange_pos[:, 2] > self.best_height + _MIN_IMPROVEMENT_M)
        self.pay_height = torch.where(improved, self.orange_pos[:, 2] - self.best_height, torch.zeros_like(improved, dtype=torch.float32))
        self.best_height = torch.where(improved, self.orange_pos[:, 2], self.best_height)

        # carry to hover: only while actually holding, or once the orange is
        # over/inside the tray (a released orange falling into the box still
        # makes progress toward the hover point and should be rewarded)
        gate_hover = self.has_grasped & (self.held_now | self.over_box | self.orange_in_box)
        improved = gate_hover & (self.dist_orange_hover < self.best_hover - _MIN_IMPROVEMENT_M)
        self.pay_hover = torch.where(improved, self.best_hover - self.dist_orange_hover, torch.zeros_like(improved, dtype=torch.float32))
        self.best_hover = torch.where(improved, self.dist_orange_hover, self.best_hover)

        # disturbance
        self.box_disp = (self.box_pos[:, :2] - self.spawn_box_xy).norm(dim=-1)
        self.orange_disp = (self.orange_pos[:, :2] - self.spawn_orange_xy).norm(dim=-1)

    def on_env_reset(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
        """Fold finished episodes into the curriculum, then clear per-episode state.

        Called by the reset event term right before the objects are teleported.
        """
        ids = env_ids
        if ids.numel() > 0:
            rate = self.succeeded_episode[ids].float().mean().item()
            self.success_ema = _EMA_DECAY * self.success_ema + (1.0 - _EMA_DECAY) * rate
            # slope 1.0: hold each difficulty level until the success rate
            # actually consolidates before the spawn region widens
            self.difficulty = min(1.0, _MIN_DIFFICULTY + 1.0 * self.success_ema)
        self.has_grasped[ids] = False
        self.just_grasped[ids] = False
        self.has_pregrasp[ids] = False
        self.just_pregrasp[ids] = False
        self.succeeded_episode[ids] = False
        self.success[ids] = False
        self.best_approach[ids] = 1.0
        self.best_hover[ids] = 1.0
        self.best_height[ids] = 0.0


# ---------------------------------------------------------------------- #
# Events (reset)
# ---------------------------------------------------------------------- #


def place_objects_and_track(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    box_yaw_half_rad: float = 0.3,
    p_in_hand: float = 0.25,
):
    """Reset-mode event: respawn box/orange and the task state machine.

    Positions follow :func:`sample_placements` (difficulty-scaled twin of the
    demo sampler); the orange gets a random yaw, the box a small one; the
    orange is dropped a few cm above the table and the box onto its footprint
    so physics settles them within the first steps.

    Spawn-under-claw curriculum: for a ``p_in_hand`` share of the resets, the
    orange starts on the table directly under the jaw (guarded to stay clear
    of the tray), so grasp-carry-release gets dense practice even before the
    approach skill is reliable. The claw sags several cm after the joint
    reset (soft servos), so pinning the orange into the claw pocket itself is
    not robust - under-the-claw keeps the curriculum without fragile offsets.
    Must run AFTER the joint reset terms - it reads the post-reset jaw pose.
    """
    state = PickState.get(env)
    state.on_env_reset(env, env_ids)

    n = env_ids.numel()
    device = env.device
    placements = sample_placements(n, state.difficulty, device)

    in_hand = torch.rand(n, device=device) < p_in_hand
    if p_in_hand > 0.0 and in_hand.any():
        robot = env.scene["robot"]
        origins = _t(env.scene.env_origins)[env_ids]
        jaw_xy = (_t(robot.data.body_pos_w)[env_ids][:, state.jaw_id] - origins)[:, :2]
        clear_of_box = (jaw_xy - placements["box"]).norm(dim=-1) >= BOX_FRUIT_CLEAR_M
        on_table = _in_rect(jaw_xy, "fruit")
        take = in_hand & clear_of_box & on_table
        placements["orange"][take] = jaw_xy[take]

    yaw_fruit = torch.rand(n, device=device) * (2.0 * math.pi)
    yaw_box = (2 * torch.rand(n, device=device) - 1) * box_yaw_half_rad * max(state.difficulty, 0.3)
    zero = torch.zeros(n, device=device)
    # write_root_pose expects world-frame poses; placements are env-local
    origins = _t(env.scene.env_origins)[env_ids]

    def pose(xy: torch.Tensor, z: float, yaw: torch.Tensor) -> torch.Tensor:
        # IsaacLab 3.0 quaternion order is (x, y, z, w); pure-z rotation
        quat = torch.stack([zero, zero, torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)
        pos = torch.stack([xy[:, 0], xy[:, 1], torch.full_like(xy[:, 0], z)], dim=-1) + origins
        return torch.cat([pos, quat], dim=-1)

    objects = {
        "cardboard_box": (placements["box"], BOX_SPAWN_Z, yaw_box),
        "orange": (placements["orange"], FRUIT_SPAWN_Z, yaw_fruit),
    }
    for name, (xy, z, yaw) in objects.items():
        asset = env.scene[name]
        asset.write_root_pose_to_sim_index(root_pose=pose(xy, z, yaw), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(n, 6, device=device), env_ids=env_ids
        )

    state.spawn_box_xy[env_ids] = placements["box"]
    state.spawn_orange_xy[env_ids] = placements["orange"]
    # stale caches are refreshed on the next PickState.update call
    state._last_step = -1


# ---------------------------------------------------------------------- #
# Observations
# ---------------------------------------------------------------------- #


def orange_state_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = PickState.get(env)
    state.update(env)
    return torch.cat([state.orange_pos, state.orange_vel], dim=-1)


def box_state_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = PickState.get(env)
    state.update(env)
    return torch.cat([state.box_pos, state.box_up_z.unsqueeze(-1)], dim=-1)


def ee_to_targets(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = PickState.get(env)
    state.update(env)
    hover = state.box_pos.clone()
    hover[:, 2] += HOVER_ABOVE_BOX_M
    # orange-to-pocket is the stage-A error signal the pre-grasp bonus pays on
    return torch.cat(
        [state.orange_pos - state.jaw_pos, state.orange_pos - state.pocket_pos, hover - state.jaw_pos],
        dim=-1,
    )


def task_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = PickState.get(env)
    state.update(env)
    return torch.stack(
        [
            state.orange_pos[:, 2] - TABLE_TOP_Z,
            state.has_pregrasp.float(),
            state.has_grasped.float(),
            state.orange_in_box.float(),
        ],
        dim=-1,
    )


def jaw_orange_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Contact-force magnitude on the orange (1-D, clipped)."""
    state = PickState.get(env)
    state.update(env)
    return state.orange_force.clamp(0.0, 20.0).unsqueeze(-1)


# ---------------------------------------------------------------------- #
# Rewards
# ---------------------------------------------------------------------- #


def approach_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Meters of claw-to-orange ground gained this step (paid once per meter).

    Active only while the orange is down (not currently carried), so the claw
    is shaped toward the orange during the approach phase.
    """
    state = PickState.get(env)
    state.update(env)
    return state.pay_approach


def approach_presence(env: ManagerBasedRLEnv, std: float = 0.1) -> torch.Tensor:
    """Small continuous proximity shaping toward the orange (pre-grasp only).

    Gives the policy a dense gradient while it is still learning to move at
    all; bounded well below the completion reward so it cannot be farmed.
    """
    state = PickState.get(env)
    state.update(env)
    gate = (~state.has_grasped).float()
    return (1.0 - torch.tanh(state.dist_jaw_orange / std)) * gate


def height_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Meters of orange climb gained this step while grasped and not over the box.

    Pays until the orange is carried high enough to clear the 5 cm tray walls.
    """
    state = PickState.get(env)
    state.update(env)
    return state.pay_height


def hover_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Meters of orange-to-hover-point ground gained this step while carrying.

    The gate also admits a released orange over/inside the tray, so dropping
    the orange into the box from a hover still earns the final approach meters.
    """
    state = PickState.get(env)
    state.update(env)
    return state.pay_hover


class grasp_onset(ManagerTermBase):
    """One-shot bonus when the orange is first pinched and lifted."""

    def __call__(self, env: ManagerBasedRLEnv, bonus: float = 4.0) -> torch.Tensor:
        state = PickState.get(env)
        state.update(env)
        return float(bonus) * state.just_grasped.float()


def gripper_openness(env: ManagerBasedRLEnv, std: float = 0.1) -> torch.Tensor:
    """Dense shaping: hold the claw capture-ready while it nears the orange.

    The main grasping failure is arriving with a half-shut claw and batting
    the orange away; this pays proportionally to claw opening times capture-
    pocket proximity, and stops permanently once state A is first reached, so
    it cannot be farmed by idling next to the fruit with an open claw.
    """

    state = PickState.get(env)
    state.update(env)
    gate = (~state.has_pregrasp).float()
    proximity = 1.0 - torch.tanh(state.dist_orange_pocket / std)
    open_fraction = (state.gripper_pos / GRIPPER_TRAVEL_RAD).clamp(0.0, 1.0)
    return proximity * open_fraction * gate


def gripper_closing(env: ManagerBasedRLEnv, pocket_std: float = 0.04, vel_ref: float = 1.5) -> torch.Tensor:
    """Dense shaping: closing the claw while the orange sits centered in the capture pocket.

    Bridges the state-A -> grasp reward cliff (observed on run 14-41: a
    low-entropy policy farmed approach/pregrasp and hovered forever because
    the +15 lift is too sparse to stumble into). Pays closing speed while the
    orange is centered in the pocket; a sloppy close that knocks the orange
    out of the pocket stops paying immediately, and the term gates off once
    the orange is grasped so it cannot be farmed afterwards.
    """

    state = PickState.get(env)
    state.update(env)
    gate = (~state.has_grasped).float()
    centered = 1.0 - torch.tanh(state.dist_orange_pocket / pocket_std)
    closing = (state.gripper_vel / vel_ref).clamp(0.0, 1.0)
    return centered * closing * gate


class pregrasp_onset(ManagerTermBase):
    """Stage-1 one-shot bonus: the arm reached capture-ready state A.

    Requires no orange contact - the open claw hovers around the fruit - and
    pays strictly less than ``grasp_onset``, so the ladder (A < grasp <
    place) always rewards advancing to the next stage.
    """

    def __call__(self, env: ManagerBasedRLEnv, bonus: float = 8.0) -> torch.Tensor:
        state = PickState.get(env)
        state.update(env)
        return float(bonus) * state.just_pregrasp.float()


def arm_collision(env: ManagerBasedRLEnv, threshold: float = ARM_CONTACT_FORCE_N) -> torch.Tensor:
    """Per-step penalty while any arm link (base and claw excluded) presses something.

    With self-collision disabled, any force on shoulder / upper_arm /
    lower_arm / wrist means the link hit the table, the tray, the orange or
    the ground - all undesired per the task spec.
    """

    state = PickState.get(env)
    state.update(env)
    return (state.arm_force > threshold).float()


def ee_bad_contact(
    env: ManagerBasedRLEnv,
    threshold: float = EE_CONTACT_FORCE_N,
    clear_of_orange: float = EE_CLEAR_OF_ORANGE_M,
    orange_rest_force: float = ORANGE_REST_FORCE_N,
) -> torch.Tensor:
    """Per-step penalty while the claw presses something that is not the orange.

    Claw contact is fine only against the orange; force on the jaw while it is
    far from the orange, or while the orange is not being squeezed (resting on
    the table it reads ~0.23 N), can only come from the table top or the tray
    walls. Only the moving jaw is watched - the gripper body carries a
    constant ~2 N table-graze at the neutral pose (see constants).
    """

    state = PickState.get(env)
    state.update(env)
    not_orange = (state.dist_jaw_orange > clear_of_orange) | (state.orange_force < orange_rest_force)
    return ((state.ee_force > threshold) & not_orange).float()


class place_success(ManagerTermBase):
    """Terminal bonus for a released, settled orange inside the box."""

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        state = PickState.get(env)
        state.update(env)
        return state.success.float()


def box_disturbed(env: ManagerBasedRLEnv, deadband: float = BOX_DEADBAND_M, span: float = BOX_SPAN_M) -> torch.Tensor:
    """Box displacement penalty, active only before the orange is grasped.

    Touching / knocking the tray before picking the orange is explicitly
    undesired behavior; after the grasp the gentler ``box_disturbed_late``
    term takes over so legitimate placement contact is barely punished.
    """
    state = PickState.get(env)
    state.update(env)
    gate = (~state.has_grasped).float()
    return (((state.box_disp - deadband) / span).clamp(0.0, 1.0)) * gate


def box_disturbed_late(
    env: ManagerBasedRLEnv, deadband: float = 2.0 * BOX_DEADBAND_M, span: float = 0.13
) -> torch.Tensor:
    """Small residual penalty for shoving the tray after the grasp."""
    state = PickState.get(env)
    state.update(env)
    gate = state.has_grasped.float()
    return (((state.box_disp - deadband) / span).clamp(0.0, 1.0)) * gate


def orange_disturbed(env: ManagerBasedRLEnv, deadband: float = ORANGE_DEADBAND_M, span: float = ORANGE_SPAN_M) -> torch.Tensor:
    """Knocking the orange away from its spawn while it is not yet held.

    The continuous counterpart of the terminal drop penalty: a sweeping claw
    that sends the orange rolling gets punished every step it stays displaced,
    so the approach itself is shaped to be careful. Small repositioning
    within the deadband (3 cm) stays free - grasping nudges the fruit.
    """
    state = PickState.get(env)
    state.update(env)
    gate = (~state.has_grasped).float()
    return (((state.orange_disp - deadband) / span).clamp(0.0, 1.0)) * gate


# ---------------------------------------------------------------------- #
# Terminations
# ---------------------------------------------------------------------- #


def orange_in_box(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Success: orange released and settled inside the tray."""
    state = PickState.get(env)
    state.update(env)
    return state.success


def orange_left_table(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = PickState.get(env)
    state.update(env)
    (x0, x1), (y0, y1) = _RECTS["fruit"]
    off_xy = (
        (state.orange_pos[:, 0] < x0 - TABLE_FALL_MARGIN_M)
        | (state.orange_pos[:, 0] > x1 + TABLE_FALL_MARGIN_M)
        | (state.orange_pos[:, 1] < y0 - TABLE_FALL_MARGIN_M)
        | (state.orange_pos[:, 1] > y1 + TABLE_FALL_MARGIN_M)
    )
    fell = state.orange_pos[:, 2] < ORANGE_GONE_Z
    return off_xy | fell


def box_tipped_or_gone(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = PickState.get(env)
    state.update(env)
    (x0, x1), (y0, y1) = _RECTS["fruit"]
    off_xy = (
        (state.box_pos[:, 0] < x0 - BOX_GONE_MARGIN_M)
        | (state.box_pos[:, 0] > x1 + BOX_GONE_MARGIN_M)
        | (state.box_pos[:, 1] < y0 - BOX_GONE_MARGIN_M)
        | (state.box_pos[:, 1] > y1 + BOX_GONE_MARGIN_M)
    )
    return (state.box_up_z < BOX_UPRIGHT_MIN) | off_xy
