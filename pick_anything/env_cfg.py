"""Scene configuration: table scene + SO101 follower arm + one graspable orange.

The table scene USD is referenced as one asset (its original cube prim was
removed; the table is reshaped to a 1.0 x 0.45 m rectangle). The orange is a
standalone rigid object converted from a GraspNet asset, scaled to ~40 mm to
fit the SO101 gripper opening. An open-top cardboard tray (16 x 16 x 5 cm)
shares the table. Each run spawns both at random positions inside the arm's
reach (see :func:`sample_object_positions`).

2026-09-25: the scene carries a single fruit — the orange (user decision:
remove the apple / its red-cube RL stand-in, keep only the orange). The
orange is the pick target everywhere downstream (mdp, reset bank, viewer);
there is no distractor fruit anymore.
"""

import math
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from pick_anything.paths import CARDBOARD_BOX_USD, ORANGE_USD, TABLE_WITH_CUBE_USD
from pick_anything.robots import SO101_FOLLOWER_CFG

TABLE_TOP_Z = 0.041  # measured top surface of the reshaped table

# Grasp-friendly size: ~56 mm diameter. 几何账（2026-09-25 教程同款爪实测）：
# 爪最低点 ≈ 垫心中点下方 20mm，桌面抓球的离桌间隙 = 球半径 - 20mm——
# 半径 <28mm 时爪尖蹭桌（40mm 球实测间隙 0.3mm，卡死压桌 24N）。56mm 球
# 给出 ~8mm 间隙（与教程 34mm vial + 矮垫的间隙一致），且仍在爪全开
# (~69mm tip gap) 的进入范围内。USD 自带 23g 质量为绝对值，缩放不变。
FRUIT_SCALE_ORANGE = 0.775
ORANGE_RADIUS_M = 0.0281  # half of the ~56.2 mm scaled bbox
ORANGE_REST_Z = TABLE_TOP_Z + ORANGE_RADIUS_M  # center height resting on the table
FRUIT_SPAWN_Z = 0.09  # small drop above the table; physics settles it

# Open-top cardboard tray 16 x 16 x 5 cm; its USD origin sits at the bottom
# face center, so it spawns just above the table top.
BOX_SPAWN_Z = TABLE_TOP_Z + 0.005

# --- random spawn region: annulus sector in front of the arm, clipped to table ---
# Arm base (0.35, -0.64) faces +y; zero-pose jaw reach is 0.30 m.
ROBOT_BASE_XY = (0.35, -0.64)
REACH_MIN_M = 0.12
REACH_MAX_M = 0.30
SECTOR_HALF_ANGLE_DEG = 55.0
# Table top x [-0.145, 0.855], y [-0.677, -0.227]; keep the orange clear of the edges.
TABLE_X_LIMITS = (0.05, 0.65)
TABLE_Y_LIMITS = (-0.60, -0.30)

# Box-specific sampling: narrower sector (the tray extends laterally) and a
# reach window shrunk so the tray body still sits in front of the arm.
BOX_REACH_MIN_M = 0.16
BOX_REACH_MAX_M = 0.24  # 0.26 时最外圈对 TRANSPORT 悬停不可达（负载 IK valid 12.5%）
BOX_SECTOR_HALF_ANGLE_DEG = 30.0
# Keep the full 16 cm footprint on the table (half-size + wall + margin).
BOX_X_LIMITS = (0.14, 0.56)
BOX_Y_LIMITS = (-0.51, -0.39)
# Orange center stays outside the tray footprint (+ radius + margin).
BOX_FRUIT_MIN_SEPARATION_M = 0.13


def sample_object_positions(rng: random.Random | None = None) -> dict[str, tuple[float, float]]:
    """Random on-table (x, y) for the tray and the orange.

    Everything is drawn inside the arm's reach (annulus sector clipped to the
    table), the tray keeps its footprint on the table, and the orange stays
    clear of the tray.
    """

    def in_reach(
        r_min: float,
        r_max: float,
        half_angle_deg: float,
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
    ) -> tuple[float, float]:
        while True:
            r = math.sqrt(rng.uniform(r_min**2, r_max**2))
            th = math.radians(rng.uniform(-half_angle_deg, half_angle_deg))
            x = ROBOT_BASE_XY[0] + r * math.sin(th)
            y = ROBOT_BASE_XY[1] + r * math.cos(th)
            if x_lim[0] <= x <= x_lim[1] and y_lim[0] <= y <= y_lim[1]:
                return (round(x, 3), round(y, 3))

    rng = rng or random.Random()

    box = in_reach(BOX_REACH_MIN_M, BOX_REACH_MAX_M, BOX_SECTOR_HALF_ANGLE_DEG, BOX_X_LIMITS, BOX_Y_LIMITS)
    orange = in_reach(REACH_MIN_M, REACH_MAX_M, SECTOR_HALF_ANGLE_DEG, TABLE_X_LIMITS, TABLE_Y_LIMITS)
    while math.hypot(orange[0] - box[0], orange[1] - box[1]) < BOX_FRUIT_MIN_SEPARATION_M:
        orange = in_reach(REACH_MIN_M, REACH_MAX_M, SECTOR_HALF_ANGLE_DEG, TABLE_X_LIMITS, TABLE_Y_LIMITS)
    return {"box": box, "orange": orange}


@configclass
class PickSceneCfg(InteractiveSceneCfg):
    """Table scene with an orange and a cardboard tray in the arm's reach."""

    env_spacing = 8.0

    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=str(TABLE_WITH_CUBE_USD)),
    )

    robot = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    orange: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Orange",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ORANGE_USD),
            scale=(FRUIT_SCALE_ORANGE,) * 3,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.28, -0.40, FRUIT_SPAWN_Z),
        ),
    )

    cardboard_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CardboardBox",
        spawn=sim_utils.UsdFileCfg(usd_path=str(CARDBOARD_BOX_USD)),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, -0.44, BOX_SPAWN_Z),  # overwritten per run by the sampler
        ),
    )

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
