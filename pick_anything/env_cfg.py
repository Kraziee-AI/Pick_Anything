"""Scene configuration: table scene + SO101 follower arm + graspable fruits.

The table scene USD is referenced as one asset (its original cube prim was
removed; the table is reshaped to a 1.0 x 0.45 m rectangle). Fruits are
standalone rigid objects converted from GraspNet assets, scaled to ~45 mm to
fit the SO101 gripper opening. An open-top cardboard tray (16 x 16 x 5 cm)
shares the table. Each run spawns everything at random positions inside the
arm's reach (see :func:`sample_object_positions`).
"""

import math
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from pick_anything.paths import APPLE_USD, CARDBOARD_BOX_USD, ORANGE_USD, TABLE_WITH_CUBE_USD
from pick_anything.robots import SO101_FOLLOWER_CFG

TABLE_TOP_Z = 0.041  # measured top surface of the reshaped table

# Grasp-friendly size: 45 mm wide (SO101 gripper opens ~50 mm).
FRUIT_SCALE_APPLE = 0.60
FRUIT_SCALE_ORANGE = 0.62
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
# Table top x [-0.145, 0.855], y [-0.677, -0.227]; keep fruits clear of the edges.
TABLE_X_LIMITS = (0.05, 0.65)
TABLE_Y_LIMITS = (-0.60, -0.30)
FRUIT_MIN_SEPARATION_M = 0.10

# Box-specific sampling: narrower sector (the tray extends laterally) and a
# reach window shrunk so the tray body still sits in front of the arm.
BOX_REACH_MIN_M = 0.16
BOX_REACH_MAX_M = 0.26
BOX_SECTOR_HALF_ANGLE_DEG = 30.0
# Keep the full 16 cm footprint on the table (half-size + wall + margin).
BOX_X_LIMITS = (0.14, 0.56)
BOX_Y_LIMITS = (-0.51, -0.39)
# Fruit centers stay outside the tray footprint (+ fruit radius + margin).
BOX_FRUIT_MIN_SEPARATION_M = 0.13


def sample_object_positions(rng: random.Random | None = None) -> dict[str, tuple[float, float]]:
    """Random on-table (x, y) for the tray and both fruits.

    Everything is drawn inside the arm's reach (annulus sector clipped to the
    table), the tray keeps its footprint on the table, and fruits stay clear
    of each other and of the tray.
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

    def fruit_clear_of(avoid: list[tuple[float, float]], min_sep: float) -> tuple[float, float]:
        while True:
            p = in_reach(REACH_MIN_M, REACH_MAX_M, SECTOR_HALF_ANGLE_DEG, TABLE_X_LIMITS, TABLE_Y_LIMITS)
            if all(math.hypot(p[0] - a[0], p[1] - a[1]) >= min_sep for a in avoid):
                return p

    box = in_reach(BOX_REACH_MIN_M, BOX_REACH_MAX_M, BOX_SECTOR_HALF_ANGLE_DEG, BOX_X_LIMITS, BOX_Y_LIMITS)
    apple = fruit_clear_of([box], BOX_FRUIT_MIN_SEPARATION_M)
    while True:
        orange = fruit_clear_of([box, apple], BOX_FRUIT_MIN_SEPARATION_M)
        if math.hypot(orange[0] - apple[0], orange[1] - apple[1]) >= FRUIT_MIN_SEPARATION_M:
            break
    return {"box": box, "apple": apple, "orange": orange}


@configclass
class PickSceneCfg(InteractiveSceneCfg):
    """Table scene with an apple, an orange and a cardboard tray in the arm's reach."""

    env_spacing = 8.0

    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=str(TABLE_WITH_CUBE_USD)),
    )

    robot = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    apple: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(APPLE_USD),
            scale=(FRUIT_SCALE_APPLE,) * 3,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.28, -0.40, FRUIT_SPAWN_Z),
        ),
    )

    orange: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Orange",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ORANGE_USD),
            scale=(FRUIT_SCALE_ORANGE,) * 3,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.43, -0.36, FRUIT_SPAWN_Z),
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
