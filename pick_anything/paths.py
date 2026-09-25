"""Asset path constants for Pick_Anything."""

from pathlib import Path

ASSETS_ROOT = Path(__file__).resolve().parent.parent / "assets"

SO101_FOLLOWER_USD = ASSETS_ROOT / "robots" / "so101_follower.usd"
# lift 工作区的同名资产拷贝：爪关节改名 gripper_joint（原名与 gripper
# body 同名干扰 PhysX 接触视图匹配），爪几何相同
SO101_FOLLOWER_GJ_USD = ASSETS_ROOT / "robots" / "so101_follower_gj.usd"
TABLE_WITH_CUBE_USD = ASSETS_ROOT / "scenes" / "table_with_cube" / "scene.usd"

# Converted from GraspNet simplified objects (012=apple, 016=orange), see
# scripts/convert_graspnet_object.py. Native sizes ~72-75mm, scaled at spawn.
APPLE_USD = ASSETS_ROOT / "objects" / "apple" / "apple.usd"
ORANGE_USD = ASSETS_ROOT / "objects" / "orange" / "orange.usd"

# Procedural open-top tray (16 x 16 x 5 cm), see scripts/make_cardboard_box.py.
CARDBOARD_BOX_USD = ASSETS_ROOT / "objects" / "cardboard_box" / "cardboard_box.usd"
