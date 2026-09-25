"""Robot configurations.

2026-09-25：换用 IsaacLabTutorial 验证过的 isaaclab_assets SO101_CFG
（TheRobotStudio SO-ARM100 复刻 + SysID Newton 变体，教程整套抓取/复位
流水线在该资产上跑通）。实测对照：leisaac 版 USD 的 V 形指对任何尺寸的
球都兜不住（41/33/52/60mm 球 0 留存——球沿 V 口滚出，立方体可被平面
卡住），而本资产的平行指可以稳定捏住 40mm 球（探针 5/12 格上提跟随
20-43mm，捕获区以两指中点为中心、x/z 向 ±1-2cm 容差）。

教程 USD 原生朝 -y（与 leisaac 相同），180° yaw 后面向 +y 的桌子。
"""

from isaaclab_assets.robots.so101 import SO101_CFG

SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Joint limits in degrees as authored in the USD asset（gripper 小值=闭合、
# 大值=全开）.
SO101_USD_JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

# 教程 workshop 实机初始姿态（IsaacLabTutorial env_cfg 同源值）
WORKSHOP_INITIAL_JOINT_POSITION = (
    -0.1221070742,
    -0.9066845838,
    0.1900876486,
    1.4797928525,
    -0.8044013083,
    0.2555162024919699,
)

SO101_FOLLOWER_CFG = SO101_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=SO101_CFG.init_state.replace(
        # 桌后站位：基座 (0.35, -0.64)，与旧 leisaac 布局一致
        pos=(0.35, -0.64, 0.01),
        # 180 deg yaw about z. IsaacLab 3.0 quaternions are (x, y, z, w);
        # the USD's native forward is -y, this rotation turns it toward +y
        # (the table).
        rot=(0.0, 0.0, 1.0, 0.0),
        joint_pos={
            "shoulder_pan": WORKSHOP_INITIAL_JOINT_POSITION[0],
            "shoulder_lift": WORKSHOP_INITIAL_JOINT_POSITION[1],
            "elbow_flex": WORKSHOP_INITIAL_JOINT_POSITION[2],
            "wrist_flex": WORKSHOP_INITIAL_JOINT_POSITION[3],
            "wrist_roll": WORKSHOP_INITIAL_JOINT_POSITION[4],
            "gripper": WORKSHOP_INITIAL_JOINT_POSITION[5],
        },
    ),
)
