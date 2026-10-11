"""Shared scene and simulation plumbing for the eight-stage picking tasks."""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

from pick_anything.env_cfg import PickSceneCfg
from pick_anything.paths import ASSETS_ROOT

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@configclass
class PickRLSceneCfg(PickSceneCfg):
    """Demo scene plus RL extras: tighter spacing, ground plane, contact sensor."""

    env_spacing = 2.5

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        # Keep the original ground and textures local so restarts do not
        # depend on the remote asset client's download cache.
        spawn=GroundPlaneCfg(
            usd_path=str(ASSETS_ROOT / "environments/default_ground_plane/default_ground_plane.usda")
        ),
        collision_group=-1,
    )

    # Net contact force on the orange: while it is airborne this can only come
    # from the claw, which is how "held" is detected (filter pairing in PhysX
    # contact views is unreliable, net forces are not - see Isaac-Lift-SO101).
    orange_contact: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Orange")

    # Net contact force on the box: staged 任务的爪-盒接触惩罚直读此传感器
    # （2026-09-26：此前爪压盒壁零成本，放橘子常蹭壁）。v0 任务不引用它。
    box_contact: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/CardboardBox")

    # Net contact force per robot rigid body (the default prim folds, so the
    # links sit directly under {ENV_REGEX_NS}/Robot). Arm links and claw are
    # selected by body name inside PickState; the base column is ignored.
    # history spans the physics substeps inside one control step so a brief
    # tap between decimations still lands in the penalty.
    robot_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*",
        history_length=3,
        update_period=0.0,
    )

    def __post_init__(self):
        # the contact reporter API must be enabled on the spawned rigid bodies
        self.robot.spawn.activate_contact_sensors = True
        # 2026-09-25: 用户定版——去掉红色立方体（原「苹果」替身），橘子是唯一
        # 抓取目标（GraspNet 016 贴图球，scale 0.55 -> ~40mm，USD 自带 23g）。
        # 抓取可行性由教程式生成器（姿态约束 IK + 物理闭合 + proof lift）把关。
        import isaaclab.sim as sim_utils

        fruit_damping = sim_utils.RigidBodyPropertiesCfg(linear_damping=0.3, angular_damping=4.0)
        self.orange.spawn.rigid_props = fruit_damping
        self.orange.spawn.activate_contact_sensors = True
        # 盒体接触上报（box_contact 传感器需要；爪-盒接触惩罚的数据源）
        self.cardboard_box.spawn.activate_contact_sensors = True
        # RL-only light table: the referenced table USD meshes instantiate per
        # env (~5 MB each) and were the main OOM driver beyond 1024 envs; a
        # plain cuboid with the same footprint and top height (z=0.041) keeps
        # the task identical while the demo keeps the pretty table
        self.scene.spawn = sim_utils.CuboidCfg(
            size=(1.0, 0.45, 0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.36, 0.30)),
        )
        self.scene.init_state.pos = (0.355, -0.452, 0.001)


@configclass
class PickTaskBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Shared scene and timing; derived tasks supply actions and task managers."""

    scene: PickRLSceneCfg = PickRLSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    def __post_init__(self):
        self.decimation = 4  # 30 Hz control at 120 Hz physics
        # 20 s: the reach-grasp-carry-release chain with one retry takes
        # 14-18 s on this arm; at 16 s over half the episodes timed out as
        # near-misses
        self.episode_length_s = 20.0
        self.is_finite_horizon = False
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_max_rigid_patch_count=4 * 5 * 2**15,
            gpu_found_lost_pairs_capacity=2**26,
        )

    def play_mode(self):
        """Viewer-friendly overrides (called by the play CLI)."""
        super().play_mode()
