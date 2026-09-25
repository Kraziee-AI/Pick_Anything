"""Manager-based RL env: SO101 picks the orange and places it in the cardboard tray.

Scene assets come unchanged from :mod:`pick_anything.env_cfg` (table USD,
SO101 follower, GraspNet orange, procedural tray); this file adds the
RL plumbing: relative joint-position actions, contact sensor, observations,
progress-based rewards with disturbance penalties, failure/success
terminations and a difficulty curriculum inside the reset event.

2026-09-25: single-fruit scene — the orange is the only pick target (the
apple / red-cube RL stand-in and the distractor orange are gone).

Reward ledger (per episode, progress terms pay once per meter of ground):
    +12/m  approach      claw-to-orange ground while the orange is down
    +25/m  height        orange climb while grasped, until clear of the walls
    +20/m  hover         orange-to-hover-point ground while carrying/releasing
    +8     pregrasp      stage 1: state A reached (orange in the capture
                         pocket, claw capture-ready - no orange contact
                         needed; a pure close grabs the orange)
    +15    grasp         stage 2: orange pinched and lifted
    +45    place         stage 3: orange released, settled in the tray
    -3/s   arm hits      any arm link (base/claw excluded) pressing anything
    -2/s   claw hits     claw pressing the table / tray (not the orange)
    -1.5/s box (pre)     tray displaced before the orange is grasped
    -0.5/s box (late)    tray displaced after the grasp
    -0.005 action-rate   smoothness regularizer
    -8/-5/-15            terminal penalties: orange off table / tray tipped /
                         abnormal joint velocities
"""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import (
    RelativeJointPositionActionCfg,
    action_rate_l2,
    is_alive,
    is_terminated_term,
    joint_pos_rel,
    joint_vel_out_of_manual_limit,
    joint_vel_rel,
    last_action,
    randomize_actuator_gains,
    randomize_joint_parameters,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_joints_by_offset,
    time_out,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

from pick_anything.env_cfg import PickSceneCfg
from pick_anything.tasks.pick_apple import mdp

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# claw travel is 0 (closed) .. 1.745 rad (open); reset spans the half-closed
# pocket band up to nearly the full open stop so some starts are already
# capture-ready and closing stays part of the action space
GRIPPER_RESET_RANGE = (0.5, 1.60)


@configclass
class PickRLSceneCfg(PickSceneCfg):
    """Demo scene plus RL extras: tighter spacing, ground plane, contact sensor."""

    env_spacing = 2.5

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
        collision_group=-1,
    )

    # Net contact force on the orange: while it is airborne this can only come
    # from the claw, which is how "held" is detected (filter pairing in PhysX
    # contact views is unreliable, net forces are not - see Isaac-Lift-SO101).
    orange_contact: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Orange")

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
class ActionsCfg:
    """Delta joint-position targets; the gripper moves faster than the arm."""

    arm_action: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        # 0.15 rad/step at 30 Hz tops out near the servo speed; timeout is the
        # dominant failure (~50%), so pace beats extra caution here
        scale=0.15,
    )
    gripper_action: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=0.35,
    )


@configclass
class ObservationsCfg:
    """Single policy group; env-local positions (world minus env origin)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=joint_pos_rel)
        joint_vel = ObsTerm(func=joint_vel_rel, clip=(-5.0, 5.0))
        last_action = ObsTerm(func=last_action)
        orange_state = ObsTerm(func=mdp.orange_state_w, clip=(-2.0, 2.0))
        box_state = ObsTerm(func=mdp.box_state_w, clip=(-2.0, 2.0))
        ee_to_targets = ObsTerm(func=mdp.ee_to_targets, clip=(-2.0, 2.0))
        task_flags = ObsTerm(func=mdp.task_flags)
        orange_force = ObsTerm(func=mdp.jaw_orange_force)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = 3

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Startup domain randomization + reset draws, in evaluation order."""

    # friction for grasp reliability (default material is 0.5/0.5)
    orange_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("orange"),
            "static_friction_range": [1.0, 1.4],
            "dynamic_friction_range": [0.8, 1.1],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 64,
        },
    )
    box_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cardboard_box"),
            "static_friction_range": [0.5, 0.8],
            "dynamic_friction_range": [0.4, 0.6],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 64,
        },
    )
    # the authored 30 g tray flips whenever the carry phase brushes a wall
    # (~15% of episodes); a 150 g loaded tray stays put through placement
    box_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cardboard_box"),
            "mass_distribution_params": [0.14, 0.16],
            "operation": "abs",
        },
    )
    claw_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper", "moving_jaw_so101_v1"]),
            "static_friction_range": [0.8, 1.1],
            "dynamic_friction_range": [0.6, 0.9],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 64,
        },
    )
    joint_gains = EventTerm(
        func=randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": [0.9, 1.1],
            "damping_distribution_params": [0.9, 1.1],
            "operation": "scale",
        },
    )
    # servo-style joint friction damps the residual PD ringing of the soft
    # hobby actuators (small absolute values, N*m*s/rad)
    joint_friction = EventTerm(
        func=randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": [0.01, 0.05],
            "operation": "abs",
        },
    )

    # reset draws; order matters - joints first, then object placement and
    # task-state bookkeeping (last term clears/stamps PickState)
    reset_arm_joints = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": list((-0.10, 0.10)),
            "velocity_range": [0.0, 0.0],
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
        },
    )
    reset_gripper = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": list(GRIPPER_RESET_RANGE),
            "velocity_range": [0.0, 0.0],
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
        },
    )
    # p_in_hand raised 0.25 -> 0.40: more spawn-under-claw resets give the
    # close-and-lift sequence extra practice while the shaping bridge teaches it
    reset_objects = EventTerm(func=mdp.place_objects_and_track, mode="reset", params={"p_in_hand": 0.40})


@configclass
class RewardsCfg:
    """Staged positives + spec-driven penalties, dt-aware scales.

    The RewardManager multiplies every step's value by dt (=1/30 s), so
    one-shot / per-meter terms carry x30 weights to land at their intended
    effective value, while per-step penalties are kept small enough that a
    single nudge costs far less than completing the task.

    Stage ladder (effective one-shot values, strictly increasing so advancing
    a stage always beats farming an earlier one):
        +8   pregrasp     state A: orange inside the capture pocket, claw
                          capture-ready (no orange contact needed)
        +15  grasp        orange pinched and lifted
        +45  place        orange released and settled in the tray
    plus +12/m approach, +25/m climb, +20/m carry progress shaping.
    """

    # positives: effective +12/m approach, +25/m climb, +20/m carry
    approach = RewTerm(func=mdp.approach_progress, weight=360.0)
    # small on purpose: a larger weight let the policy idle next to the orange
    # (farming proximity) instead of risking the grasp
    approach_presence = RewTerm(func=mdp.approach_presence, weight=0.25, params={"std": 0.1})
    # dense nudge to arrive with an open claw (stops once state A is reached)
    gripper_openness = RewTerm(func=mdp.gripper_openness, weight=0.5, params={"std": 0.1})
    # dense bridge over the state-A -> grasp cliff: closing with the orange
    # centered in the pocket pays ~0.03/step, a full close sequence ~0.5 per
    # episode - far below grasp +15, so completing the lift still dominates
    gripper_closing = RewTerm(func=mdp.gripper_closing, weight=1.0)
    height = RewTerm(func=mdp.height_progress, weight=750.0)
    hover = RewTerm(func=mdp.hover_progress, weight=600.0)
    # one-shot ladder: weight x dt = 1, so the bonus param IS the effective
    # payout (pregrasp +8 < grasp +15 < place +45); state A needs no orange
    # contact, so nothing rewards touching the fruit before the close
    pregrasp = RewTerm(func=mdp.pregrasp_onset, weight=30.0, params={"bonus": 8.0})
    grasp = RewTerm(func=mdp.grasp_onset, weight=30.0, params={"bonus": 15.0})
    place = RewTerm(func=mdp.place_success, weight=1350.0)
    # gentle pace pressure: effective -0.013/step, ~-8 for a full-length
    # episode, so finishing early beats wandering within the timeout
    time_pressure = RewTerm(func=is_alive, weight=-0.4)

    # robot collision penalties (task spec): arm links may touch nothing
    # (effective -0.1/step while pressing), claw may touch nothing but the
    # orange (table / tray contact, effective -0.067/step)
    arm_collision = RewTerm(func=mdp.arm_collision, weight=-3.0)
    ee_collision = RewTerm(func=mdp.ee_bad_contact, weight=-2.0)

    # disturbance penalties: effective about -0.008/step while displaced
    box_disturbed = RewTerm(func=mdp.box_disturbed, weight=-0.25)
    box_disturbed_late = RewTerm(func=mdp.box_disturbed_late, weight=-0.1)
    orange_disturbed = RewTerm(func=mdp.orange_disturbed, weight=-0.5)

    action_rate = RewTerm(func=action_rate_l2, weight=-0.005)
    # terminal failures: effective -25 / -15 / -15; the light orange gets
    # flung off the table in ~46% of episodes, so the drop penalty leads
    orange_dropped = RewTerm(
        func=is_terminated_term, weight=-750.0, params={"term_keys": "orange_left_table"}
    )
    box_ruined = RewTerm(
        func=is_terminated_term, weight=-600.0, params={"term_keys": "box_tipped_or_gone"}
    )
    abnormal_robot = RewTerm(
        func=is_terminated_term, weight=-450.0, params={"term_keys": "abnormal_robot"}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    orange_in_box = DoneTerm(func=mdp.orange_in_box)
    orange_left_table = DoneTerm(func=mdp.orange_left_table)
    box_tipped_or_gone = DoneTerm(func=mdp.box_tipped_or_gone)
    # the soft hobby-servo PD rings well past the USD's 10 rad/s soft limit on
    # ordinary moves; only treat a real blow-up as abnormal
    abnormal_robot = DoneTerm(
        func=joint_vel_out_of_manual_limit,
        params={
            "max_velocity": 25.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
        },
    )


@configclass
class PickAppleEnvCfg(ManagerBasedRLEnvCfg):
    """SO101: pick the orange off the table, place it in the cardboard tray."""

    scene: PickRLSceneCfg = PickRLSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

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
