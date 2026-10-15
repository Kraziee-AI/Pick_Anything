"""Independent multi-object task; no orange reset bank or eight-stage rewards."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp import action_rate_l2, joint_vel_l2, time_out
from isaaclab.managers import (
    EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils import PresetCfg

from pick_anything.grounded_pick.config import GoalImageCfg
from pick_anything.grounded_pick.language import goal_caption
from pick_anything.tabletop.catalog import OBJECTS
from pick_anything.tasks.pick_apple.camera_env_cfg import (
    CameraClosureHdSceneCfg, CameraProprioceptionV2Cfg, scan_home_closure_robot_cfg,
)
from pick_anything.tasks.pick_apple.agents.camera_ppo_cfg import CameraPPOBaseRunnerCfg
from pick_anything.tasks.pick_apple.pick_env_cfg import PickTaskBaseEnvCfg
from pick_anything.tasks.pick_apple.staged_env_cfg import StagedActionsCfg, make_newton_physics, _register_contact_model
from . import mdp
from .observations import OpenVocabRGBMask

OBJECT_KEYS = tuple(OBJECTS)


@configclass
class OpenVocabSceneCfg(CameraClosureHdSceneCfg):
    yellow_duck = None
    blue_cube = None
    red_cylinder = None
    yellow_duck_contact = None
    blue_cube_contact = None
    red_cylinder_contact = None

    def __post_init__(self):
        super().__post_init__()
        self.robot = scan_home_closure_robot_cfg()
        for key in OBJECT_KEYS:
            obj = OBJECTS[key]
            # All procedural objects have bottom origins; grasp centres and
            # projection centres are explicitly offset in the new task.
            setattr(self, key, RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{obj.prim_name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(obj.asset), scale=(obj.scale,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.3, angular_damping=4.0),
                    activate_contact_sensors=True, semantic_tags=[("class", key)],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2 + 0.1 * OBJECT_KEYS.index(key), -0.32, obj.spawn_z)),
            ))
            setattr(self, f"{key}_contact", ContactSensorCfg(prim_path=f"{{ENV_REGEX_NS}}/{obj.prim_name}"))
        self.wrist_camera.width, self.wrist_camera.height = 256, 192


@configclass
class OpenVocabImageCfg(GoalImageCfg):
    def __post_init__(self):
        super().__post_init__()
        self.image.func = OpenVocabRGBMask


@configclass
class CriticCfg(CameraProprioceptionV2Cfg):
    state = ObservationTermCfg(func=mdp.critic_features)

    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    rgb_mask: OpenVocabImageCfg = OpenVocabImageCfg()
    proprioception: CameraProprioceptionV2Cfg = CameraProprioceptionV2Cfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    reset_scene = EventTermCfg(func=mdp.reset_scene, mode="reset")


@configclass
class RewardsCfg:
    approach = RewardTermCfg(func=mdp.approach_progress, weight=6.0)
    grasp = RewardTermCfg(func=mdp.grasp_reward, weight=4.0)
    lift = RewardTermCfg(func=mdp.lift_progress, weight=8.0)
    milestones = RewardTermCfg(func=mdp.milestone_reward, weight=30.0)
    wrong_target = RewardTermCfg(func=mdp.wrong_object, weight=-8.0)
    action_rate = RewardTermCfg(func=action_rate_l2, weight=-0.05)
    joint_velocity = RewardTermCfg(func=joint_vel_l2, weight=-0.005)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=time_out, time_out=True)
    success = TerminationTermCfg(func=mdp.success)
    dropped_or_invalid = TerminationTermCfg(func=mdp.dropped_or_invalid)


@configclass
class OpenVocabPickEnvCfg(PickTaskBaseEnvCfg):
    scene: OpenVocabSceneCfg = OpenVocabSceneCfg(num_envs=64)
    actions: StagedActionsCfg = StagedActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    object_keys: tuple[str, ...] = OBJECT_KEYS
    target_key: str = "random"
    grounding_caption: str = ""
    lift_height: float = 0.08
    lift_confirm_steps: int = 5
    evaluate_first_episode: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = make_newton_physics()
        # Same backend/contact model as the validated four-channel orange task.
        _register_contact_model()
        self.actions.arm_action.scale = 0.033
        self.actions.gripper_action.scale = 0.02
        self.episode_length_s = 30.0
        self.scene.wrist_camera.update_period = self.sim.dt * self.decimation
        self.num_rerenders_on_reset = 1

    def validate_config(self):
        if tuple(self.object_keys) != OBJECT_KEYS:
            raise ValueError("当前训练场景固定包含 orange/yellow_duck/blue_cube/red_cylinder。")
        if self.target_key != "random" and self.target_key not in self.object_keys:
            raise ValueError("target_key 必须是训练场景中的一个目标或 random。")
        params = self.observations.rgb_mask.image.params
        goal_caption(params["instruction"], self.grounding_caption)
        if params["backend"] not in {"gt", "dino"}:
            raise ValueError("mask backend must be gt or dino")
        if params["image_history"] not in (1, 3):
            raise ValueError("image_history must be 1 or 3")
        if params["backend"] == "dino" and self.target_key == "random":
            raise ValueError("DINO 使用固定文本；请指定 target_key，避免随机奖励目标与文本不一致。")
        if self.evaluate_first_episode and self.target_key == "random":
            raise ValueError("评估必须指定 target_key。")
        if self.lift_height <= 0 or self.lift_confirm_steps < 1:
            raise ValueError("抬升高度和连续确认步数必须为正。")
        if not 0 <= params["box_jitter"] < 1 or not 0 <= params["mask_dropout"] < 1:
            raise ValueError("box_jitter and mask_dropout must be in [0,1)")
        if min(params["policy_height"], params["policy_width"], params["dump_every"]) <= 0:
            raise ValueError("图像尺寸和 dump_every 必须为正。")

    def play_mode(self):
        super().play_mode()
        self.observations.rgb_mask.enable_corruption = False
        self.observations.proprioception.enable_corruption = False


@configclass
class OpenVocabPickRunnerCfg(PresetCfg):
    default = CameraPPOBaseRunnerCfg().replace(
        experiment_name="pick_anything_so101_open_vocab",
        obs_groups={"actor": ["rgb_mask", "proprioception"], "critic": ["critic"]},
    )
