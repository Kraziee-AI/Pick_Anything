"""Isolated environment/runner configuration for RGB + goal mask PPO."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils import PresetCfg

from pick_anything.env_cfg import ORANGE_RADIUS_M
from pick_anything.tasks.pick_apple.camera_env_cfg import (
    CameraClosureFinetuneEnvCfg, CameraProprioceptionV2Cfg, CameraFrontRewardsCfg,
)
from pick_anything.tasks.pick_apple.staged_env_cfg import StagedObservationsCfg
from pick_anything.tasks.pick_apple.agents.camera_ppo_cfg import CameraPPOBaseRunnerCfg
from pick_anything.tasks.pick_apple.stages import PREGRASP_POCKET_M, GRASP_PROXIMITY_M

from .grounding import DEFAULT_CACHE, DEFAULT_MODEL
from .masks import orange_caption
from .observations import GroundedRGBMask
from .rewards import PregraspOpeningProgress, GroundedApproachProgress, empty_gripper_before_lift


@configclass
class GoalImageCfg(ObservationGroupCfg):
    image = ObservationTermCfg(
        func=GroundedRGBMask,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "backend": "gt", "instruction": "抓橘子", "model_name": DEFAULT_MODEL,
            "grounding_device": "", "cache_dir": str(DEFAULT_CACHE), "local_files_only": True,
            "box_threshold": 0.25, "text_threshold": 0.25, "radius": ORANGE_RADIUS_M,
            "policy_height": 96, "policy_width": 128,
            "image_history": 1,
            "box_jitter": 0.10, "mask_dropout": 0.02,
            "dump_dir": "", "dump_every": 30, "dump_max": 100,
        },
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = True


@configclass
class GroundedObservationsCfg:
    rgb_mask: GoalImageCfg = GoalImageCfg()
    proprioception: CameraProprioceptionV2Cfg = CameraProprioceptionV2Cfg()
    critic: StagedObservationsCfg.CriticCfg = StagedObservationsCfg.CriticCfg()


@configclass
class GroundedRewardsCfg(CameraFrontRewardsCfg):
    # Parent env applies its common 1/30 scale; effective configured weight=0.2.
    pregrasp_open_progress = RewardTermCfg(func=PregraspOpeningProgress, weight=6.0)
    approach_progress = RewardTermCfg(func=GroundedApproachProgress, weight=6.0)
    empty_gripper = RewardTermCfg(func=empty_gripper_before_lift, weight=-3.0)


@configclass
class GroundedPickEnvCfg(CameraClosureFinetuneEnvCfg):
    observations: GroundedObservationsCfg = GroundedObservationsCfg()
    rewards: GroundedRewardsCfg = GroundedRewardsCfg()
    evaluate_first_episode: bool = False
    evaluation_diagnostics_dir: str = ""
    evaluation_trace_every: int = 5
    grasp_min_gripper_rad: float = 0.0
    grasp_confirm_proximity_m: float = PREGRASP_POCKET_M

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        # Detector sees 256x192; policy receives 128x96 RGB plus a binary mask.
        self.scene.wrist_camera.width = 256
        self.scene.wrist_camera.height = 192
        self.scene.wrist_camera.update_period = self.sim.dt * self.decimation
        self.scene.wrist_camera.update_latest_camera_pose = True
        # Render reset poses before computing their masks (no preceding episode frame).
        self.num_rerenders_on_reset = 1

    def validate_config(self):
        if self.evaluation_trace_every < 1:
            raise ValueError("evaluation_trace_every must be positive")
        if not 0 <= self.grasp_min_gripper_rad < 0.75:
            raise ValueError("grasp_min_gripper_rad must be in [0,0.75)")
        if not 0 < self.grasp_confirm_proximity_m <= GRASP_PROXIMITY_M:
            raise ValueError("grasp_confirm_proximity_m must be in (0,0.075]")
        params = self.observations.rgb_mask.image.params
        orange_caption(params["instruction"])
        if params["backend"] not in ("gt", "dino"):
            raise ValueError("mask backend must be gt or dino")
        if not 0 <= params["box_jitter"] < 1 or not 0 <= params["mask_dropout"] < 1:
            raise ValueError("box_jitter and mask_dropout must be in [0,1)")
        if min(params["policy_height"], params["policy_width"], params["dump_every"]) <= 0:
            raise ValueError("Image dimensions and dump_every must be positive")
        if params["image_history"] not in (1, 3):
            raise ValueError("image_history must be 1 or 3")


@configclass
class GroundedPickRunnerCfg(PresetCfg):
    default = CameraPPOBaseRunnerCfg().replace(
        experiment_name="pick_orange_so101_grounded",
        obs_groups={"actor": ["rgb_mask", "proprioception"], "critic": ["critic"]},
    )
