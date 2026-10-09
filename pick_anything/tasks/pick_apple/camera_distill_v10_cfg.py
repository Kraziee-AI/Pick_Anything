"""Standalone v10 camera distillation with raw teacher labels and actions."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg
from isaaclab_tasks.utils import PresetCfg

from .agents.camera_ppo_cfg import CAMERA_ACTOR_CFG, CameraBoundedGaussianCfg
from .agents.staged_ppo_cfg import STAGED_ACTOR_CFG
from .camera_env_cfg import (
    CameraClosureHdSceneCfg,
    CameraProprioceptionV2Cfg,
    CameraStagedPickAppleEnvCfg,
    HOME_BOX_WINDOW,
    SCAN_ORANGE_WINDOW,
    WristImageCfg,
    scan_home_closure_robot_cfg,
)
from .staged_env_cfg import StagedObservationsCfg


@configclass
class CameraDistillV10ObservationsCfg:
    wrist_rgb: WristImageCfg = WristImageCfg()
    proprioception: CameraProprioceptionV2Cfg = CameraProprioceptionV2Cfg()
    teacher: StagedObservationsCfg.PolicyCfg = StagedObservationsCfg.PolicyCfg()


@configclass
class CameraDistillV10PickAppleEnvCfg(CameraStagedPickAppleEnvCfg):
    """256×192 closure camera, fine control, uniform resets and a 90s episode."""

    observations: CameraDistillV10ObservationsCfg = CameraDistillV10ObservationsCfg()
    scene: CameraClosureHdSceneCfg = CameraClosureHdSceneCfg(num_envs=1024)

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = scan_home_closure_robot_cfg()
        self.scene.wrist_camera.width = 256
        self.scene.wrist_camera.height = 192
        self.actions.arm_action.scale = 0.033
        self.actions.gripper_action.scale = 0.02
        self.episode_length_s = 90.0
        reset = self.events.reset_staged.params
        reset["home_orange_window"] = SCAN_ORANGE_WINDOW
        reset["home_box_window"] = HOME_BOX_WINDOW
        for phase in ("early", "mid", "late"):
            reset[f"curriculum_{phase}"] = (0.125,) * 8
        self.events.reset_arm_joints.params["position_range"] = [-0.05, 0.05]


@configclass
class CameraDistillV10BaseRunnerCfg(RslRlDistillationRunnerCfg):
    """Standard DAgger; match the unclipped state teacher's action contract."""

    seed = 42
    num_steps_per_env = 32
    max_iterations = 4000
    save_interval = 250
    experiment_name = "pick_apple_staged_so101_camera_distill"
    run_name = ""
    obs_groups = {"student": ["wrist_rgb", "proprioception"], "teacher": ["teacher"]}
    clip_actions = None
    student = CAMERA_ACTOR_CFG.replace(
        distribution_cfg=CameraBoundedGaussianCfg(init_std=0.05, std_type="log")
    )
    teacher = STAGED_ACTOR_CFG
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=4,
        learning_rate=5.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )


@configclass
class CameraDistillV10RunnerCfg(PresetCfg):
    default = CameraDistillV10BaseRunnerCfg()
