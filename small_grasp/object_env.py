"""Small-grasp-only single object scene. No old RL state, masks or stage scoring."""
import math
import random

TABLE_Z = .041
TERMINATIONS = ('time_out', 'object_invalid_or_dropped')
LAYOUT_PROTOCOL = 'single-observation-windows-v1'
TARGET_XY_BOUNDS_M = (.290, .320, -.380, -.340)
TRAY_XY_BOUNDS_M = (.440, .490, -.450, -.400)


def sample_layout(spec, seed):
    """Retain validated single-object observation windows for every target class."""
    rng = random.Random(seed)
    radius = spec['footprint_radius_m']
    for _ in range(100):
        bx, by = rng.uniform(*TRAY_XY_BOUNDS_M[:2]), rng.uniform(*TRAY_XY_BOUNDS_M[2:])
        for _ in range(300):
            x, y = rng.uniform(*TARGET_XY_BOUNDS_M[:2]), rng.uniform(*TARGET_XY_BOUNDS_M[2:])
            if not (-.135 + radius <= x <= .845 - radius and -.667 + radius <= y <= -.237 - radius):
                continue
            if math.hypot(max(abs(x - bx) - .08, 0.), max(abs(y - by) - .08, 0.)) < radius + .012:
                continue
            yaw = rng.uniform(-math.pi, math.pi)
            return {'target_object': {'position': [x, y, spec['spawn_z_m']],
                                     'quaternion_xyzw': [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]},
                    'cardboard_box': {'position': [bx, by, TABLE_Z + .005],
                                     'quaternion_xyzw': [0., 0., 0., 1.]}, 'seed': seed,
                    'layout_protocol': LAYOUT_PROTOCOL,
                    'bounds_m': {'target_xy': list(TARGET_XY_BOUNDS_M),
                                 'tray_xy': list(TRAY_XY_BOUNDS_M)}}
    raise ValueError('Cannot place the registered target and tray without overlap')


def reset_scene(env, env_ids):
    import torch
    from .sim_eval import tensor
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    seed = int(getattr(env, 'object_episode_seed', env.cfg.seed))
    layouts = [sample_layout(env.cfg.object_spec, seed + int(i)) for i in ids.tolist()]
    for key in ('target_object', 'cardboard_box'):
        pose = torch.tensor([row[key]['position'] + row[key]['quaternion_xyzw'] for row in layouts], device=env.device)
        pose[:, :3] += tensor(env.scene.env_origins)[ids]
        asset = env.scene[key]
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
    robot = env.scene['robot']
    position = tensor(robot.data.default_joint_pos)[ids].clone()
    robot.write_joint_position_to_sim_index(position=position, env_ids=ids)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(position), env_ids=ids)
    robot.set_joint_position_target_index(target=position, env_ids=ids)
    env.object_spawn_layout = layouts[0]


def invalid_or_dropped(env):
    import torch
    from .sim_eval import tensor
    obj, robot = env.scene['target_object'], env.scene['robot']
    position = tensor(obj.data.root_pos_w) - tensor(env.scene.env_origins)
    q, velocity = tensor(robot.data.joint_pos), tensor(robot.data.joint_vel)
    finite = torch.isfinite(position).all(-1) & torch.isfinite(q).all(-1) & torch.isfinite(velocity).all(-1)
    finite &= torch.isfinite(tensor(obj.data.root_quat_w)).all(-1)
    return (~finite) | (velocity.abs().amax(-1) > 25.) | (position[:, 2] < .02)


def make_config(target_key):
    """Isaac imports are deferred until AppLauncher initialized the runtime."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils import configclass
    from isaaclab.envs.mdp import joint_pos_rel, joint_vel_rel, time_out as termination_time_out
    from pick_anything.tasks.pick_apple.camera_env_cfg import CameraClosureHdSceneCfg, scan_home_closure_robot_cfg
    from pick_anything.tasks.pick_apple.pick_env_cfg import PickTaskBaseEnvCfg
    from pick_anything.tasks.pick_apple.staged_env_cfg import StagedActionsCfg, make_newton_physics
    from .object_registry import get_object_spec, PHYSICS_MATERIAL
    spec = get_object_spec(target_key)
    if spec['mass_kg'] <= .01:
        raise ValueError('Registered target mass must exceed 0.01 kg for the fixed contact score')

    @configclass
    class Proprio(ObservationGroupCfg):
        position = ObservationTermCfg(func=joint_pos_rel)
        velocity = ObservationTermCfg(func=joint_vel_rel)

    @configclass
    class Observations:
        proprioception: Proprio = Proprio()

    @configclass
    class Events:
        reset_object = EventTermCfg(func=reset_scene, mode='reset')

    @configclass
    class Terminations:
        time_out = TerminationTermCfg(func=termination_time_out, time_out=True)
        object_invalid_or_dropped = TerminationTermCfg(func=invalid_or_dropped)

    @configclass
    class Config(PickTaskBaseEnvCfg):
        scene: CameraClosureHdSceneCfg = CameraClosureHdSceneCfg(num_envs=1)
        actions: StagedActionsCfg = StagedActionsCfg()
        observations: Observations = Observations()
        events: Events = Events()
        terminations: Terminations = Terminations()
        rewards: dict = {}

        def __post_init__(self):
            super().__post_init__()
            self.sim.physics = make_newton_physics()
            self.episode_length_s = 90.
            self.scene.robot = scan_home_closure_robot_cfg()
            # Parent scene setup references orange before this instance removes it.
            self.scene.orange = self.scene.orange_contact = None
            self.scene.target_object = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{spec['prim_name']}",
                spawn=sim_utils.UsdFileCfg(usd_path=spec['usd_path'], scale=(spec['scale'],) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=.3, angular_damping=4.),
                    physics_material=sim_utils.RigidBodyMaterialCfg(**PHYSICS_MATERIAL),
                    activate_contact_sensors=True, semantic_tags=[('class', target_key)]),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(.28, -.40, spec['spawn_z_m'])))
            self.scene.target_contact = ContactSensorCfg(prim_path=f"{{ENV_REGEX_NS}}/{spec['prim_name']}")
            self.scene.wrist_camera.width, self.scene.wrist_camera.height = 256, 192
            self.scene.wrist_camera.update_period = self.sim.dt * self.decimation
            self.scene.wrist_camera.update_latest_camera_pose = True
            self.num_rerenders_on_reset = 1
            self.object_spec = spec
            self.target_key = target_key

    return Config()
