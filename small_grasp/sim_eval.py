"""Independent SO101 RGB-D grasp execution in unchanged single/multi task scenes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .infer import ROOT, configure_runtime


def tensor(value):
    return value.torch if hasattr(value, 'torch') else value


def zero_image(env, sensor_cfg=None, backend='gt', instruction='', model_name='', grounding_device='',
               cache_dir='', local_files_only=True, box_threshold=.25, text_threshold=.25, radius=.028,
               policy_height=96, policy_width=128, image_history=1, box_jitter=0., mask_dropout=0.,
               dump_dir='', dump_every=30, dump_max=100):
    import torch
    return torch.zeros((env.num_envs, 4 * image_history, policy_height, policy_width), device=env.device)


def main():
    configure_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=['single', 'multi', 'object-single'], required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--inference-seed', type=int, default=42)
    parser.add_argument('--width-scale', type=float, default=10.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--capture-only', action='store_true')
    parser.add_argument('--capture-gripper-rad', type=float, help='Diagnostic only: command jaw before capture-only')
    parser.add_argument('--diagnostic-contacts', action='store_true')
    parser.add_argument('--mask-backend', choices=['dino', 'gt'], default='dino', help='gt is diagnostic oracle only')
    parser.add_argument('--segmentation-backend', choices=['sam', 'depth', 'box'], default='box')
    parser.add_argument('--segmentation-model', type=Path)
    parser.add_argument('--collision-check', choices=['off', 'capsule'], default='off')
    parser.add_argument('--closure', choices=['first_contact', 'full_close'], default='first_contact')
    parser.add_argument('--controller', choices=['baseline', 'repaired', 'servo'], default='repaired')
    parser.add_argument('--arm-tracking-feedback', action='store_true', help='Experimental bounded robot-joint feedback; requires servo')
    parser.add_argument('--close-tracking-integral', action='store_true', help='Experimental bounded close-only integral; requires servo arm tracking')
    parser.add_argument('--close-frame-policy', choices=['fixed-body', 'planned-line', 'reachable-line'], default='fixed-body',
                        help='Experimental planned-line or whole-path reachable partial line compensation')
    parser.add_argument('--close-integral-transfer', choices=['reset', 'adjacent'], default='reset',
                        help='Experimental transfer only between explicitly authorized adjacent preflight close waypoints')
    parser.add_argument('--closure-latch-policy', choices=['legacy', 'no_reopen'], default='legacy',
                        help='Experimental no_reopen retains the last issued jaw target at contact stall; requires servo arm tracking')
    parser.add_argument('--candidate-contact-ranking', action='store_true', help='Experimental observed contact-band preference among feasible plans; requires servo')
    parser.add_argument('--candidate-contact-policy', choices=['soft8', 'band_first'], default='soft8')
    parser.add_argument('--grasp-point', choices=['network', 'sphere'], default='network')
    parser.add_argument('--contact-frame', choices=['open', 'contact'], default='open')
    parser.add_argument('--instruction', default='orange')
    parser.add_argument('--viz', choices=['none', 'kit'], default='none')
    parser.add_argument('--record-video', action='store_true', help='Record each trial with a passive overview camera')
    parser.add_argument('--live-debug', action='store_true', help='Live browser scene / DINO+SAM split screen')
    parser.add_argument('--live-debug-port', type=int, default=8765)
    parser.add_argument('--live-debug-device', default='cpu')
    parser.add_argument('--live-debug-interval', type=float, default=1.)
    parser.add_argument('--video-fps', type=float, default=15.)
    parser.add_argument('--wrist-camera-profile', choices=['legacy', 'open-jaws'], default='legacy')
    parser.add_argument('--network-region', choices=['auto', 'bbox', 'target-mask'], default='auto')
    parser.add_argument('--network-frame', choices=['camera', 'legacy-optical'], default='camera',
                        help='Experimental model input basis; exported poses stay in the real camera frame')
    parser.add_argument('--candidate-pool-size', type=int,
                        help='NMS pool: defaults to 100 for open-jaws target-mask, otherwise 20')
    args = parser.parse_args()
    from .instructions import resolve_pick_instruction
    try:
        language = resolve_pick_instruction(args.instruction, allow_objects=args.scene == 'object-single')
    except ValueError as error:
        parser.error(str(error))
    if args.episodes < 1:
        parser.error('episodes must be positive')
    if args.capture_gripper_rad is not None and (not args.capture_only or
            not math.isfinite(args.capture_gripper_rad) or not .05 <= args.capture_gripper_rad <= 1.4):
        parser.error('capture-gripper-rad requires capture-only and a finite angle in [0.05, 1.4]')
    if not math.isfinite(args.width_scale) or args.width_scale <= 0:
        parser.error('width scale must be finite and positive')
    if not math.isfinite(args.video_fps) or args.video_fps <= 0:
        parser.error('video FPS must be finite and positive')
    if args.controller in ('repaired', 'servo') and args.mask_backend != 'dino':
        parser.error('repaired/servo controllers use real language detection; GT controls require baseline')
    from .pipeline_options import validate_options, resolve_network_region, resolve_candidate_pool_size
    network_region = resolve_network_region(args.wrist_camera_profile, args.network_region)
    try:
        candidate_pool_size = resolve_candidate_pool_size(args.wrist_camera_profile, network_region, args.candidate_pool_size)
        validate_options(args)
        from small_grasp.network_frame import frame_metadata
        network_input_frame = frame_metadata(args.wrist_camera_profile, args.network_frame)
    except ValueError as error:
        parser.error(str(error))
    root = args.output_dir.resolve()
    if not root.is_relative_to(ROOT):
        parser.error('outputs must stay inside small_grasp')
    root.mkdir(parents=True, exist_ok=False)
    os.environ['ISAACSIM_ASSET_ROOT'] = str(ROOT.parent / 'assets/isaac_cache')
    os.environ['WARP_CACHE_PATH'] = str(ROOT / 'cache/warp')
    os.environ['CUDA_CACHE_PATH'] = str(ROOT / 'cache/cuda')
    (root / 'tmp').mkdir()
    os.environ['TMPDIR'] = str(root / 'tmp')
    from isaaclab.app import AppLauncher
    kit_root = ROOT / 'cache/kit'
    kit_root.mkdir(parents=True, exist_ok=True)
    launcher = AppLauncher({'viz': args.viz, 'device': args.device,
                           'kit_args': f'--portable-root {kit_root} --/app/settings/persistent=false'})
    import torch
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    object_single = args.scene == 'object-single'
    if object_single:
        from .object_env import make_config
        from .object_registry import get_object_spec
        from .object_scoring import sample_env, LiftTracker, SCHEMA
        from isaaclab.envs import ManagerBasedRLEnv as Base
        object_spec = get_object_spec(language['target_key'])
        Config = lambda: make_config(language['target_key'])
    elif args.scene == 'single':
        from pick_anything.grounded_pick.config import GroundedPickEnvCfg as Config
        from pick_anything.grounded_pick.env import GroundedPickEnv as Base
    else:
        from pick_anything.open_vocab_pick.config import OpenVocabPickEnvCfg as Config
        from pick_anything.open_vocab_pick.env import OpenVocabPickEnv as Base

    class ScoredEnv(Base):
        tracking = False
        terminal_result = None
        diagnostic_rows = None
        diagnostic_physics = None
        diagnostic_physics_rows = None
        video_recorder = None
        live_debug = None
        live_frames = None
        object_tracker = None

        def step(self, action):
            result = super().step(action)
            if object_single and self.tracking and self.object_tracker is not None:
                self.object_tracker.observe(int(self.episode_length_buf[0]), sample_env(self, object_spec))
            if args.diagnostic_contacts and self.tracking and self.diagnostic_rows is not None:
                # This privileged channel is write-only evidence. Neither
                # perception nor the executor reads these records.
                sensor = self.scene.sensors['robot_contact']
                force = sensor.data.force_matrix_w
                if force is None:
                    raise RuntimeError('Diagnostic contact filters produced no force matrix')
                force = tensor(force)[0]
                magnitude = force.norm(dim=-1)
                pairs = (magnitude > .05).nonzero()
                counterpart_ids = sensor.contact_view.counterpart_indices[0]
                labels = sensor._get_model_labels('shape')
                self.diagnostic_rows.append({
                    'step': int(self.episode_length_buf[0]),
                    'contacts': [{'body': sensor.body_names[int(i)],
                                  'partner_shape': labels[counterpart_ids[int(j)]],
                                  'force_world_n': force[i, j].tolist(),
                                  'force_norm_n': float(magnitude[i, j])} for i, j in pairs]})
                if self.diagnostic_physics is not None:
                    self.diagnostic_physics_rows.append(self.diagnostic_physics.sample(self))
            if self.tracking and self.video_recorder is not None:
                self.video_recorder.after_step()
            if self.tracking and self.live_debug is not None and self.live_frames is not None:
                self.live_debug.offer(self.live_frames, self.live_trial, self.live_seed,
                                      float(self.episode_length_buf[0]) * self.step_dt)
            return result

        def _reset_idx(self, env_ids):
            if self.tracking and bool((self.episode_length_buf[env_ids] > 0).any()):
                if object_single:
                    from .object_env import TERMINATIONS
                    names = TERMINATIONS
                    if self.object_tracker is not None:
                        self.object_tracker.observe(int(self.episode_length_buf[0]), sample_env(self, object_spec))
                else:
                    names = ('success', 'time_out', 'staged_orange_dropped', 'staged_box_tipped', 'staged_abnormal_robot') if args.scene == 'single' else ('success', 'time_out', 'dropped_or_invalid')
                self.terminal_result = {name: bool(self.termination_manager.get_term(name)[0]) for name in names}
                if self.video_recorder is not None:
                    # Capture the terminal physics state before automatic reset.
                    self.video_recorder.after_step(terminal=True)
                if self.live_debug is not None and self.live_frames is not None:
                    self.live_debug.offer(self.live_frames, self.live_trial, self.live_seed,
                                          float(self.episode_length_buf[0]) * self.step_dt, force=True)
                self.tracking = False
            super()._reset_idx(env_ids)

    from isaaclab_tasks.utils.hydra import resolve_presets
    cfg = resolve_presets(Config(), selected=('newton', 'newton_renderer'))
    cfg.play_mode()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.evaluate_first_episode = False
    if args.controller in ('repaired', 'servo'):
        # Original collision-only tray meshes are hidden by Newton's visual
        # importer. Add equivalent non-colliding visuals in this instance only.
        cfg.scene.cardboard_box.spawn.usd_path = str(ROOT / 'assets/cardboard_box_visible.usda')
    cfg.scene.wrist_camera.data_types = ['rgb', 'distance_to_image_plane']
    from .wrist_view import configure_wrist_view
    configure_wrist_view(cfg, args.wrist_camera_profile)
    # The learned policy uses incremental actions; this independent IK adapter
    # must send absolute joint targets, including during physical settling.
    from isaaclab.envs.mdp.actions import JointPositionActionCfg
    cfg.actions.arm_action = JointPositionActionCfg(asset_name='robot',
        joint_names=['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'],
        scale=1.0, use_default_offset=False)
    cfg.actions.gripper_action = JointPositionActionCfg(asset_name='robot', joint_names=['gripper'],
        scale=1.0, use_default_offset=False)
    if not object_single:
        cfg.observations.rgb_mask.image.func = zero_image
        cfg.observations.rgb_mask.image.params['backend'] = 'gt'
    if args.diagnostic_contacts:
        cfg.scene.robot_contact.filter_shape_prim_expr = [
            '{ENV_REGEX_NS}/Scene.*',
            '{ENV_REGEX_NS}/' + (object_spec['prim_name'] if object_single else 'Orange') + '.*',
            '{ENV_REGEX_NS}/CardboardBox.*']
    if args.scene == 'single':
        cfg.events.reset_staged.params['forced_stage'] = 0
    elif not object_single:
        cfg.target_key = language['target_key']
    if args.record_video or args.live_debug:
        from isaaclab.sensors import CameraCfg
        from isaaclab_newton.renderers import NewtonWarpRendererCfg
        import isaaclab.sim as sim_utils
        from .video import VIDEO_WIDTH, VIDEO_HEIGHT
        cfg.scene.recording_camera = CameraCfg(
            prim_path='{ENV_REGEX_NS}/RecordingCamera', update_period=0.,
            width=VIDEO_WIDTH, height=VIDEO_HEIGHT, data_types=['rgb'],
            spawn=sim_utils.PinholeCameraCfg(focal_length=22., clipping_range=(.01, 20.)),
            renderer_cfg=NewtonWarpRendererCfg())
    env = ScoredEnv(cfg)
    from pick_anything.tasks.pick_apple.staged_mdp import silence_mjwarp_overflow_warning
    silence_mjwarp_overflow_warning(env)
    original_hashes = json.loads((ROOT / 'rl_source_before.json').read_text())
    rows = []

    def capture(directory):
        directory.mkdir(parents=True, exist_ok=False)
        sensor = env.scene.sensors['wrist_camera']
        data = sensor.data
        rgb = tensor(data.output['rgb'])[0, ..., :3].cpu().numpy().astype(np.uint8)
        depth = tensor(data.output['distance_to_image_plane'])[0].cpu().numpy().squeeze()
        if not np.isfinite(depth).any() or not (depth > 0).any():
            raise RuntimeError('Camera did not produce valid optical-axis depth')
        matrix = tensor(data.intrinsic_matrices)[0].cpu().numpy()
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(tensor(data.quat_w_ros)[0].cpu().numpy()).as_matrix()
        transform[:3, 3] = tensor(data.pos_w)[0].cpu().numpy()
        intrinsics = dict(width=rgb.shape[1], height=rgb.shape[0], fx=float(matrix[0, 0]), fy=float(matrix[1, 1]),
                          cx=float(matrix[0, 2]), cy=float(matrix[1, 2]), depth_scale=1.0)
        Image.fromarray(rgb).save(directory / 'rgb.png')
        np.save(directory / 'depth.npy', depth)
        (directory / 'intrinsics.json').write_text(json.dumps(intrinsics, indent=2) + '\n')
        (directory / 'camera_settings.json').write_text(json.dumps({
            'wrist_camera_profile': args.wrist_camera_profile,
            'capture_gripper_rad': args.capture_gripper_rad,
            'intrinsics_source': 'live sensor; optical-axis depth; ROS world_from_camera'}, indent=2) + '\n')
        (directory / 'world_from_camera.json').write_text(json.dumps(transform.tolist(), indent=2) + '\n')
        # Privileged state is saved for scoring/pairing diagnostics, never fed
        # to the language/GraspNet path. The GT mask is explicitly an oracle.
        robot = env.scene['robot']
        (directory / 'robot_joint_positions.json').write_text(json.dumps({
            'joint_names': list(robot.joint_names),
            'joint_position': tensor(robot.data.joint_pos)[0].tolist()}, indent=2) + '\n')
        state = {'joint_position': tensor(robot.data.joint_pos)[0].tolist(),
                 'objects': {key: tensor(env.scene[key].data.root_pos_w)[0].tolist()
                             for key in ('target_object', 'orange', 'cardboard_box', 'yellow_duck', 'blue_cube', 'red_cylinder') if key in env.scene.keys()}}
        if object_single:
            target = env.scene['target_object']
            state.update(target_key=language['target_key'], target_asset_key='target_object',
                         object_spec=object_spec, spawn_layout=env.object_spawn_layout,
                         role='privileged_scoring_diagnostic_only_not_used_by_control',
                         scoring_diagnostic=sample_env(env, object_spec),
                         target_root_state={
                             'position_world_m': tensor(target.data.root_pos_w)[0].tolist(),
                             'quaternion_xyzw': tensor(target.data.root_quat_w)[0].tolist(),
                             'linear_velocity_world_m_s': tensor(target.data.root_lin_vel_w)[0].tolist(),
                             'angular_velocity_world_rad_s': tensor(target.data.root_ang_vel_w)[0].tolist()},
                         target_contact_net_forces_world_n=tensor(
                             env.scene.sensors['target_contact'].data.net_forces_w)[0].tolist())
        (directory / 'evaluation_state.json').write_text(json.dumps(state, indent=2) + '\n')
        if args.mask_backend == 'gt':
            from pick_anything.grounded_pick.masks import sphere_boxes, boxes_to_masks
            boxes, available = sphere_boxes(tensor(env.scene['orange'].data.root_pos_w), tensor(data.pos_w),
                 tensor(data.quat_w_ros), tensor(data.intrinsic_matrices), cfg.observations.rgb_mask.image.params['radius'])
            oracle = boxes_to_masks(boxes, available, rgb.shape[0], rgb.shape[1])[0, 0].byte().cpu().numpy() * 255
            Image.fromarray(oracle).save(directory / 'oracle_mask.png')
        return transform

    def score_state():
        if object_single:
            return sample_env(env, object_spec)
        robot = env.scene['robot']
        gripper = int(robot.find_bodies(['gripper'])[0][0])
        grip_joint = int(robot.find_joints(['gripper'])[0][0])
        from isaaclab.utils.math import quat_apply
        position = tensor(robot.data.body_pos_w)[:, gripper]
        pocket = position + quat_apply(tensor(robot.data.body_quat_w)[:, gripper], position.new_tensor([.0408, -.0002, -.0709])[None])
        orange = tensor(env.scene['orange'].data.root_pos_w)
        rel_velocity = (tensor(env.scene['orange'].data.root_lin_vel_w) - tensor(robot.data.body_lin_vel_w)[:, gripper]).norm(dim=-1)
        sensor = env.scene.sensors['robot_contact']
        columns = [i for i, name in enumerate(sensor.body_names) if name in ('gripper', 'moving_jaw_so101_v1')]
        claw_force = tensor(sensor.data.net_forces_w)[:, columns].norm(dim=-1).amax(-1)
        orange_force = tensor(env.scene.sensors['orange_contact'].data.net_forces_w).norm(dim=-1).sum(-1)
        grip = tensor(robot.data.joint_pos)[:, grip_joint]
        distance = (orange - pocket).norm(dim=-1)
        held = (distance < .045) & (claw_force > 2.2) & (orange_force > .1) & (grip > .05) & (grip < 1.6) & (rel_velocity < .15)
        return {'orange_z': float(orange[0, 2]), 'distance': float(distance[0]), 'grip': float(grip[0]),
                'held': bool(held[0]), 'claw_force': float(claw_force[0])}

    def scan_views(directory, home):
        from .perception import inspect_view
        robot = env.scene['robot']
        pan = int(robot.find_joints(['shoulder_pan'])[0][0])
        grip = int(robot.find_joints(['gripper'])[0][0])
        limits = tensor(robot.data.soft_joint_pos_limits)[0]
        selected, destination, target_centre = None, None, None
        records = []
        for index, offset in enumerate((0., .60, -.60, 1.10, -1.10)):
            if index:
                target = home.clone()
                target[:, pan] += offset
                target[:, grip] = 1.4
                target = torch.maximum(torch.minimum(target, limits[:, 1]), limits[:, 0])
                start = tensor(robot.data.joint_pos).clone()
                for step in range(45):
                    if not env.tracking:
                        break
                    env.step(torch.lerp(start, target, min(1., (step + 1) / 30)))
                if not env.tracking:
                    break
                view = directory / f'view_{index:02d}'
                capture(view)
            else:
                view = directory
            result = inspect_view(view, args.device, need_box=args.scene in ('single', 'object-single'),
                                  instruction=language['grounding_instruction'],
                                  segmentation_backend=args.segmentation_backend,
                                  segmentation_model=args.segmentation_model)
            records.append({'view': str(view), 'pan_offset_rad': offset, **result})
            if selected is None and result['target_quality']['accepted']:
                selected = view
                target_centre = result['target_quality'].get('sphere', {}).get('centre_world_m')
            if args.scene in ('single', 'object-single') and result['box_quality']['accepted']:
                destination = result['box_quality']['world_base_m']
            print('PERCEPTION_VIEW=' + json.dumps(records[-1]), flush=True)
            if selected is not None and (args.scene == 'multi' or destination is not None):
                break
        (directory / 'scan.json').write_text(json.dumps({'views': records,
             'selected_view': str(selected) if selected is not None else None,
             'destination_world_m': destination, 'target_centre_world_m': target_centre}, indent=2) + '\n')
        return selected, destination, target_centre

    try:
        if args.live_debug:
            from .live_debug import LiveDebug
            env.live_debug = LiveDebug(root / 'live_debug', language['grounding_instruction'],
                                      args.live_debug_port, args.live_debug_device,
                                      args.segmentation_model, args.live_debug_interval)
            print('LIVE_DEBUG_URL=' + env.live_debug.url, flush=True)
        for trial in range(args.episodes):
            seed = args.seed + trial
            env.tracking = False
            env.terminal_result = None
            env.diagnostic_rows = []
            env.diagnostic_physics = None
            env.diagnostic_physics_rows = []
            env.object_tracker = None
            if object_single:
                env.object_episode_seed = seed
            env.seed(seed)
            env.reset()
            env.tracking = True
            if args.record_video or args.live_debug:
                from .video import EpisodeVideo, OBSERVER_EYE, OBSERVER_TARGET
                observer = env.scene.sensors['recording_camera']
                origin = tensor(env.scene.env_origins)[0]
                observer.set_world_poses_from_view(
                    torch.tensor([OBSERVER_EYE], device=env.device) + origin,
                    torch.tensor([OBSERVER_TARGET], device=env.device) + origin)
                def video_frames():
                    def rgb(camera):
                        return tensor(camera.data.output['rgb'])[0, ..., :3].cpu().numpy().astype(np.uint8)
                    return rgb(observer), rgb(env.scene.sensors['wrist_camera'])
                env.live_frames, env.live_trial, env.live_seed = video_frames, trial + 1, seed
            if args.record_video:
                video_context = {}
                if object_single or args.capture_only:
                    video_context = {'target_label': language['target_key'],
                                     'score_schema': SCHEMA if object_single else None,
                                     'lift_only': object_single, 'capture_only': args.capture_only}
                env.video_recorder = EpisodeVideo(
                    root / 'videos' / f'trial_{trial + 1:02d}_seed{seed}.mp4', video_frames,
                    env.step_dt, trial + 1, args.episodes, seed, args.video_fps, **video_context)
            # Perception must observe the settled object rather than its spawn
            # clearance in mid-air. These are ordinary physical control steps.
            home = tensor(env.scene['robot'].data.joint_pos).clone()
            for _ in range(60):
                env.step(home)
            if args.capture_gripper_rad is not None:
                grip = int(env.scene['robot'].find_joints(['gripper'])[0][0])
                target = home.clone()
                target[:, grip] = args.capture_gripper_rad
                for step in range(90):
                    if not env.tracking:
                        raise RuntimeError('Episode terminated during diagnostic jaw capture')
                    env.step(torch.lerp(home, target, min(1., (step + 1) / 60)))
            if not env.tracking:
                raise RuntimeError('Episode terminated while settling before perception')
            if object_single:
                env.object_tracker = LiftTracker(score_state()['target_center_z'], env.step_dt)
            directory = root / f'trial_{trial:02d}_seed{seed}'
            capture(directory)
            if args.capture_only:
                if env.video_recorder is not None:
                    env.video_recorder.finish({'failure_stage': 'capture_only'})
                    env.video_recorder = None
                print('CAPTURED=' + str(directory), flush=True)
                break
            view, destination, target_centre = directory, None, None
            perception_failed = False
            if args.controller in ('repaired', 'servo'):
                view, destination, target_centre = scan_views(directory, home)
                perception_failed = view is None
                view = view or directory
            command = [sys.executable, '-B', '-m', 'small_grasp.infer', '--rgb', str(view / 'rgb.png'),
                       '--depth', str(view / 'depth.npy'), '--intrinsics', str(view / 'intrinsics.json'),
                       '--world-from-camera', str(view / 'world_from_camera.json'), '--mask-backend', 'dino',
                       '--instruction', language['grounding_instruction'], '--device', args.device,
                       '--seed', str(args.inference_seed), '--width-scale', str(args.width_scale),
                       '--network-region', network_region, '--network-frame', args.network_frame,
                       '--wrist-camera-profile', args.wrist_camera_profile,
                       '--camera-settings', str(view / 'camera_settings.json'),
                       '--candidate-pool-size', str(candidate_pool_size),
                       '--output-dir', str(directory / 'proposals')]
            if args.mask_backend == 'gt' or args.controller in ('repaired', 'servo'):
                index = command.index('--mask-backend')
                command[index + 1] = 'saved'
                command += ['--mask', str(view / ('target_mask.png' if args.controller in ('repaired', 'servo') else 'oracle_mask.png'))]
                if args.controller in ('repaired', 'servo'):
                    command += ['--mask-provenance', str(view / 'perception.json')]
            else:
                command += ['--segmentation-backend', args.segmentation_backend]
                if args.segmentation_model:
                    command += ['--segmentation-model', str(args.segmentation_model)]
            with (directory / 'inference.log').open('w') as stream:
                subprocess.run(command, cwd=ROOT.parent, stdout=stream, stderr=subprocess.STDOUT, check=True)
            proposals = json.loads((directory / 'proposals/result.json').read_text())
            if env.live_debug is not None:
                env.live_debug.publish_candidates(directory / 'proposals', view / 'intrinsics.json',
                                                  trial + 1, proposals['candidates'])
            row = {'trial': trial, 'seed': seed, 'inference_seed': args.inference_seed,
                   'width_scale': args.width_scale, 'scene': args.scene, **language,
                   'counts': proposals['counts'], 'grounding': proposals['grounding'], 'success': False,
                   'grasp_lift_success': False, 'failure_stage': proposals['failure_stage'], 'robot_executed': False,
                   'terminal': None, 'command': command}
            row.update(mask_backend=args.mask_backend, closure=args.closure, controller=args.controller,
                       arm_tracking_feedback_requested=args.arm_tracking_feedback,
                       closure_latch_policy=args.closure_latch_policy,
                       close_tracking_integral_requested=args.close_tracking_integral,
                       close_integral_transfer=args.close_integral_transfer,
                       close_frame_policy=args.close_frame_policy,
                       candidate_contact_ranking_requested=args.candidate_contact_ranking,
                       candidate_contact_policy=args.candidate_contact_policy,
                       wrist_camera_profile=args.wrist_camera_profile,
                       network_region=network_region, network_frame=args.network_frame,
                       network_input_frame=proposals['network_input'],
                       candidate_pool_size=candidate_pool_size,
                       segmentation_backend=args.segmentation_backend, collision_check=args.collision_check,
                       grasp_point=args.grasp_point, contact_frame=args.contact_frame,
                       effective_closure='centred_sustained_close' if args.controller in ('repaired', 'servo') else args.closure,
                       selected_view=str(view), destination_world_m=destination)
            if object_single:
                row.update(object_spec=object_spec, score_schema=SCHEMA,
                           target_asset_key='target_object', place_success=None, place_evaluated=False)
            if perception_failed:
                row['failure_stage'] = 'target_not_verified_after_scan'
            if proposals['candidates'] and env.tracking and not perception_failed:
                from .sim_robot import GraspExecutor
                executor = GraspExecutor(env, score_state, directory, closure=args.closure,
                                         controller=args.controller, destination=destination,
                                         target_centre=target_centre if args.grasp_point == 'sphere' else None,
                                         contact_frame=args.contact_frame, collision_check=args.collision_check,
                                         observation_dir=view, arm_tracking_feedback=args.arm_tracking_feedback,
                                         candidate_contact_ranking=args.candidate_contact_ranking,
                                         candidate_contact_policy=args.candidate_contact_policy,
                                         closure_latch_policy=args.closure_latch_policy,
                                         close_tracking_integral=args.close_tracking_integral,
                                         close_integral_transfer=args.close_integral_transfer,
                                         close_frame_policy=args.close_frame_policy)
                if args.diagnostic_contacts and object_single:
                    from .physics_diagnostics import PhysicsDiagnostics
                    env.diagnostic_physics = PhysicsDiagnostics('/World/envs/env_0/' + object_spec['prim_name'])
                row.update(executor.execute(proposals['candidates'], args.scene))
            if env.live_debug is not None:
                env.live_debug.publish_candidates(directory / 'proposals', view / 'intrinsics.json',
                                                  trial + 1, proposals['candidates'], row)
            env.tracking = False
            row['terminal'] = env.terminal_result
            if env.terminal_result and not object_single:
                row['success'] = env.terminal_result['success']
            if object_single:
                trace_path = directory / 'execution_trace.json'
                execution_trace = json.loads(trace_path.read_text()) if trace_path.is_file() else []
                generic = env.object_tracker.summary(execution_trace)
                row['executor_grasp_lift_success'] = row['grasp_lift_success']
                row.update(generic)
                row['grasp_lift_success'] = generic['generic_grasp_lift_success']
                row['success'] = generic['generic_grasp_lift_success']
                row['success_semantics'] = 'generic lift only; original place task not evaluated'
                if row['success']:
                    row['failure_stage'] = None
                (directory / 'object_scoring_trace.json').write_text(json.dumps({
                    **generic, 'samples': env.object_tracker.rows}, indent=2, allow_nan=False) + '\n')
            if env.video_recorder is not None:
                row['video'] = env.video_recorder.finish(row)
                env.video_recorder = None
            (directory / 'episode.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
            if args.diagnostic_contacts:
                from .physics_diagnostics import SCHEMA, phase_rows
                trace_path = directory / 'execution_trace.json'
                issued_trace = json.loads(trace_path.read_text()) if trace_path.exists() else []
                (directory / 'contact_diagnostics.json').write_text(json.dumps({
                    'role': 'privileged_diagnostic_only_not_used_by_control',
                    'force_semantics': 'per-body/per-observed-filter-shape total force (normal plus friction)',
                    'minimum_recorded_force_n': .05,
                    'physics': {'schema': SCHEMA, 'scope': 'object-single close/hold/lift issued steps only',
                                'role': 'privileged diagnostic only; no control or score input',
                                'rows': phase_rows(env.diagnostic_physics_rows, issued_trace)},
                    'rows': env.diagnostic_rows}, indent=2, allow_nan=False) + '\n')
            rows.append(row)
            result = {'scene': args.scene, 'episodes': len(rows), 'requested_episodes': args.episodes,
                      'successes': sum(r['success'] for r in rows), 'success_rate': sum(r['success'] for r in rows) / len(rows),
                      'grasp_lift_successes': sum(r['grasp_lift_success'] for r in rows), 'rows': rows,
                      'mask_backend': args.mask_backend, 'closure': args.closure, 'checkpoint_sha256': proposals['checkpoint_sha256'],
                      'controller': args.controller,
                      'arm_tracking_feedback_requested': args.arm_tracking_feedback,
                      'closure_latch_policy': args.closure_latch_policy,
                      'close_tracking_integral_requested': args.close_tracking_integral,
                      'close_integral_transfer': args.close_integral_transfer,
                      'close_frame_policy': args.close_frame_policy,
                      'candidate_contact_ranking_requested': args.candidate_contact_ranking,
                      'candidate_contact_policy': args.candidate_contact_policy,
                      'wrist_camera_profile': args.wrist_camera_profile,
                      'network_region': network_region, 'network_frame': args.network_frame,
                      'network_input_frame': network_input_frame,
                      'candidate_pool_size': candidate_pool_size,
                      'inference_seed': args.inference_seed,
                      'width_scale': args.width_scale,
                      'segmentation_backend': args.segmentation_backend, 'collision_check': args.collision_check,
                      'language': language, 'viz': args.viz,
                      'grasp_point': args.grasp_point, 'contact_frame': args.contact_frame,
                      'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'argv': sys.argv,
                      'metric': (SCHEMA if object_single else 'original HOME full place/retreat'
                                 if args.scene == 'single' else 'original correct held lift 8cm for 5 steps')}
            if object_single:
                result.update(score_schema=SCHEMA, place_evaluated=False, place_success=None,
                              original_task_success=None,
                              generic_end_hold_successes=sum(r['generic_end_hold_success'] for r in rows))
            (root / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            print('SMALL_GRASP_EPISODE=' + json.dumps(row, ensure_ascii=False), flush=True)
        unchanged = all(hashlib.sha256((ROOT.parent / p).read_bytes()).hexdigest() == digest for p, digest in original_hashes.items())
        if not unchanged:
            raise RuntimeError('Existing RL source hashes changed during evaluation')
    finally:
        try:
            if env.video_recorder is not None:
                env.video_recorder.finish()
        finally:
            try:
                if env.live_debug is not None:
                    env.live_debug.close()
            finally:
                env.close()
                launcher.app.close()


if __name__ == '__main__':
    main()
