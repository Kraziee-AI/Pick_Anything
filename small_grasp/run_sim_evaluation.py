"""Run independent SO101 small-model evaluations and keep complete subprocess logs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ''):
    # Preserve direct-script usage from any working directory.
    sys.path.insert(0, str(ROOT.parent))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', nargs='+', choices=['single', 'multi'], default=['single', 'multi'])
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--inference-seed', type=int, default=42,
                        help='Point sampling seed, independent of the scene seed')
    parser.add_argument('--width-scale', type=float, default=10.,
                        help='Experimental GraspNet width decoder multiplier; legacy default 10')
    parser.add_argument('--evaluation-role', choices=['smoke', 'regression', 'holdout'], default='regression')
    parser.add_argument('--segmentation-backend', choices=['sam', 'depth', 'box'], default='box')
    parser.add_argument('--segmentation-model', type=Path)
    parser.add_argument('--collision-check', choices=['off', 'capsule'], default='off')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--capture-only', action='store_true')
    parser.add_argument('--capture-gripper-rad', type=float)
    parser.add_argument('--diagnostic-contacts', action='store_true',
                        help='Save privileged contact partners in a separate diagnostic file; never used by control')
    parser.add_argument('--mask-backend', choices=['dino', 'gt'], default='dino')
    parser.add_argument('--closure', choices=['first_contact', 'full_close'], default='first_contact')
    parser.add_argument('--controller', choices=['baseline', 'repaired', 'servo'], default='repaired')
    parser.add_argument('--grasp-point', choices=['network', 'sphere'], default='network')
    parser.add_argument('--contact-frame', choices=['open', 'contact'], default='open')
    parser.add_argument('--instruction', default='orange')
    parser.add_argument('--viz', choices=['none', 'kit'], default='none')
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--live-debug', action='store_true')
    parser.add_argument('--live-debug-port', type=int, default=8765)
    parser.add_argument('--live-debug-device', default='cpu')
    parser.add_argument('--live-debug-interval', type=float, default=1.)
    parser.add_argument('--video-fps', type=float, default=15.)
    parser.add_argument('--wrist-camera-profile', choices=['legacy', 'open-jaws'], default='legacy')
    parser.add_argument('--network-region', choices=['auto', 'bbox', 'target-mask'], default='auto')
    parser.add_argument('--candidate-pool-size', type=int,
                        help='NMS pool: defaults to 100 for open-jaws target-mask, otherwise 20')
    args = parser.parse_args(argv)
    from small_grasp.instructions import resolve_pick_instruction
    try:
        language = resolve_pick_instruction(args.instruction)
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
    from small_grasp.pipeline_options import validate_options, model_manifest, resolve_network_region, resolve_candidate_pool_size
    network_region = resolve_network_region(args.wrist_camera_profile, args.network_region)
    try:
        candidate_pool_size = resolve_candidate_pool_size(args.wrist_camera_profile, network_region, args.candidate_pool_size)
        validate_options(args)
    except ValueError as error:
        parser.error(str(error))
    root = args.output_dir.resolve()
    if not root.is_relative_to(ROOT):
        parser.error('outputs must stay inside small_grasp')
    root.mkdir(parents=True, exist_ok=False)
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in ROOT.rglob('*.py') if 'outputs' not in path.parts and 'cache' not in path.parts}
    snapshot = root / 'source'
    for relative in sources:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    assets = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (ROOT / 'assets').glob('*.usd*')}
    (root / 'run.json').write_text(json.dumps({'argv': sys.argv, 'source_sha256': sources, 'asset_sha256': assets,
        'runner_arguments': argv if argv is not None else sys.argv[1:], 'language': language, 'viz': args.viz,
        'seeds': list(range(args.seed, args.seed + args.episodes)), 'scenes': args.scenes,
        'inference_seed': args.inference_seed, 'width_scale': args.width_scale, 'controller': args.controller,
        'evaluation_role': args.evaluation_role,
        'segmentation_backend': args.segmentation_backend, 'collision_check': args.collision_check,
        'segmentation_model': model_manifest(args), 'capture_only': args.capture_only,
        'diagnostic_contacts': args.diagnostic_contacts,
        'record_video': args.record_video, 'video_fps': args.video_fps,
        'live_debug': {'enabled': args.live_debug, 'port': args.live_debug_port,
                       'device': args.live_debug_device, 'interval_s': args.live_debug_interval},
        'wrist_camera_profile': args.wrist_camera_profile,
        'network_region_requested': args.network_region, 'network_region': network_region,
        'candidate_pool_size_requested': args.candidate_pool_size, 'candidate_pool_size': candidate_pool_size,
        'capture_gripper_rad': args.capture_gripper_rad}, indent=2) + '\n')
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    for scene in args.scenes:
        command = [sys.executable, '-B', '-m', 'small_grasp.sim_eval', '--scene', scene,
                   '--episodes', str(args.episodes), '--device', 'cuda:0', '--output-dir', str(root / scene),
                   '--mask-backend', args.mask_backend, '--closure', args.closure, '--controller', args.controller,
                   '--grasp-point', args.grasp_point, '--contact-frame', args.contact_frame,
                   '--instruction', language['instruction'], '--viz', args.viz, '--seed', str(args.seed),
                   '--inference-seed', str(args.inference_seed),
                   '--width-scale', str(args.width_scale),
                   '--segmentation-backend', args.segmentation_backend, '--collision-check', args.collision_check,
                   '--wrist-camera-profile', args.wrist_camera_profile, '--network-region', network_region,
                   '--candidate-pool-size', str(candidate_pool_size)]
        if args.segmentation_model:
            command += ['--segmentation-model', str(args.segmentation_model.resolve())]
        if args.capture_only:
            command.append('--capture-only')
        if args.capture_gripper_rad is not None:
            command += ['--capture-gripper-rad', str(args.capture_gripper_rad)]
        if args.diagnostic_contacts:
            command.append('--diagnostic-contacts')
        if args.record_video:
            command += ['--record-video', '--video-fps', str(args.video_fps)]
        if args.live_debug:
            command += ['--live-debug', '--live-debug-port', str(args.live_debug_port),
                        '--live-debug-device', args.live_debug_device,
                        '--live-debug-interval', str(args.live_debug_interval)]
            print('LIVE_DEBUG=' + (f'http://127.0.0.1:{args.live_debug_port} (ready after scene initialization)'
                  if args.live_debug_port else str(root / scene / 'live_debug/viewer.json')), flush=True)
        print(json.dumps({'command': command, 'log': str(root / f'{scene}.log')}), flush=True)
        try:
            with (root / f'{scene}.log').open('w') as stream:
                subprocess.run(command, cwd=ROOT.parent, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            if not args.capture_only and not (root / scene / 'results.json').is_file():
                raise RuntimeError(f'{scene} exited without results.json; inspect {root / f"{scene}.log"}')
        except (subprocess.CalledProcessError, RuntimeError) as error:
            (root / 'run_failure.json').write_text(json.dumps({
                'scene': scene, 'requested_episodes': args.episodes,
                'failure_stage': 'simulator_process', 'error': str(error),
                'log': str(root / f'{scene}.log'), 'counts_as_success': False}, indent=2) + '\n')
            raise
        if not args.capture_only:
            data = json.loads((root / scene / 'results.json').read_text())
            assert data['episodes'] == args.episodes
            print(json.dumps({k: data[k] for k in ('scene', 'episodes', 'successes', 'success_rate', 'grasp_lift_successes')}), flush=True)


if __name__ == '__main__':
    main()
