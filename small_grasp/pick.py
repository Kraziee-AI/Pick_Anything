"""Language -> Grounding -> target mask -> GraspNet proposals -> SO101 execution."""
import argparse
from datetime import datetime
from pathlib import Path

from .instructions import resolve_pick_instruction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instruction', required=True, help='Target description: orange or 橘子')
    parser.add_argument('--scene', choices=['single', 'multi'], default='multi')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--inference-seed', type=int, default=42)
    parser.add_argument('--width-scale', type=float, default=10.)
    parser.add_argument('--controller', choices=['repaired', 'servo'], default='repaired')
    parser.add_argument('--evaluation-role', choices=['smoke', 'regression', 'holdout'], default='smoke')
    parser.add_argument('--segmentation-backend', choices=['sam', 'depth', 'box'], default='sam')
    parser.add_argument('--segmentation-model', type=Path)
    parser.add_argument('--collision-check', choices=['off', 'capsule'], default='off',
                        help='capsule enables experimental sampled whole-arm collision checks')
    parser.add_argument('--viz', choices=['none', 'kit'], default='none')
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--video-fps', type=float, default=15.)
    parser.add_argument('--wrist-camera-profile', choices=['legacy', 'open-jaws'], default='legacy')
    parser.add_argument('--network-region', choices=['auto', 'bbox', 'target-mask'], default='auto')
    parser.add_argument('--candidate-pool-size', type=int,
                        help='NMS pool: defaults to 100 for open-jaws target-mask, otherwise 20')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        language = resolve_pick_instruction(args.instruction)
    except ValueError as error:
        parser.error(str(error))
    if args.episodes < 1:
        parser.error('episodes must be positive')
    from .run_sim_evaluation import ROOT, main as evaluate
    output = args.output_dir or ROOT / 'outputs' / datetime.now().strftime('language_pick_%Y%m%d_%H%M%S_%f')
    command = ['--scenes', args.scene, '--episodes', str(args.episodes), '--seed', str(args.seed),
              '--instruction', language['instruction'], '--viz', args.viz,
              '--controller', args.controller, '--mask-backend', 'dino',
              '--inference-seed', str(args.inference_seed),
              '--width-scale', str(args.width_scale),
              '--grasp-point', 'network', '--contact-frame', 'open', '--output-dir', str(output),
              '--segmentation-backend', args.segmentation_backend, '--collision-check', args.collision_check,
              '--evaluation-role', args.evaluation_role,
              '--wrist-camera-profile', args.wrist_camera_profile, '--network-region', args.network_region]
    if args.candidate_pool_size is not None:
        command += ['--candidate-pool-size', str(args.candidate_pool_size)]
    if args.segmentation_model:
        command += ['--segmentation-model', str(args.segmentation_model.resolve())]
    if args.record_video:
        command += ['--record-video', '--video-fps', str(args.video_fps)]
    evaluate(command)
    print('PICK_OUTPUT=' + str(output.resolve()), flush=True)


if __name__ == '__main__':
    main()
