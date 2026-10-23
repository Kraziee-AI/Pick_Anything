"""Language -> Grounding -> target mask -> GraspNet proposals -> SO101 execution."""
import argparse
from datetime import datetime
from pathlib import Path

from .instructions import resolve_pick_instruction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instruction', required=True, help='Target description: orange or 橘子')
    parser.add_argument('--scene', choices=['single', 'multi', 'object-single'], default='multi')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--inference-seed', type=int, default=42)
    parser.add_argument('--width-scale', type=float, default=10.)
    parser.add_argument('--controller', choices=['repaired', 'servo'], default='repaired')
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
    parser.add_argument('--evaluation-role', choices=['smoke', 'regression', 'holdout'], default='smoke')
    parser.add_argument('--segmentation-backend', choices=['sam', 'depth', 'box'], default='sam')
    parser.add_argument('--segmentation-model', type=Path)
    parser.add_argument('--collision-check', choices=['off', 'capsule'], default='off',
                        help='capsule enables experimental sampled whole-arm collision checks')
    parser.add_argument('--viz', choices=['none', 'kit'], default='none')
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--live-debug', action='store_true', help='Local live scene / DINO+SAM split screen')
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
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        language = resolve_pick_instruction(args.instruction, allow_objects=args.scene == 'object-single')
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
              '--wrist-camera-profile', args.wrist_camera_profile, '--network-region', args.network_region,
              '--network-frame', args.network_frame]
    if args.candidate_pool_size is not None:
        command += ['--candidate-pool-size', str(args.candidate_pool_size)]
    if args.arm_tracking_feedback:
        command += ['--arm-tracking-feedback']
    if args.close_tracking_integral:
        if args.controller != 'servo' or not args.arm_tracking_feedback:
            parser.error('--close-tracking-integral requires --controller servo --arm-tracking-feedback')
        command += ['--close-tracking-integral']
    if args.close_frame_policy != 'fixed-body':
        if (args.controller != 'servo' or not args.arm_tracking_feedback or args.close_tracking_integral or
                args.close_integral_transfer != 'reset' or args.closure_latch_policy != 'legacy'):
            parser.error('--close-frame-policy planned-line/reachable-line requires servo tracking, integral off, reset transfer and legacy latch')
        command += ['--close-frame-policy', args.close_frame_policy]
    if args.close_integral_transfer != 'reset':
        if args.controller != 'servo' or not args.arm_tracking_feedback or not args.close_tracking_integral:
            parser.error('--close-integral-transfer adjacent requires --controller servo --arm-tracking-feedback --close-tracking-integral')
        command += ['--close-integral-transfer', args.close_integral_transfer]
    if args.closure_latch_policy != 'legacy':
        if args.controller != 'servo' or not args.arm_tracking_feedback:
            parser.error('--closure-latch-policy no_reopen requires --controller servo --arm-tracking-feedback')
        command += ['--closure-latch-policy', args.closure_latch_policy]
    if args.candidate_contact_ranking:
        command += ['--candidate-contact-ranking', '--candidate-contact-policy', args.candidate_contact_policy]
    elif args.candidate_contact_policy != 'soft8':
        parser.error('--candidate-contact-policy requires --candidate-contact-ranking')
    if args.segmentation_model:
        command += ['--segmentation-model', str(args.segmentation_model.resolve())]
    if args.record_video:
        command += ['--record-video', '--video-fps', str(args.video_fps)]
    if args.live_debug:
        command += ['--live-debug', '--live-debug-port', str(args.live_debug_port),
                    '--live-debug-device', args.live_debug_device,
                    '--live-debug-interval', str(args.live_debug_interval)]
    evaluate(command)
    print('PICK_OUTPUT=' + str(output.resolve()), flush=True)


if __name__ == '__main__':
    main()
