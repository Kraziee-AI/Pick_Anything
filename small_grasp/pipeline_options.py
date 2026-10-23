"""Validate model resources and execution combinations before starting Isaac."""
import hashlib
import math
from pathlib import Path


def resolve_network_region(profile, region):
    if region not in ('auto', 'bbox', 'target-mask'):
        raise ValueError(f'Unknown network region: {region}')
    return ('target-mask' if profile == 'open-jaws' else 'bbox') if region == 'auto' else region


def resolve_candidate_pool_size(profile, region, requested=None):
    """Keep the inference pool independent of the primary execution budget."""
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError('candidate pool size must be a positive integer')
        return requested
    return 100 if profile == 'open-jaws' and region == 'target-mask' else 20


def sam_path(args):
    from .segmentation import default_model
    return Path(args.segmentation_model or default_model()).expanduser().resolve()


def validate_options(args):
    close_frame = getattr(args, 'close_frame_policy', 'fixed-body')
    if close_frame not in ('fixed-body', 'planned-line', 'reachable-line'):
        raise ValueError('Unsupported --close-frame-policy')
    if close_frame in ('planned-line', 'reachable-line') and (args.controller != 'servo' or not getattr(args, 'arm_tracking_feedback', False) or
            args.contact_frame != 'open' or getattr(args, 'close_tracking_integral', False) or
            getattr(args, 'close_integral_transfer', 'reset') != 'reset' or
            getattr(args, 'closure_latch_policy', 'legacy') != 'legacy'):
        raise ValueError('--close-frame-policy planned-line/reachable-line requires servo tracking, open frame, integral off, reset transfer and legacy latch')
    transfer = getattr(args, 'close_integral_transfer', 'reset')
    if transfer not in ('reset', 'adjacent'):
        raise ValueError('Unsupported --close-integral-transfer')
    if transfer == 'adjacent' and (args.controller != 'servo' or not getattr(args, 'arm_tracking_feedback', False) or
                                   not getattr(args, 'close_tracking_integral', False)):
        raise ValueError('--close-integral-transfer adjacent requires --controller servo --arm-tracking-feedback --close-tracking-integral')
    if getattr(args, 'close_tracking_integral', False) and (args.controller != 'servo' or not getattr(args, 'arm_tracking_feedback', False)):
        raise ValueError('--close-tracking-integral requires --controller servo --arm-tracking-feedback')
    latch_policy = getattr(args, 'closure_latch_policy', 'legacy')
    if latch_policy not in ('legacy', 'no_reopen'):
        raise ValueError('Unsupported --closure-latch-policy')
    if latch_policy != 'legacy' and (args.controller != 'servo' or not getattr(args, 'arm_tracking_feedback', False)):
        raise ValueError('--closure-latch-policy no_reopen requires --controller servo --arm-tracking-feedback')
    if getattr(args, 'arm_tracking_feedback', False) and args.controller != 'servo':
        raise ValueError('--arm-tracking-feedback requires --controller servo')
    if getattr(args, 'candidate_contact_ranking', False) and args.controller != 'servo':
        raise ValueError('--candidate-contact-ranking requires --controller servo')
    if getattr(args, 'candidate_contact_policy', 'soft8') != 'soft8' and not getattr(args, 'candidate_contact_ranking', False):
        raise ValueError('--candidate-contact-policy requires --candidate-contact-ranking')
    scenes = getattr(args, 'scenes', [getattr(args, 'scene', '')])
    if 'object-single' in scenes:
        if (args.controller != 'servo' or args.mask_backend != 'dino' or
                args.segmentation_backend != 'sam' or args.grasp_point != 'network' or
                args.contact_frame != 'open'):
            raise ValueError('object-single requires servo, real DINO/SAM, network poses and open contact frame')
    from .network_frame import frame_metadata
    frame_metadata(getattr(args, 'wrist_camera_profile', None), getattr(args, 'network_frame', 'camera'))
    if getattr(args, 'live_debug', False):
        if not 0 <= args.live_debug_port <= 65535:
            raise ValueError('live-debug-port must be between 0 and 65535')
        if not math.isfinite(args.live_debug_interval) or args.live_debug_interval <= 0:
            raise ValueError('live-debug-interval must be positive and finite')
    if args.controller in ('repaired', 'servo') and args.mask_backend != 'dino':
        raise ValueError('repaired/servo controllers require real language detection; GT diagnostics require baseline')
    if args.segmentation_model and args.segmentation_backend != 'sam':
        raise ValueError('--segmentation-model requires --segmentation-backend sam')
    if args.collision_check == 'capsule' and args.controller not in ('repaired', 'servo'):
        raise ValueError('capsule collision checking requires a repaired/servo controller')
    if args.controller == 'servo' and args.grasp_point != 'network':
        raise ValueError('servo requires network grasp poses without sphere overrides')
    if getattr(args, 'inference_seed', 42) < 0:
        raise ValueError('inference seed must be nonnegative')
    if args.segmentation_backend != 'box' and args.mask_backend == 'gt':
        raise ValueError('GT diagnostics cannot be combined with learned/depth segmentation')
    if args.segmentation_backend == 'sam' and args.grasp_point == 'sphere':
        raise ValueError('SAM uses generic masks; sphere grasp points require the orange depth baseline')
    if args.segmentation_backend == 'sam' and not args.capture_only:
        path = sam_path(args)
        missing = [name for name in ('config.json', 'preprocessor_config.json', 'model.safetensors')
                   if not (path / name).is_file()]
        if missing:
            raise ValueError(f'Local SAM checkpoint incomplete at {path}: {", ".join(missing)}')


def model_manifest(args):
    if args.segmentation_backend != 'sam' or args.capture_only:
        return None
    path = sam_path(args)
    return {'path': str(path), 'files_sha256': {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in ('config.json', 'preprocessor_config.json', 'model.safetensors')}}
