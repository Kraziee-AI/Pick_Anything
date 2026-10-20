"""Validate model resources and execution combinations before starting Isaac."""
import hashlib
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
