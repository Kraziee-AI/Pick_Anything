"""Standalone pretrained RGB-D GraspNet proposals, never robot execution."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def configure_runtime():
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    for name, suffix in [('XDG_CACHE_HOME', ''), ('HF_HOME', 'huggingface'), ('TORCH_HOME', 'torch'),
                         ('MPLCONFIGDIR', 'matplotlib')]:
        path = ROOT / 'cache' / suffix
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='Bundled real RGB-D pear and saved RefTR target mask')
    parser.add_argument('--rgb', type=Path)
    parser.add_argument('--depth', type=Path)
    parser.add_argument('--intrinsics', type=Path)
    parser.add_argument('--mask', type=Path, help='Registered single-channel binary target mask (0/1 or 0/255)')
    parser.add_argument('--mask-provenance', type=Path,
                        help='perception.json recording the same saved target_mask.png and its SHA256')
    parser.add_argument('--instruction')
    parser.add_argument('--mask-backend', choices=['saved', 'dino'])
    parser.add_argument('--grounding-model')
    parser.add_argument('--segmentation-backend', choices=['sam', 'depth', 'box'], default='box',
                        help='DINO box refinement; box preserves the legacy rectangular ROI')
    parser.add_argument('--segmentation-model', type=Path, help='Local checkpoint for the SAM segmentation backend')
    parser.add_argument('--box-threshold', type=float, default=0.25)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/graspnet_epoch18.tar')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-points', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top-k', '--candidate-pool-size', dest='top_k', type=int, default=20,
                        help='NMS proposal retention pool; independent of the executor primary budget')
    parser.add_argument('--min-score', type=float, default=0.05)
    parser.add_argument('--max-width', type=float, default=0.075, help='Grasp proposal width cap in metres; not a SO101 calibration')
    parser.add_argument('--width-scale', type=float, default=10.0,
                        help='Width head multiplier before the unchanged 0.075m decoder cap; 1.0 is an explicit native-unit experiment')
    parser.add_argument('--max-depth', type=float, default=2.0)
    parser.add_argument('--depth-tolerance', type=float, default=0.04)
    parser.add_argument('--crop-padding', type=int, default=20)
    parser.add_argument('--network-region', choices=['bbox', 'target-mask'], default='bbox',
                        help='Network input only; target association and full-scene collision checks remain enabled')
    parser.add_argument('--network-frame', choices=['camera', 'legacy-optical'], default='camera',
                        help='Experimental model input basis; all exported poses remain in the real camera frame')
    parser.add_argument('--wrist-camera-profile', choices=['legacy', 'open-jaws'],
                        help='Known capture mount; required for legacy-optical model coordinates')
    parser.add_argument('--camera-settings', type=Path,
                        help='Capture camera_settings.json; otherwise use the intrinsics sibling when present')
    parser.add_argument('--collision-threshold', type=float, default=0.01)
    parser.add_argument('--collision-voxel', type=float, default=0.005)
    parser.add_argument('--world-from-camera', type=Path, help='JSON 4x4 T_world_camera; otherwise export camera poses only')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.demo:
        if any(value is not None for value in (args.rgb, args.depth, args.intrinsics, args.mask)):
            parser.error('--demo cannot be combined with custom images or mask')
        sample = ROOT / 'examples/pear'
        args.rgb, args.depth, args.intrinsics = [sample / name for name in ('rgb.png', 'depth.png', 'intrinsics.json')]
        if args.mask_backend != 'dino' and args.segmentation_backend == 'box':
            if args.instruction not in (None, 'large yellow pear'):
                parser.error('The saved demo mask belongs to large yellow pear; use --mask-backend dino for a new instruction')
            args.mask = sample / 'target_mask.png'
        args.instruction = args.instruction or 'large yellow pear'
    if not all((args.rgb, args.depth, args.intrinsics)):
        parser.error('Provide --demo or --rgb/--depth/--intrinsics')
    args.mask_backend = args.mask_backend or ('saved' if args.mask else 'dino')
    if args.mask_backend == 'saved' and not args.mask:
        parser.error('saved backend requires --mask')
    if args.mask_backend == 'dino' and (not args.instruction or args.mask):
        parser.error('dino backend requires --instruction and no --mask')
    if args.mask_provenance and args.mask_backend != 'saved':
        parser.error('--mask-provenance requires --mask-backend saved')
    if args.mask_backend == 'saved' and args.segmentation_backend != 'box':
        parser.error('sam/depth segmentation requires --mask-backend dino')
    if args.segmentation_model and args.segmentation_backend != 'sam':
        parser.error('--segmentation-model requires --segmentation-backend sam')
    args.instruction = args.instruction or 'externally supplied target mask'
    if args.network_frame != 'camera' and args.wrist_camera_profile is None:
        parser.error('--network-frame legacy-optical requires explicit --wrist-camera-profile')
    if not (args.num_points >= 2048 and args.top_k >= 1 and 0 < args.max_width <= 0.075
            and args.max_depth > 0 and args.depth_tolerance > 0 and args.crop_padding >= 0
            and args.collision_threshold > 0 and args.collision_voxel > 0 and args.min_score >= 0
            and 0 < args.box_threshold < 1 and 0 < args.text_threshold < 1
            and math.isfinite(args.width_scale) and args.width_scale > 0):
        parser.error('Invalid sampling, filtering or threshold parameters')
    output = args.output_dir or ROOT / 'outputs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    args.output_dir = output.expanduser().resolve()
    if not args.output_dir.is_relative_to(ROOT):
        parser.error('Output directory must stay inside small_grasp/')
    return args


def load_mask_provenance(path, mask_path, mask, instruction):
    """Associate a saved mask only with its verified, same-view perception record."""
    path, mask_path = Path(path).resolve(), Path(mask_path).resolve()
    if mask_path.name != 'target_mask.png' or path.name != 'perception.json' or path.parent != mask_path.parent:
        raise ValueError('Mask provenance requires same-view perception.json and target_mask.png')
    source_bytes = path.read_bytes()
    record = json.loads(source_bytes)
    if not isinstance(record, dict):
        raise ValueError('Mask provenance must be a JSON object')
    for key in ('target_detection', 'target_segmentation', 'target_quality'):
        if not isinstance(record.get(key), dict):
            raise ValueError(f'Mask provenance missing {key} object')
    expected_sha = record.get('target_mask_sha256')
    actual_sha = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    if not isinstance(expected_sha, str) or expected_sha != actual_sha:
        raise ValueError('Saved target mask SHA256 does not match perception provenance')
    detection, segmentation, quality = (record[key] for key in
                                        ('target_detection', 'target_segmentation', 'target_quality'))
    if detection.get('backend') != 'dino' or detection.get('instruction') != instruction:
        raise ValueError('Mask provenance detection backend or instruction does not match this request')
    if segmentation.get('backend') not in ('sam', 'depth', 'box') or not segmentation.get('provenance'):
        raise ValueError('Mask provenance has no recognized segmentation source')
    accepted, pixels = quality.get('accepted'), quality.get('pixels')
    if not isinstance(accepted, bool):
        raise ValueError('Mask provenance target_quality must state acceptance')
    # Orange rejection records can omit pixels when the saved mask is empty.
    if ((pixels is None and (accepted or mask.any())) or
            (pixels is not None and (isinstance(pixels, bool) or pixels != int(mask.sum())))):
        raise ValueError('Saved target mask pixels disagree with perception provenance')
    return {'backend': 'saved_verified_perception', 'instruction': instruction,
            'detection': detection, 'segmentation': segmentation, 'mask_refinement': quality,
            'mask_source': str(mask_path), 'mask_sha256': actual_sha,
            'provenance_file': str(path),
            'provenance_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'live_language_detection': True, 'detection_recomputed': False,
            'source_relationship': 'saved_mask_from_verified_live_perception'}


def run(args):
    import numpy as np
    import torch
    from PIL import Image
    from .geometry import backproject, diverse_grasp_indices, load_binary_mask, pose_matrix, target_crop, target_membership, validate_intrinsics, validate_transform
    from .visualize import save_overlay
    from .network_frame import frame_metadata, grasps_to_camera
    network_frame = frame_metadata(args.wrist_camera_profile, args.network_frame)
    network_frame['profile_source'] = 'explicit_cli_declaration' if args.wrist_camera_profile else 'unspecified_identity_only'
    model_from_camera = np.asarray(network_frame['model_from_camera'])
    settings_path = args.camera_settings or args.intrinsics.parent / 'camera_settings.json'
    if args.camera_settings or settings_path.is_file():
        settings_bytes = settings_path.read_bytes()
        settings = json.loads(settings_bytes)
        if args.wrist_camera_profile is not None and settings.get('wrist_camera_profile') != args.wrist_camera_profile:
            raise ValueError('Capture camera profile disagrees with --wrist-camera-profile')
        network_frame['capture_settings'] = {'path': str(settings_path.resolve()),
                                             'sha256': hashlib.sha256(settings_bytes).hexdigest(),
                                             'wrist_camera_profile': settings.get('wrist_camera_profile')}
        network_frame['profile_source'] = 'capture_settings_verified_against_cli' if args.wrist_camera_profile else 'capture_settings_identity_only'
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; launch on GPU 0 with the isaaclab_61 Conda interpreter')
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(args.seed)
    rgb = Image.open(args.rgb).convert('RGB')
    raw_depth = np.load(args.depth, allow_pickle=False) if args.depth.suffix == '.npy' else np.asarray(Image.open(args.depth))
    if raw_depth.ndim != 2 or raw_depth.shape != (rgb.height, rgb.width):
        raise ValueError('RGB and axial depth must be registered, with the same image dimensions')
    intrinsics = validate_intrinsics(json.loads(args.intrinsics.read_text()), raw_depth.shape)
    transform = validate_transform(json.loads(args.world_from_camera.read_text())) if args.world_from_camera else None
    start = time.perf_counter()
    if args.mask_backend == 'saved':
        mask = load_binary_mask(args.mask)
        if args.mask_provenance:
            grounding = load_mask_provenance(args.mask_provenance, args.mask, mask, args.instruction)
        else:
            grounding = {'backend': 'saved_reftr_box' if args.demo else 'saved', 'instruction': args.instruction,
                         'mask_source': str(args.mask),
                         'mask_sha256': hashlib.sha256(args.mask.read_bytes()).hexdigest(),
                         'live_language_detection': False,
                         'segmentation': {'backend': 'external', 'provenance': 'saved_binary_mask'}}
    else:
        from .grounding import default_model
        from .perception import detect_and_segment
        mask, grounding = detect_and_segment(
            rgb, args.instruction, args.grounding_model or default_model(), args.device,
            segmentation_backend=args.segmentation_backend,
            segmentation_model=str(args.segmentation_model) if args.segmentation_model else None,
            depth_m=raw_depth / intrinsics['depth_scale'],
            box_threshold=args.box_threshold, text_threshold=args.text_threshold)
        grounding['live_language_detection'] = True
        grounding['detection_recomputed'] = True
    if mask.shape != raw_depth.shape:
        raise ValueError('Target mask must align with RGB/depth')
    rejected_by_quality = bool(args.mask_provenance and not grounding['mask_refinement']['accepted'])
    if rejected_by_quality:
        grounding['source_target_pixels'] = int(mask.sum())
        grounding['effective_mask'] = 'empty_due_to_rejected_target_quality'
        mask = np.zeros_like(mask, dtype=bool)
    cloud, valid = backproject(raw_depth, intrinsics, args.max_depth)
    from .geometry import network_input_mask
    crop = network_input_mask(mask, valid, args.network_region, args.crop_padding)
    points = cloud[crop]
    model_points = points if args.network_frame == 'camera' else (points @ model_from_camera.T).astype(np.float32)
    sampled_indices = np.empty(0, dtype=np.int64)
    counts = {'scene_points': int(valid.sum()), 'target_pixels': int(mask.sum()), 'network_crop_points': len(points),
              'raw': 0, 'finite_score_width': 0, 'target_filtered': 0, 'collision_free': 0, 'exported': 0}
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    default_checkpoint = ROOT / 'models/graspnet_epoch18.tar'
    if args.checkpoint.resolve() == default_checkpoint.resolve() and digest != '60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868':
        raise ValueError('Bundled checkpoint hash mismatch')
    selected = np.zeros((0, 17), dtype=np.float32)
    raw_rows = np.zeros((0, 17), dtype=np.float32)
    raw_model_rows = np.zeros((0, 17), dtype=np.float32)
    width_metadata = {key: np.empty(0, dtype=np.float32) for key in
                      ('width_unscaled_m', 'width_scale', 'width_preclamp_m')}
    width_metadata['width_saturated'] = np.empty(0, dtype=bool)
    filtered_rows = np.zeros((0, 17), dtype=np.float32)
    selected_indices = np.empty(0, dtype=np.int64)
    filtered_indices = np.empty(0, dtype=np.int64)
    rejection_reasons = np.empty(0, dtype='<U32')
    inference_seconds = 0.0
    parameters = None
    epoch = None
    failure = ('target_quality_rejected' if rejected_by_quality else
               'empty_target_mask' if not mask.any() else 'empty_depth_crop' if not len(points) else None)
    if failure is None:
        from .vendor.models.graspnet import GraspNet, pred_decode
        from .vendor.utils.collision_detector import ModelFreeCollisionDetector
        net = GraspNet(is_training=False).to(device).eval()
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)
        parameters = sum(parameter.numel() for parameter in net.parameters())
        epoch = checkpoint.get('epoch')
        sampled_indices = rng.choice(len(points), args.num_points, replace=len(points) < args.num_points)
        sampled = torch.from_numpy(model_points[sampled_indices][None].astype(np.float32)).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        with torch.inference_mode():
            output = net({'point_clouds': sampled})
            decoded, decoded_widths = pred_decode(output, width_scale=args.width_scale,
                                                  return_width_metadata=True)
            raw_model_rows = decoded[0].cpu().numpy().astype(np.float32, copy=False).reshape(-1, 17)
            raw_rows = grasps_to_camera(raw_model_rows, model_from_camera)
            width_metadata = {key: value.cpu().numpy() for key, value in decoded_widths[0].items()}
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_start
        counts['raw'] = len(raw_rows)
        rejection_reasons = np.full(len(raw_rows), 'unprocessed', dtype='<U32')
        finite = np.isfinite(raw_rows).all(-1)
        rejection_reasons[~finite] = 'non_finite'
        rejection_reasons[finite & (raw_rows[:, 0] < args.min_score)] = 'low_score'
        rejection_reasons[finite & (raw_rows[:, 0] >= args.min_score) & (raw_rows[:, 1] <= 0)] = 'invalid_width'
        rejection_reasons[finite & (raw_rows[:, 0] >= args.min_score) & (raw_rows[:, 1] > args.max_width)] = 'width_over_limit'
        good_indices = np.flatnonzero(rejection_reasons == 'unprocessed')
        counts['finite_score_width'] = len(good_indices)
        on_target = target_membership(raw_rows[good_indices, 13:16],
                                      raw_depth / intrinsics['depth_scale'], mask,
                                      intrinsics, args.depth_tolerance)
        rejection_reasons[good_indices[~on_target]] = 'target_mismatch'
        target_indices = good_indices[on_target]
        counts['target_filtered'] = len(target_indices)
        detector = ModelFreeCollisionDetector(cloud[valid], voxel_size=args.collision_voxel)
        for chunk_indices in np.array_split(target_indices, max(1, (len(target_indices) + 31) // 32)):
            if not len(chunk_indices):
                continue
            collision, empty = detector.detect(GraspRows(raw_rows[chunk_indices]), approach_dist=0.05,
                                               collision_thresh=args.collision_threshold,
                                               return_empty_grasp=True, empty_thresh=0.01)
            collision, empty = np.asarray(collision, dtype=bool), np.asarray(empty, dtype=bool)
            rejection_reasons[chunk_indices[collision]] = 'collision'
            rejection_reasons[chunk_indices[~collision & empty]] = 'empty_grasp'
            rejection_reasons[chunk_indices[~collision & ~empty]] = 'collision_free'
        filtered_indices = np.flatnonzero(rejection_reasons == 'collision_free')
        filtered_rows = raw_rows[filtered_indices]
        counts['collision_free'] = len(filtered_rows)
        kept_local, nms_reasons = diverse_grasp_indices(filtered_rows, args.top_k)
        rejection_reasons[filtered_indices] = nms_reasons
        selected_indices = filtered_indices[kept_local]
        selected = raw_rows[selected_indices]
        counts['exported'] = len(selected)
        if not len(selected):
            failure = 'no_valid_grasp_candidates'
    candidates = []
    for rank, (raw_index, row) in enumerate(zip(selected_indices, selected)):
        camera_pose = pose_matrix(row)
        item = {'candidate_id': f'g{raw_index:06d}', 'raw_index': int(raw_index), 'rank': rank,
                'overlay_index': rank + 1,
                'source': {'model': 'graspnet', 'checkpoint_sha256': digest,
                           'network_frame': args.network_frame, 'wrist_camera_profile': args.wrist_camera_profile,
                           'target_mask': grounding.get('mask_source', str(args.mask) if args.mask else None)},
                'coordinate_frame': 'camera_optical',
                'score': float(row[0]), 'score_semantics': 'model_rank_score_not_success_probability',
                'width_m': float(row[1]), 'height_m': float(row[2]), 'depth_m': float(row[3]),
                'width_decoding': {
                    key: bool(values[raw_index]) if values.dtype.kind == 'b' else
                    float(values[raw_index]) if np.isfinite(values[raw_index]) else None
                    for key, values in width_metadata.items()},
                'position_camera_m': camera_pose[:3, 3].tolist(),
                'rotation_camera_grasp': camera_pose[:3, :3].tolist(),
                'T_camera_grasp': camera_pose.tolist()}
        if transform is not None:
            item['T_world_grasp'] = (transform @ camera_pose).tolist()
        candidates.append(item)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rgb.save(args.output_dir / 'rgb.png')
    Image.fromarray(mask.astype(np.uint8) * 255).save(args.output_dir / 'target_mask.png')
    rgb_array = np.asarray(rgb)
    np.savez_compressed(args.output_dir / 'point_clouds.npz',
                        scene_xyz_camera_m=cloud[valid], scene_rgb=rgb_array[valid],
                        target_xyz_camera_m=cloud[valid & mask], target_rgb=rgb_array[valid & mask],
                        network_crop_xyz_camera_m=points, network_crop_rgb=rgb_array[crop],
                        network_crop_xyz_model_m=model_points,
                        sampled_network_indices=sampled_indices,
                        sampled_xyz_model_m=model_points[sampled_indices],
                        model_from_camera=model_from_camera)
    np.savez_compressed(args.output_dir / 'raw_candidates.npz', grasps=raw_rows,
                        grasps_model=raw_model_rows,
                        raw_indices=np.arange(len(raw_rows), dtype=np.int64),
                        rejection_reasons=rejection_reasons, **width_metadata)
    np.savez_compressed(args.output_dir / 'filtered_candidates.npz', grasps=filtered_rows,
                        raw_indices=filtered_indices,
                        **{key: values[filtered_indices] for key, values in width_metadata.items()})
    np.savez_compressed(args.output_dir / 'candidates.npz', grasps=selected,
                        raw_indices=selected_indices,
                        candidate_ids=np.asarray([item['candidate_id'] for item in candidates], dtype='<U16'),
                        T_camera_grasp=np.asarray([item['T_camera_grasp'] for item in candidates], dtype=np.float64).reshape(-1, 4, 4),
                        **{key: values[selected_indices] for key, values in width_metadata.items()})
    save_overlay(rgb, mask, candidates, intrinsics, args.output_dir / 'grasp_overlay.png')
    report = {'status': 'candidates_ready' if candidates else 'no_candidates', 'failure_stage': failure,
              'robot_executed': False, 'ik_checked': False, 'grasp_success': None,
              'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': digest,
              'checkpoint_epoch': epoch, 'model_parameters': parameters,
              'width_decoding': {'width_scale': args.width_scale, 'decoder_max_width_m': 0.075,
                                 'raw_saturated_count': int(width_metadata['width_saturated'].sum()),
                                 'semantics': 'upper-cap saturation is preclamp width >= decoder cap; no SO101 joint calibration',
                                 'metadata_row_alignment': 'NPZ width arrays share grasps/raw_indices row order'},
              'pointnet2_backend': 'pure_torch_forward', 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'torch_version': torch.__version__, 'device': str(device), 'seed': args.seed,
              'grounding': grounding, 'counts': counts, 'intrinsics': intrinsics,
              'candidate_pool_size': args.top_k,
              'network_input': {'region': args.network_region, 'points': len(points),
                                **network_frame,
                                'point_cloud_provenance': {
                                    'camera': 'point_clouds.npz:network_crop_xyz_camera_m (registered axial depth, real camera intrinsics)',
                                    'model': 'point_clouds.npz:network_crop_xyz_model_m (camera crop rotated by model_from_camera)',
                                    'sample': 'point_clouds.npz:sampled_xyz_model_m (indexed by sampled_network_indices)',
                                    'raw_predictions': 'raw_candidates.npz:grasps_model (model basis); grasps (real camera basis)'},
                                'target_points': int((crop & mask).sum()),
                                'background_points': int((crop & ~mask).sum()),
                                'collision_input': 'full_scene_valid_depth'},
              'rejection_counts': {reason: int(count) for reason, count in zip(*np.unique(rejection_reasons, return_counts=True))},
              'candidate_files': {'raw': 'raw_candidates.npz', 'filtered': 'filtered_candidates.npz',
                                  'selected': 'candidates.npz', 'point_clouds': 'point_clouds.npz'},
              'frame': 'camera optical: +X right, +Y down, +Z forward',
              'gripper_axes': 'local X approach, local Y width/closing, local Z height; NOT a SO101 TCP transform',
              'collision_geometry': {'type': 'GraspNet parallel jaw approximation, not SO101 mesh',
                                     'finger_width_m': 0.01, 'finger_length_m': 0.06},
              'inference_seconds': inference_seconds, 'pipeline_seconds': time.perf_counter() - start,
              'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              'candidates': candidates}
    (args.output_dir / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'counts': counts, 'inference_seconds': inference_seconds,
                      'result': str(args.output_dir / 'result.json')}, indent=2), flush=True)


class GraspRows:
    """Small data adapter for upstream collision detection; no graspnetAPI install."""
    def __init__(self, rows):
        self.translations = rows[:, 13:16]
        self.rotation_matrices = rows[:, 4:13].reshape(-1, 3, 3)
        self.widths, self.heights, self.depths = rows[:, 1], rows[:, 2], rows[:, 3]


def main():
    configure_runtime()
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
