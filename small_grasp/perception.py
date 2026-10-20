"""RGB-D checks for the independent tabletop executor; no simulator object poses."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
from scipy.optimize import least_squares

from .geometry import backproject

TABLE_Z = .041  # calibrated fixed table plane, not a target object's pose


def fit_orange_center(points):
    """Refine the spherical orange centre from visible depth, not a surface median."""
    points = np.asarray(points, dtype=float)
    if len(points) < 30 or not np.isfinite(points).all():
        return None
    local_origin = points.mean(0)
    local = points - local_origin
    seed = np.linalg.lstsq(np.column_stack([2 * local, np.ones(len(local))]),
                           np.sum(local ** 2, axis=1), rcond=None)[0]
    radius = np.sqrt(max(0., seed[3] + np.sum(seed[:3] ** 2)))
    initial = np.r_[seed[:3] + local_origin, np.clip(radius, .0201, .0399)]
    lower = np.r_[points.min(0) - .04, .02]
    upper = np.r_[points.max(0) + .04, .04]
    initial = np.maximum(lower + 1e-6, np.minimum(upper - 1e-6, initial))
    fit = least_squares(lambda p: np.linalg.norm(points - p[:3], axis=1) - p[3],
                        initial, bounds=(lower, upper), loss='soft_l1', f_scale=.002)
    residual = float(np.sqrt(np.mean(fit.fun ** 2)))
    if not fit.success or residual > .003 or not TABLE_Z + .015 < fit.x[2] < TABLE_Z + .045:
        return None
    return {'centre_world_m': fit.x[:3].tolist(), 'radius_m': float(fit.x[3]), 'rms_m': residual,
            'method': 'orange_sphere_fit_to_language_mask_depth'}


def world_cloud(directory):
    directory = Path(directory)
    k = json.loads((directory / 'intrinsics.json').read_text())
    cloud, valid = backproject(np.load(directory / 'depth.npy'), k)
    transform = np.asarray(json.loads((directory / 'world_from_camera.json').read_text()))
    return cloud @ transform[:3, :3].T + transform[:3, 3], valid


def detect_and_segment(rgb, instruction, grounding_model, device, *, segmentation_backend='sam',
                       segmentation_model=None, depth_m=None, box_threshold=.25,
                       text_threshold=.25):
    """Detect one language target and produce a mask with explicit provenance.

    RGB is a PIL image; depth_m, when supplied, is registered optical-axis
    depth in metres. No detection returns an empty mask. A SAM loading or
    inference failure propagates to the caller and never becomes a box ROI.
    """
    from .grounding import detect
    from .segmentation import segment

    if depth_m is not None and np.asarray(depth_m).shape != (rgb.height, rgb.width):
        raise ValueError('Optical-axis depth must align with RGB')
    detection, detection_meta = detect(rgb, instruction, grounding_model, device,
                                       box_threshold, text_threshold)
    if detection is None:
        mask = np.zeros((rgb.height, rgb.width), dtype=bool)
        segmentation = {'backend': segmentation_backend, 'provenance': 'none',
                        'model': str(segmentation_model) if segmentation_model else None,
                        'prompt_box_xyxy': None, 'predicted_iou': None,
                        'selected_mask_index': None, 'raw_pixels': 0, 'pixels': 0,
                        'valid_depth_fraction': None, 'accepted': False, 'reason': 'no_detection'}
    else:
        mask, segmentation = segment(rgb, detection['box_xyxy'], backend=segmentation_backend,
                                     model_path=segmentation_model, device=device, depth_m=depth_m)
    return mask, {'detection': detection_meta, 'segmentation': segmentation,
                  'mask_source': segmentation['provenance'], 'mask_pixels': int(mask.sum())}


def extract_point_clouds(depth_m, intrinsics, mask, max_depth_m=2.0):
    """Back-project target and surrounding scene from optical-axis metres.

    Returned points are in the camera optical frame; the surrounding cloud
    excludes target pixels and retains observed obstacles for collision tests.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    target = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or target.shape != depth.shape:
        raise ValueError('Target mask and optical-axis depth must align')
    camera = dict(intrinsics, depth_scale=1.0)
    cloud, valid = backproject(depth, camera, max_depth_m)
    return cloud[target & valid], cloud[~target & valid], valid


def generic_target_quality(mask, points_world, valid, segmentation, min_pixels=16,
                           edge_margin=2, min_valid_fraction=.7):
    """Physical gating for an observed object, without category geometry.

    The learned segmenter's acceptance means it produced a mask. This check
    separately rejects clipped, largely unobserved, or table-plane regions
    before a robot attempts a grasp. TABLE_Z is calibrated scene geometry.
    """
    mask = np.asarray(mask, dtype=bool)
    points = np.asarray(points_world)
    valid = np.asarray(valid, dtype=bool)
    if points.shape != (*mask.shape, 3) or valid.shape != mask.shape:
        raise ValueError('Target mask, world points and valid depth must align')
    selected = mask & valid & np.isfinite(points).all(-1)
    pixels = int(selected.sum())
    raw_pixels = int(segmentation.get('raw_pixels', int(mask.sum())))
    valid_fraction = float(pixels / raw_pixels) if raw_pixels else 0.0
    quality = {'accepted': False, 'reason': None, 'method': 'generic_observed_object',
               'pixels': pixels, 'valid_depth_fraction': valid_fraction,
               'edge_margin_pixels': edge_margin, 'top_z_m': None, 'median_z_m': None,
               'segmentation_accepted': bool(segmentation.get('accepted', False))}
    if not quality['segmentation_accepted']:
        quality['reason'] = segmentation.get('reason') or 'segmentation_rejected'
    elif pixels < min_pixels:
        quality['reason'] = 'insufficient_valid_target_pixels'
    elif valid_fraction < min_valid_fraction:
        quality['reason'] = 'insufficient_valid_depth_fraction'
    else:
        yy, xx = np.nonzero(selected)
        if (xx.min() < edge_margin or yy.min() < edge_margin or
                xx.max() >= mask.shape[1] - edge_margin or
                yy.max() >= mask.shape[0] - edge_margin):
            quality['reason'] = 'target_clipped_by_image_edge'
        else:
            heights = points[selected, 2]
            quality['top_z_m'] = float(np.quantile(heights, .90))
            quality['median_z_m'] = float(np.median(heights))
            if quality['top_z_m'] <= TABLE_Z + .010 or quality['median_z_m'] <= TABLE_Z + .005:
                quality['reason'] = 'target_near_table_plane'
            else:
                quality['accepted'] = True
    return quality


def refine_target(mask, points, valid, rgb=None):
    """Reject clipped/flat detections; isolate a compact raised depth component.

    Size gates are for the current orange evaluation, not arbitrary objects.
    The language detector must supply the mask; this never searches by GT/hue.
    """
    foreground = (mask & valid & (points[..., 2] > TABLE_Z + .008)
                  & (points[..., 2] < TABLE_Z + .12)
                  & (points[..., 0] > -.12) & (points[..., 0] < .83)
                  & (points[..., 1] > -.60) & (points[..., 1] < -.20))
    components, count = label(foreground)
    choices = []
    for index in range(1, count + 1):
        selected = components == index
        if selected.sum() < 30:
            continue
        yy, xx = np.nonzero(selected)
        xyz = points[selected]
        span = np.quantile(xyz, .98, axis=0) - np.quantile(xyz, .02, axis=0)
        top = float(np.quantile(xyz[:, 2], .95))
        if min(xx.min(), yy.min()) < 2 or xx.max() >= mask.shape[1] - 2 or yy.max() >= mask.shape[0] - 2:
            continue
        if not (.022 < span[0] < .09 and .022 < span[1] < .09 and top > TABLE_Z + .030):
            continue
        # Current instruction is explicitly orange. Geometry alone cannot
        # reject a similarly sized blue cube which DINO sometimes mislabels.
        # A broad warm-colour prior includes the dark brown shaded fruit.
        if rgb is not None:
            red, green, blue = np.median(np.asarray(rgb)[selected], axis=0)
            if not (red > 25 and red > 1.25 * green and green > 1.3 * blue + 2):
                continue
        choices.append((int(selected.sum()), selected, span, top))
    if not choices:
        return np.zeros_like(mask), {'accepted': False, 'reason': 'clipped_flat_or_missing_target'}
    _, selected, span, top = max(choices, key=lambda x: x[0])
    sphere = fit_orange_center(points[selected])
    if sphere is None:
        return np.zeros_like(mask), {'accepted': False, 'reason': 'orange_depth_not_spherical'}
    return selected, {'accepted': True, 'pixels': int(selected.sum()), 'sphere': sphere,
                      'span_m': span.tolist(), 'top_z_m': top,
                      'visible_surface_median_m': np.median(points[selected], axis=0).tolist()}


def locate_box(mask, points, valid):
    """Use low box geometry, excluding the table and high robot foreground.

    Require enough of the 16 cm tray in both horizontal directions; a corner
    alone cannot produce a destination. The plane height is calibrated.
    """
    foreground = (mask & valid & (points[..., 2] > TABLE_Z + .0015)
                  & (points[..., 2] < TABLE_Z + .060)
                  & (points[..., 0] > -.12) & (points[..., 0] < .83)
                  & (points[..., 1] > -.57) & (points[..., 1] < -.22))
    components, count = label(foreground)
    choices = []
    for index in range(1, count + 1):
        selected = components == index
        xyz = points[selected]
        if len(xyz) < 80:
            continue
        yy, xx = np.nonzero(selected)
        if min(xx.min(), yy.min()) < 2 or xx.max() >= mask.shape[1] - 2 or yy.max() >= mask.shape[0] - 2:
            continue
        low, high = np.quantile(xyz, [.02, .98], axis=0)
        span = high - low
        if not (.10 < span[0] < .23 and .10 < span[1] < .23):
            continue
        centre = (low + high) / 2
        centre[2] = TABLE_Z
        choices.append((len(xyz), centre, span))
    if not choices:
        return None, {'accepted': False, 'reason': 'box_incomplete_or_occluded'}
    size, centre, span = max(choices, key=lambda x: x[0])
    return centre, {'accepted': True, 'pixels': size, 'world_base_m': centre.tolist(),
                    'observed_span_m': span.tolist(), 'method': 'low_geometry_component_bounds'}


def _save_detection_overlay(rgb, box_xyxy, mask, destination):
    overlay = rgb.copy()
    if box_xyxy is not None:
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(tuple(box_xyxy), outline=(255, 210, 0), width=2)
    tint = Image.new('RGB', rgb.size, (0, 220, 100))
    overlay.paste(Image.blend(overlay, tint, .35), (0, 0), Image.fromarray(mask.astype(np.uint8) * 255))
    overlay.save(destination)


def inspect_view(directory, device, need_box=False, instruction='orange',
                 segmentation_backend='box', segmentation_model=None):
    from .grounding import default_model
    from .segmentation import box_roi
    directory = Path(directory)
    rgb = Image.open(directory / 'rgb.png').convert('RGB')
    points, valid = world_cloud(directory)
    k = json.loads((directory / 'intrinsics.json').read_text())
    depth_m = np.load(directory / 'depth.npy', allow_pickle=False) / k['depth_scale']
    mask, provenance = detect_and_segment(rgb, instruction, default_model(), device,
                                          segmentation_backend=segmentation_backend,
                                          segmentation_model=segmentation_model, depth_m=depth_m)
    detection = provenance['detection']
    if segmentation_backend == 'sam':
        # Generic learned masks are checked for usable depth, without orange
        # colour, radius, or sphere assumptions.
        refined = mask & valid
        quality = generic_target_quality(refined, points, valid, provenance['segmentation'])
    elif instruction.strip().lower().rstrip('.') in ('orange', 'orange fruit'):
        refined, quality = refine_target(mask, points, valid, np.asarray(rgb))
        quality['method'] = 'orange_depth_component_sphere'
    else:
        refined = mask & valid
        quality = generic_target_quality(refined, points, valid, provenance['segmentation'])
    roi = (box_roi((rgb.height, rgb.width), detection['box_xyxy']) if detection['detected']
           else np.zeros((rgb.height, rgb.width), dtype=bool))
    Image.fromarray(roi.astype(np.uint8) * 255).save(directory / 'dino_mask.png')
    target_mask_path = directory / 'target_mask.png'
    Image.fromarray(refined.astype(np.uint8) * 255).save(target_mask_path)
    _save_detection_overlay(rgb, detection['box_xyxy'], refined, directory / 'target_detection_overlay.png')
    result = {'target_detection': detection, 'target_segmentation': provenance['segmentation'],
              'target_quality': quality,
              'target_mask_sha256': hashlib.sha256(target_mask_path.read_bytes()).hexdigest()}
    if need_box:
        box_mask, box_provenance = detect_and_segment(rgb, 'cardboard box', default_model(), device,
                            segmentation_backend=segmentation_backend,
                            segmentation_model=segmentation_model, depth_m=depth_m)
        _, box_quality = locate_box(box_mask, points, valid)
        Image.fromarray(box_mask.astype(np.uint8) * 255).save(directory / 'box_mask.png')
        _save_detection_overlay(rgb, box_provenance['detection']['box_xyxy'], box_mask,
                                directory / 'box_detection_overlay.png')
        result.update(box_detection=box_provenance['detection'],
                      box_segmentation=box_provenance['segmentation'], box_quality=box_quality)
    (directory / 'perception.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
