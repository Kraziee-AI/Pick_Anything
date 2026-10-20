"""Box-prompted masks with explicit source and RGB-D quality metadata."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import label


def default_model():
    """Workspace-local SAM ViT-base checkpoint; model loading is offline only."""
    return str(Path(__file__).resolve().parent / 'models/sam-vit-base')


def box_roi(shape, box_xyxy):
    box = np.asarray(box_xyxy, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError('Segmentation prompt must be a finite, non-degenerate xyxy box')
    height, width = shape
    x0, y0 = np.floor(box[:2]).astype(int)
    x1, y1 = np.ceil(box[2:]).astype(int)
    x0, x1 = np.clip([x0, x1], 0, width)
    y0, y1 = np.clip([y0, y1], 0, height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Segmentation prompt does not intersect the image')
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


@lru_cache(maxsize=2)
def _load_sam(model_path, device):
    from transformers import SamModel, SamProcessor

    path = Path(model_path)
    if not path.is_dir() or not (path / 'config.json').is_file():
        raise FileNotFoundError(f'SAM checkpoint directory is unavailable: {path}')
    kwargs = {'local_files_only': True}
    processor = SamProcessor.from_pretrained(str(path), **kwargs)
    model = SamModel.from_pretrained(str(path), **kwargs).to(device).eval()
    return processor, model


def _sam_mask(rgb, box, model_path, device):
    import torch

    processor, model = _load_sam(str(model_path), str(device))
    # SamProcessor expects image -> prompts -> boxes in original pixel xyxy.
    inputs = processor(images=rgb, input_boxes=[[box]], return_tensors='pt')
    model_inputs = {key: value.to(device) for key, value in inputs.items()
                    if key in ('pixel_values', 'input_boxes')}
    with torch.inference_mode():
        outputs = model(**model_inputs, multimask_output=True)
    scores = outputs.iou_scores[0, 0].detach().cpu().numpy()
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise RuntimeError('SAM returned no finite mask quality scores')
    selected = int(np.argmax(scores))
    masks = processor.post_process_masks(outputs.pred_masks.detach().cpu(),
                                         inputs['original_sizes'], inputs['reshaped_input_sizes'])
    mask = masks[0][0, selected].detach().cpu().numpy().astype(bool)
    if mask.shape != (rgb.height, rgb.width):
        raise RuntimeError(f'SAM mask shape {mask.shape} does not match RGB size {rgb.size}')
    return mask, float(scores[selected]), selected


def _depth_component(roi, depth_m):
    """Comparison backend: largest depth-contiguous component around box centre.

    This is an RGB-D heuristic, not a learned instance segmentation model.
    """
    depth = np.asarray(depth_m, dtype=float)
    valid = roi & np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros_like(roi)
    yy, xx = np.nonzero(valid)
    centre_y = (yy.min() + yy.max()) / 2
    centre_x = (xx.min() + xx.max()) / 2
    near_centre = (xx - centre_x) ** 2 + (yy - centre_y) ** 2
    seed_depth = float(depth[yy[np.argmin(near_centre)], xx[np.argmin(near_centre)]])
    # A 3 cm axial band removes most background while allowing visible depth
    # variation. It is deliberately recorded as a heuristic in provenance.
    foreground = valid & (np.abs(depth - seed_depth) <= .03)
    components, count = label(foreground)
    if not count:
        return np.zeros_like(roi)
    sizes = np.bincount(components[foreground], minlength=count + 1)
    sizes[0] = 0
    return components == int(np.argmax(sizes))


def segment(rgb, box_xyxy, backend='sam', model_path=None, device='cpu', depth_m=None,
            min_pixels=16):
    """Return an aligned bool mask and JSON-safe source/quality metadata.

    depth_m is optical-axis depth in metres. Invalid depth pixels are removed
    from the output when depth is supplied. SAM failure never falls back to a
    rectangle; callers must explicitly request another backend.
    """
    if backend not in ('sam', 'depth', 'box'):
        raise ValueError(f'Unknown segmentation backend: {backend}')
    shape = (rgb.height, rgb.width)
    roi = box_roi(shape, box_xyxy)
    box = np.asarray(box_xyxy, dtype=float).tolist()
    if depth_m is not None:
        depth = np.asarray(depth_m, dtype=float)
        if depth.shape != shape:
            raise ValueError('Optical-axis depth must align with RGB')
        valid_depth = np.isfinite(depth) & (depth > 0)
    else:
        valid_depth = None
    if backend == 'sam':
        model_path = str(model_path or default_model())
        raw, predicted_iou, selected_index = _sam_mask(rgb, box, model_path, device)
        source = 'sam_box_prompt'
    elif backend == 'depth':
        if depth_m is None:
            raise ValueError('Depth segmentation requires optical-axis depth in metres')
        raw = _depth_component(roi, depth_m)
        predicted_iou = selected_index = None
        source = 'depth_component_heuristic'
    else:
        raw = roi
        predicted_iou = selected_index = None
        source = 'detector_box_roi'
    raw_pixels = int(raw.sum())
    box_overlap_pixels = int((raw & roi).sum())
    mask = raw & valid_depth if valid_depth is not None else raw
    pixels = int(mask.sum())
    valid_fraction = float(pixels / raw_pixels) if raw_pixels else 0.0
    reason = ('mask_misses_detection_box' if raw_pixels and box_overlap_pixels / raw_pixels < .1
              else None if pixels >= min_pixels else 'insufficient_valid_target_pixels')
    metadata = {'backend': backend, 'provenance': source,
                'model': str(model_path) if backend == 'sam' else None,
                'prompt_box_xyxy': box, 'predicted_iou': predicted_iou,
                'selected_mask_index': selected_index, 'raw_pixels': raw_pixels,
                'pixels': pixels, 'box_overlap_pixels': box_overlap_pixels,
                'valid_depth_fraction': valid_fraction if depth_m is not None else None,
                'accepted': reason is None, 'reason': reason}
    if reason:
        mask = np.zeros(shape, dtype=bool)
    return mask.astype(bool, copy=False), metadata
