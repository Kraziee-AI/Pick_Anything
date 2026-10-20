"""Independent optional GroundingDINO adapter, with caches inside this folder."""
from pathlib import Path
from functools import lru_cache
import numpy as np


def default_model():
    root = Path(__file__).resolve().parents[1]
    snapshots = root / 'outputs/grounded_pick/cache/huggingface/models--IDEA-Research--grounding-dino-tiny/snapshots'
    found = sorted(snapshots.glob('*/config.json'))
    return str(found[0].parent) if found else 'IDEA-Research/grounding-dino-tiny'


@lru_cache(maxsize=2)
def _load_detector(model_path, device):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    cache = Path(__file__).resolve().parent / 'cache/huggingface'
    cache.mkdir(parents=True, exist_ok=True)
    kwargs = dict(local_files_only=True, cache_dir=str(cache))
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_path, disable_custom_kernels=True, **kwargs).to(device).eval()
    return processor, model


def detect(rgb, instruction, model_path, device, threshold=0.25, text_threshold=0.25):
    """Select one GroundingDINO box; return its record and JSON-safe provenance.

    The highest scoring finite, non-degenerate box intersecting RGB is chosen. A missing
    detection is represented explicitly and is never turned into a mask.
    """
    import torch
    caption = instruction.strip().lower().rstrip('.')
    if not caption:
        raise ValueError('Grounding instruction is empty')
    caption = ('orange fruit' if caption == 'orange' else caption) + '.'
    processor, model = _load_detector(str(model_path), str(device))
    inputs = processor(images=rgb, text=caption, return_tensors='pt').to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, input_ids=inputs.input_ids, threshold=threshold, text_threshold=text_threshold,
            target_sizes=[(rgb.height, rgb.width)])[0]
    metadata = {'backend': 'dino', 'instruction': instruction, 'caption': caption,
                'model': str(model_path), 'box_threshold': threshold, 'text_threshold': text_threshold,
                'detected': False, 'box_xyxy': None, 'score': None, 'label': None,
                'selected_index': None, 'selection_reason': 'no_valid_detection'}
    boxes, scores = result['boxes'].cpu().numpy(), result['scores'].cpu().numpy()
    valid = np.isfinite(boxes).all(-1) & np.isfinite(scores)
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    valid &= ((np.minimum(boxes[:, 2], rgb.width) > np.maximum(boxes[:, 0], 0))
              & (np.minimum(boxes[:, 3], rgb.height) > np.maximum(boxes[:, 1], 0)))
    if valid.any():
        ids = np.flatnonzero(valid)
        idx = ids[np.argmax(scores[ids])]
        box = boxes[idx].astype(float).tolist()
        labels = result.get('text_labels', result.get('labels', []))
        label = labels[idx] if len(labels) > idx else None
        metadata.update(detected=True, box_xyxy=box, score=float(scores[idx]),
                        label=str(label) if label is not None else None,
                        selected_index=int(idx), selection_reason='highest_valid_score')
    return (dict(box_xyxy=metadata['box_xyxy'], score=metadata['score'],
                 label=metadata['label'], index=metadata['selected_index'])
            if metadata['detected'] else None), metadata


def predict_mask(rgb, instruction, model_path, device, threshold=0.25, text_threshold=0.25):
    """Legacy rectangular ROI adapter; its metadata identifies the ROI source."""
    detection, metadata = detect(rgb, instruction, model_path, device, threshold, text_threshold)
    mask = np.zeros((rgb.height, rgb.width), dtype=bool)
    if detection is not None:
        x0, y0 = np.floor(detection['box_xyxy'][:2]).astype(int)
        x1, y1 = np.ceil(detection['box_xyxy'][2:]).astype(int)
        mask[max(0, y0):min(rgb.height, y1), max(0, x0):min(rgb.width, x1)] = True
    metadata.update(mask_source='dino_box_roi', mask_pixels=int(mask.sum()))
    return mask, metadata
