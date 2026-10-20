"""Copy the audited local GraspNet sources and example; no environment installs."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/user123/ViewInvariantSE3/OVGNet_full_repro/graspnet/graspnet')
VL = Path('/home/user123/ViewInvariantSE3/vlgrasp_demo')


def main():
    manifest = {'source': str(SOURCE), 'upstream': 'https://github.com/cv516Buaa/OVGNet',
                'upstream_commit': 'caaf6f695895aaca13aa010ec93d32d27e34c049', 'files': {},
                'adaptations': ['Relative imports only; remove training-only imports.',
                                'PointNet2 forward kernels use local pure PyTorch operations.',
                                'Architecture, learned tensors and decoder retained.']}
    rl_paths = list((ROOT.parent / 'pick_anything').rglob('*.py'))
    rl_paths += list((ROOT.parent / 'scripts').glob('*.py'))
    baseline = {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in rl_paths}
    audit = ROOT / 'rl_source_before.json'
    if not audit.exists():
        audit.write_text(json.dumps(baseline, indent=2) + '\n')
    files = ['models/backbone.py', 'models/modules.py', 'models/graspnet.py',
             'pointnet2/pointnet2_modules.py', 'pointnet2/pointnet2_utils.py', 'pointnet2/pytorch_utils.py',
             'utils/loss_utils.py', 'utils/collision_detector.py', 'LICENSE']
    for relative in files:
        src = SOURCE / relative
        dst = ROOT / 'vendor' / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = src.read_bytes()
        manifest['files'][str(dst.relative_to(ROOT))] = {'source_sha256': hashlib.sha256(raw).hexdigest()}
        text = raw.decode().replace('\r\n', '\n')
        if relative.endswith('.py'):
            text = text.replace('\nimport pointnet2_utils\n', '\nfrom . import pointnet2_utils\n')
            text = text.replace('\nimport pytorch_utils as pt_utils\n', '\nfrom . import pytorch_utils as pt_utils\n')
            # Avoid putting legacy generic modules into another package's search path.
            lines = text.splitlines()
            lines = [line for line in lines if not line.startswith(('BASE_DIR =', 'ROOT_DIR =', 'sys.path.append('))]
            text = '\n'.join(lines) + '\n'
        if relative == 'models/graspnet.py':
            text = text.replace('from .loss import get_loss\n', '')
            text = text.replace('from ..utils.label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix',
                                'from ..utils.loss_utils import batch_viewpoint_params_to_matrix')
            text = text.replace('self.is_training = is_training',
                                'self.is_training = is_training\n        if is_training:\n            raise ValueError("This isolated baseline supports inference only")')
        if relative == 'pointnet2/pointnet2_utils.py':
            start = text.index('try:\n    import builtins')
            end = text.index('if False:')
            text = text[:start] + 'from . import torch_ops as _ext\n\n' + text[end:]
        dst.write_text(text)
        manifest['files'][str(dst.relative_to(ROOT))]['local_sha256'] = hashlib.sha256(dst.read_bytes()).hexdigest()
    for directory in ('models', 'pointnet2', 'utils'):
        (ROOT / 'vendor' / directory / '__init__.py').touch()
    copies = {
        'models/graspnet_epoch18.tar': SOURCE / 'logs/log_rs/checkpoint.tar',
        'examples/pear/rgb.png': VL / 'outputs/official_demo/input_rgb.png',
        'examples/pear/depth.png': VL / 'VL-Grasp/example/depth/0003000.png',
    }
    for relative, src in copies.items():
        dst = ROOT / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        manifest['files'][relative] = {'source': str(src), 'sha256': hashlib.sha256(dst.read_bytes()).hexdigest()}
    # The old target_mask.png is a coloured overlay, NOT a machine-readable mask.
    # Use its saved RefTR-predicted bounding box to create an explicit binary ROI.
    from PIL import Image
    import numpy as np
    prediction = VL / 'outputs/official_demo/prediction.json'
    box = json.loads(prediction.read_text())['grounding']['bbox_xyxy_pixels']
    binary = np.zeros((480, 640), dtype=np.uint8)
    x0, y0 = np.floor(box[:2]).astype(int)
    x1, y1 = np.ceil(box[2:]).astype(int)
    binary[y0:y1, x0:x1] = 255
    mask_path = ROOT / 'examples/pear/target_mask.png'
    Image.fromarray(binary).save(mask_path)
    manifest['files']['examples/pear/target_mask.png'] = {
        'source': str(prediction), 'source_sha256': hashlib.sha256(prediction.read_bytes()).hexdigest(),
        'generation': 'Rasterized saved RefTR bbox, not segmentation or live language detection',
        'box_xyxy': box, 'sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest()}
    intrinsics = {'width': 640, 'height': 480, 'fx': 592.566, 'fy': 592.566, 'cx': 319.132, 'cy': 246.937,
                  'depth_scale': 1000.0, 'frame': 'camera optical: +X right, +Y down, +Z forward'}
    (ROOT / 'examples/pear/intrinsics.json').write_text(json.dumps(intrinsics, indent=2) + '\n')
    (ROOT / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared local model, licensed architecture and saved RGB-D/mask example in', ROOT)


if __name__ == '__main__':
    main()
