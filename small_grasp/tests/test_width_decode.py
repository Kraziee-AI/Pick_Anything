"""Small CPU tensors exercise width units and exported raw-row provenance."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import torch

from small_grasp.infer import ROOT, parse_args, run
from small_grasp.vendor.models.graspnet import pred_decode


def predictions():
    # Different selected angle/depth per foreground point, with a background
    # point between them, catch incorrect width metadata gathers and masking.
    shape = (1, 12, 3, 2)
    scores = torch.zeros(shape)
    angles = torch.zeros(shape)
    widths = torch.full(shape, .01)
    scores[0, 2, 0, 1], angles[0, 2, 0, 1], widths[0, 2, 0, 1] = .8, 1., .04
    scores[0, 7, 2, 0], angles[0, 7, 2, 0], widths[0, 7, 2, 0] = .7, 1., .08
    return {'point_clouds': torch.zeros(1, 3, 3),
            'objectness_score': torch.tensor([[[0., 1., 0.], [1., 0., 1.]]]),
            'grasp_score_pred': scores,
            'fp2_xyz': torch.tensor([[[0., 0., 1.], [.05, 0., 1.], [.1, 0., 1.]]]),
            'grasp_top_view_xyz': torch.tensor([[[-1., 0., 0.]] * 3]),
            'grasp_angle_cls_pred': angles, 'grasp_width_pred': widths,
            'grasp_tolerance_pred': torch.full(shape, .05)}


class WidthDecodeTests(unittest.TestCase):
    def test_historical_default_and_native_width_only_difference(self):
        default = pred_decode(predictions())[0]
        explicit = pred_decode(predictions(), width_scale=10.)[0]
        torch.testing.assert_close(default, explicit, rtol=0, atol=0)
        native, meta = pred_decode(predictions(), width_scale=1., return_width_metadata=True)
        torch.testing.assert_close(default[:, 1], torch.tensor([.075, .075]))
        torch.testing.assert_close(native[0][:, 1], torch.tensor([.04, .075]))
        keep = [i for i in range(17) if i != 1]
        torch.testing.assert_close(default[:, keep], native[0][:, keep], rtol=0, atol=0)
        torch.testing.assert_close(meta[0]['width_unscaled_m'], torch.tensor([.04, .08]))
        torch.testing.assert_close(meta[0]['width_preclamp_m'], torch.tensor([.04, .08]))
        self.assertEqual(meta[0]['width_saturated'].tolist(), [False, True])
        torch.testing.assert_close(default[:, 13:16], torch.tensor([[0., 0., 1.], [.1, 0., 1.]]))

    def test_empty_foreground_and_invalid_scale(self):
        output = predictions()
        output['objectness_score'][:] = torch.tensor([[[1.], [0.]]])
        rows, meta = pred_decode(output, return_width_metadata=True)
        self.assertEqual(tuple(rows[0].shape), (0, 17))
        self.assertTrue(all(tuple(v.shape) == (0,) for v in meta[0].values()))
        for scale in [0., -1., float('inf'), float('nan')]:
            with self.assertRaisesRegex(ValueError, 'finite and positive'):
                pred_decode(predictions(), width_scale=scale)

    def test_infer_exports_width_arrays_and_candidate_metadata_in_row_order(self):
        cache = ROOT / 'cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            directory = Path(directory)
            Image.new('RGB', (10, 10), (80, 90, 100)).save(directory/'rgb.png')
            Image.fromarray(np.ones((10, 10), dtype=np.uint8)*255).save(directory/'mask.png')
            np.save(directory/'depth.npy', np.ones((10, 10), dtype=np.float32))
            (directory/'intrinsics.json').write_text(json.dumps(dict(width=10,height=10,fx=10,fy=10,cx=5,cy=5,depth_scale=1)))
            (directory/'checkpoint.tar').write_bytes(b'mock weights, no inference')
            argv=['infer', '--rgb',str(directory/'rgb.png'),'--depth',str(directory/'depth.npy'),
                  '--intrinsics',str(directory/'intrinsics.json'),'--mask',str(directory/'mask.png'),
                  '--checkpoint',str(directory/'checkpoint.tar'),'--device','cpu','--num-points','2048',
                  '--width-scale','1','--output-dir',str(directory/'out')]
            with patch('sys.argv', argv):
                args = parse_args()
            fake = MagicMock()
            fake.to.return_value.eval.return_value = fake
            fake.parameters.return_value = [torch.zeros(1)]
            fake.return_value = predictions()
            collision = MagicMock()
            collision.detect.side_effect = lambda rows, **kwargs: (np.zeros(len(rows.widths),dtype=bool), np.zeros(len(rows.widths),dtype=bool))
            with patch('small_grasp.vendor.models.graspnet.GraspNet',return_value=fake), \
                 patch('small_grasp.vendor.utils.collision_detector.ModelFreeCollisionDetector',return_value=collision), \
                 patch('torch.load',return_value={'model_state_dict':{},'epoch':18}), \
                 contextlib.redirect_stdout(io.StringIO()):
                run(args)
            out=directory/'out'
            raw=np.load(out/'raw_candidates.npz')
            for filename in ['raw_candidates.npz','filtered_candidates.npz','candidates.npz']:
                data=np.load(out/filename)
                self.assertEqual(data['grasps'].shape[1],17)
                ids=data['raw_indices']
                for key in ['width_unscaled_m','width_scale','width_preclamp_m','width_saturated']:
                    np.testing.assert_array_equal(data[key],raw[key][ids])
            report=json.loads((out/'result.json').read_text())
            self.assertEqual(report['width_decoding']['width_scale'],1.)
            self.assertEqual(report['width_decoding']['raw_saturated_count'],1)
            self.assertEqual(report['counts']['exported'],2)
            for c in report['candidates']:
                i=c['raw_index']
                self.assertEqual(c['candidate_id'],f'g{i:06d}')
                self.assertAlmostEqual(c['width_decoding']['width_unscaled_m'],float(raw['width_unscaled_m'][i]))
                self.assertEqual(c['width_decoding']['width_saturated'],bool(raw['width_saturated'][i]))
