import hashlib
import json
from pathlib import Path
import unittest
import tempfile

import numpy as np
import torch

from small_grasp.geometry import backproject, diverse_grasps, load_binary_mask, target_crop, target_membership, validate_intrinsics, validate_transform
from small_grasp.vendor.pointnet2 import torch_ops


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.k = dict(width=3, height=2, fx=2, fy=2, cx=1, cy=0, depth_scale=1000)

    def test_metric_backprojection_invalid_depth_and_target(self):
        depth = np.array([[1000, 1000, 0], [2000, np.nan, 1000]])
        cloud, valid = backproject(depth, self.k, max_depth=1.5)
        np.testing.assert_allclose(cloud[0, 0], [-0.5, 0, 1])
        self.assertEqual(int(valid.sum()), 3)
        mask = np.zeros((2, 3), dtype=bool)
        mask[0, 0] = True
        got = target_membership(np.array([[-.5, 0, 1], [0, 0, 1], [-.6, 0, 1.2], [0, 0, -1]]),
                                depth / 1000, mask, self.k, tolerance=.04)
        np.testing.assert_array_equal(got, [True, False, False, False])

    def test_resolution_and_scale_contract(self):
        validate_intrinsics(self.k, (2, 3))
        with self.assertRaises(ValueError):
            validate_intrinsics(self.k, (3, 2))
        for key in ('fx', 'depth_scale'):
            with self.assertRaises(ValueError):
                validate_intrinsics({**self.k, key: 0}, (2, 3))

    def test_empty_mask_and_edge_crop(self):
        mask = np.zeros((5, 5), dtype=bool)
        self.assertFalse(target_crop(mask).any())
        mask[0, 0] = True
        self.assertEqual(int(target_crop(mask, 1).sum()), 4)

    def test_mask_loader_rejects_visualization_overlay(self):
        from PIL import Image
        root = Path(__file__).resolve().parents[1] / 'cache'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / 'mask.png'
            Image.fromarray(np.full((3, 4, 3), 128, dtype=np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError, 'not an RGB overlay'):
                load_binary_mask(path)
            Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(path)
            np.testing.assert_array_equal(load_binary_mask(path), [[False, True], [True, False]])

    def test_se3_transform_rejects_reflections(self):
        world_from_camera = np.eye(4)
        world_from_camera[:3, 3] = [1, 2, 3]
        np.testing.assert_allclose(validate_transform(world_from_camera) @ [0, 0, .5, 1], [1, 2, 3.5, 1])
        bad = np.eye(4)
        bad[0, 0] = -1
        with self.assertRaises(ValueError):
            validate_transform(bad)

    def test_gripper_symmetry_nms(self):
        row = np.zeros(17, dtype=np.float32)
        row[0], row[1], row[2], row[3] = .8, .05, .02, .03
        row[4:13] = np.eye(3).reshape(-1)
        row[13:16] = [0, 0, .5]
        flipped = row.copy()
        flipped[0] = .7
        flipped[4:13] = np.diag([1, -1, -1]).reshape(-1)
        distinct = row.copy()
        distinct[0], distinct[13] = .6, .1
        result = diverse_grasps(np.stack([flipped, distinct, row]), 20)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(float(result[0, 0]), .8, places=5)


class OperatorTests(unittest.TestCase):
    def test_queries_preserve_source_order_pad_and_strict_boundaries(self):
        xyz = torch.tensor([[[.03, .02, 1], [.01, 0, 1], [0, 0, 1], [.1, 0, 1]]])
        centre = torch.tensor([[[0, 0, 1.]]])
        ids = torch_ops.ball_query(centre, xyz, .1, 6)
        self.assertEqual(ids.tolist(), [[[0, 1, 2, 0, 0, 0]]])
        rotation = torch.eye(3).reshape(1, 1, 3, 3)
        ids = torch_ops.cylinder_query(centre, xyz, rotation, .05, 0, .1, 4)
        self.assertEqual(ids.tolist(), [[[0, 1, 0, 0]]])
        # A 90-degree orientation makes the radial/approach distinction observable.
        rotation = torch.tensor([[[[0., -1, 0], [1, 0, 0], [0, 0, 1]]]])
        ids = torch_ops.cylinder_query(centre, xyz, rotation, .05, .01, .04, 3)
        self.assertEqual(ids.tolist(), [[[0, 0, 0]]])

    def test_fps_skips_origin_and_3nn_interpolation(self):
        xyz = torch.tensor([[[0., 0, 1], [1, 0, 1], [.2, 0, 1], [0, 0, 0]]])
        self.assertEqual(torch_ops.furthest_point_sampling(xyz, 3).tolist(), [[0, 1, 2]])
        known = torch.tensor([[[0., 0, 1], [1., 0, 1], [3., 0, 1]]])
        dist, indices = torch_ops.three_nn(torch.tensor([[[.1, 0, 1.]]]), known)
        torch.testing.assert_close(dist, torch.tensor([[[.01, .81, 8.41]]]))
        feature = torch.tensor([[[2., 4., 10.]]])
        got = torch_ops.three_interpolate(feature, indices, torch.tensor([[[.5, .3, .2]]]))
        torch.testing.assert_close(got, torch.tensor([[[4.2]]]))

    def test_query_against_scalar_reference(self):
        rng = np.random.default_rng(19)
        xyz = rng.normal(size=(1, 53, 3)).astype('float32')
        centres = xyz[:, :7].copy()
        expected = []
        for centre in centres[0]:
            ids = [i for i, point in enumerate(xyz[0]) if np.sum((point - centre) ** 2) < 1.1 ** 2][:8]
            expected.append((ids + [ids[0]] * 8)[:8])
        got = torch_ops.ball_query(torch.from_numpy(centres), torch.from_numpy(xyz), 1.1, 8)
        self.assertEqual(got.tolist(), [expected])


class IsolationTests(unittest.TestCase):
    def test_existing_rl_sources_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads((root / 'rl_source_before.json').read_text())
        for relative, digest in baseline.items():
            self.assertEqual(hashlib.sha256((root.parent / relative).read_bytes()).hexdigest(), digest, relative)

    def test_pretrained_architecture_loads_strictly(self):
        from small_grasp.vendor.models.graspnet import GraspNet
        root = Path(__file__).resolve().parents[1]
        saved = torch.load(root / 'models/graspnet_epoch18.tar', map_location='cpu', weights_only=True)
        model = GraspNet(is_training=False).eval()
        model.load_state_dict(saved['model_state_dict'], strict=True)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 1_000_000)
        with self.assertRaisesRegex(ValueError, 'inference only'):
            GraspNet(is_training=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
