"""Contract checks for detector boxes, mask provenance and failure handling."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from small_grasp import grounding, perception, segmentation


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.rgb = Image.new('RGB', (40, 30), (80, 100, 120))
        self.depth = np.full((30, 40), .5, dtype=float)

    def detector_result(self, boxes, scores):
        import torch
        from types import SimpleNamespace

        result = {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  'scores': torch.tensor(scores),
                  'text_labels': [f'pear {index}' for index in range(len(scores))]}

        class Inputs(dict):
            input_ids = torch.tensor([[1]])

            def to(self, device):
                return self

        processor = SimpleNamespace(
            post_process_grounded_object_detection=lambda *args, **kwargs: [result])

        class Processor:
            def __call__(self, **kwargs):
                return Inputs()

            post_process_grounded_object_detection = processor.post_process_grounded_object_detection

        with patch.object(grounding, '_load_detector', return_value=(Processor(), lambda **kwargs: None)):
            return grounding.detect(self.rgb, 'pear', 'local-dino', 'cpu')

    def test_detector_selects_highest_valid_intersecting_box(self):
        detection, metadata = self.detector_result(
            [[-12, 4, -2, 20], [4, 5, 20, 21], [8, 8, 25, 25],
             [1, 1, 1, 10], [float('nan'), 1, 10, 10]],
            [.99, .6, .8, .98, .97])
        self.assertEqual(detection['index'], 2)
        self.assertEqual(detection['box_xyxy'], [8., 8., 25., 25.])
        self.assertEqual(metadata['label'], 'pear 2')
        self.assertEqual(metadata['selection_reason'], 'highest_valid_score')

    def test_detector_empty_and_all_invalid_results_are_explicit(self):
        for boxes, scores in (([], []),
                              ([[-12, 4, -2, 20], [40, 2, 50, 20], [2, 4, 10, 20]],
                               [.99, .98, float('nan')])):
            with self.subTest(boxes=boxes):
                detection, metadata = self.detector_result(boxes, scores)
                self.assertIsNone(detection)
                self.assertFalse(metadata['detected'])
                self.assertEqual(metadata['selection_reason'], 'no_valid_detection')

    def test_box_roi_and_depth_provenance(self):
        self.depth[8:10, 8:10] = np.nan
        mask, info = segmentation.segment(self.rgb, [5, 5, 25, 25], 'box', depth_m=self.depth)
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(mask.shape, self.depth.shape)
        self.assertFalse(mask[8, 8])
        self.assertEqual(info['provenance'], 'detector_box_roi')
        self.assertLess(info['valid_depth_fraction'], 1)
        self.depth[5:25, 5:25] = 1.5
        self.depth[10:20, 10:20] = .5
        mask, info = segmentation.segment(self.rgb, [5, 5, 25, 25], 'depth', depth_m=self.depth)
        self.assertEqual(info['provenance'], 'depth_component_heuristic')
        self.assertGreater(mask.sum(), 0)
        self.assertLess(mask.sum(), 400)

    def test_sam_mask_and_quality_are_kept_separate_from_roi(self):
        learned = np.zeros((30, 40), dtype=bool)
        learned[8:20, 9:23] = True
        with patch.object(segmentation, '_sam_mask', return_value=(learned, .81, 2)):
            mask, info = segmentation.segment(self.rgb, [5, 5, 25, 25], 'sam',
                                               model_path='local-sam', depth_m=self.depth)
        np.testing.assert_array_equal(mask, learned)
        self.assertEqual(info['provenance'], 'sam_box_prompt')
        self.assertEqual(info['selected_mask_index'], 2)
        self.assertAlmostEqual(info['predicted_iou'], .81)

    def test_sam_adapter_selects_best_model_mask_in_original_resolution(self):
        import torch
        from types import SimpleNamespace

        masks = torch.zeros((1, 3, 30, 40), dtype=torch.bool)
        masks[0, 1, 8:20, 9:23] = True

        class FakeProcessor:
            def __call__(self, **kwargs):
                self.boxes = kwargs['input_boxes']
                return {'pixel_values': torch.zeros((1, 3, 2, 2)),
                        'input_boxes': torch.tensor(self.boxes),
                        'original_sizes': torch.tensor([[30, 40]]),
                        'reshaped_input_sizes': torch.tensor([[2, 2]])}

            def post_process_masks(self, raw, original, reshaped):
                self.sizes = (original.tolist(), reshaped.tolist())
                return [masks]

        class FakeModel:
            def __call__(self, **kwargs):
                self.multimask = kwargs['multimask_output']
                return SimpleNamespace(iou_scores=torch.tensor([[[.2, .9, .7]]]),
                                       pred_masks=torch.zeros((1, 1, 3, 1, 1)))

        processor, model = FakeProcessor(), FakeModel()
        with patch.object(segmentation, '_load_sam', return_value=(processor, model)):
            result, score, index = segmentation._sam_mask(self.rgb, [5., 5., 25., 25.],
                                                           'local-sam', 'cpu')
        self.assertEqual(processor.boxes, [[[5., 5., 25., 25.]]])
        self.assertEqual(processor.sizes, ([[30, 40]], [[2, 2]]))
        self.assertTrue(model.multimask)
        self.assertEqual(index, 1)
        self.assertAlmostEqual(score, .9)
        self.assertEqual(result.shape, (30, 40))
        self.assertEqual(result.sum(), 168)

    def test_sam_failure_does_not_fall_back_to_box(self):
        with patch.object(segmentation, '_sam_mask', side_effect=FileNotFoundError('weights absent')):
            with self.assertRaisesRegex(FileNotFoundError, 'weights absent'):
                segmentation.segment(self.rgb, [5, 5, 25, 25], 'sam', depth_m=self.depth)

    def test_no_detection_is_explicit_and_never_runs_segmenter(self):
        no_detection = {'detected': False, 'box_xyxy': None, 'score': None}
        with patch.object(grounding, 'detect', return_value=(None, no_detection)):
            with patch.object(segmentation, 'segment') as model:
                mask, info = perception.detect_and_segment(self.rgb, 'pear', 'dino', 'cpu',
                                                            depth_m=self.depth)
        self.assertFalse(mask.any())
        self.assertEqual(info['segmentation']['reason'], 'no_detection')
        model.assert_not_called()

    def test_missing_depth_or_bad_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'requires optical-axis depth'):
            segmentation.segment(self.rgb, [5, 5, 25, 25], 'depth')
        with self.assertRaisesRegex(ValueError, 'align'):
            segmentation.segment(self.rgb, [5, 5, 25, 25], 'box', depth_m=np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, 'non-degenerate'):
            segmentation.segment(self.rgb, [5, 5, 5, 25], 'box')
        for outside in ([-10, 5, -2, 25], [42, 5, 50, 25],
                        [5, -10, 25, -2], [5, 32, 25, 40]):
            with self.subTest(outside=outside):
                with self.assertRaisesRegex(ValueError, 'does not intersect'):
                    segmentation.segment(self.rgb, outside, 'box')

    def test_generic_physical_quality_rejects_clipping_flatness_and_missing_depth(self):
        mask = np.zeros((30, 40), dtype=bool)
        mask[8:20, 9:23] = True
        points = np.zeros((30, 40, 3), dtype=float)
        points[..., 2] = .08
        valid = np.ones((30, 40), dtype=bool)
        info = {'accepted': True, 'raw_pixels': int(mask.sum())}
        self.assertTrue(perception.generic_target_quality(mask, points, valid, info)['accepted'])
        clipped = np.zeros_like(mask)
        clipped[0:12, 9:23] = True
        self.assertEqual(perception.generic_target_quality(clipped, points, valid, info)['reason'],
                         'target_clipped_by_image_edge')
        points[..., 2] = perception.TABLE_Z + .002
        self.assertEqual(perception.generic_target_quality(mask, points, valid, info)['reason'],
                         'target_near_table_plane')
        points[..., 2] = .08
        valid[8:14, 9:23] = False
        self.assertEqual(perception.generic_target_quality(mask, points, valid, info)['reason'],
                         'insufficient_valid_depth_fraction')

    def test_inspect_view_hashes_exact_saved_target_mask(self):
        mask = np.zeros((30, 40), dtype=bool)
        mask[8:20, 9:23] = True
        metadata = {'detection': {'detected': True, 'box_xyxy': [9., 8., 23., 20.]},
                    'segmentation': {'accepted': True, 'reason': None,
                                     'raw_pixels': int(mask.sum()), 'provenance': 'sam_box_prompt'}}
        with tempfile.TemporaryDirectory(dir='small_grasp/outputs') as tmp:
            directory = Path(tmp)
            self.rgb.save(directory / 'rgb.png')
            np.save(directory / 'depth.npy', np.full(mask.shape, .08, dtype=np.float32))
            (directory / 'intrinsics.json').write_text(json.dumps(
                {'fx': 20., 'fy': 20., 'cx': 20., 'cy': 15., 'depth_scale': 1.}))
            (directory / 'world_from_camera.json').write_text(json.dumps(np.eye(4).tolist()))
            with patch.object(perception, 'detect_and_segment', return_value=(mask, metadata)):
                result = perception.inspect_view(directory, 'cpu', instruction='pear',
                                                 segmentation_backend='sam')
            digest = hashlib.sha256((directory / 'target_mask.png').read_bytes()).hexdigest()
            self.assertEqual(result['target_mask_sha256'], digest)
            self.assertEqual(json.loads((directory / 'perception.json').read_text())['target_mask_sha256'], digest)
            self.assertTrue(result['target_quality']['accepted'])
            self.assertNotIn('sphere', result['target_quality'])

    def test_target_and_surrounding_camera_clouds_exclude_invalid_depth(self):
        depth = np.array([[1., np.nan], [0., 2.]], dtype=float)
        mask = np.array([[True, True], [False, False]])
        k = {'fx': 2., 'fy': 2., 'cx': 0., 'cy': 0.}
        target, surrounding, valid = perception.extract_point_clouds(depth, k, mask, 3.)
        self.assertEqual(len(target), 1)
        self.assertEqual(len(surrounding), 1)
        np.testing.assert_array_equal(valid, [[True, False], [False, True]])
        np.testing.assert_allclose(target[0], [0., 0., 1.])
        np.testing.assert_allclose(surrounding[0], [1., 1., 2.])

    def test_legacy_predict_mask_remains_rectangular_and_identified(self):
        detection = {'box_xyxy': [5.1, 6.2, 15.9, 20.1]}
        info = {'detected': True, 'box_xyxy': detection['box_xyxy']}
        with patch.object(grounding, 'detect', return_value=(detection, info)):
            mask, metadata = grounding.predict_mask(self.rgb, 'pear', 'dino', 'cpu')
        self.assertEqual(mask.sum(), 11 * 15)
        self.assertEqual(metadata['mask_source'], 'dino_box_roi')


if __name__ == '__main__':
    unittest.main()
