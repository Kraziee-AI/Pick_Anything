"""Candidate source mapping and rejection decisions without model inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from small_grasp.geometry import diverse_grasp_indices, diverse_grasps
from small_grasp.infer import load_mask_provenance, parse_args


class CandidateTraceTests(unittest.TestCase):
    def test_standalone_infer_default_and_existing_top_k_remain_compatible(self):
        for flags, expected in [([], 20), (['--candidate-pool-size', '100'], 100), (['--top-k', '7'], 7)]:
            with self.subTest(flags=flags), patch('sys.argv', ['small_grasp.infer', '--demo', *flags]):
                self.assertEqual(parse_args().top_k, expected)

    def test_ranked_indices_preserve_raw_mapping_and_reasons(self):
        def proposal(score, x):
            row = np.zeros(17, dtype=np.float32)
            row[0:4] = [score, .04, .02, .03]
            row[4:13] = np.eye(3, dtype=np.float32).reshape(-1)
            row[13:16] = [x, 0, .5]
            return row

        rows = np.stack([proposal(.5, .0), proposal(.9, .1),
                         proposal(.8, .101), proposal(.7, .2), proposal(.6, .3)])
        indices, reasons = diverse_grasp_indices(rows, top_k=2)
        np.testing.assert_array_equal(indices, [1, 3])
        self.assertEqual(reasons.tolist(), ['beyond_top_k', 'selected', 'duplicate_pose',
                                            'selected', 'beyond_top_k'])
        np.testing.assert_array_equal(diverse_grasps(rows, 2), rows[indices])

    def test_empty_candidates_keep_17_column_shape(self):
        rows = np.empty((0, 17), dtype=np.float32)
        indices, reasons = diverse_grasp_indices(rows, top_k=5)
        self.assertEqual(indices.shape, (0,))
        self.assertEqual(reasons.shape, (0,))
        self.assertEqual(diverse_grasps(rows, 5).shape, (0, 17))

    def test_larger_pool_preserves_first_twenty_rows_and_raw_indices(self):
        rng = np.random.default_rng(42)
        rows = np.zeros((160, 17), dtype=np.float32)
        rows[:, 0] = rng.uniform(.05, .9, len(rows))
        rows[:, 1:4] = [.04, .02, .03]
        rows[:, 4:13] = np.eye(3).reshape(1, 9)
        rows[:, 13] = np.arange(len(rows)) * .011
        rows[:, 15] = .5
        # Include near-duplicates and equal scores to exercise NMS and tie order.
        rows[100:110] = rows[:10]
        raw_indices = np.arange(len(rows)) * 7 + 3
        baseline, _ = diverse_grasp_indices(rows, 20)
        expanded, _ = diverse_grasp_indices(rows, 100)
        self.assertEqual(len(expanded), 100)
        np.testing.assert_array_equal(expanded[:20], baseline)
        np.testing.assert_array_equal(raw_indices[expanded[:20]], raw_indices[baseline])
        np.testing.assert_array_equal(rows[expanded[:20]], rows[baseline])

    def test_saved_mask_requires_matching_record_bytes_and_view(self):
        cache = Path(__file__).resolve().parents[1] / 'cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            view = Path(directory)
            mask = np.array([[False, True], [True, False]])
            mask_path = view / 'target_mask.png'
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            record = {'target_detection': {'backend': 'dino', 'instruction': 'orange', 'detected': True},
                      'target_segmentation': {'backend': 'sam', 'provenance': 'sam_box_prompt'},
                      'target_quality': {'accepted': True, 'pixels': 2},
                      'target_mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest()}
            provenance_path = view / 'perception.json'
            provenance_path.write_text(json.dumps(record))
            source = load_mask_provenance(provenance_path, mask_path, mask, 'orange')
            self.assertTrue(source['live_language_detection'])
            self.assertFalse(source['detection_recomputed'])
            self.assertEqual(source['segmentation']['backend'], 'sam')
            self.assertEqual(source['mask_sha256'], record['target_mask_sha256'])
            with self.assertRaisesRegex(ValueError, 'instruction'):
                load_mask_provenance(provenance_path, mask_path, mask, 'pear')
            Image.fromarray(np.zeros_like(mask, dtype=np.uint8)).save(mask_path)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_mask_provenance(provenance_path, mask_path, mask, 'orange')
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            record.pop('target_segmentation')
            provenance_path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'target_segmentation'):
                load_mask_provenance(provenance_path, mask_path, mask, 'orange')
            with self.assertRaisesRegex(ValueError, 'same-view'):
                load_mask_provenance(view / 'other.json', mask_path, mask, 'orange')

    def test_rejected_empty_orange_mask_accepts_missing_pixel_count(self):
        cache = Path(__file__).resolve().parents[1] / 'cache'
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            view = Path(directory)
            mask_path = view / 'target_mask.png'
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(mask_path)
            record = {'target_detection': {'backend': 'dino', 'instruction': 'orange'},
                      'target_segmentation': {'backend': 'box', 'provenance': 'detector_box_roi'},
                      'target_quality': {'accepted': False, 'reason': 'missing_target'},
                      'target_mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest()}
            provenance_path = view / 'perception.json'
            provenance_path.write_text(json.dumps(record))
            got = load_mask_provenance(provenance_path, mask_path,
                                       np.zeros((2, 2), dtype=bool), 'orange')
            self.assertFalse(got['mask_refinement']['accepted'])


if __name__ == '__main__':
    unittest.main()
