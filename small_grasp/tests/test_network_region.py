"""Targeted inference must exclude background without dropping collision context."""
import unittest
import numpy as np
from small_grasp.geometry import network_input_mask, target_crop
from small_grasp.pipeline_options import resolve_network_region, resolve_candidate_pool_size


class NetworkRegionTests(unittest.TestCase):
    def test_irregular_target_and_depth_holes_exclude_background(self):
        mask = np.zeros((9, 12), dtype=bool)
        mask[2:7, 4:8] = True
        mask[2, 4] = False
        valid = np.ones_like(mask)
        valid[4, 6] = False
        before = valid.copy()
        got = network_input_mask(mask, valid, 'target-mask')
        self.assertEqual(int(got.sum()), 18)
        self.assertFalse(got[2, 4])
        self.assertFalse(got[4, 6])
        np.testing.assert_array_equal(valid, before)  # Full scene remains intact.
        np.testing.assert_array_equal(network_input_mask(mask, valid, 'bbox', 2), target_crop(mask, 2) & valid)
        self.assertFalse(network_input_mask(np.zeros_like(mask), valid, 'target-mask').any())

    def test_profile_defaults_preserve_legacy_and_explicit_ablation(self):
        self.assertEqual(resolve_network_region('legacy', 'auto'), 'bbox')
        self.assertEqual(resolve_network_region('open-jaws', 'auto'), 'target-mask')
        self.assertEqual(resolve_network_region('open-jaws', 'bbox'), 'bbox')
        self.assertEqual(resolve_network_region('legacy', 'target-mask'), 'target-mask')
        with self.assertRaises(ValueError):
            resolve_network_region('open-jaws', 'unknown')

    def test_pool_defaults_and_explicit_baseline_are_independent_of_primary_budget(self):
        self.assertEqual(resolve_candidate_pool_size('legacy', 'bbox'), 20)
        self.assertEqual(resolve_candidate_pool_size('legacy', 'target-mask'), 20)
        self.assertEqual(resolve_candidate_pool_size('open-jaws', 'bbox'), 20)
        self.assertEqual(resolve_candidate_pool_size('open-jaws', 'target-mask'), 100)
        self.assertEqual(resolve_candidate_pool_size('open-jaws', 'target-mask', 20), 20)
        self.assertEqual(resolve_candidate_pool_size('legacy', 'bbox', 80), 80)
        for invalid in [0, -1, True, 1.5]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_candidate_pool_size('open-jaws', 'target-mask', invalid)
