import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from small_grasp.candidate_contact import (ContactBandConfig, contact_band_cost,
                                         contact_priority_key, prepare_target_cloud)


class CandidateContactTests(unittest.TestCase):
    def setUp(self):
        self.config = ContactBandConfig(enabled=True)
        grid = np.stack(np.meshgrid(np.linspace(-.02, .02, 9),
                                    np.linspace(-.02, .02, 9), [.002, .009]), axis=-1).reshape(-1, 3)
        self.cloud = prepare_target_cloud(grid)

    def score(self, pocket=(0, 0, 0), rotation=None, offset=(0, -.00022, 0), cloud=None):
        return contact_band_cost(self.cloud if cloud is None else cloud, pocket,
                                 np.eye(3) if rotation is None else rotation, offset, config=self.config)

    def test_disabled_is_exact_noop_without_geometry(self):
        score = contact_band_cost(object(), object(), object(), object())
        self.assertEqual(score['penalty_deg_equivalent'], 0)
        self.assertEqual(score['status'], 'disabled')

    def test_centered_vs_misaligned_changes_feasible_preference_not_pose(self):
        centered = self.score()
        pocket = np.array([0, .03, 0])
        saved = pocket.copy()
        missed = self.score(pocket)
        np.testing.assert_array_equal(pocket, saved)
        self.assertLess(centered['penalty_deg_equivalent'], missed['penalty_deg_equivalent'])
        self.assertEqual(missed['status'], 'scored')
        self.assertLessEqual(missed['penalty_deg_equivalent'], 8.)

    def test_body_rotation_and_tool_reference_are_applied(self):
        R = Rotation.from_euler('xyz', [.4, -.2, .8]).as_matrix()
        pocket = np.array([.3, -.4, .1])
        offset = np.array([.04, -.00022, -.07])
        # Preserve exact prepared samples; prepare's world voxel grid is not rotation invariant.
        points = self.cloud.points_world_m
        world = points @ R.T + pocket
        rotated = type(self.cloud)(world, dict(self.cloud.evidence))
        a = self.score(offset=offset)
        b = contact_band_cost(rotated, pocket, R, offset, config=self.config)
        self.assertAlmostEqual(a['observed_median_y_offset_m'], b['observed_median_y_offset_m'], places=12)
        self.assertAlmostEqual(a['penalty_deg_equivalent'], b['penalty_deg_equivalent'], places=12)

    def test_nonfinite_and_repeated_pixels_do_not_supply_false_sample_confidence(self):
        repeated = np.tile([[0, 0, 0]], (1000, 1))
        cloud = prepare_target_cloud(np.r_[repeated, [[np.nan, 0, 0], [np.inf, 0, 0]]])
        self.assertEqual(len(cloud.points_world_m), 1)
        self.assertEqual(cloud.evidence['nonfinite_removed'], 2)
        self.assertEqual(self.score(cloud=cloud)['status'], 'insufficient_observed_points')

    def test_small_cloud_missing_and_invalid_pose_fall_back(self):
        missing = contact_band_cost(None, None, None, None, config=self.config)
        self.assertEqual(missing['status'], 'missing_target_cloud')
        self.assertEqual(missing['penalty_deg_equivalent'], 0)
        for R in [np.zeros((3, 3)), np.diag([-1, 1, 1]), np.full((3, 3), np.nan)]:
            score = self.score(rotation=R)
            self.assertEqual(score['status'], 'invalid_plan_geometry')
            self.assertEqual(score['penalty_deg_equivalent'], 0)
            json.dumps(score, allow_nan=False)

    def test_preparation_and_scoring_preserve_input_arrays(self):
        points = self.cloud.points_world_m.copy()
        source = points.copy()
        self.score(cloud=prepare_target_cloud(points))
        np.testing.assert_array_equal(source, points)
        with self.assertRaises(ValueError):
            self.cloud.points_world_m[0] = 0

    def test_partial_confidence_and_finite_json(self):
        cloud = prepare_target_cloud(self.cloud.points_world_m[:40])
        score = self.score(cloud=cloud)
        self.assertEqual(score['point_count'], 40)
        self.assertEqual(score['confidence'], .4)
        json.dumps(score, allow_nan=False)

    def test_configuration_and_coordinate_shape_validation(self):
        for kwargs in [dict(offset_scale_m=0), dict(half_width_m=np.nan), dict(penalty_weight_deg=-1),
                       dict(coverage_reference=1.1), dict(confidence_points=5), dict(offset_weight=np.nan),
                       dict(policy='contact_first')]:
            with self.assertRaises(ValueError):
                ContactBandConfig(**kwargs)
        with self.assertRaises(ValueError):
            prepare_target_cloud(np.zeros((3, 4)))
        with self.assertRaises(TypeError):
            contact_band_cost(np.zeros((100, 3)), [0, 0, 0], np.eye(3), [0, 0, 0], config=self.config)

    def test_default_soft_policy_and_disabled_priority_preserve_legacy(self):
        self.assertEqual(ContactBandConfig().policy, 'soft8')
        diag = self.score(pocket=(0, .03, 0))
        self.assertEqual(contact_priority_key(4., diag), (0, 4.))
        self.assertEqual(contact_priority_key(4., diag, config=self.config),
                         (0, 4. + diag['penalty_deg_equivalent']))

    def test_band_first_prefers_observed_band_and_uses_base_within_group(self):
        cfg = ContactBandConfig(enabled=True, policy='band_first')
        far, near = self.score(pocket=(0, .03, 0)), self.score()
        original = [(4., far, 'far'), (10., near, 'near_expensive'), (8., near, 'near_cheaper')]
        ranked = sorted(original, key=lambda p: contact_priority_key(p[0], p[1], config=cfg))
        self.assertEqual([p[2] for p in ranked], ['near_cheaper', 'near_expensive', 'far'])
        self.assertEqual(len(ranked), len(original))
        self.assertEqual(contact_priority_key(8., near, config=cfg), (0, 8.))

    def test_no_band_scored_plan_falls_back_to_same_soft_order(self):
        cfg = ContactBandConfig(enabled=True, policy='band_first')
        plans = [(7., self.score(pocket=(0, .03, 0))),
                 (9., self.score(pocket=(0, .05, 0))),
                 (12., {'enabled': True, 'status': 'insufficient_observed_points',
                         'penalty_deg_equivalent': 0.})]
        soft = sorted(range(len(plans)), key=lambda i: contact_priority_key(*plans[i], config=self.config))
        band = sorted(range(len(plans)), key=lambda i: contact_priority_key(*plans[i], config=cfg))
        self.assertEqual(soft, band)
        self.assertTrue(all(contact_priority_key(*p, config=cfg)[0] == 1 for p in plans))

    def test_band_boundary_missing_evidence_and_nonfinite_keys(self):
        cfg = ContactBandConfig(enabled=True, policy='band_first')
        diag = {'enabled': True, 'status': 'scored', 'observed_median_y_offset_m': .006,
                'penalty_deg_equivalent': 3.}
        self.assertEqual(contact_priority_key(4., diag, config=cfg), (0, 4.))
        self.assertEqual(contact_priority_key(4., {**diag, 'observed_median_y_offset_m': .006001}, config=cfg), (1, 7.))
        self.assertEqual(contact_priority_key(4., {**diag, 'status': 'invalid_plan_geometry'}, config=cfg), (1, 7.))
        for base, penalty in [(np.nan, 0.), (0., np.inf), (0., 8.1)]:
            with self.assertRaises(ValueError):
                contact_priority_key(base, {**diag, 'penalty_deg_equivalent': penalty}, config=cfg)


if __name__ == '__main__':
    unittest.main()
