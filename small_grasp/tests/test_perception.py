"""Regression cases for false boxes, clipped targets and robot foreground."""
import unittest

import numpy as np

from small_grasp.perception import TABLE_Z, fit_orange_center, locate_box, refine_target


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.indices((80, 100))
        self.points = np.stack([.2 + xx * .002, -.5 + yy * .002,
                                np.full_like(xx, TABLE_Z, dtype=float)], axis=-1)
        self.valid = np.ones((80, 100), dtype=bool)

    def orange_patch(self):
        centre = np.array([.264, -.436, TABLE_Z + .027])
        radius2 = np.sum((self.points[..., :2] - centre[:2]) ** 2, axis=-1)
        mask = radius2 < .026 ** 2
        self.points[mask, 2] = centre[2] + np.sqrt(.028 ** 2 - radius2[mask])
        return mask, centre

    def test_language_box_over_flat_small_distractor_is_rejected(self):
        mask = np.zeros_like(self.valid)
        mask[10:25, 10:25] = True
        self.points[mask, 2] = TABLE_Z + .009
        selected, quality = refine_target(mask, self.points, self.valid)
        self.assertFalse(quality['accepted'])
        self.assertFalse(selected.any())

    def test_raised_orange_component_excludes_table_and_rejects_clipped_view(self):
        mask = np.zeros_like(self.valid)
        mask[15:50, 15:50] = True
        sphere, _ = self.orange_patch()
        selected, quality = refine_target(mask, self.points, self.valid)
        self.assertTrue(quality['accepted'])
        np.testing.assert_array_equal(selected, sphere)
        self.points = np.roll(self.points, -20, axis=0)
        mask = np.roll(mask, -20, axis=0)
        _, quality = refine_target(mask, self.points, self.valid)
        self.assertFalse(quality['accepted'])

    def test_box_estimate_ignores_high_robot_foreground_and_table(self):
        mask = np.ones_like(self.valid)
        self.points[5:75, 10:90, 2] = TABLE_Z + .004
        self.points[25:55, 35:65, 2] = .24
        centre, quality = locate_box(mask, self.points, self.valid)
        self.assertTrue(quality['accepted'])
        np.testing.assert_allclose(centre, [.299, -.421, TABLE_Z], atol=.005)

    def test_corner_only_cannot_be_a_destination(self):
        mask = np.zeros_like(self.valid)
        mask[10:30, 10:30] = True
        self.points[mask, 2] = TABLE_Z + .004
        centre, quality = locate_box(mask, self.points, self.valid)
        self.assertIsNone(centre)
        self.assertFalse(quality['accepted'])

    def test_orange_instruction_rejects_same_size_blue_cube(self):
        mask, _ = self.orange_patch()
        rgb = np.full((*mask.shape, 3), [87, 156, 223], dtype=np.uint8)
        selected, quality = refine_target(mask, self.points, self.valid, rgb)
        self.assertFalse(quality['accepted'])
        self.assertFalse(selected.any())
        rgb[:] = [43, 14, 2]
        _, quality = refine_target(mask, self.points, self.valid, rgb)
        self.assertTrue(quality['accepted'])

    def test_partial_depth_surface_estimates_centre_not_visible_surface_median(self):
        mask, centre = self.orange_patch()
        points = self.points[mask]
        points = points[points[:, 0] > centre[0] - .01]
        fit = fit_orange_center(points)
        self.assertIsNotNone(fit)
        np.testing.assert_allclose(fit['centre_world_m'], centre, atol=1e-5)
        self.assertGreater(np.linalg.norm(np.median(points, axis=0) - centre), .015)


if __name__ == '__main__':
    unittest.main()
