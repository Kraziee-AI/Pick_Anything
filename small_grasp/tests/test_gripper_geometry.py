"""Independent convex-halfspace checks for the mesh-ray contact diagnostic."""
import unittest

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from small_grasp.gripper_geometry import SO101GripperGeometry, FIXED_REFERENCE_M, MOVING_REFERENCE_M


class GripperGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = SO101GripperGeometry()

    def _convex_exit(self, origin, direction, triangles, names, prim):
        vertices = np.unique(triangles[np.array(names) == prim].reshape(-1, 3), axis=0)
        planes = ConvexHull(vertices).equations
        signed = planes[:, :3] @ origin + planes[:, 3]
        self.assertLessEqual(float(signed.max()), 1e-8, 'pad reference must be inside the hit solid')
        slopes = planes[:, :3] @ direction
        forward = slopes > 1e-10
        return float(np.min(-signed[forward] / slopes[forward]))

    def test_ray_gap_matches_independent_convex_plane_clipping(self):
        g = self.geometry
        for q in [.05, .23, .4, .6, 1., 1.4]:
            with self.subTest(joint=q):
                m = g.measure(q)
                # Independent USD joint conversion, explicitly expanded in column convention.
                r = g.parent_rotation @ Rotation.from_rotvec(g.axis * q).as_matrix() @ g.child_rotation.T
                moving = g.parent_pivot + r @ (MOVING_REFERENCE_M - g.child_pivot)
                vector = moving - FIXED_REFERENCE_M
                direction = vector / np.linalg.norm(vector)
                fixed_triangles, fnames = g.meshes['fixed']
                child_triangles, mnames = g.meshes['moving']
                moving_triangles = np.einsum('ij,nkj->nki', r, child_triangles - g.child_pivot) + g.parent_pivot
                a = self._convex_exit(FIXED_REFERENCE_M, direction, fixed_triangles, fnames, m['fixed_collision_prim'])
                b = self._convex_exit(moving, -direction, moving_triangles, mnames, m['moving_collision_prim'])
                self.assertAlmostEqual(m['local_surface_gap_m'], np.linalg.norm(vector) - a - b, places=9)
                self.assertGreater(m['fixed_surface_patch']['area_m2'], 0)
                self.assertGreater(m['moving_surface_patch']['area_m2'], 0)
                for side in ['fixed', 'moving']:
                    patch = m[side + '_surface_patch']
                    vertices = np.array(patch['vertices_body_m'])
                    normal = np.array(m[side + '_outward_normal_body'])
                    hit = np.array(m[side + '_surface_body_m'])
                    np.testing.assert_allclose((vertices - hit) @ normal, 0, atol=1e-7)

    def test_reference_span_is_not_inner_gap(self):
        m = self.geometry.measure(.4)
        self.assertAlmostEqual(m['reference_span_m'], .056691597833833124)
        self.assertAlmostEqual(m['local_surface_gap_m'], .04736902344528689)
        self.assertGreater(m['normal_opposition_error_deg'], 20)
        self.assertGreater(m['reference_span_m'] - m['local_surface_gap_m'], .009)

    def test_opening_face_changes_on_large_rotation(self):
        narrow = self.geometry.measure(1.)
        wide = self.geometry.measure(1.4)
        # Both are the same distal solid, but a different finite planar face is hit.
        self.assertEqual(narrow['moving_collision_prim'], wide['moving_collision_prim'])
        self.assertGreater(narrow['normal_opposition_error_deg'], 50)
        self.assertLess(wide['normal_opposition_error_deg'], 15)
        self.assertNotEqual(narrow['moving_surface_patch']['area_m2'], wide['moving_surface_patch']['area_m2'])
