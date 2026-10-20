"""CPU regressions for live USD variant collision coverage, without Isaac/GT."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from small_grasp.gripper_geometry import DEFAULT_USD, DEFAULT_VARIANTS, SO101GripperGeometry
from small_grasp.sim_robot import GraspExecutor, _GripperMeshClearance


class RobotSensorCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.live = SO101GripperGeometry()
        cls.authored = SO101GripperGeometry(variants={})

    def test_default_matches_live_and_covers_camera_both_meshes(self):
        self.assertEqual(self.live.selected_variants, DEFAULT_VARIANTS)
        self.assertEqual(self.live.evidence['mesh_counts'], {'fixed': 9, 'moving': 9})
        self.assertEqual(self.authored.evidence['mesh_counts'], {'fixed': 7, 'moving': 9})
        paths = self.live.evidence['mesh_paths']['fixed']
        for suffix in ['camera_mount/Cube/Cube', 'camera_mount/Cylinder/Cylinder']:
            self.assertTrue(any(path.endswith(suffix) for path in paths))
        json.dumps(self.live.evidence)

    def test_original_sixteen_meshes_and_joint_frames_are_unchanged(self):
        for side, (old_triangles, old_names) in self.authored.meshes.items():
            triangles, names = self.live.meshes[side]
            for path in set(old_names):
                np.testing.assert_array_equal(triangles[np.array(names) == path],
                                              old_triangles[np.array(old_names) == path])
        for attr in ['parent_pivot', 'child_pivot', 'parent_rotation', 'child_rotation', 'axis']:
            np.testing.assert_array_equal(getattr(self.live, attr), getattr(self.authored, attr))

    def test_pad_rays_axes_normals_and_reference_midpoints_are_unchanged(self):
        for q in [.05, .23, .4, .6, .8, 1., 1.4]:
            with self.subTest(joint=q):
                self.assertEqual(self.live.measure(q), self.authored.measure(q))

    def test_invalid_variants_fail_instead_of_silently_selecting_defaults(self):
        for variants in [{'Sensor': 'typo'}, {'UnknownSet': 'sensors'}]:
            with self.subTest(variants=variants), self.assertRaises(ValueError):
                SO101GripperGeometry(variants=variants)

    def test_explicit_sensor_selection_and_source_asset_not_modified(self):
        before = hashlib.sha256(DEFAULT_USD.read_bytes()).hexdigest()
        disabled = SO101GripperGeometry(variants={'Robot': 'robot', 'Physics': 'physics'})
        self.assertEqual(disabled.selected_variants['Sensor'], '')
        self.assertEqual(disabled.evidence['mesh_counts']['fixed'], 7)
        enabled = SO101GripperGeometry(variants=DEFAULT_VARIANTS)
        self.assertEqual(enabled.evidence['mesh_counts']['fixed'], 9)
        self.assertEqual(hashlib.sha256(DEFAULT_USD.read_bytes()).hexdigest(), before)
        # Opening a fresh stage still sees authored selections, not the other session's overrides.
        self.assertEqual(SO101GripperGeometry(variants={}).selected_variants['Robot'], 'none')

    def test_executor_uses_runtime_variants_instead_of_geometry_defaults(self):
        executor = GraspExecutor.__new__(GraspExecutor)
        executor.controller = 'servo'
        executor.robot = SimpleNamespace(cfg=SimpleNamespace(spawn=SimpleNamespace(
            usd_path=str(DEFAULT_USD), variants={})))
        executor._configure_tool()
        self.assertEqual(executor.gripper_geometry.evidence['mesh_counts']['fixed'], 7)
        self.assertEqual(executor.tool_evidence['robot_geometry']['selected_variants']['Sensor'], '')
        executor.destination = None
        executor.model = SimpleNamespace(state=lambda: None)
        executor.observation_dir = Path('.')
        executor.mesh_evidence = {}
        # Verify the second initializer forwards variants too when no tool cache exists.
        del executor.gripper_geometry
        executor._init_mesh_clearance()
        self.assertEqual(executor.mesh_evidence['robot_geometry']['mesh_counts']['fixed'], 7)

    def test_saved_robot_pose_detects_previously_missing_table_and_box_mesh(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/robot_sensor_collision_306_320.json').read_text())
        for row in fixture['rows']:
            with self.subTest(seed=row['seed'], step=row['step']):
                args = row['pocket'], row['rotation'], row['joint_rad']
                self.assertIsNone(_GripperMeshClearance(self.authored, row['box_points']).check(*args))
                hit = _GripperMeshClearance(self.live, row['box_points']).check(*args)
                self.assertEqual(hit['reason'], row['expected_live']['reason'])
                if row['seed'] == 320:
                    self.assertAlmostEqual(hit['minimum_mesh_z_m'], .04064668552409737, places=9)
                    self.assertGreater(hit['required_vertical_correction_m'], .008)
                else:
                    self.assertEqual(hit['mesh'], row['expected_live']['mesh'])
                    self.assertEqual(hit['body'], 'fixed')


if __name__ == '__main__':
    unittest.main()
