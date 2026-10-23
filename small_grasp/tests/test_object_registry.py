"""CPU contracts for independent class metadata and actual local USD geometry."""
import hashlib
import math
import subprocess
import sys
from pathlib import Path
import unittest

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Sdf

from small_grasp.object_registry import (
    OBJECT_KEYS, PHYSICS_MATERIAL, NEWTON_CONTACT_FRICTION, PROJECT_ROOT,
    TABLE_TOP_Z_M, SPAWN_CLEARANCE_M, get_object_spec, resolve_object_key,
)


class ObjectRegistryTests(unittest.TestCase):
    def test_fixed_classes_sizes_and_native_pose(self):
        self.assertEqual(OBJECT_KEYS, ('orange', 'apple', 'pear', 'blue_cube'))
        for key, scale in {'orange': .775, 'apple': .775, 'pear': .8, 'blue_cube': 1.}.items():
            spec = get_object_spec(key)
            self.assertEqual(spec['scale'], scale)
            self.assertEqual(spec['initial_orientation_wxyz'], [1., 0, 0, 0])
            self.assertTrue(Path(spec['usd_path']).is_file())
            self.assertTrue(Path(spec['usd_path']).is_relative_to(PROJECT_ROOT))
        self.assertGreater(get_object_spec('pear')['collision_max_m'][1] -
                           get_object_spec('pear')['collision_min_m'][1], .08)
        spec = get_object_spec('apple')
        spec['aliases'].append('banana')
        self.assertNotIn('banana', get_object_spec('apple')['aliases'])
        with self.assertRaises(ValueError):
            get_object_spec('banana')

    def test_language_aliases_are_unambiguous_without_visual_claims(self):
        for text, key in [('抓橘子', 'orange'), ('pick a red apple', 'apple'),
                          ('拿梨子', 'pear'), ('抓蓝色方块', 'blue_cube'),
                          ('APPLE FRUIT.', 'apple'), ('pick the blue block', 'blue_cube')]:
            self.assertEqual(resolve_object_key(text), key)
        for text in ('', 'banana', 'apple and pear', '抓苹果和橘子', 'orangeish', 'pineapple'):
            with self.assertRaises(ValueError):
                resolve_object_key(text)

    def test_collision_bounds_mass_origin_and_dependency_match_real_assets(self):
        for key in OBJECT_KEYS:
            spec = get_object_spec(key)
            path = Path(spec['usd_path'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), spec['asset_sha256'])
            source = Usd.Stage.Open(str(path))
            self.assertEqual(sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in source.Traverse()), 1)
            self.assertAlmostEqual(UsdPhysics.MassAPI(source.GetDefaultPrim()).GetMassAttr().Get(),
                                   spec['mass_kg'], places=8)
            # Compose into an explicit metre/Z stage, mirroring reference numeric
            # assembly, without starting Isaac, a renderer or a physics engine.
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageMetersPerUnit(stage, 1.)
            UsdGeom.SetStageUpAxis(stage, 'Z')
            obj = UsdGeom.Xform.Define(stage, '/TargetObject')
            obj.GetPrim().GetReferences().AddReference(str(path))
            obj.AddScaleOp().Set((spec['scale'],) * 3)
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'], False, True)
            bounds = [cache.ComputeWorldBound(p).ComputeAlignedRange() for p in stage.Traverse()
                      if p.HasAPI(UsdPhysics.CollisionAPI)]
            self.assertTrue(bounds)
            lo = np.min([list(b.GetMin()) for b in bounds], axis=0)
            hi = np.max([list(b.GetMax()) for b in bounds], axis=0)
            np.testing.assert_allclose(lo, spec['collision_min_m'], atol=2e-9)
            np.testing.assert_allclose(hi, spec['collision_max_m'], atol=2e-9)
            np.testing.assert_allclose((lo + hi) / 2, spec['center_offset_m'], atol=2e-9)
            self.assertAlmostEqual(-lo[2], spec['resting_root_height_m'], places=8)
            self.assertAlmostEqual(spec['spawn_z_m'] + lo[2], TABLE_TOP_Z_M + SPAWN_CLEARANCE_M, places=8)
            self.assertAlmostEqual(hi[2] - lo[2], spec['height_m'], places=8)
            self.assertGreaterEqual(spec['footprint_radius_m'] + 2e-9,
                                    math.hypot(max(abs(lo[0]), abs(hi[0])), max(abs(lo[1]), abs(hi[1]))))
            self.assertFalse(source.GetRootLayer().subLayerPaths)
            for prim in source.Traverse():
                self.assertFalse(prim.HasAuthoredReferences() or prim.HasAuthoredPayloads())
                for attr in prim.GetAttributes():
                    if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                        continue
                    value = attr.Get()
                    if value and value.path.endswith('.png'):
                        texture = Path(value.path)
                        texture = texture if texture.is_absolute() else path.parent / texture
                        self.assertTrue(texture.resolve().is_relative_to(PROJECT_ROOT))
                        self.assertTrue(texture.is_file())

    def test_material_is_uniform_and_preserves_known_newton_contact_values(self):
        self.assertEqual(PHYSICS_MATERIAL, {'static_friction': .9, 'dynamic_friction': .9, 'restitution': 0.})
        self.assertEqual(NEWTON_CONTACT_FRICTION, {'friction': .9, 'rolling_friction': .5, 'torsional_friction': .1})

    def test_registry_import_is_cpu_only_and_needs_no_research_directory(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            'import sys; from small_grasp.object_registry import get_object_spec; '
            'assert get_object_spec("pear")["scale"] == .8; '
            'assert not any(k.split(".")[0] in ("torch", "isaaclab", "pxr", "numpy") '
            'for k in sys.modules)'], cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
