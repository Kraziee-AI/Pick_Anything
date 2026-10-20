"""Visual-only tray repair must preserve every original physical mesh and mass."""
from pathlib import Path
import unittest

from pxr import Usd, UsdGeom, UsdPhysics


class RenderAssetTests(unittest.TestCase):
    def test_physics_identical_and_visual_meshes_have_no_collision(self):
        root = Path(__file__).resolve().parents[1]
        original = Usd.Stage.Open(str(root.parent / 'assets/objects/cardboard_box/cardboard_box.usd'))
        repaired = Usd.Stage.Open(str(root / 'assets/cardboard_box_visible.usda'))
        original_body = original.GetPrimAtPath('/cardboard_box')
        repaired_body = repaired.GetPrimAtPath('/cardboard_box')
        self.assertEqual(UsdPhysics.MassAPI(original_body).GetMassAttr().Get(),
                         UsdPhysics.MassAPI(repaired_body).GetMassAttr().Get())
        colliders = [p for p in repaired.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        self.assertEqual(len(colliders), 5)
        for prim in colliders:
            before = original.GetPrimAtPath(prim.GetPath())
            self.assertTrue(before)
            for name in ('points', 'faceVertexCounts', 'faceVertexIndices', 'extent', 'physics:approximation'):
                self.assertEqual(before.GetAttribute(name).Get(), prim.GetAttribute(name).Get())
            visual = repaired.GetPrimAtPath('/cardboard_box/Visible_' + prim.GetName())
            self.assertTrue(visual.IsA(UsdGeom.Mesh))
            self.assertFalse(visual.HasAPI(UsdPhysics.CollisionAPI))
            self.assertFalse(visual.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertEqual(visual.GetAttribute('points').Get(), prim.GetAttribute('points').Get())
