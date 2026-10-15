"""CPU contracts for assets, collision-free placement and training isolation."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
for key, folder in {"XDG_CACHE_HOME": "xdg", "WARP_CACHE_PATH": "warp",
                    "MPLCONFIGDIR": "matplotlib", "TORCH_HOME": "torch"}.items():
    os.environ[key] = str(PROJECT_ROOT / "outputs/tabletop_objects/cache" / folder)

from pick_anything.tabletop.catalog import (
    BOX_ASSET, BOX_KEY, DEFAULT_OBJECTS, OBJECTS, resolve_target, sample_layout,
)


class TabletopTests(unittest.TestCase):
    def test_language_selection_and_absent_or_ambiguous_target(self):
        for text, key in [("抓小黄鸭", "yellow_duck"), ("拿蓝色方块", "blue_cube"),
                          ("pick up the orange", "orange"), ("pick the rubber duck", "yellow_duck"),
                          ("抓红色圆柱", "red_cylinder")]:
            self.assertEqual(resolve_target(text, tuple(OBJECTS)).key, key)
        self.assertEqual(resolve_target("orangeish block").key, "blue_cube")
        for text in ("", "抓香蕉", "抓小黄鸭和橘子", "pick the duckling"):
            with self.assertRaises(ValueError):
                resolve_target(text)
        with self.assertRaises(ValueError):
            resolve_target("抓红色圆柱")

    def test_many_layouts_fit_reach_and_keep_bounding_circles_apart(self):
        self.assertEqual(sample_layout(seed=42), sample_layout(seed=42))
        for seed in range(200):
            layout = sample_layout(tuple(OBJECTS), seed)
            bx, by, bz = layout[BOX_KEY]["position"]
            self.assertTrue(0.14 <= bx <= 0.56 and -0.51 <= by <= -0.39)
            self.assertTrue(0.16 <= math.hypot(bx - 0.35, by + 0.64) <= 0.24)
            self.assertAlmostEqual(bz, 0.046)
            self.assertEqual(layout[BOX_KEY]["orientation_xyzw"], (0, 0, 0, 1))
            for key in OBJECTS:
                pose = layout[key]
                x, y, z = pose["position"]
                obj = OBJECTS[key]
                radius = obj.footprint_radius
                self.assertTrue(-0.135 + radius <= x <= 0.845 - radius)
                self.assertTrue(-0.667 + radius <= y <= -0.237 - radius)
                self.assertTrue(0.14 <= math.hypot(x - 0.35, y + 0.64) <= 0.29)
                self.assertAlmostEqual(z - obj.origin_to_bottom, 0.045)
                self.assertAlmostEqual(sum(v * v for v in pose["orientation_xyzw"]), 1)
                tray_distance = math.hypot(max(abs(x - bx) - 0.08, 0), max(abs(y - by) - 0.08, 0))
                self.assertGreaterEqual(tray_distance, radius + 0.012)
                for other, p in layout.items():
                    if other in (key, BOX_KEY):
                        continue
                    self.assertGreaterEqual(math.hypot(x - p["position"][0], y - p["position"][1]),
                                            radius + OBJECTS[other].footprint_radius + 0.012)
        self.assertNotIn(BOX_KEY, sample_layout(seed=42, include_box=False))
        for keys in ((), ("orange", "orange"), ("unknown",)):
            with self.assertRaises(ValueError):
                sample_layout(keys)
        with self.assertRaises(ValueError):
            sample_layout(gap=1.0)

    def test_assets_are_single_rigid_bodies_with_bounded_geometry_and_local_textures(self):
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

        for key in ("yellow_duck", "blue_cube", "red_cylinder"):
            obj = OBJECTS[key]
            stage = Usd.Stage.Open(str(obj.asset))
            root = stage.GetDefaultPrim()
            self.assertTrue(root.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertGreater(UsdPhysics.MassAPI(root).GetMassAttr().Get(), 0)
            self.assertEqual(sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()), 1)
            self.assertGreater(sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse()), 0)
            bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeLocalBound(root).ComputeAlignedRange()
            lo, hi = bounds.GetMin(), bounds.GetMax()
            self.assertAlmostEqual(lo[2], 0, places=6)
            self.assertAlmostEqual(hi[2], obj.height, places=6)
            radius = math.hypot(max(abs(lo[0]), abs(hi[0])), max(abs(lo[1]), abs(hi[1])))
            # Cylinder's rectangular AABB corners are outside its circular footprint.
            if key != "red_cylinder":
                self.assertLessEqual(radius, obj.footprint_radius)
            for prim in stage.Traverse():
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                    self.assertTrue(material)
                    self.assertGreater(UsdPhysics.MaterialAPI(material.GetPrim()).GetStaticFrictionAttr().Get(), 0)
                if prim.IsA(UsdShade.Shader):
                    texture = UsdShade.Shader(prim).GetInput("diffuse_texture")
                    if texture:
                        asset = texture.Get()
                        self.assertFalse(Path(asset.path).is_absolute())
                        self.assertTrue((obj.asset.parent / asset.path).is_file())

    def test_new_scene_leaves_both_training_configs_and_registry_unchanged(self):
        import gymnasium as gym
        from pick_anything.env_cfg import PickSceneCfg
        from pick_anything.tasks.pick_apple.camera_env_cfg import CameraClosureFinetuneEnvCfg
        from pick_anything.grounded_pick.config import GroundedPickEnvCfg

        original_scene = PickSceneCfg(num_envs=1).to_dict()
        original_camera = CameraClosureFinetuneEnvCfg().to_dict()
        original_grounded = GroundedPickEnvCfg().to_dict()
        registered = set(gym.registry)
        from pick_anything.tabletop.scene_cfg import make_scene_cfg

        cfg = make_scene_cfg(seed=42)
        cfg.validate()
        self.assertIsNotNone(cfg.cardboard_box)
        self.assertEqual(cfg.cardboard_box.spawn.usd_path, str(BOX_ASSET))
        self.assertEqual(cfg.cardboard_box.init_state.pos, sample_layout(seed=42)[BOX_KEY]["position"])
        self.assertIsNone(cfg.red_cylinder)
        for key in DEFAULT_OBJECTS:
            self.assertTrue(getattr(cfg, key).spawn.activate_contact_sensors)
            self.assertEqual(getattr(cfg, key).spawn.semantic_tags, [("class", key)])
        cfg.orange.spawn.scale = (2.0,) * 3
        cfg.robot.init_state.pos = (0, 0, 1)
        cfg.cardboard_box.init_state.pos = (0, 0, 1)
        self.assertEqual(original_scene, PickSceneCfg(num_envs=1).to_dict())
        self.assertEqual(original_camera, CameraClosureFinetuneEnvCfg().to_dict())
        self.assertEqual(original_grounded, GroundedPickEnvCfg().to_dict())
        self.assertEqual(registered, set(gym.registry))
        all_objects = make_scene_cfg(tuple(OBJECTS), seed=1)
        all_objects.validate()
        self.assertIsNotNone(all_objects.red_cylinder)
        orange_only = make_scene_cfg(("orange",), seed=1)
        self.assertIsNone(orange_only.yellow_duck)
        self.assertIsNone(orange_only.blue_cube)
        self.assertIsNotNone(orange_only.cardboard_box)
        self.assertIsNone(make_scene_cfg(include_box=False).cardboard_box)


if __name__ == "__main__":
    unittest.main(verbosity=2)
