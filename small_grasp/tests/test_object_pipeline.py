"""CPU coverage for independent object scene routing, provenance and spawn safety."""
import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from small_grasp.instructions import resolve_pick_instruction
from small_grasp.object_env import sample_layout, LAYOUT_PROTOCOL, TARGET_XY_BOUNDS_M, TRAY_XY_BOUNDS_M
from small_grasp.object_registry import OBJECT_KEYS, get_object_spec
from small_grasp.run_sim_evaluation import ROOT, main


class ObjectPipelineTests(unittest.TestCase):
    def test_language_expansion_is_explicit_and_uses_caption(self):
        for key, text in [('orange', '抓橘子'), ('apple', '抓苹果'), ('pear', '抓梨子'), ('blue_cube', '抓蓝色方块')]:
            language = resolve_pick_instruction(text, allow_objects=True)
            self.assertEqual(language['target_key'], key)
            self.assertEqual(language['grounding_instruction'], get_object_spec(key)['caption'])
            if key != 'orange':
                with self.assertRaises(ValueError):
                    resolve_pick_instruction(text)
        self.assertEqual(resolve_pick_instruction('orange')['grounding_instruction'], 'orange')
        with self.assertRaises(ValueError):
            resolve_pick_instruction('apple and pear', allow_objects=True)

    def test_one_target_layout_is_deterministic_and_clear_of_tray(self):
        for key in OBJECT_KEYS:
            spec = get_object_spec(key)
            self.assertGreater(spec['mass_kg'], .01)
            for seed in range(4000, 4010):
                layout = sample_layout(spec, seed)
                self.assertEqual(layout, sample_layout(spec, seed))
                self.assertEqual(set(layout), {'seed', 'target_object', 'cardboard_box', 'layout_protocol', 'bounds_m'})
                self.assertEqual(layout['layout_protocol'], LAYOUT_PROTOCOL)
                self.assertEqual(layout['bounds_m'], {'target_xy': list(TARGET_XY_BOUNDS_M),
                                                     'tray_xy': list(TRAY_XY_BOUNDS_M)})
                x, y, z = layout['target_object']['position']
                bx, by, _ = layout['cardboard_box']['position']
                self.assertTrue(.290 <= x <= .320 and -.380 <= y <= -.340)
                self.assertTrue(.440 <= bx <= .490 and -.450 <= by <= -.400)
                self.assertGreaterEqual((max(abs(x-bx)-.08, 0)**2 + max(abs(y-by)-.08, 0)**2)**.5,
                                        spec['footprint_radius_m'] + .012)
                self.assertAlmostEqual(z, spec['spawn_z_m'])
                self.assertEqual(layout['target_object']['quaternion_xyzw'][:2], [0., 0.])

    def test_runner_capture_preparation_records_generic_manifest_without_models(self):
        cache = ROOT / 'cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / 'run'
            with patch('small_grasp.run_sim_evaluation.subprocess.run') as launch, redirect_stdout(io.StringIO()):
                main(['--scenes', 'object-single', '--instruction', '抓梨子', '--capture-only',
                      '--controller', 'servo', '--segmentation-backend', 'sam', '--seed', '4000',
                      '--wrist-camera-profile', 'open-jaws', '--network-frame', 'legacy-optical',
                      '--output-dir', str(output)])
            manifest = json.loads((output / 'run.json').read_text())
            self.assertEqual(manifest['language']['grounding_instruction'], 'pear fruit.')
            self.assertEqual(manifest['object_single']['spec']['scale'], .8)
            self.assertIn('assets/objects/pear/pear.usd', manifest['object_single']['files_sha256'])
            self.assertIn('small_grasp/object_assets.json', manifest['object_single']['files_sha256'])
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index('--scene')+1], 'object-single')
            self.assertEqual(command[command.index('--instruction')+1], '抓梨子')
            self.assertIn('--capture-only', command)

    def test_generic_options_reject_legacy_oracle_and_mixed_scene_scoring(self):
        for extra in (['--controller', 'baseline'], ['--mask-backend', 'gt'], ['--segmentation-backend', 'box'],
                      ['--scenes', 'object-single', 'single']):
            with patch('small_grasp.run_sim_evaluation.subprocess.run') as launch, redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                main(['--scenes', 'object-single', '--instruction', 'orange', '--controller', 'servo',
                      '--segmentation-backend', 'sam', '--output-dir', str(ROOT/'cache/must_not_create'), *extra])
            launch.assert_not_called()

    def test_executor_is_not_given_gt_center_and_old_scoring_branch_remains(self):
        tree = ast.parse((ROOT/'sim_eval.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'GraspExecutor']
        self.assertEqual(len(calls), 1)
        keyword = next(k for k in calls[0].keywords if k.arg == 'target_centre')
        self.assertIsInstance(keyword.value, ast.IfExp)
        # Only an explicitly selected sphere experiment may pass a perceived fit;
        # object-single validates network-only and does not pass scoring center.
        self.assertEqual(ast.unparse(keyword.value.test), "args.grasp_point == 'sphere'")
        self.assertIsNone(keyword.value.orelse.value)

    def test_summary_never_labels_generic_lift_as_original_place_task(self):
        from small_grasp.summarize_run import summarize_row
        from small_grasp.object_scoring import SCHEMA
        row = {'trial': 0, 'seed': 4000, 'success': True, 'grasp_lift_success': True,
               'score_schema': SCHEMA, 'generic_grasp_lift_success': True,
               'generic_end_hold_success': False, 'terminal': None}
        item = summarize_row(ROOT/'cache/nonexistent_evidence_for_cpu_test', 'object-single', row, {})
        self.assertTrue(item['stages']['stable_lift'])
        self.assertIsNone(item['stages']['task'])
        self.assertIsNone(item['original_task_success'])
        self.assertNotIn('scoring_issue', item)


if __name__ == '__main__':
    unittest.main()
