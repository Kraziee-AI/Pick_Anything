"""Language provenance and fixed real-perception settings across entry points."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from small_grasp.instructions import resolve_pick_instruction
from small_grasp.run_sim_evaluation import ROOT, main as evaluate


class LanguageEntryTests(unittest.TestCase):
    def test_supported_descriptions_keep_original_text_and_canonical_detector_target(self):
        for text in ('orange', 'orange fruit', '抓橘子', '橙子'):
            with self.subTest(text=text):
                language = resolve_pick_instruction(text)
                self.assertEqual(language['instruction'], text)
                self.assertEqual(language['target_key'], 'orange')
                self.assertEqual(language['grounding_instruction'], 'orange')
                self.assertEqual(language['caption'], 'orange fruit.')
        for text in ('', 'duck', 'orange and blue cube', 'banana'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                resolve_pick_instruction(text)

    def test_unified_entry_forwards_language_and_real_pipeline_options(self):
        from small_grasp.pick import main
        output = ROOT / 'outputs/example_only_not_created'
        with patch('sys.argv', ['small_grasp.pick', '--instruction', '抓橘子', '--scene', 'single',
                               '--viz', 'kit', '--network-frame', 'legacy-optical', '--wrist-camera-profile', 'open-jaws',
                               '--live-debug', '--live-debug-port', '0',
                               '--live-debug-interval', '2', '--output-dir', str(output)]), \
             patch('small_grasp.run_sim_evaluation.main') as runner, redirect_stdout(io.StringIO()):
            main()
        argv = runner.call_args.args[0]
        self.assertEqual(argv[argv.index('--network-frame') + 1], 'legacy-optical')
        self.assertEqual(argv[argv.index('--wrist-camera-profile') + 1], 'open-jaws')
        self.assertIn('--live-debug', argv)
        self.assertNotIn('--arm-tracking-feedback', argv)
        self.assertNotIn('--closure-latch-policy', argv)
        self.assertNotIn('--close-tracking-integral', argv)
        self.assertNotIn('--close-integral-transfer', argv)
        self.assertNotIn('--close-frame-policy', argv)
        self.assertNotIn('--candidate-contact-ranking', argv)
        self.assertEqual(argv[argv.index('--live-debug-port') + 1], '0')
        self.assertEqual(argv[argv.index('--live-debug-interval') + 1], '2.0')
        for flag, expected in {'--instruction': '抓橘子', '--mask-backend': 'dino',
                               '--controller': 'repaired', '--grasp-point': 'network',
                               '--contact-frame': 'open', '--viz': 'kit', '--episodes': '1',
                               '--segmentation-backend': 'sam', '--seed': '42', '--collision-check': 'off'}.items():
            self.assertEqual(argv[argv.index(flag) + 1], expected)

    def test_runner_records_raw_language_and_passes_it_to_simulator_on_gpu0(self):
        cache = ROOT / 'cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / 'run'
            with patch('small_grasp.run_sim_evaluation.subprocess.run') as launch, redirect_stdout(io.StringIO()):
                evaluate(['--scenes', 'multi', '--episodes', '1', '--capture-only',
                          '--instruction', '抓橘子', '--seed', '100', '--segmentation-backend', 'sam',
                          '--controller', 'servo', '--inference-seed', '43',
                          '--arm-tracking-feedback',
                          '--closure-latch-policy', 'no_reopen',
                          '--close-tracking-integral',
                          '--close-integral-transfer', 'adjacent',
                          '--candidate-contact-ranking',
                          '--candidate-contact-policy', 'band_first',
                          '--wrist-camera-profile', 'open-jaws', '--network-frame', 'legacy-optical', '--candidate-pool-size', '20',
                          '--width-scale', '1',
                          '--diagnostic-contacts',
                          '--live-debug', '--live-debug-port', '0', '--live-debug-interval', '2',
                          '--collision-check', 'capsule', '--output-dir', str(output)])
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index('--instruction') + 1], '抓橘子')
            self.assertEqual(command[command.index('--seed') + 1], '100')
            self.assertEqual(command[command.index('--inference-seed') + 1], '43')
            self.assertEqual(command[command.index('--width-scale') + 1], '1.0')
            self.assertEqual(command[command.index('--candidate-pool-size') + 1], '20')
            self.assertEqual(command[command.index('--network-frame') + 1], 'legacy-optical')
            record = json.loads((output / 'run.json').read_text())
            self.assertEqual(record['network_frame'], 'legacy-optical')
            self.assertTrue(record['arm_tracking_feedback'])
            self.assertIn('--arm-tracking-feedback', command)
            self.assertEqual(record['closure_latch_policy'], 'no_reopen')
            self.assertTrue(record['close_tracking_integral'])
            self.assertIn('--close-tracking-integral', command)
            self.assertEqual(record['close_integral_transfer'], 'adjacent')
            self.assertEqual(command[command.index('--close-integral-transfer')+1], 'adjacent')
            self.assertEqual(command[command.index('--closure-latch-policy') + 1], 'no_reopen')
            self.assertTrue(record['candidate_contact_ranking'])
            self.assertIn('--candidate-contact-ranking',command)
            self.assertEqual(record['candidate_contact_policy'],'band_first')
            self.assertEqual(command[command.index('--candidate-contact-policy')+1],'band_first')
            self.assertEqual(record['network_input_frame']['wrist_camera_profile'], 'open-jaws')
            self.assertEqual(command[command.index('--controller') + 1], 'servo')
            self.assertIn('--diagnostic-contacts', command)
            self.assertIn('--live-debug', command)
            self.assertEqual(command[command.index('--live-debug-port') + 1], '0')
            self.assertEqual(json.loads((output / 'run.json').read_text())['live_debug']['interval_s'], 2.)
            self.assertEqual(command[command.index('--segmentation-backend') + 1], 'sam')
            self.assertEqual(command[command.index('--collision-check') + 1], 'capsule')
            self.assertEqual(json.loads((output / 'run.json').read_text())['seeds'], [100])
            self.assertEqual(json.loads((output / 'run.json').read_text())['inference_seed'], 43)
            self.assertEqual(json.loads((output / 'run.json').read_text())['width_scale'], 1.)
            self.assertEqual(json.loads((output / 'run.json').read_text())['candidate_pool_size'], 20)
            self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
            self.assertEqual(json.loads((output / 'run.json').read_text())['language']['instruction'], '抓橘子')
            rejected = Path(temporary) / 'unsupported'
            with patch('small_grasp.run_sim_evaluation.subprocess.run') as launch, \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                evaluate(['--instruction', 'duck', '--output-dir', str(rejected)])
            self.assertEqual(error.exception.code, 2)
            launch.assert_not_called()
            self.assertFalse(rejected.exists())
            # Preserve direct-script usage without PYTHONPATH or cwd imports.
            completed = subprocess.run([sys.executable, '-B', str(ROOT / 'run_sim_evaluation.py'),
                 '--instruction', 'duck', '--output-dir', str(rejected)], cwd=temporary,
                 env={**os.environ, 'PYTHONPATH': ''}, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            self.assertIn('currently supports only one orange', completed.stderr)
            self.assertFalse(rejected.exists())

    def test_invalid_combinations_fail_before_creating_output_or_launching(self):
        cache = ROOT / 'cache'
        cache.mkdir(exist_ok=True)
        for flags in (['--segmentation-backend', 'sam', '--grasp-point', 'sphere'],
                      ['--controller', 'servo', '--grasp-point', 'sphere'],
                      ['--controller', 'servo', '--mask-backend', 'gt'],
                      ['--arm-tracking-feedback', '--controller', 'repaired'],
                      ['--closure-latch-policy', 'no_reopen'],
                      ['--close-tracking-integral'],
                      ['--close-integral-transfer', 'adjacent'],
                      ['--close-integral-transfer', 'adjacent', '--controller', 'servo', '--arm-tracking-feedback'],
                      ['--close-integral-transfer', 'adjacent', '--close-tracking-integral', '--controller', 'repaired'],
                      ['--close-integral-transfer', 'unknown'],
                      ['--close-frame-policy', 'planned-line'],
                      ['--close-frame-policy', 'planned-line', '--controller', 'servo'],
                      ['--close-frame-policy', 'planned-line', '--controller', 'servo', '--arm-tracking-feedback', '--close-tracking-integral'],
                      ['--close-frame-policy', 'planned-line', '--controller', 'servo', '--arm-tracking-feedback', '--contact-frame', 'contact'],
                      ['--close-frame-policy', 'planned-line', '--controller', 'servo', '--arm-tracking-feedback', '--closure-latch-policy', 'no_reopen'],
                      ['--close-frame-policy', 'planned-line', '--controller', 'servo', '--arm-tracking-feedback', '--close-tracking-integral', '--close-integral-transfer', 'adjacent'],
                      ['--close-frame-policy', 'reachable-line'],
                      ['--close-frame-policy', 'reachable-line', '--controller', 'servo', '--arm-tracking-feedback', '--close-tracking-integral'],
                      ['--close-frame-policy', 'reachable-line', '--controller', 'servo', '--arm-tracking-feedback', '--contact-frame', 'contact'],
                      ['--close-frame-policy', 'reachable-line', '--controller', 'servo', '--arm-tracking-feedback', '--closure-latch-policy', 'no_reopen'],
                      ['--close-tracking-integral', '--controller', 'servo'],
                      ['--closure-latch-policy', 'no_reopen', '--controller', 'servo'],
                      ['--closure-latch-policy', 'no_reopen', '--arm-tracking-feedback', '--controller', 'repaired'],
                      ['--candidate-contact-ranking', '--controller', 'repaired'],
                      ['--candidate-contact-policy', 'band_first'],
                      ['--inference-seed', '-1'],
                      ['--candidate-pool-size', '0'],
                      ['--network-frame', 'unknown'],
                      ['--width-scale', 'nan'],
                      ['--width-scale', '0'],
                      ['--live-debug', '--live-debug-port', '-1'],
                      ['--live-debug', '--live-debug-interval', 'nan'],
                      ['--collision-check', 'capsule', '--controller', 'baseline'],
                      ['--segmentation-backend', 'sam', '--mask-backend', 'gt'],
                      ['--segmentation-backend', 'sam', '--segmentation-model', str(cache / 'absent_sam')]):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory(dir=cache) as temporary:
                output = Path(temporary) / 'invalid'
                with patch('small_grasp.run_sim_evaluation.subprocess.run') as launch, \
                     redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    evaluate(flags + ['--output-dir', str(output)])
                launch.assert_not_called()
                self.assertFalse(output.exists())

    def test_unified_entry_forwards_explicit_tracking_feedback(self):
        from small_grasp.pick import main
        with patch('sys.argv', ['small_grasp.pick', '--instruction', 'orange', '--scene', 'single',
                               '--controller', 'servo', '--arm-tracking-feedback', '--candidate-contact-ranking',
                               '--closure-latch-policy', 'no_reopen',
                               '--close-tracking-integral',
                               '--close-integral-transfer', 'adjacent',
                               '--candidate-contact-policy', 'band_first']), \
             patch('small_grasp.run_sim_evaluation.main') as runner, redirect_stdout(io.StringIO()):
            main()
        self.assertIn('--arm-tracking-feedback', runner.call_args.args[0])
        self.assertIn('--close-tracking-integral', runner.call_args.args[0])
        self.assertEqual(runner.call_args.args[0][runner.call_args.args[0].index('--close-integral-transfer')+1], 'adjacent')
        self.assertEqual(runner.call_args.args[0][runner.call_args.args[0].index('--closure-latch-policy')+1], 'no_reopen')
        self.assertIn('--candidate-contact-ranking',runner.call_args.args[0])
        self.assertEqual(runner.call_args.args[0][runner.call_args.args[0].index('--candidate-contact-policy')+1],'band_first')

    def test_planned_line_entry_and_runner_preserve_single_mechanism_manifest(self):
        from small_grasp.pick import main
        with patch('sys.argv',['pick','--instruction','orange','--controller','servo','--arm-tracking-feedback',
                               '--close-frame-policy','planned-line']), \
             patch('small_grasp.run_sim_evaluation.main') as runner,redirect_stdout(io.StringIO()):
            main()
        argv=runner.call_args.args[0]
        self.assertEqual(argv[argv.index('--close-frame-policy')+1],'planned-line')
        self.assertNotIn('--close-tracking-integral',argv)
        self.assertNotIn('--close-integral-transfer',argv)
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as temp:
            output=Path(temp)/'planned_line'
            with patch('small_grasp.run_sim_evaluation.subprocess.run') as launch,redirect_stdout(io.StringIO()):
                evaluate(['--capture-only','--episodes','1','--controller','servo','--arm-tracking-feedback',
                          '--close-frame-policy','planned-line','--output-dir',str(output)])
            record=json.loads((output/'run.json').read_text());command=launch.call_args.args[0]
            self.assertEqual(record['close_frame_policy'],'planned-line')
            self.assertFalse(record['close_tracking_integral'])
            self.assertEqual(record['close_integral_transfer'],'reset')
            self.assertEqual(record['closure_latch_policy'],'legacy')
            self.assertEqual(command[command.index('--close-frame-policy')+1],'planned-line')

    def test_reachable_line_entry_and_runner_keep_integral_off_and_default_mechanisms(self):
        from small_grasp.pick import main
        with patch('sys.argv',['pick','--instruction','orange','--controller','servo','--arm-tracking-feedback',
                               '--close-frame-policy','reachable-line']), \
             patch('small_grasp.run_sim_evaluation.main') as runner,redirect_stdout(io.StringIO()):
            main()
        argv=runner.call_args.args[0]
        self.assertEqual(argv[argv.index('--close-frame-policy')+1],'reachable-line')
        self.assertNotIn('--close-tracking-integral',argv)
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as temp:
            output=Path(temp)/'reachable_line'
            with patch('small_grasp.run_sim_evaluation.subprocess.run') as launch,redirect_stdout(io.StringIO()):
                evaluate(['--capture-only','--episodes','1','--controller','servo','--arm-tracking-feedback',
                          '--close-frame-policy','reachable-line','--output-dir',str(output)])
            record=json.loads((output/'run.json').read_text());command=launch.call_args.args[0]
            self.assertEqual(record['close_frame_policy'],'reachable-line')
            self.assertFalse(record['close_tracking_integral'])
            self.assertEqual(record['close_integral_transfer'],'reset')
            self.assertEqual(record['closure_latch_policy'],'legacy')
            self.assertEqual(command[command.index('--close-frame-policy')+1],'reachable-line')
