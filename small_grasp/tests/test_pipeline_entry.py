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
                               '--viz', 'kit', '--output-dir', str(output)]), \
             patch('small_grasp.run_sim_evaluation.main') as runner, redirect_stdout(io.StringIO()):
            main()
        argv = runner.call_args.args[0]
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
                          '--wrist-camera-profile', 'open-jaws', '--candidate-pool-size', '20',
                          '--width-scale', '1',
                          '--diagnostic-contacts',
                          '--collision-check', 'capsule', '--output-dir', str(output)])
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index('--instruction') + 1], '抓橘子')
            self.assertEqual(command[command.index('--seed') + 1], '100')
            self.assertEqual(command[command.index('--inference-seed') + 1], '43')
            self.assertEqual(command[command.index('--width-scale') + 1], '1.0')
            self.assertEqual(command[command.index('--candidate-pool-size') + 1], '20')
            self.assertEqual(command[command.index('--controller') + 1], 'servo')
            self.assertIn('--diagnostic-contacts', command)
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
                      ['--inference-seed', '-1'],
                      ['--candidate-pool-size', '0'],
                      ['--width-scale', 'nan'],
                      ['--width-scale', '0'],
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
