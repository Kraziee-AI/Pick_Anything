"""Exercise the real local encoder and recording time/terminal-frame contract."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from small_grasp.video import EpisodeVideo, VIDEO_HEIGHT, VIDEO_WIDTH


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.cache = Path(__file__).resolve().parents[1] / 'cache' / 'video_tests'
        self.cache.mkdir(parents=True, exist_ok=True)

    def test_real_h264_sampling_terminal_and_result_hold(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as directory:
            path = Path(directory) / 'trial.mp4'
            frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
            frame[:, :VIDEO_WIDTH // 2, 1] = 180
            wrist = np.full((192, 256, 3), 120, dtype=np.uint8)
            calls = []
            def supply():
                calls.append(True)
                return frame, wrist
            video = EpisodeVideo(path, supply, 1 / 30, 1, 30, 500)
            for _ in range(3):
                video.after_step()
            video.after_step(terminal=True)  # Step4 must be recorded despite stride2.
            metadata = video.finish({'grasp_lift_success': True, 'success': False,
                                     'failure_stage': 'placement_incomplete'})
            self.assertEqual(len(calls), 3)
            self.assertEqual(metadata['frames'], 18)
            self.assertEqual(metadata['control_steps'], 4)
            self.assertEqual(metadata['sample_stride'], 2)
            self.assertTrue(metadata['completed'])
            self.assertNotIn('target_label', metadata)
            self.assertNotIn('score_schema', metadata)
            self.assertEqual(video._default_status(), 'DINO + SAM + RGB-D + GraspNet | single-orange simulation')
            self.assertEqual(video._result_status({'grasp_lift_success': True, 'success': False,
                                                   'failure_stage': 'placement_incomplete'}),
                             'GRASP: SUCCESS | TASK: FAIL | placement_incomplete')
            probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                            '-show_entries', 'stream=codec_name,width,height,nb_frames',
                                            '-of', 'json', str(path)], text=True)
            stream = json.loads(probe)['streams'][0]
            self.assertEqual(stream['codec_name'], 'h264')
            self.assertEqual((stream['width'], stream['height']), (1280, 720))
            self.assertEqual(int(stream['nb_frames']), 18)
            self.assertEqual(video.finish(), metadata)

    def test_no_overwrite_and_invalid_fps(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as directory:
            path = Path(directory) / 'existing.mp4'
            path.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                EpisodeVideo(path, None, 1 / 30, 1, 1, 1)
            self.assertEqual(path.read_bytes(), b'keep')
            for fps in (0, -1, float('nan'), 60):
                with self.assertRaises(ValueError):
                    EpisodeVideo(Path(directory) / 'bad.mp4', None, 1 / 30, 1, 1, 1, fps)

    def test_generic_and_capture_context_draws_correct_text_and_encodes(self):
        from unittest.mock import patch
        from PIL import ImageDraw
        from small_grasp.object_scoring import SCHEMA
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        wrist = np.zeros((288, 384, 3), dtype=np.uint8)
        for capture_only in (False, True):
            with self.subTest(capture_only=capture_only), tempfile.TemporaryDirectory(dir=self.cache) as directory:
                path = Path(directory) / 'generic.mp4'
                video = EpisodeVideo(path, lambda: (frame, wrist), 1/30, 1, 1, 3900,
                    target_label='pear', score_schema=SCHEMA, lift_only=True, capture_only=capture_only)
                texts = []
                original = ImageDraw.ImageDraw.text
                def record(draw, xy, text, *args, **kwargs):
                    texts.append(text)
                    return original(draw, xy, text, *args, **kwargs)
                result = ({'failure_stage': 'capture_only'} if capture_only else
                          {'success': True, 'generic_grasp_lift_success': True, 'generic_end_hold_success': False})
                with patch.object(ImageDraw.ImageDraw, 'text', record):
                    video.after_step()
                    metadata = video.finish(result)
                self.assertIn('single-object | target: pear', metadata['default_status'])
                expected = ('pear | CAPTURE ONLY | NO GRASP ATTEMPTED | PLACE NOT EVALUATED' if capture_only else
                            'pear | GENERIC GRASP: SUCCESS | END HOLD: FAIL | PLACE NOT EVALUATED')
                self.assertIn(expected, texts)
                self.assertFalse(any('TASK:' in text or 'single-orange' in text for text in texts))
                self.assertEqual(metadata['result_status'], expected)
                self.assertEqual(metadata['score_schema'], SCHEMA)
                self.assertFalse(metadata['place_evaluated'])
                self.assertEqual(metadata['capture_only'], capture_only)
                self.assertEqual(json.loads(path.with_suffix('.json').read_text()), metadata)
                probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=nb_frames', '-of', 'json', str(path)], text=True)
                self.assertEqual(int(json.loads(probe)['streams'][0]['nb_frames']), metadata['frames'])
                subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'], check=True)


if __name__ == '__main__':
    unittest.main()
