"""Exercise the real local encoder and recording time/terminal-frame contract."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from small_grasp.video import EpisodeVideo, VIDEO_HEIGHT, VIDEO_WIDTH


class VideoTests(unittest.TestCase):
    def test_real_h264_sampling_terminal_and_result_hold(self):
        with tempfile.TemporaryDirectory() as directory:
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
            probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                            '-show_entries', 'stream=codec_name,width,height,nb_frames',
                                            '-of', 'json', str(path)], text=True)
            stream = json.loads(probe)['streams'][0]
            self.assertEqual(stream['codec_name'], 'h264')
            self.assertEqual((stream['width'], stream['height']), (1280, 720))
            self.assertEqual(int(stream['nb_frames']), 18)
            self.assertEqual(video.finish(), metadata)

    def test_no_overwrite_and_invalid_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'existing.mp4'
            path.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                EpisodeVideo(path, None, 1 / 30, 1, 1, 1)
            self.assertEqual(path.read_bytes(), b'keep')
            for fps in (0, -1, float('nan'), 60):
                with self.assertRaises(ValueError):
                    EpisodeVideo(Path(directory) / 'bad.mp4', None, 1 / 30, 1, 1, 1, fps)


if __name__ == '__main__':
    unittest.main()
