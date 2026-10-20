"""Frame identity, HTTP transport and lifecycle of passive visualization."""
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from collections import deque
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np
from PIL import Image

from small_grasp.live_debug import (LiveDebug, State, overlays, process_frames,
                                    split_image, handler_for)
from http.server import ThreadingHTTPServer


class LiveDebugTests(unittest.TestCase):
    def test_overlay_uses_matching_frame_and_mask_without_modifying_rgb(self):
        rgb = np.full((50, 80, 3), 80, dtype=np.uint8)
        before = rgb.copy()
        mask = np.zeros((50, 80), bool)
        mask[20:35, 30:45] = True
        d = {'detected': True, 'box_xyxy': [25, 15, 50, 40], 'score': .7, 'label': 'orange'}
        images = overlays(rgb, d, mask)
        np.testing.assert_array_equal(rgb, before)
        np.testing.assert_array_equal(np.asarray(images['processed_raw']), rgb)
        self.assertEqual(int(np.asarray(images['mask'])[25, 35, 0]), 255)
        self.assertNotEqual(list(np.asarray(images['overlay'])[25, 35]), [80, 80, 80])
        self.assertEqual(list(np.asarray(images['grounding'])[15, 25]), [255, 226, 107])
        with self.assertRaises(ValueError):
            overlays(rgb, d, mask[:20])
        self.assertEqual(split_image(rgb, images['overlay'], 'test').size, (1280, 560))

    def test_async_result_keeps_source_identity_and_does_not_cross_reset(self):
        state = State()
        pending, condition, stop = deque(maxlen=1), threading.Condition(), threading.Event()
        started, release = threading.Event(), threading.Event()
        packet = {'wrist': np.full((30, 40, 3), 32, np.uint8),
                  'overview': np.full((60, 80, 3), 64, np.uint8),
                  'trial': 1, 'seed': 500, 'frame': 7, 'sim_time_s': 2.3, 'wall_time': time.time()}
        state.info['trial'] = 1

        def fake(rgb, *args):
            started.set()
            self.assertTrue(release.wait(3))
            return {'detected': False}, np.zeros(rgb.shape[:2], bool), {'backend': 'sam'}

        with tempfile.TemporaryDirectory() as tmp, patch('small_grasp.live_debug.diagnostic_perception', fake):
            pending.append(packet)
            thread = threading.Thread(target=process_frames, args=(state, pending, condition, stop,
                                      Path(tmp), 'orange', 'cpu', None, .01))
            thread.start()
            try:
                self.assertTrue(started.wait(3))
                # Simulation has reset while the old frame is still processing.
                state.info['trial'] = 2
                original = state.images['overlay']
                release.set()
                deadline = time.monotonic() + 3
                while not (Path(tmp) / 'last_perception.json').exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                saved = json.loads((Path(tmp) / 'last_perception.json').read_text())
                self.assertEqual((saved['trial'], saved['frame']), (1, 7))
                self.assertEqual(state.images['overlay'], original)
                self.assertNotIn('processed_frame', state.info)
                self.assertTrue((Path(tmp) / 'last_split.png').is_file())
            finally:
                stop.set()
                with condition:
                    condition.notify_all()
                thread.join(3)
                self.assertFalse(thread.is_alive())

    def test_spawned_loopback_server_and_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            viewer = LiveDebug(Path(tmp) / 'debug', 'orange', port=0)
            try:
                self.assertTrue(viewer.process.is_alive())
                self.assertTrue(viewer.url.startswith('http://127.0.0.1:'))
                html = urlopen(viewer.url, timeout=3).read().decode()
                self.assertIn('SAM', html)
                status = json.load(urlopen(viewer.url + '/status', timeout=3))
                self.assertEqual(status['message'], 'loading models on first frame')
                jpg = urlopen(viewer.url + '/frame/overlay', timeout=3).read()
                self.assertTrue(jpg.startswith(b'\xff\xd8'))
                with self.assertRaises(RuntimeError):
                    LiveDebug(Path(tmp) / 'conflict', 'orange', port=int(viewer.url.rsplit(':', 1)[1]))
            finally:
                viewer.close()
                viewer.close()
            self.assertFalse(viewer.process.is_alive())

    def test_display_backpressure_is_bounded_and_inactive_does_not_capture(self):
        viewer = object.__new__(LiveDebug)
        viewer.closed, viewer.last, viewer.period = False, 0, 0
        viewer.frames, viewer.dropped = 0, 0
        viewer.incoming = queue.Queue(maxsize=1)
        viewer.process = type('Alive', (), {'is_alive': lambda self: True})()
        calls = []
        def frames():
            calls.append(1)
            return np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8)
        viewer.offer(frames, 1, 500, 1.)
        viewer.offer(frames, 1, 500, 2.)
        self.assertEqual(viewer.incoming.qsize(), 1)
        self.assertEqual(viewer.dropped, 1)
        self.assertEqual(viewer.incoming.get()['sim_time_s'], 1.)
        viewer.closed = True
        viewer.offer(frames, 1, 500, 3.)
        self.assertEqual(len(calls), 2)
