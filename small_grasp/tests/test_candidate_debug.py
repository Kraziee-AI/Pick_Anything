"""Pose projection and source/selection identity, independent of model inference."""
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from PIL import Image

from small_grasp.candidate_debug import CandidateSnapshot, execution_summary, project
from small_grasp.live_debug import LiveDebug, State, refresh_candidates, handler_for


class CandidateDebugTests(unittest.TestCase):
    def fixture(self, root):
        proposals = root / 'proposals'
        proposals.mkdir()
        Image.new('RGB', (160, 120), '#545454').save(proposals / 'rgb.png')
        Image.new('L', (160, 120), 255).save(proposals / 'target_mask.png')
        intrinsic = {'fx': 200., 'fy': 200., 'cx': 80., 'cy': 60.}
        (root / 'intrinsics.json').write_text(json.dumps(intrinsic))
        candidates = []
        for i, rotation in enumerate((np.eye(3), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))):
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = [0, 0, .5]
            candidates.append({'candidate_id': f'g{i:06d}', 'rank': i, 'score': .9 - .1*i,
                               'width_m': .04, 'depth_m': .03, 'T_camera_grasp': pose.tolist()})
        (proposals / 'result.json').write_text(json.dumps({'candidates': candidates}))
        spec = {'proposals_dir': str(proposals), 'intrinsics_path': str(root / 'intrinsics.json'),
                'trial': 1, 'execution': execution_summary(candidates)}
        return candidates, spec, intrinsic

    def test_projection_and_columns_show_actual_grasp_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates, spec, intrinsic = self.fixture(Path(tmp))
            snapshot = CandidateSnapshot(spec)
            np.testing.assert_allclose(project([.03, 0, .5], intrinsic), [92, 60])
            np.testing.assert_allclose(project([0, .03, .5], intrinsic), [80, 72])
            self.assertIsNone(project([0, 0, -.1], intrinsic))
            self.assertIsNone(project([0, float('nan'), .5], intrinsic))
            second = snapshot.data()['candidates'][1]
            self.assertEqual(second['approach_camera'], [0, 1, 0])
            self.assertEqual(second['closing_camera'], [-1, 0, 0])
            self.assertEqual(snapshot.render(1).size, (1280, 800))
            self.assertEqual(Image.open(io.BytesIO(snapshot.jpeg(1))).size, (1280, 800))
            with self.assertRaises(ValueError): snapshot.render(-1)

    def test_accepted_attempt_is_not_marked_rejected_and_selected_is_not_rank_one(self):
        candidates = [{'candidate_id': 'a'}, {'candidate_id': 'b'}, {'candidate_id': 'c'}]
        result = execution_summary(candidates, {'robot_executed': True, 'chosen': {'candidate_id': 'b'},
                   'ik_attempts': [{'candidate': 0, 'rejected': 'ik'}, {'candidate': 0, 'accepted': True},
                                   {'candidate': 1, 'rejected': 'collision'}]})
        self.assertEqual(result['candidates']['a']['status'], 'plan_accepted')
        self.assertEqual(result['candidates']['b']['status'], 'selected')
        self.assertEqual(result['candidates']['c']['status'], 'not_checked')

    def test_publish_source_snapshot_http_selection_and_reset_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates, spec, intrinsic = self.fixture(root)
            viewer = object.__new__(LiveDebug)
            viewer.directory = root
            viewer.publish_candidates(root / 'proposals', root / 'intrinsics.json', 1, candidates,
                                      {'chosen': {'candidate_id': candidates[1]['candidate_id']}, 'robot_executed': True})
            state = State()
            state.info['trial'] = 2
            self.assertIsNone(refresh_candidates(state, root, None))
            self.assertIsNone(state.candidate_snapshot)
            state.info['trial'] = 1
            stamp = refresh_candidates(state, root, None)
            self.assertEqual(state.candidate_snapshot.default, 1)
            self.assertEqual(refresh_candidates(state, root, stamp), stamp)
            self.assertTrue((root / 'last_candidates.png').is_file())
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(state))
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                url = f'http://127.0.0.1:{server.server_port}'
                data = json.load(urlopen(url + '/candidate-data'))
                self.assertEqual(data['default_index'], 1)
                image = Image.open(io.BytesIO(urlopen(url + '/frame/candidates?candidate=0').read()))
                self.assertEqual(image.size, (1280, 800))
                with self.assertRaises(HTTPError) as error:
                    urlopen(url + '/frame/candidates?candidate=-1')
                self.assertEqual(error.exception.code, 400)
            finally:
                server.shutdown(); server.server_close(); thread.join(3)


if __name__ == '__main__':
    unittest.main()
