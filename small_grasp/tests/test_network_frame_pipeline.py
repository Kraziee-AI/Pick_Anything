"""CPU integration: input basis, inverse poses and real-camera filter context."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import torch

from small_grasp.infer import ROOT, parse_args, run
from small_grasp.network_frame import network_from_camera


class NetworkFramePipelineTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'cache').mkdir(parents=True, exist_ok=True)

    def make_args(self, folder, flags):
        Image.new('RGB', (10, 10), (80, 90, 100)).save(folder / 'rgb.png')
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:8, 3:8] = 255
        Image.fromarray(mask).save(folder / 'mask.png')
        np.save(folder / 'depth.npy', np.ones((10, 10), dtype=np.float32))
        (folder / 'intrinsics.json').write_text(json.dumps(dict(width=10, height=10, fx=10, fy=10, cx=5, cy=5, depth_scale=1)))
        world = np.eye(4)
        world[:3, 3] = [.4, -.2, .1]
        (folder / 'world.json').write_text(json.dumps(world.tolist()))
        (folder / 'checkpoint.tar').write_bytes(b'mocked checkpoint')
        argv = ['infer', '--rgb', str(folder / 'rgb.png'), '--depth', str(folder / 'depth.npy'),
                '--intrinsics', str(folder / 'intrinsics.json'), '--mask', str(folder / 'mask.png'),
                '--world-from-camera', str(folder / 'world.json'),
                '--checkpoint', str(folder / 'checkpoint.tar'), '--device', 'cpu', '--num-points', '2048',
                '--network-region', 'target-mask', '--output-dir', str(folder / 'out'), *flags]
        with patch('sys.argv', argv):
            return parse_args()

    def test_input_and_full_pose_inverse_precede_real_camera_filters(self):
        for frame in ['camera', 'legacy-optical']:
            with self.subTest(frame=frame), tempfile.TemporaryDirectory(dir=ROOT / 'cache') as tmp:
                folder = Path(tmp)
                args = self.make_args(folder, ['--network-frame', frame, '--wrist-camera-profile', 'open-jaws'])
                (folder / 'camera_settings.json').write_text(json.dumps({'wrist_camera_profile': 'open-jaws'}))
                c = network_from_camera('open-jaws', frame)
                camera_rows = np.zeros((1, 17), dtype=np.float32)
                camera_rows[0, :4] = [.8, .05, .02, .03]
                camera_rows[0, 4:13] = np.eye(3).ravel()
                camera_rows[0, 13:16] = [0., 0., 1.]
                camera_rows[0, 16] = -1
                model_rows = camera_rows.copy()
                model_rows[:, 13:16] = camera_rows[:, 13:16] @ c.T
                model_rows[:, 4:13] = (c @ camera_rows[:, 4:13].reshape(-1, 3, 3)).reshape(-1, 9)
                widths = {key: torch.tensor([value]) for key, value in
                          [('width_unscaled_m', .005), ('width_scale', 10.), ('width_preclamp_m', .05), ('width_saturated', False)]}
                net = MagicMock()
                net.to.return_value.eval.return_value = net
                net.parameters.return_value = [torch.zeros(1)]
                detector = MagicMock()
                detector.detect.return_value = (np.array([False]), np.array([False]))
                with patch('small_grasp.vendor.models.graspnet.GraspNet', return_value=net), \
                     patch('small_grasp.vendor.models.graspnet.pred_decode', return_value=([torch.from_numpy(model_rows)], [widths])), \
                     patch('small_grasp.vendor.utils.collision_detector.ModelFreeCollisionDetector', return_value=detector) as collision_init, \
                     patch('torch.load', return_value={'model_state_dict': {}, 'epoch': 18}), redirect_stdout(io.StringIO()):
                    run(args)
                report = json.loads((folder / 'out/result.json').read_text())
                self.assertEqual(report['counts']['exported'], 1)  # Fails if target projection sees model axes.
                self.assertEqual(report['network_input']['profile_source'], 'capture_settings_verified_against_cli')
                with np.load(folder / 'out/point_clouds.npz') as clouds:
                    real = clouds['network_crop_xyz_camera_m']
                    np.testing.assert_allclose(clouds['network_crop_xyz_model_m'], real @ c.T, atol=1e-7)
                    sample = clouds['sampled_xyz_model_m']
                    np.testing.assert_array_equal(net.call_args.args[0]['point_clouds'].numpy()[0], sample)
                    np.testing.assert_array_equal(sample, clouds['network_crop_xyz_model_m'][clouds['sampled_network_indices']])
                    np.testing.assert_array_equal(collision_init.call_args.args[0], clouds['scene_xyz_camera_m'])
                    self.assertEqual(len(clouds['scene_xyz_camera_m']), 100)
                    self.assertEqual(len(real), 25)
                np.testing.assert_allclose(detector.detect.call_args.args[0].translations, camera_rows[:, 13:16], atol=1e-7)
                np.testing.assert_allclose(detector.detect.call_args.args[0].rotation_matrices, np.eye(3)[None], atol=1e-7)
                with np.load(folder / 'out/raw_candidates.npz') as raw:
                    np.testing.assert_array_equal(raw['grasps_model'], model_rows)
                    np.testing.assert_allclose(raw['grasps'], camera_rows, atol=1e-7)
                    if frame == 'camera':
                        np.testing.assert_array_equal(raw['grasps'], model_rows)
                pose = np.array(report['candidates'][0]['T_world_grasp'])
                np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-7)
                np.testing.assert_allclose(pose[:3, 3], [.4, -.2, 1.1], atol=1e-7)

    def test_unknown_mount_and_capture_profile_mismatch_fail_without_forward(self):
        with patch('sys.argv', ['infer', '--demo', '--network-frame', 'legacy-optical']), \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args()
        with tempfile.TemporaryDirectory(dir=ROOT / 'cache') as tmp:
            folder = Path(tmp)
            args = self.make_args(folder, ['--network-frame', 'legacy-optical', '--wrist-camera-profile', 'open-jaws'])
            (folder / 'camera_settings.json').write_text(json.dumps({'wrist_camera_profile': 'legacy'}))
            with patch('small_grasp.vendor.models.graspnet.GraspNet') as model, \
                 self.assertRaisesRegex(ValueError, 'camera profile disagrees'):
                run(args)
            model.assert_not_called()
            self.assertFalse((folder / 'out').exists())
