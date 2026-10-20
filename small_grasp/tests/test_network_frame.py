import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from small_grasp.network_frame import network_from_camera, grasps_to_camera, frame_metadata


class NetworkFrameTests(unittest.TestCase):
    def test_mount_semantics_match_usd_local_post_rotations_and_ros_basis(self):
        # Independent composition mirrors old*local_yaw*local_pitch in USD.
        old_usd = Rotation.from_euler('xyz', [17, -43, 22], degrees=True).as_matrix()
        delta = Rotation.from_euler('Y', -30, degrees=True).as_matrix() @ Rotation.from_euler('X', 8, degrees=True).as_matrix()
        b = np.diag([1., -1., -1.])
        c = network_from_camera('open-jaws', 'legacy-optical')
        old_ros = old_usd @ b
        actual_ros = old_usd @ delta @ b
        np.testing.assert_allclose(old_ros @ c, actual_ros, atol=1e-12)
        points = np.array([[.03, -.02, .3], [-.11, .2, .7]])
        np.testing.assert_allclose((points @ c.T) @ old_ros.T, points @ actual_ros.T, atol=1e-12)
        with self.assertRaises(ValueError):
            frame_metadata(None, 'legacy-optical')
        np.testing.assert_array_equal(frame_metadata(None, 'camera')['model_from_camera'], np.eye(3))

    def test_world_pose_invariant_and_grasp_semantics_preserved(self):
        c = network_from_camera('open-jaws', 'legacy-optical')
        world_actual = Rotation.from_euler('xyz', [22, -34, 17], degrees=True).as_matrix()
        model_rotation = Rotation.from_euler('xyz', [51, 11, -9], degrees=True).as_matrix()
        rows = np.zeros((1, 17))
        rows[0, :4] = [.42, .06, .02, .03]
        rows[0, 4:13] = model_rotation.ravel()
        rows[0, 13:16] = [.07, -.13, .4]
        rows[0, 16] = -1
        before = rows.copy()
        actual = grasps_to_camera(rows, c)
        np.testing.assert_allclose(world_actual @ actual[0, 4:13].reshape(3, 3),
                                   (world_actual @ c.T) @ model_rotation, atol=1e-12)
        np.testing.assert_allclose(world_actual @ actual[0, 13:16],
                                   (world_actual @ c.T) @ rows[0, 13:16], atol=1e-12)
        np.testing.assert_array_equal(actual[:, :4], rows[:, :4])
        np.testing.assert_array_equal(actual[:, 16], rows[:, 16])
        np.testing.assert_array_equal(rows, before)
        np.testing.assert_allclose(grasps_to_camera(actual, c.T), rows, atol=1e-12)

    def test_default_identity_empty_and_invalid_transforms(self):
        for profile, frame in [('legacy', 'camera'), ('legacy', 'legacy-optical'), ('open-jaws', 'camera')]:
            np.testing.assert_array_equal(network_from_camera(profile, frame), np.eye(3))
        self.assertEqual(grasps_to_camera(np.empty((0, 17)), np.eye(3)).shape, (0, 17))
        for bad in [np.zeros((3, 3)), np.diag([1., 1., -1.]), np.full((3, 3), np.nan)]:
            with self.assertRaises(ValueError):
                grasps_to_camera(np.empty((0, 17)), bad)
