import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from small_grasp.arm_collision import (Capsule, add_finger_capsules, capsules_from_bodies, check_capsules,
                                       joint_samples, load_capture_joint_positions, load_observed_cloud,
                                       rejection_counts, remove_robot_returns, width_rejection)


class ArmCollisionTests(unittest.TestCase):
    def setUp(self):
        self.bodies = {name: np.array([0., i * .13, .25]) for i, name in enumerate(
            ('base', 'shoulder', 'upper_arm', 'lower_arm', 'wrist', 'gripper', 'jaw'))}
        self.links = capsules_from_bodies(self.bodies)

    def test_observed_point_and_target_contact_are_distinct(self):
        point = np.array([[0., .39, .25]])
        self.assertEqual(check_capsules(self.links, point, np.empty((0, 3)))['reason'], 'observed_obstacle')
        self.assertEqual(check_capsules(self.links, np.empty((0, 3)), point)['reason'],
                         'target_contact_outside_gripper')
        at_jaw = np.array([[0., .71, .25]])
        self.assertIsNone(check_capsules(self.links, np.empty((0, 3)), at_jaw))

    def test_table_and_nonadjacent_self_collision(self):
        lowered = [*self.links]
        lowered[3] = Capsule(lowered[3].name, np.array([0., -.39, .05]),
                             np.array([0., -.52, .05]), .027)
        self.assertEqual(check_capsules(lowered, np.empty((0, 3)), np.empty((0, 3)),
                                        table_z=.041)['reason'], 'table')
        outside = [*self.links]
        outside[3] = Capsule(outside[3].name, np.array([0., .39, .05]),
                             np.array([0., .52, .05]), .027)
        self.assertIsNone(check_capsules(outside, np.empty((0, 3)), np.empty((0, 3))))
        folded = [*self.links]
        folded[4] = Capsule(folded[4].name, np.array([0., .13, .25]),
                            np.array([0., .26, .25]), .028)
        self.assertEqual(check_capsules(folded, np.empty((0, 3)), np.empty((0, 3)))['reason'], 'self')

    def test_joint_interpolation_and_robot_return_removal(self):
        samples = joint_samples(np.zeros(6), np.array([.17, 0, 0, 0, 0, 0]), [0])
        self.assertEqual(len(samples), 4)
        self.assertTrue(np.allclose(samples[-1], [.17, 0, 0, 0, 0, 0]))
        points = np.array([[0., .39, .25], [.5, .39, .25]])
        clean = remove_robot_returns(points, self.links)
        self.assertEqual(len(clean), 1)
        self.assertTrue(np.allclose(clean[0], points[1]))

    def test_finger_offsets_follow_body_orientation(self):
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_euler('z', 90, degrees=True).as_quat()
        poses = {name: np.r_[position, rotation] for name, position in self.bodies.items()}
        fingers = add_finger_capsules([*self.links], poses)[-2:]
        self.assertEqual([finger.name for finger in fingers], ['gripper:fixed_pad', 'jaw:moving_pad'])
        expected = self.bodies['gripper'] + Rotation.from_quat(rotation).apply([-.01289, -.00022, -.0986])
        self.assertTrue(np.allclose(fingers[0].end, expected))

    def test_saved_rgbd_has_target_and_non_target_points(self):
        path = Path('small_grasp/outputs/20261015_repair_final_eval10/single/trial_00_seed42')
        if not path.exists():
            self.skipTest('Historical saved RGB-D absent')
        obstacles, target = load_observed_cloud(path)
        self.assertGreater(len(obstacles), 0)
        self.assertGreater(len(target), 0)
        self.assertTrue(np.isfinite(obstacles).all())
        self.assertTrue(np.isfinite(target).all())

    def test_capture_robot_joint_file_precedes_legacy_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            view = Path(temporary)
            (view / 'evaluation_state.json').write_text(json.dumps({
                'joint_position': [9, 9], 'objects': {'orange': [1, 2, 3]}}))
            (view / 'robot_joint_positions.json').write_text(json.dumps({
                'joint_names': ['gripper', 'shoulder_pan'], 'joint_position': [1.4, .2]}))
            q, source = load_capture_joint_positions(view, ['shoulder_pan', 'gripper'])
            self.assertEqual(source, 'robot_joint_positions.json')
            self.assertTrue(np.allclose(q, [.2, 1.4]))
            (view / 'robot_joint_positions.json').unlink()
            q, source = load_capture_joint_positions(view, ['shoulder_pan', 'gripper'])
            self.assertEqual(source, 'evaluation_state.json:joint_position')
            self.assertTrue(np.allclose(q, [9, 9]))

    def test_invalid_named_capture_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            view = Path(temporary)
            (view / 'evaluation_state.json').write_text(json.dumps({'joint_position': [.1, .2]}))
            (view / 'robot_joint_positions.json').write_text(json.dumps({
                'joint_names': ['shoulder_pan', 'wrong'], 'joint_position': [.1, .2]}))
            with self.assertRaisesRegex(ValueError, 'do not match'):
                load_capture_joint_positions(view, ['shoulder_pan', 'gripper'])

    def test_capture_above_soft_limit_uses_measured_fk_without_clamping(self):
        import torch
        from small_grasp.sim_robot import GraspExecutor

        names = ['base', 'shoulder', 'upper_arm', 'lower_arm', 'wrist', 'gripper', 'jaw']
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        capture = np.array([0., 0., 0., 0., 0., 1.7449817657470703])
        poses = {name: np.array([0., 0., .3, 0., 0., 0., 1.]) for name in names}
        executor = GraspExecutor.__new__(GraspExecutor)
        executor.model = SimpleNamespace(body_label=names, state=lambda: None)
        executor.robot = SimpleNamespace(joint_names=joint_names,
            find_bodies=lambda requested: ([names.index(requested[0])], None),
            data=SimpleNamespace(joint_pos=torch.zeros((1, 6)),
                body_pos_w=torch.tensor([[[0., 0., .3]] * 7]),
                body_quat_w=torch.tensor([[[0., 0., 0., 1.]] * 7]),
                soft_joint_pos_limits=torch.tensor([[[-1.5, 1.5]] * 6])))
        executor.jaw = 6
        executor.observation_dir = Path('unused')
        executor.collision_evidence = {}
        executor._fk_body_poses = Mock(return_value=poses)
        executor._fk_capsules = Mock(return_value=[])
        with patch('small_grasp.arm_collision.load_capture_joint_positions',
                   return_value=(capture, 'robot_joint_positions.json')), \
             patch('small_grasp.arm_collision.load_observed_cloud',
                   return_value=(np.empty((0, 3)), np.array([[0., 0., .1]]))):
            executor._init_collision()
        np.testing.assert_array_equal(executor._fk_capsules.call_args.args[0], capture)
        self.assertAlmostEqual(executor.collision_evidence['capture_soft_limit_excess_rad']['gripper'],
                               capture[-1] - 1.5)
        self.assertEqual(executor.collision_evidence['capture_robot_state_source'], 'robot_joint_positions.json')

    def test_all_rejected_reasons_remain_countable(self):
        attempts = [{'candidate_id': 'a', 'rejected': 'grasp_ik'},
                    {'candidate_id': 'a', 'rejected': 'arm_collision'},
                    {'candidate_id': 'b', 'rejected': 'arm_collision'}]
        self.assertEqual(rejection_counts(attempts), {'grasp_ik': 1, 'arm_collision': 2})

    def test_width_validation_does_not_equate_reference_spans_to_inner_gap(self):
        for invalid in (None, 0., -.01, float('nan'), float('inf')):
            self.assertEqual(width_rejection(invalid), 'invalid_candidate_width')
        for width in (.005, .056, .2):
            self.assertIsNone(width_rejection(width))


if __name__ == '__main__':
    unittest.main()
