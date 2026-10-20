"""Check the contact frame against SO101 USD joint kinematics, independently."""
from pathlib import Path
import unittest

import numpy as np
from pxr import Usd, UsdPhysics
from scipy.spatial.transform import Rotation

from small_grasp.contact_frame import contact_frame
from small_grasp.arm_collision import CLOSED_PAD_SPAN_M, OPEN_PAD_SPAN_M


class ContactFrameTests(unittest.TestCase):
    def test_contact_normal_tracks_closed_jaw_instead_of_open_jaw(self):
        root = Path(__file__).resolve().parents[2]
        stage = Usd.Stage.Open(str(root / 'assets/isaac_cache/Isaac/Robots_Multiphysics/RobotStudio/so101_new_calib_SysID/so101_new_calib.usda'))
        joint = next(UsdPhysics.RevoluteJoint(p) for p in stage.Traverse()
                     if p.GetName() == 'gripper' and p.IsA(UsdPhysics.RevoluteJoint))
        q = joint.GetLocalRot0Attr().Get()
        parent_r = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
        pivot = np.asarray(joint.GetLocalPos0Attr().Get())
        fixed = np.array([-.01289, -.00022, -.0986])
        pad_child = np.array([-.0068, -.0766, .0189])
        axis = parent_r @ [0., 0., 1.]
        moving_open = pivot + parent_r @ Rotation.from_rotvec([0, 0, 1.4]).apply(pad_child)
        moving_closed_command = pivot + parent_r @ Rotation.from_rotvec([0, 0, .05]).apply(pad_child)
        self.assertAlmostEqual(float(np.linalg.norm(moving_open - fixed)), OPEN_PAD_SPAN_M)
        self.assertAlmostEqual(float(np.linalg.norm(moving_closed_command - fixed)), CLOSED_PAD_SPAN_M)
        axes, metadata = contact_frame(fixed, moving_open, pivot, axis, 1.4, .056)
        angle = metadata['contact_joint_rad']
        moving_closed = pivot + parent_r @ Rotation.from_rotvec([0, 0, angle]).apply(pad_child)
        closed_normal = moving_closed - fixed
        closed_normal /= np.linalg.norm(closed_normal)
        np.testing.assert_allclose(axes[:, 1], closed_normal, atol=1e-7)
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.det(axes)), 1.)
        open_normal = moving_open - fixed
        open_normal /= np.linalg.norm(open_normal)
        self.assertGreater(float(np.rad2deg(np.arccos(np.dot(open_normal, closed_normal)))), 20.)
        self.assertTrue(.3 < angle < .5)
