import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from small_grasp.sim_robot import GraspExecutor


class ClosureOrientationTests(unittest.TestCase):
    def test_feedback_preserves_full_planned_rotation_and_only_logs_measured_error(self):
        e=GraspExecutor.__new__(GraspExecutor)
        e.arm_tracking_feedback=True;e.motion_diagnostics=[]
        planned=Rotation.from_euler('xyz',[.3,-.2,.6]).as_matrix()
        actual=planned@Rotation.from_euler('y',.7152524864324324,degrees=True).as_matrix()
        e._body_rotation=Mock(return_value=actual)
        e.scorer=Mock(side_effect=AssertionError('Target scoring may not choose the pose'))
        nominal=e._closure_rotation(planned)
        np.testing.assert_array_equal(nominal,planned)
        self.assertFalse(np.allclose(nominal,actual))
        self.assertAlmostEqual(e.motion_diagnostics[-1]['measured_orientation_error_deg'],.7152524864324324)
        nominal[:]=0
        self.assertGreater(np.linalg.det(planned),.999)
        e.scorer.assert_not_called()

    def test_disabled_mode_keeps_original_measured_rotation_without_planned_input_reads(self):
        e=GraspExecutor.__new__(GraspExecutor)
        e.arm_tracking_feedback=False
        actual=Rotation.from_euler('x',.2).as_matrix()
        e._body_rotation=Mock(return_value=actual)
        self.assertIs(e._closure_rotation(None),actual)
        e._body_rotation.assert_called_once()
        self.assertFalse(hasattr(e,'motion_diagnostics'))


if __name__=='__main__':unittest.main()
