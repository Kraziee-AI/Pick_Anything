import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from small_grasp.closure_envelope import FixedMidpointClosureEnvelope
from small_grasp.gripper_geometry import SO101GripperGeometry, FIXED_REFERENCE_M, MOVING_REFERENCE_M
from small_grasp.sim_robot import _GripperMeshClearance


class ClosureEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry=SO101GripperGeometry()
        cls.parts=_GripperMeshClearance(cls.geometry).parts
        cls.envelope=FixedMidpointClosureEnvelope(cls.geometry,cls.parts)

    def test_analytic_interior_minimum_matches_dense_actual_usd_replay(self):
        centre=np.array([.295,-.343,.0694])
        rotation=Rotation.from_euler('xyz',[.5,.4,-.2]).as_matrix()
        result=self.envelope.minimum(centre,rotation,1.4,.05)
        dense=float('inf')
        # Independent direct transform of actual hull geometry, including
        # interior jaw angles. Analytic minimum must not miss a sampled low.
        for angle in np.linspace(.05,1.4,1201):
            midpoint=(FIXED_REFERENCE_M+self.geometry.moving_to_fixed(MOVING_REFERENCE_M,angle))/2
            for body,_,vertices,_ in self.parts:
                vectors=self.geometry.moving_to_fixed(vertices,angle) if body=='moving' else vertices
                dense=min(dense,float(((vectors-midpoint)@rotation.T+centre)[:,2].min()))
        self.assertLessEqual(result['minimum_mesh_z_m'],dense+1e-10)
        self.assertLess(dense-result['minimum_mesh_z_m'],1e-7)
        self.assertTrue(.05<=result['critical_jaw_joint_rad']<=1.4)

    def test_translation_and_interval_endpoints(self):
        r=np.eye(3);p=np.array([.3,-.4,.1])
        original=self.envelope.minimum(p,r,.8,.8)
        shifted=self.envelope.minimum(p+[0,0,.004],r,.8,.8)
        self.assertAlmostEqual(shifted['minimum_mesh_z_m']-original['minimum_mesh_z_m'],.004)
        self.assertEqual(original['critical_jaw_joint_rad'],.8)

    def test_invalid_pose_never_produces_feasible_clearance(self):
        with self.assertRaises(ValueError):self.envelope.minimum([0,0,float('nan')],np.eye(3),1.4,.05)
        with self.assertRaises(ValueError):self.envelope.minimum([0,0,0],np.zeros((3,3)),1.4,.05)


if __name__=='__main__':unittest.main()
