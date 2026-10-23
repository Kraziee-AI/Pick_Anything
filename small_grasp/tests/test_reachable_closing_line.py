"""Whole-path, fixed-order mechanical compensation with tighter nominal gates."""
import unittest
import json
from unittest.mock import Mock

import numpy as np

from small_grasp.closing_line import PlannedClosingLine, REACHABLE_LINE_BETAS
from small_grasp.gripper_geometry import SO101GripperGeometry
from small_grasp.sim_robot import POCKET, SERVO_CONFIG, CONFIG
from small_grasp.tests import test_planned_closing_line as planned_tests


class ReachableClosingLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        planned_tests.PlannedClosingLineTests.setUpClass()
        cls.geometry=planned_tests.PlannedClosingLineTests.geometry

    def helper(self):
        helper=planned_tests.PlannedClosingLineTests()
        original=helper.executor
        def executor():
            e=original();e.close_frame_policy='reachable-line'
            solve=e._solve.side_effect
            def with_pose(*args,**kwargs):
                q,d=solve(*args,**kwargs)
                d.update(position_error_m=0.,orientation_error_deg=0.)
                return q,d
            e._solve=Mock(side_effect=with_pose)
            return e
        helper.executor=executor
        return helper

    def test_beta_endpoints_start_continuity_and_partial_line_residual_are_honest(self):
        from scipy.spatial.transform import Rotation
        p=np.array([.3,-.38,.3]);r0=Rotation.from_euler('xyz',[.2,-.3,.5]).as_matrix()
        lines={b:PlannedClosingLine(self.geometry,p,r0,1.4,POCKET,beta=b)for b in REACHABLE_LINE_BETAS}
        full=PlannedClosingLine(self.geometry,p,r0,1.4,POCKET)
        for b,line in lines.items():
            np.testing.assert_array_equal(line.pose(1.4)[1],r0)
            for q in (.9,.5,.05):
                tool,r,d=line.pose(q)
                np.testing.assert_allclose(tool,p+r@(np.array(POCKET)-d['reference_midpoint_body_m']),atol=1e-12)
                self.assertEqual(d['exact_line_preservation_claim'],b==1.)
                if b==0:np.testing.assert_array_equal(r,r0)
                if b==1:np.testing.assert_array_equal(r,full.pose(q)[1])
                else:self.assertGreater(d['closing_line_residual_deg'],0.)
                self.assertAlmostEqual(d['applied_compensation_angle_deg'],b*d['full_compensation_angle_deg'],places=10)
                observation=line.observed_residual(r,q)
                self.assertAlmostEqual(observation['actual_closing_line_residual_deg'],d['closing_line_residual_deg'],places=10)
                self.assertEqual(line.observed_residual(r,float('nan'))['status'],'unavailable')
        residuals=[lines[b].pose(.3)[2]['closing_line_residual_deg']for b in REACHABLE_LINE_BETAS]
        self.assertEqual(residuals,sorted(residuals))

    def test_fixed_order_search_evaluates_complete_paths_and_selects_first_fully_valid_beta(self):
        e=self.helper().executor();original=e._solve.side_effect;calls=[]
        def solve(*args,**kwargs):
            attempt=len(calls)//56;index=len(calls)%56
            calls.append((args[2].clone(),args[1].copy()))
            q,d=original(*args,**kwargs)
            d['orientation_error_deg']=13.0001 if attempt<2 and index==55 else 13.
            return q,d
        e._solve=Mock(side_effect=solve)
        start=e.robot.data.joint_pos.clone();p=np.array([.3,-.38,.3])
        path,d=e._prepare_synchronized_closure(p,np.eye(3),start,55)
        self.assertEqual(d['selected_beta'],.5)
        self.assertEqual([x['beta']for x in d['beta_attempts']],[1.,.75,.5])
        self.assertEqual([len(x['waypoints'])for x in d['beta_attempts']],[56,56,56])
        self.assertEqual(e._solve.call_count,168);self.assertEqual(len(path),56)
        self.assertTrue(all(not x['valid']for x in d['beta_attempts'][:2]))
        for index in (0,56,112):
            np.testing.assert_array_equal(calls[index][0],start)
        line=PlannedClosingLine(self.geometry,p,np.eye(3),float(start[0,5]),POCKET,beta=.5)
        for item,record in zip(path,d['waypoints']):
            np.testing.assert_array_equal(item[2],line.pose(record['jaw_joint_target_rad'])[1])
        self.assertEqual(e._planning_mesh_at_q.call_count,168)
        self.assertEqual(sum(not c.kwargs for c in e._mesh_path.call_args_list),3)
        self.assertTrue(all(c.kwargs in ({},{'table_buffer_m':.008})for c in e._mesh_path.call_args_list))

    def test_all_beta_failure_and_missing_pose_never_issue_alignment_or_close_actions(self):
        for error in (13.00001,None,float('nan')):
            e=self.helper().executor();original=e._solve.side_effect
            def solve(*args,**kwargs):
                q,d=original(*args,**kwargs);d['orientation_error_deg']=error;return q,d
            e._solve=Mock(side_effect=solve);e._move=Mock();e._step=Mock()
            self.assertIsNone(e._close_centered_synchronized(np.array([.3,-.38,.3]),np.eye(3)))
            self.assertEqual(e._solve.call_count,280)
            self.assertEqual(e.motion_failure,'close_reachable_line_no_feasible_beta')
            evidence=e.motion_diagnostics[-1]
            self.assertEqual([a['beta']for a in evidence['beta_attempts']],list(REACHABLE_LINE_BETAS))
            self.assertTrue(all(len(a['waypoints'])==56 for a in evidence['beta_attempts']))
            json.dumps(evidence,allow_nan=False)
            e._move.assert_not_called();e._step.assert_not_called()

    def test_ik_position_jump_and_mesh_failures_remain_rejections_even_at_beta_zero(self):
        for gate in ('ik','position','jump','endpoint','connector','coverage'):
            e=self.helper().executor();original=e._solve.side_effect
            def solve(*args,**kwargs):
                q,d=original(*args,**kwargs)
                if gate=='ik':return None,dict(d,valid=False)
                if gate=='position':d['position_error_m']=CONFIG['ik_position_tolerance_m']
                if gate=='jump':q[:,0]+=1.
                return q,d
            e._solve=Mock(side_effect=solve)
            if gate=='endpoint':e._planning_mesh_at_q.return_value={'reason':'table_mesh'}
            if gate=='connector':e._mesh_path.return_value={'reason':'observed_box_mesh'}
            if gate=='coverage':e.mesh_clearance=type('Incomplete',(),{'parts':[None]*17})()
            e._move=Mock()
            self.assertIsNone(e._close_centered_synchronized(np.array([.3,-.38,.3]),np.eye(3)))
            self.assertEqual(len(e.motion_diagnostics[-1]['beta_attempts']),5)
            e._move.assert_not_called()

    def test_runtime_fifteen_degree_gate_and_lock_hold_lift_inheritance_are_unchanged(self):
        helper=self.helper()
        e,result,issued=helper.run_synchronized(contact=True)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(e._closure_followup_rotation(np.eye(3)),issued[-1][2])
        e,result,issued=helper.run_synchronized(wrong_angle=True)
        self.assertIsNone(result);self.assertEqual(len(issued),360)
        self.assertEqual(SERVO_CONFIG['orientation_tolerance_deg'],15.)
        self.assertEqual(SERVO_CONFIG['position_tolerance_m'],.012)
        helper.test_actual_execute_hold_lift_and_lift_hold_use_final_plan_and_never_measured_r()


if __name__=='__main__':unittest.main()
