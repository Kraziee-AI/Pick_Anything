"""Real USD line identities and executor pose/safety inheritance on CPU."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from small_grasp.closing_line import PlannedClosingLine
from small_grasp.gripper_geometry import SO101GripperGeometry
from small_grasp.sim_robot import POCKET, SERVO_CONFIG, _GripperMeshClearance
from small_grasp.tests import test_motion_synchronization as synchronized


class PlannedClosingLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry=SO101GripperGeometry()
        cls.mesh=_GripperMeshClearance(cls.geometry)

    def test_usd_reference_line_start_continuity_and_correct_rigid_tool_offset(self):
        r0=Rotation.from_euler('xyz',[.2,-.3,.5]).as_matrix();p=np.array([.3,-.38,.3])
        for q0 in (1.4,1.,.8):
            line=PlannedClosingLine(self.geometry,p,r0,q0,np.array(POCKET))
            np.testing.assert_array_equal(line.pose(q0)[1],r0)
            for jaw in np.linspace(q0,.05,56):
                tool,r,g=line.pose(float(jaw))
                self.assertLess(g['fixed_line_identity_error_m'],1e-12)
                self.assertLess(g['moving_line_identity_error_m'],1e-12)
                np.testing.assert_allclose(r@line.axes(jaw),line.frame0,atol=1e-12)
                np.testing.assert_allclose(tool,p+r@(np.array(POCKET)-g['reference_midpoint_body_m']),atol=1e-12)
                self.assertIsNone(self.mesh.check(tool,r,jaw,table_buffer_m=.008))
            tool,r,g=line.pose(.3)
            incorrect=p+r0@(np.array(POCKET)-g['reference_midpoint_body_m'])
            self.assertGreater(np.linalg.norm(incorrect-tool),.001)
        self.assertEqual(len(self.mesh.parts),18)

    def executor(self):
        e=synchronized.MotionSynchronizationTests().executor()
        e.close_frame_policy='planned-line';e.gripper_geometry=self.geometry
        e.mesh_clearance=self.mesh;e.tool_offset_b=np.array(POCKET)
        e._check_path=Mock(return_value=None);e._planning_mesh_at_q=Mock(return_value=None)
        def solve(point,r,seed,prefer_near=False):
            self.assertTrue(prefer_near)
            q=seed.clone();q[0,0]=float(Rotation.from_matrix(r).as_rotvec()[1])
            return q,{'valid':True,'requested_pocket_world_m':np.asarray(point).tolist()}
        e._solve=Mock(side_effect=solve)
        return e

    def test_every_changed_rotation_gets_fresh_ik_and_original_full_mesh_path_reserve(self):
        e=self.executor();e._planning_closure_envelope=Mock(side_effect=AssertionError('fixed-R cache cannot approve new path'))
        path,d=e._prepare_synchronized_closure(np.array([.3,-.38,.3]),np.eye(3),e.robot.data.joint_pos,55)
        self.assertTrue(d['valid']);self.assertEqual(len(path),56);self.assertEqual(e._solve.call_count,56)
        for (tool,q,r),call in zip(path,e._solve.call_args_list):
            np.testing.assert_array_equal(call.args[0],tool);np.testing.assert_array_equal(call.args[1],r)
        self.assertFalse(np.allclose(path[0][2],path[-1][2]))
        self.assertEqual(e._planning_mesh_at_q.call_count,56)
        self.assertEqual(e._mesh_path.call_args_list[0].kwargs,{})
        self.assertTrue(all(c.kwargs=={'table_buffer_m':.008}for c in e._mesh_path.call_args_list[1:]))
        self.assertEqual(e._check_path.call_count,56)
        self.assertEqual(d['future_required_table_margin_m'],.016)
        self.assertEqual(d['origin_runtime_margin_m'],.008)

    def test_late_rotated_path_ik_and_future_clearance_reject_before_actions(self):
        for gate in ('path','endpoint','ik','coverage'):
            e=self.executor();e._move=Mock()
            if gate=='path':e._mesh_path.side_effect=[None,None,{'reason':'table_mesh'}]
            elif gate=='endpoint':e._planning_mesh_at_q.side_effect=[None,{'reason':'table_mesh'}]
            elif gate=='ik':e._solve.side_effect=[(e.robot.data.joint_pos.clone(),{'valid':True}),(None,{'valid':False})]
            else:e.mesh_clearance=SimpleNamespace(parts=[None]*17)
            self.assertIsNone(e._close_centered_synchronized(np.array([.3,-.38,.3]),np.eye(3)))
            e._move.assert_not_called()
        self.assertEqual(SERVO_CONFIG['position_tolerance_m'],.012)
        self.assertEqual(SERVO_CONFIG['orientation_tolerance_deg'],15.)
        self.assertEqual(SERVO_CONFIG['mesh_max_vertical_correction_m'],.008)

    def run_synchronized(self, contact=False, wrong_angle=False):
        e=self.executor();e._move=Mock(return_value=True);e._arrive=Mock(return_value=True)
        issued=[];actual_r=np.eye(3);actual_tool=np.zeros(3)
        e._pad_midpoint=lambda:(actual_tool.copy(),actual_tool.copy())
        e._body_rotation=lambda:actual_r.copy()
        e.scorer=Mock(return_value={'claw_force':10. if contact else 0.})
        def step(target,phase):
            nonlocal actual_r,actual_tool
            issued.append((target.clone(),np.array(e.motion_goal[0]),np.array(e.motion_goal[1])))
            actual_tool=np.array(e.motion_goal[0]);actual_r=np.array(e.motion_goal[1])
            if wrong_angle:actual_r=actual_r@Rotation.from_euler('z',15.001,degrees=True).as_matrix()
            e.robot.data.joint_pos.copy_(target)
            if contact:e.robot.data.joint_pos[0,5]=max(.9,float(target[0,5]))
            e.robot.data.joint_vel.zero_();e.env.episode_length_buf+=1
            return True
        e._step=step
        # The synthetic measured start is exactly the alignment goal.
        line=PlannedClosingLine(self.geometry,np.array([.3,-.38,.3]),np.eye(3),1.4,np.array(POCKET))
        actual_tool=line.pose(1.4)[0]
        if wrong_angle:actual_r=Rotation.from_euler('z',15.001,degrees=True).as_matrix()
        result=e._close_centered_synchronized(np.array([.3,-.38,.3]),np.eye(3))
        return e,result,issued

    def test_runtime_waypoint_and_actual_jaw_latch_select_matching_planned_rotation(self):
        for contact in (False,True):
            e,result,issued=self.run_synchronized(contact=contact)
            self.assertIsNotNone(result)
            evidence=e.motion_diagnostics[-1]
            line=PlannedClosingLine(self.geometry,[.3,-.38,.3],np.eye(3),1.4,POCKET)
            for target,tool,r in issued[:5]:
                desired,expected,_=line.pose(float(target[0,5]))
                np.testing.assert_allclose(r,expected,atol=1e-7)
                np.testing.assert_allclose(tool,desired,atol=1e-7)
            if contact:
                latch=next(x for x in e.close_control_events if x['event']=='contact_stall_latched')
                # Final arm reference uses the nearest prepared actual .9 jaw,
                # while the jaw target retains the original legacy .88 latch.
                jaws=np.linspace(1.4,.05,56);index=int(np.argmin(abs(jaws-.9)))
                expected=line.pose(float(jaws[index]))[1]
                np.testing.assert_allclose(issued[-1][2],expected,atol=1e-7)
            np.testing.assert_array_equal(e._last_closure_rotation,issued[-1][2])
            np.testing.assert_array_equal(e._closure_followup_rotation(np.eye(3)),issued[-1][2])
        e,result,issued=self.run_synchronized(wrong_angle=True)
        self.assertIsNone(result);self.assertEqual(len(issued),360)
        self.assertTrue(all(abs(float(x[0][0,5])-1.4)<1e-6 for x in issued))

    def test_actual_execute_hold_lift_and_lift_hold_use_final_plan_and_never_measured_r(self):
        e=self.executor();r0=np.eye(3);rf=Rotation.from_euler('y',.2).as_matrix();point=np.array([.3,-.38,.3])
        candidate={'candidate_id':'network-source','T_world_grasp':np.eye(4).tolist(),'depth_m':0.,'width_m':.075}
        q=torch.zeros((1,6));q[0,5]=.5
        e._grasp_axes_in_body=Mock(return_value=r0);e._pad_midpoint=Mock(return_value=(point,point))
        e._plan_servo_openings=Mock(return_value=(0.,0,candidate,point,q.clone(),q.clone(),r0,False))
        e.ik_attempts=[{'accepted':True,'candidate':0,'symmetric_flip':False,'network_pocket_world_m':point.tolist(),
            'executed_pocket_offset_m':[0,0,0],'opening_stage_index':0}]
        e.target_centre=None;e.contact_frame='open';e.servo_routes[(0,False)]=(q.clone(),point)
        e.servo_descents[(0,False)]=[None];e._arrive=Mock(return_value=True)
        e._cartesian_descend=Mock(return_value=True);e._closure_rotation=lambda r:r
        def close(p,r):
            e._last_closure_rotation=rf.copy();e._last_closure_tool=point.copy();return q.clone()
        e._close_centered=close
        phases=[]
        def move(target,phase,*args):
            if phase=='lift':phases.append((phase,np.array(e.motion_goal[1])))
            return True
        def step(target,phase):phases.append((phase,np.array(e.motion_goal[1])));return True
        e._move=move;e._step=step;e._solve=Mock(return_value=(q.clone(),{'valid':True}))
        e._body_rotation=Mock(side_effect=AssertionError('do not absorb measured R'))
        class Completed(Exception):pass
        def stop(*args):raise Completed()
        e._stop_grasp_hold=stop
        with self.assertRaises(Completed):e._execute([candidate],'object-single')
        self.assertEqual([p for p,r in phases].count('hold'),15)
        self.assertEqual([p for p,r in phases].count('lift_hold'),30)
        self.assertEqual([p for p,r in phases].count('lift'),1)
        self.assertTrue(all(np.array_equal(r,rf)for p,r in phases))
        np.testing.assert_array_equal(e._solve.call_args.args[1],rf)
        e._body_rotation.assert_not_called()

    def test_fixed_body_unchanged_commands_and_original_rotation_identity(self):
        helper=synchronized.MotionSynchronizationTests()
        for kwargs in ({},{'contact':True},{'forever_fast':True}):
            e,result,commands,_=helper.run_closure(**kwargs)
            explicit=helper.executor();explicit.close_frame_policy='fixed-body'
            with patch.object(helper,'executor',return_value=explicit):new,other,other_commands,_=helper.run_closure(**kwargs)
            self.assertTrue(all(torch.equal(a,b)for a,b in zip(commands,other_commands)))
            self.assertEqual(len(commands),len(other_commands));self.assertEqual(e.close_control_events,new.close_control_events)
        original=np.eye(3);self.assertIs(explicit._closure_followup_rotation(original),original)


if __name__=='__main__':unittest.main()
