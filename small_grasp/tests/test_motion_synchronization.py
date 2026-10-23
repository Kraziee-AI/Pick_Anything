"""Finite real-robot progression gates, without reading target truth or scores."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from small_grasp.tests import test_servo_motion
from small_grasp.arm_tracking import MotionSynchronizationConfig


class MotionSynchronizationTests(unittest.TestCase):
    def executor(self):
        e=test_servo_motion.ServoMotionTests().executor()
        e.arm_tracking_feedback=True
        e.env.step_dt=1/30
        e.env.episode_length_buf=torch.tensor([0])
        e.robot.data.joint_pos[0,5]=1.4
        e.robot.data.joint_vel=torch.zeros((1,6))
        e._pad_midpoint=Mock(return_value=(np.zeros(3),np.zeros(3)))
        e.close_control_events=[]
        e.close_ik_attempts=[]
        return e

    def test_descent_arrival_requires_measured_speed_settle(self):
        e=self.executor()
        e.robot.data.joint_vel.fill_(.2)
        count=[]
        def step(target,phase):
            count.append(1)
            e.robot.data.joint_vel.fill_(.02 if len(count)>1 else .1)
            return True
        e._step=step
        self.assertTrue(e._arrive(e.robot.data.joint_pos,np.zeros(3),np.eye(3),'descend'))
        self.assertEqual(len(count),4)
        self.assertLess(e.motion_diagnostics[-1]['max_arm_joint_speed_rad_s'],.05)
        e.scorer.assert_not_called()

    def test_descent_waits_at_every_waypoint_and_stops_before_next_on_failure(self):
        e=self.executor();e._check_path=Mock(return_value=None)
        e._move=Mock(return_value=True);e._arrive=Mock(side_effect=[True,False])
        path=[(np.array([0.,0.,.2-i*.008]),torch.full((1,6),i*.02)) for i in range(3)]
        self.assertFalse(e._cartesian_descend(path[-1][0],np.eye(3),path[-1][1],prepared=path))
        self.assertEqual(e._move.call_count,2)
        self.assertEqual(e._arrive.call_count,2)

    def test_complete_closure_preflight_rejects_late_collision_before_any_action(self):
        e=self.executor();e.tool_offset_b=np.zeros(3)
        e.gripper_geometry=SimpleNamespace(moving_to_fixed=lambda point,jaw:np.array([jaw*.01,0.,0.]))
        e._solve=Mock(return_value=(torch.zeros((1,6)),{'valid':True}))
        e._check_path=Mock(return_value=None)
        e._mesh_path=Mock(side_effect=[None,None,{'reason':'table_mesh','sample':1}])
        e._step=Mock()
        path,diag=e._prepare_synchronized_closure(np.zeros(3),np.eye(3),e.robot.data.joint_pos,55)
        self.assertIsNone(path)
        self.assertEqual(diag['failure_reason'],'close_preflight_collision')
        self.assertEqual(len(diag['waypoints']),3)
        e._step.assert_not_called()

    def test_future_nominal_endpoint_reserve_does_not_replace_actual_origin_gate(self):
        e=self.executor();e.tool_offset_b=np.zeros(3)
        e.gripper_geometry=SimpleNamespace(moving_to_fixed=lambda point,jaw:np.array([jaw*.01,0.,0.]))
        e._solve=Mock(return_value=(torch.zeros((1,6)),{'valid':True}))
        e._planning_mesh_at_q=Mock(return_value=None)
        e._check_path=Mock(return_value=None);e._mesh_path=Mock(return_value=None)
        start=e.robot.data.joint_pos.clone()
        path,diag=e._prepare_synchronized_closure(np.zeros(3),np.eye(3),start,1)
        self.assertIsNotNone(path)
        self.assertEqual(e._mesh_path.call_args_list[0].kwargs,{})
        self.assertEqual(e._mesh_path.call_args_list[1].kwargs,{'table_buffer_m':.008})
        self.assertEqual(diag['origin_runtime_margin_m'],.008)
        self.assertEqual(diag['future_required_table_margin_m'],.016)
        e._planning_mesh_at_q.return_value={'reason':'table_mesh','minimum_mesh_z_m':.055}
        e._mesh_path.reset_mock()
        path,diag=e._prepare_synchronized_closure(np.zeros(3),np.eye(3),start,1)
        self.assertIsNone(path)
        self.assertEqual(diag['failure_reason'],'close_preflight_tracking_clearance')
        e._mesh_path.assert_not_called()

    def run_closure(self,forever_fast=False,contact=False):
        e=self.executor();config=MotionSynchronizationConfig()
        path=[]
        for jaw in np.linspace(1.4,.05,56):
            q=torch.zeros((1,6));q[0,5]=float(jaw);path.append((np.zeros(3),q))
        e._prepare_synchronized_closure=Mock(return_value=(path,{'phase':'close_preflight','valid':True}))
        e._move=Mock(return_value=True);e._arrive=Mock(return_value=True)
        class RobotOnly(dict):
            def __getitem__(self,key):
                if key!='claw_force':raise AssertionError('No target scoring may drive closure')
                return 10. if contact else 0.
        e.scorer=Mock(return_value=RobotOnly())
        issued=[]
        def step(target,phase):
            self.assertEqual(phase,'close_centered')
            issued.append(target.clone())
            previous=float(e.robot.data.joint_pos[0,5])
            e.robot.data.joint_pos.copy_(target)
            if contact:e.robot.data.joint_pos[0,5]=max(.9,float(target[0,5]))
            e.robot.data.joint_vel[0,:5]=.2 if forever_fast or len(issued)<10 else 0.
            e.robot.data.joint_vel[0,5]=(float(e.robot.data.joint_pos[0,5])-previous)/e.env.step_dt
            e.env.episode_length_buf+=1
            return True
        e.robot.data.joint_vel[0,:5]=.2
        e._step=step
        result=e._close_centered_synchronized(np.zeros(3),np.eye(3))
        return e,result,issued,config

    def test_waiting_does_not_advance_jaw_and_progress_has_finite_budget(self):
        e,result,issued,config=self.run_closure()
        self.assertIsNotNone(result)
        self.assertTrue(all(abs(float(q[0,5])-1.4)<1e-6 for q in issued[:9]))
        events=[x for x in e.close_control_events if x['event']=='closure_synchronization']
        self.assertEqual(events[-1]['jaw_advance_steps'],55)
        self.assertLessEqual(len(issued),config.close_max_steps)
        self.assertAlmostEqual(float(result[0,5]),.05,places=6)

    def test_persistent_motion_stops_without_closing_after_budget(self):
        e,result,issued,config=self.run_closure(forever_fast=True)
        self.assertIsNone(result)
        self.assertEqual(e.motion_failure,'close_synchronization_timeout')
        self.assertEqual(len(issued),config.close_max_steps)
        self.assertTrue(all(abs(float(q[0,5])-1.4)<1e-6 for q in issued))

    def test_original_contact_stall_locks_target_and_stops_progress(self):
        e,result,issued,config=self.run_closure(contact=True)
        latch=next(x for x in e.close_control_events if x['event']=='contact_stall_latched')
        self.assertAlmostEqual(latch['locked_joint_target_rad'],.88,places=6)
        self.assertAlmostEqual(float(result[0,5]),.88,places=6)
        self.assertLess(latch['close_step'],55)
        self.assertTrue(all(abs(float(q[0,5])-.88)<1e-6 for q in issued[-15:]))


if __name__=='__main__':unittest.main()
