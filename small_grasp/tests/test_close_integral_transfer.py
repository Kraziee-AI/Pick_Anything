"""Explicit one-use close-path authority, bounded I, and frozen reset commands."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from small_grasp.arm_tracking import BoundedArmTrackingFeedback
from small_grasp.tests import test_close_tracking_integral as integral_tests
from small_grasp.tests import test_motion_synchronization as synchronization_tests


class CloseIntegralTransferTests(unittest.TestCase):
    def setUp(self):
        self.nominal = np.array([0., 0., .3, 0., 0., 1.])
        self.path = [self.nominal.copy() for _ in range(4)]
        for i, q in enumerate(self.path):
            q[2] += i*.002; q[5] -= i*.02
        self.limits = np.tile([-3., 3.], (6, 1))

    def controller(self):
        c = BoundedArmTrackingFeedback(range(5), close_integral=True, integral_transfer='adjacent')
        c.begin_close(self.path)
        for _ in range(12):
            d = self.propose(c, self.path[0]); c.commit(d, True)
        self.assertGreater(c.integral[2], 0)
        return c

    def propose(self, c, nominal, token=None, phase='close_centered', velocity=0.):
        actual = nominal.copy(); actual[2] -= .03
        return c.propose(nominal, actual, self.limits, phase=phase, dt_s=1/30,
            arm_velocity=np.full(5, velocity), integral_transfer=token)

    def test_authorized_adjacent_transfer_carries_once_and_redebounces(self):
        c = self.controller(); prior = c.integral.copy(); bias = c.bias.copy()
        token = c.authorize_adjacent_transfer(self.path, 0, 1, self.path[0], self.path[1])
        self.assertIsNotNone(token)
        d = self.propose(c, self.path[1], token)
        self.assertTrue(d.integral_transfer_used)
        self.assertEqual(d.integral_reason, 'adjacent_transfer')
        self.assertEqual(d.stationary_steps, 0)
        np.testing.assert_array_equal(d.integral, prior)
        np.testing.assert_array_equal(c.integral, prior)  # propose never commits I
        self.assertLessEqual(max(abs(d.bias-bias)), .004+1e-12)
        # The same capability cannot be used for another uncommitted proposal.
        duplicate = self.propose(c, self.path[1], token)
        self.assertFalse(duplicate.integral_transfer_used)
        np.testing.assert_array_equal(duplicate.integral, np.zeros(5))
        c.commit(d, True)
        for expected in (1, 2):
            d = self.propose(c, self.path[1]); c.commit(d, True)
            self.assertEqual(d.stationary_steps, expected)
            np.testing.assert_array_equal(c.integral, prior)
        d = self.propose(c, self.path[1], velocity=.051); c.commit(d, True)
        np.testing.assert_array_equal(c.integral, prior)
        self.assertEqual(d.integral_reason, 'arm_moving')
        d = self.propose(c, self.path[1]); c.commit(d, True)
        self.assertGreater(c.integral[2], prior[2])

    def test_nearby_nominal_jump_reverse_wrong_path_and_forged_context_never_carry(self):
        for invalid in ('unannounced', 'jump', 'reverse', 'other_path', 'wrong_nominal', 'forged'):
            with self.subTest(invalid=invalid):
                c = self.controller(); token = None; target = self.path[1]
                if invalid == 'jump':
                    token = c.authorize_adjacent_transfer(self.path, 0, 2, self.path[0], self.path[2]); target=self.path[2]
                elif invalid == 'reverse':
                    token = c.authorize_adjacent_transfer(self.path, 1, 0, self.path[1], self.path[0]); target=self.path[1]
                elif invalid == 'other_path':
                    token = c.authorize_adjacent_transfer(list(self.path), 0, 1, self.path[0], self.path[1])
                elif invalid == 'wrong_nominal':
                    token = c.authorize_adjacent_transfer(self.path, 0, 1, self.path[0], self.path[2])
                elif invalid == 'forged':
                    token = object()
                d = self.propose(c, target, token)
                self.assertFalse(d.integral_transfer_used)
                np.testing.assert_array_equal(d.integral, np.zeros(5))

    def test_safety_rejection_new_close_exit_and_invalid_state_clear_integral_and_tokens(self):
        for boundary in ('rejected', 'new_close', 'end_close', 'outside', 'invalid'):
            with self.subTest(boundary=boundary):
                c = self.controller()
                token = c.authorize_adjacent_transfer(self.path, 0, 1, self.path[0], self.path[1])
                if boundary == 'new_close':
                    c.begin_close(self.path)
                elif boundary == 'end_close':
                    c.end_close()
                elif boundary == 'outside':
                    d = self.propose(c, self.path[1], token, phase='lift'); c.commit(d, True)
                elif boundary == 'invalid':
                    d = c.propose(self.path[1], self.path[1], self.limits, phase='close_centered',
                        dt_s=float('nan'), arm_velocity=np.zeros(5), integral_transfer=token)
                    c.commit(d, False)
                else:
                    d = self.propose(c, self.path[1], token); c.commit(d, False)
                np.testing.assert_array_equal(c.integral, np.zeros(5))
                d = self.propose(c, self.path[1], token)
                self.assertFalse(d.integral_transfer_used)
                np.testing.assert_array_equal(d.integral, np.zeros(5))

    def test_reset_default_matches_frozen_pretransfer_commands_exactly(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/close_integral_reset_compatibility.json').read_text())
        for case in fixture['cases']:
            for kwargs in ({}, {'integral_transfer': 'reset'}):
                c = BoundedArmTrackingFeedback(range(5), close_integral=True, **kwargs)
                initial = case['initial']
                c.bias[:] = initial['bias']; c.integral[:] = initial['integral']
                c.integral_nominal = np.array(initial['reference']); c.stationary_steps=initial['stationary']
                for row in case['rows']:
                    d = c.propose(row['nominal'], row['actual'], self.limits, phase='close_centered',
                        dt_s=1/30, arm_velocity=row['velocity'])
                    np.testing.assert_array_equal(d.command, row['command'])
                    np.testing.assert_array_equal(d.bias, row['bias'])
                    np.testing.assert_array_equal(d.integral, row['integral'])
                    self.assertEqual(d.stationary_steps, row['stationary'])
                    self.assertEqual(d.integral_reason, row['reason'])
                    c.commit(d, True)

    def synchronized(self, mode='adjacent', contact=False, position_error=0., fail=False,
                     angle_deg=0., arm_error=0., arm_speed=0.):
        e = synchronization_tests.MotionSynchronizationTests().executor()
        e.close_integral_transfer = mode; e.close_tracking_integral=True
        e._arm_tracking = BoundedArmTrackingFeedback(e.arm, close_integral=True, integral_transfer=mode)
        path=[]
        for jaw in np.linspace(1.4,.05,56):
            q=torch.zeros((1,6));q[0,5]=float(jaw);path.append((np.zeros(3),q))
        e._prepare_synchronized_closure=Mock(return_value=(path,{'valid':True}))
        e._move=Mock(return_value=True);e._arrive=Mock(return_value=True)
        e._pad_midpoint=Mock(return_value=(np.zeros(3), np.array([position_error,0.,0.])))
        from scipy.spatial.transform import Rotation
        e._body_rotation=Mock(return_value=Rotation.from_euler('z',angle_deg,degrees=True).as_matrix())
        e.robot.data.joint_pos[0,0]=arm_error
        e.robot.data.joint_vel[0,:5]=arm_speed
        e.scorer=Mock(return_value={'claw_force':10. if contact else 0.})
        issued=[]; decisions=[]
        def step(target,phase):
            token=getattr(e,'_pending_close_integral_transfer',None);e._pending_close_integral_transfer=None
            actual=target[0].numpy().copy();actual[2]-=.03
            d=e._arm_tracking.propose(target[0].numpy(),actual,self.limits,phase=phase,
                dt_s=1/30,arm_velocity=np.full(5,arm_speed),integral_transfer=token)
            e._arm_tracking.commit(d,not fail);decisions.append(d)
            issued.append(target.clone());e.robot.data.joint_pos.copy_(target)
            if contact:e.robot.data.joint_pos[0,5]=max(.9,float(target[0,5]))
            e.robot.data.joint_pos[0,0]=arm_error
            e.robot.data.joint_vel.zero_();e.robot.data.joint_vel[0,:5]=arm_speed
            e.env.episode_length_buf+=1
            if fail:raise RuntimeError('test rejected/exception cleanup')
            return True
        e._step=step
        if fail:
            with self.assertRaises(RuntimeError):e._close_centered_synchronized(np.zeros(3),np.eye(3))
            result=None
        else:result=e._close_centered_synchronized(np.zeros(3),np.eye(3))
        return e,result,issued,decisions

    def test_executor_issues_authority_only_on_original_full_settle_advance_and_clears_on_exit(self):
        e,result,issued,decisions=self.synchronized()
        self.assertIsNotNone(result)
        authorities=[x for x in e.close_control_events if x['event']=='integral_transfer_authorized']
        self.assertEqual(len(authorities),55)
        self.assertEqual(sum(d.integral_transfer_used for d in decisions),55)
        for x in authorities:
            self.assertTrue(x['authorized']);self.assertTrue(x['path_preflight_valid'])
            self.assertEqual(x['to_index'],x['from_index']+1)
            self.assertTrue(x['settled']);self.assertGreaterEqual(x['consecutive_settled_steps'],3)
            self.assertFalse(x['latched'])
        np.testing.assert_array_equal(e._arm_tracking.integral,np.zeros(5))
        self.assertIsNone(e._arm_tracking._close_path)
        self.assertIsNone(e._pending_close_integral_transfer)
        # Original 12mm gate and finite timeout cannot grant any transfer.
        for gate in ({'position_error':.012001},{'angle_deg':15.001},{'arm_error':.14001},{'arm_speed':.05001}):
            with self.subTest(gate=gate):
                e,result,issued,decisions=self.synchronized(**gate)
                self.assertIsNone(result);self.assertEqual(len(issued),360)
                self.assertFalse(any(d.integral_transfer_used for d in decisions))
                self.assertFalse(any(x['event']=='integral_transfer_authorized'for x in e.close_control_events))
                np.testing.assert_array_equal(e._arm_tracking.integral,np.zeros(5))

    def test_lock_and_exception_clear_integral_and_default_reset_keeps_synchronized_commands(self):
        e,result,issued,decisions=self.synchronized(contact=True)
        self.assertIsNotNone(result)
        latch=next(x for x in e.close_control_events if x['event']=='contact_stall_latched')
        later=[d for d,x in zip(decisions,issued)][-15:]
        self.assertTrue(all(not d.integral_transfer_used for d in later))
        self.assertTrue(all(np.array_equal(d.integral,np.zeros(5))for d in later))
        failed,_,_,_=self.synchronized(fail=True)
        self.assertIsNone(failed._arm_tracking._close_path)
        self.assertIsNone(failed._pending_close_integral_transfer)
        np.testing.assert_array_equal(failed._arm_tracking.integral,np.zeros(5))
        helper=synchronization_tests.MotionSynchronizationTests()
        for kwargs in ({},{'contact':True},{'forever_fast':True}):
            old,result,commands,_=helper.run_closure(**kwargs)
            explicit=helper.executor();explicit.close_integral_transfer='reset'
            with patch.object(helper,'executor',return_value=explicit):
                new,other,other_commands,_=helper.run_closure(**kwargs)
            self.assertEqual(len(commands),len(other_commands))
            self.assertTrue(all(torch.equal(a,b)for a,b in zip(commands,other_commands)))
            self.assertEqual(old.close_control_events,new.close_control_events)

    def test_executor_rejection_consumes_token_without_commit_and_nonclose_scope_stays_compatible(self):
        fixture=integral_tests.CloseTrackingIntegralTests();fixture.setUp()
        c=self.controller();token=c.authorize_adjacent_transfer(self.path,0,1,self.path[0],self.path[1])
        actual=self.path[1].copy();actual[2]-=.03
        e=fixture.executor(actual);e.close_integral_transfer='adjacent';e._arm_tracking=c
        e._pending_close_integral_transfer=token
        e._mesh_path.side_effect=[None,{'reason':'table_mesh'}]
        self.assertIsNone(e._tracking_command(torch.tensor(self.path[1])[None],'close_centered'))
        self.assertIsNone(e._pending_close_integral_transfer)
        np.testing.assert_array_equal(c.integral,np.zeros(5))
        self.assertFalse(e.arm_tracking_events[-1]['command_issued'])
        for phase in ('approach_high','close_align','hold','lift','lift_hold'):
            e=fixture.executor(actual);e.close_integral_transfer='adjacent'
            e._arm_tracking=self.controller()
            e._mesh_path.side_effect=[None,{'reason':'table_mesh'}]
            target=torch.tensor(self.path[1])[None]
            self.assertIs(e._tracking_command(target,phase),target)
            self.assertTrue(e.collision_ready)
            np.testing.assert_array_equal(e._arm_tracking.integral,np.zeros(5))

    def test_origin_and_nominal_rejection_cancel_pending_transfer_before_proposal(self):
        fixture=integral_tests.CloseTrackingIntegralTests();fixture.setUp()
        for gate in ('origin','nominal'):
            c=self.controller();token=c.authorize_adjacent_transfer(self.path,0,1,self.path[0],self.path[1])
            e=fixture.executor(self.path[1]);e.close_integral_transfer='adjacent';e._arm_tracking=c
            e._pending_close_integral_transfer=token
            if gate=='origin':e._mesh_at_q.return_value={'reason':'table_mesh'}
            else:e._mesh_path.return_value={'reason':'table_mesh'}
            self.assertIsNone(e._tracking_command(torch.tensor(self.path[1])[None],'close_centered'))
            self.assertIsNone(e._pending_close_integral_transfer)
            np.testing.assert_array_equal(c.integral,np.zeros(5))
            self.assertFalse(self.propose(c,self.path[1],token).integral_transfer_used)


if __name__=='__main__':unittest.main()
