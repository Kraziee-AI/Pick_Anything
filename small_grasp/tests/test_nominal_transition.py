"""Feedback trajectories inherit issued nominal commands without load-error resets."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from small_grasp.sim_robot import GraspExecutor
from small_grasp.tests import test_servo_motion


class NominalTransitionTests(unittest.TestCase):
    def executor(self):
        e=test_servo_motion.ServoMotionTests().executor()
        e.arm_tracking_feedback=True
        e.robot.data.joint_pos.fill_(.04)
        e._last_nominal_target=torch.zeros((1,6))
        e._last_issued_target=torch.full((1,6),-.02)
        return e

    def test_move_inherits_nominal_not_actual_or_biased_command(self):
        e=self.executor();commands=[]
        e._step=lambda target,phase:commands.append(target.clone()) or True
        goal=torch.full((1,6),.1)
        self.assertTrue(e._move(goal,'close_align',20))
        torch.testing.assert_close(commands[0],torch.full((1,6),.1/18))
        self.assertEqual(e.motion_diagnostics[0]['start_source'],'last_issued_nominal')
        np.testing.assert_allclose(e.motion_diagnostics[0]['last_issued_command_rad'],[-.02]*6,atol=1e-8)

    def test_explicit_start_and_disabled_mode_preserve_existing_semantics(self):
        e=self.executor();commands=[]
        e._step=lambda target,phase:commands.append(target.clone()) or True
        goal=torch.full((1,6),.1)
        e._move(goal,'descend',20,start=torch.full((1,6),.02))
        torch.testing.assert_close(commands[0],torch.full((1,6),.02+.08/18))
        self.assertEqual(e.motion_diagnostics[0]['start_source'],'explicit_caller')
        e.arm_tracking_feedback=False;commands.clear()
        e._move(goal,'close_align',20)
        torch.testing.assert_close(commands[0],torch.full((1,6),.04+.06/18))

    def test_first_cartesian_connector_inherits_cache_then_prepared_nominal(self):
        e=self.executor();e._check_path=Mock(return_value=None)
        e._move=Mock(return_value=True);e._arrive=Mock(return_value=True)
        path=[(np.array([0.,0.,.2]),torch.full((1,6),.06)),
              (np.array([0.,0.,.19]),torch.full((1,6),.08))]
        self.assertTrue(e._cartesian_descend(path[-1][0],np.eye(3),path[-1][1],prepared=path))
        torch.testing.assert_close(e._move.call_args_list[0].kwargs['start'],e._last_nominal_target)
        torch.testing.assert_close(e._move.call_args_list[1].kwargs['start'],path[0][1])
        event=next(x for x in e.motion_diagnostics if x.get('event')=='nominal_connector_start')
        self.assertEqual(event['start_source'],'last_issued_nominal')

    def step_executor(self):
        e=self.executor();e.device='cpu';e.executed=False
        e.env=SimpleNamespace(tracking=True,step=Mock())
        e.arm_tracking_events=[{'nominal_cache_updated':False}]
        e._apply_grasp_hold=lambda target,phase:target
        e._stop_grasp_hold=Mock()
        e._tracking_command=Mock(side_effect=lambda nominal,phase:nominal-.01)
        return e

    def test_only_successfully_issued_step_updates_both_distinct_caches(self):
        e=self.step_executor();nominal=torch.full((1,6),.2)
        e.env.step.side_effect=lambda action:setattr(e.env,'tracking',False)
        self.assertFalse(e._step(nominal,'close_align'))
        torch.testing.assert_close(e._last_nominal_target,nominal)
        torch.testing.assert_close(e._last_issued_target,nominal-.01)
        self.assertTrue(e.arm_tracking_events[-1]['nominal_cache_updated'])
        nominal.fill_(.5)
        torch.testing.assert_close(e._last_nominal_target,torch.full((1,6),.2))

    def test_unissued_and_exception_steps_leave_cache_unchanged(self):
        for rejected in (True,False):
            with self.subTest(rejected=rejected):
                e=self.step_executor();nominal=torch.full((1,6),.2)
                before_nominal=e._last_nominal_target.clone();before_issued=e._last_issued_target.clone()
                if rejected:
                    e._tracking_command=Mock(return_value=None)
                    self.assertFalse(e._step(nominal,'close_align'))
                    e.env.step.assert_not_called()
                else:
                    e.env.step.side_effect=RuntimeError('execution failed')
                    with self.assertRaises(RuntimeError):e._step(nominal,'close_align')
                torch.testing.assert_close(e._last_nominal_target,before_nominal)
                torch.testing.assert_close(e._last_issued_target,before_issued)
                self.assertFalse(e.arm_tracking_events[-1]['nominal_cache_updated'])

    def test_new_execute_clears_previous_execution_caches(self):
        e=self.executor();e._arm_tracking=SimpleNamespace(reset=Mock())
        e._stop_grasp_hold=Mock()
        def execute(candidates,scene):
            self.assertIsNone(e._last_nominal_target)
            self.assertIsNone(e._last_issued_target)
            return {'ok':True}
        e._execute=execute
        self.assertEqual(e.execute([],'object-single'),{'ok':True})


if __name__=='__main__':unittest.main()
