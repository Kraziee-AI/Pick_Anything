import json
import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from small_grasp.physics_diagnostics import (PhysicsDiagnostics, contacts_snapshot,
                                            phase_rows, solver_joint_effort)

ROBOT = '/World/envs/env_0/Robot'
TARGET = '/World/envs/env_0/BlueCube'


def scene():
    labels = [ROBOT+'/gripper/mesh', TARGET+'/Cube', '/World/envs/env_0/Other/mesh',
              ROBOT+'/moving_jaw_so101_v1/mesh']
    bodies = [ROBOT+'/gripper', ROBOT+'/moving_jaw_so101_v1', TARGET, '/Other/gripper']
    poses = np.zeros((4,7)); poses[:,6] = 1
    poses[0,:3] = [1,2,0]; poses[0,3:] = [0,0,2**-.5,2**-.5]
    contact = NS(geom=np.array([[0,1],[1,3],[0,2],[2,1]]), worldid=np.zeros(4,int),
        pos=np.tile([1.,3.,0.],(4,1)), frame=np.tile(np.eye(3),(4,1,1)),
        dist=np.array([-.001,.003,-.002,-.003]),dim=np.array([3,3,3,3]),efc_address=np.array([[0],[-1],[2],[3]]))
    data = NS(nacon=np.array([4]),naconmax=4,contact=contact,
              qfrc_actuator=np.array([[10.,20.]]),qfrc_constraint=np.array([[30.,40.]]))
    model = NS(shape_body=np.array([0,2,3,1]),shape_label=labels,body_label=bodies,
        joint_label=[ROBOT+'/joint_a',ROBOT+'/joint_b'],joint_qd_start=np.array([0,1]),
        joint_dof_dim=np.array([[0,1],[0,1]]))
    solver = NS(mjw_data=data,mjc_geom_to_newton_shape=np.array([[0,1,2,3]]),
                mjc_dof_to_newton_dof=np.array([[1,0]]))
    return model,solver,poses


class PhysicsDiagnosticTests(unittest.TestCase):
    def test_actual_pair_filter_order_normal_and_world_to_body_coordinates(self):
        model,solver,poses=scene()
        r=contacts_snapshot(solver.mjw_data,solver.mjc_geom_to_newton_shape,model.shape_body,
                            model.shape_label,model.body_label,poses,ROBOT,TARGET)
        self.assertEqual((r['detected_count'],r['constraint_included_count'],r['selected_pair_count']),(4,3,2))
        a,b=r['records'];self.assertTrue(a['constraint_active']);self.assertFalse(b['constraint_active'])
        np.testing.assert_allclose(a['point_robot_body_at_sample_m'],[1,0,0],atol=1e-12)
        self.assertEqual(a['normal_world_robot_to_target'],[1.,0.,0.])
        self.assertEqual(b['normal_world_robot_to_target'],[-1.,0.,0.])
        self.assertEqual(b['robot_side'],1)
        self.assertEqual(a['newton_body_ids'],[0,2])
        self.assertEqual(len(json.dumps(r,allow_nan=False))>0,True)

    def test_active_count_capacity_record_limit_and_bad_geometry_are_explicit(self):
        m,s,p=scene()
        def read():return contacts_snapshot(s.mjw_data,s.mjc_geom_to_newton_shape,m.shape_body,
                                           m.shape_label,m.body_label,p,ROBOT,TARGET,maximum=1)
        r=read();self.assertTrue(r['truncated']);self.assertEqual(r['selected_pair_count'],2)
        s.mjw_data.nacon[0]=5
        with self.assertRaises(ValueError):read()
        s.mjw_data.nacon[0]=4;s.mjw_data.contact.pos[0,0]=np.nan
        with self.assertRaises(ValueError):read()

    def test_solver_effort_mapping_uses_real_dof_ids_and_rejects_ambiguity(self):
        model,solver,_=scene()
        r=solver_joint_effort(solver,model,['joint_a','joint_b'],ROBOT)
        self.assertEqual([x['actuator_effort_nm']for x in r['rows']],[20.,10.])
        self.assertEqual([x['all_constraint_effort_nm']for x in r['rows']],[40.,30.])
        solver.mjc_dof_to_newton_dof[:]=0
        with self.assertRaises(ValueError):solver_joint_effort(solver,model,['joint_a'],ROBOT)

    def test_opt_in_reader_does_not_write_control_and_labels_approximate_effort(self):
        model,solver,poses=scene()
        manager=NS(get_model=lambda:model,_solver=solver,get_solver_dt=lambda:1/240,
                   get_state_0=lambda:NS(body_q=poses))
        package=types.ModuleType('isaaclab_newton');physics=types.ModuleType('isaaclab_newton.physics')
        physics.NewtonManager=manager
        data=NS(joint_pos=np.zeros((1,2)),joint_vel=np.zeros((1,2)),joint_pos_target=np.zeros((1,2)),
                joint_effort_limits=np.full((1,2),3.35),joint_vel_limits=np.full((1,2),30.),
                computed_torque=np.array([[4.,2.]]),applied_torque=np.array([[3.35,2.]]))
        for value in vars(data).values():value.setflags(write=False)
        robot=NS(data=data,joint_names=['joint_a','joint_b'],set_joint_position_target=Mock())
        env=NS(scene={'robot':robot},episode_length_buf=np.array([12]),step_dt=1/30,step=Mock(),scorer=Mock())
        with patch.dict(sys.modules,{'isaaclab_newton':package,'isaaclab_newton.physics':physics}):
            r=PhysicsDiagnostics(TARGET).sample(env)
        self.assertEqual(r['solver_joint_effort']['status'],'available')
        self.assertEqual(r['contacts']['status'],'available')
        self.assertEqual(r['joint_telemetry']['computed_torque']['values'],[4.,2.])
        self.assertIn('approximate',r['joint_telemetry']['implicit_effort_semantics'])
        env.step.assert_not_called();env.scorer.assert_not_called();robot.set_joint_position_target.assert_not_called()
        json.dumps(r,allow_nan=False)

    def test_missing_solver_or_torque_is_unavailable_and_never_filled_with_zero(self):
        physics=types.ModuleType('isaaclab_newton.physics');physics.NewtonManager=NS()
        env=NS(scene={'robot':NS(data=NS(),joint_names=['elbow_flex'])},episode_length_buf=np.array([1]),step_dt=1/30)
        with patch.dict(sys.modules,{'isaaclab_newton.physics':physics}):r=PhysicsDiagnostics(TARGET).sample(env)
        for field in ['computed_torque','applied_torque','joint_effort_limits']:
            self.assertEqual(r['joint_telemetry'][field]['status'],'unavailable')
            self.assertNotIn('values',r['joint_telemetry'][field])
        self.assertEqual(r['contacts']['status'],'unavailable')
        self.assertEqual(r['solver_joint_effort']['status'],'unavailable')

    def test_phase_join_uses_actual_issued_step_and_does_not_modify_inputs(self):
        snapshots=[{'step':1},{'step':2},{'step':3},{'step':4}]
        trace=[{'step':1,'phase':'descend','joint_target':[0]},
               {'step':2,'phase':'close_centered','joint_target':[0]},
               {'step':3,'phase':'lift','command_issued':False},
               {'step':4,'phase':'lift_hold','joint_target':[0]}]
        out=phase_rows(snapshots,trace)
        self.assertEqual(out,[{'step':2,'phase':'close_centered'},{'step':4,'phase':'lift_hold'}])
        self.assertNotIn('phase',snapshots[1])


if __name__=='__main__':unittest.main()
