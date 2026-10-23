"""Independent safety reserve and ranking integration boundaries (CPU only)."""
import sys
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from scipy.spatial import ConvexHull

from small_grasp.sim_robot import POCKET, SERVO_CONFIG, _GripperMeshClearance
from small_grasp.tests import test_servo_motion


class TrackingPlanningTests(unittest.TestCase):
    def executor(self):
        executor = test_servo_motion.ServoMotionTests().executor()
        executor.target_centre, executor.destination, executor.contact_frame = None, None, 'open'
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.zeros(3)))
        def solve(point, rotation, seed=None, **kwargs):
            return torch.zeros((1, 6)), {'valid': True, 'achieved_rotation': np.eye(3).tolist(),
                'orientation_error_deg': 0., 'achieved_pocket_world_m': np.asarray(point).tolist()}
        executor._solve = Mock(side_effect=solve)
        executor._plan_cartesian_descent = Mock(return_value=([(np.zeros(3), torch.zeros((1, 6)))], {'valid': True}))
        executor.candidate_contact_events = []
        executor._planning_closure_envelope=Mock(return_value={'valid':True,'required_vertical_correction_m':0.})
        return executor

    def test_buffer_only_tightens_original_mesh_gate(self):
        vertices = np.array([[x,y,z] for x in (-.005,.005) for y in (-.005,.005) for z in (-.01,.01)])
        triangles = vertices[ConvexHull(vertices).simplices]
        geometry = SimpleNamespace(meshes={label:(triangles,['mesh_'+label]*len(triangles)) for label in ('fixed','moving')},
            parent_pivot=np.array([.1,0.,0.]), child_pivot=np.zeros(3),
            parent_rotation=np.eye(3), child_rotation=np.eye(3), axis=np.array([0.,0.,1.]),
            moving_to_fixed=lambda points,q: points+[.1,0.,0.])
        clearance = _GripperMeshClearance(geometry)
        pocket = np.asarray(POCKET)+[0.,-.4,.062]  # lowest vertex z=.052
        self.assertIsNone(clearance.check(pocket,np.eye(3),0.))
        collision = clearance.check(pocket,np.eye(3),0.,table_buffer_m=.008)
        self.assertAlmostEqual(collision['required_table_margin_m'],.016)
        self.assertAlmostEqual(collision['required_vertical_correction_m'],.005)
        self.assertIn('mesh',collision)
        self.assertIn('minimum_vertex_world_m',collision)
        self.assertEqual(SERVO_CONFIG['mesh_table_margin_m'],.008)

    def test_feedback_buffer_is_explicit_and_off_calls_original_path(self):
        executor = self.executor()
        q = torch.zeros((1,6))
        executor.arm_tracking_feedback = False
        executor._planning_mesh_path(q,q,'descend')
        executor._mesh_path.assert_called_once_with(q,q,'descend')
        executor._mesh_path.reset_mock()
        executor.arm_tracking_feedback = True
        executor._planning_mesh_path(q,q,'descend')
        executor._mesh_path.assert_called_once_with(q,q,'descend',table_buffer_m=.008)

    def test_tighter_margin_never_expands_network_point_correction_limit(self):
        executor = self.executor()
        executor.arm_tracking_feedback = True
        executor._mesh_at_q.return_value = {'reason':'table_mesh','required_vertical_correction_m':.009}
        pose = np.eye(4);pose[:3,3]=[.25,-.4,.1]
        candidate = {'T_world_grasp':pose.tolist(),'depth_m':0.,'width_m':.075}
        with patch.dict(sys.modules,{'isaaclab.utils.math':SimpleNamespace(quat_apply_inverse=lambda q,v:v)}):
            selected = executor._plan_repaired([candidate],np.eye(3))
        self.assertIsNone(selected)
        self.assertEqual(executor._solve.call_count,4)  # no shifted re-solve
        self.assertTrue(all(e['rejected']=='table_mesh' for e in executor.ik_attempts))
        self.assertTrue(all(e['required_table_margin_m']==.016 for e in executor.ik_attempts))
        self.assertEqual(SERVO_CONFIG['mesh_max_vertical_correction_m'],.008)

    def test_contact_ranking_only_changes_cost_between_accepted_full_poses(self):
        executor = self.executor()
        candidates=[]
        for i,x in enumerate((.25,.3)):
            pose=np.eye(4);pose[:3,3]=[x,-.4,.1]
            candidates.append({'candidate_id':f'g{i}','T_world_grasp':pose.tolist(),'depth_m':0.,'width_m':.075})
        original=[np.array(x['T_world_grasp']) for x in candidates]
        executor._contact_band_diagnostic=Mock(side_effect=lambda point,rotation:{'penalty_deg_equivalent':8. if point[0]<.26 else 0.})
        with patch.dict(sys.modules,{'isaaclab.utils.math':SimpleNamespace(quat_apply_inverse=lambda q,v:v)}):
            executor.candidate_contact_ranking=False
            baseline=executor._plan_repaired(candidates,np.eye(3))
            executor._contact_band_diagnostic.assert_not_called()
            executor.candidate_contact_ranking=True
            selected=executor._plan_repaired(candidates,np.eye(3))
        self.assertEqual(baseline[1],0)
        self.assertEqual(selected[1],1)
        self.assertEqual(executor.candidate_contact_events[-1]['baseline_candidate'],0)
        self.assertEqual(executor.candidate_contact_events[-1]['selected_candidate'],1)
        for before,candidate in zip(original,candidates):
            np.testing.assert_array_equal(before,candidate['T_world_grasp'])
        np.testing.assert_array_equal(selected[3],original[1][:3,3])

    def test_contact_cloud_uses_saved_real_camera_target_points_and_calibration(self):
        from small_grasp.infer import ROOT
        executor=self.executor()
        executor.candidate_contact_ranking=True
        with tempfile.TemporaryDirectory(dir=ROOT/'cache') as tmp:
            executor.directory=Path(tmp)
            executor.observation_dir=Path(tmp)/'actual_view'
            executor.observation_dir.mkdir()
            (executor.directory/'proposals').mkdir()
            points=np.array([[x*.004,0.,.1] for x in range(40)])
            np.savez(executor.directory/'proposals/point_clouds.npz',target_xyz_camera_m=points)
            transform=np.eye(4);transform[:3,3]=[.3,-.4,.02]
            (executor.observation_dir/'world_from_camera.json').write_text(json.dumps(transform.tolist()))
            diagnostic=executor._contact_band_diagnostic(np.array([.3,-.4,.12]),np.eye(3))
            self.assertEqual(diagnostic['status'],'scored')
            self.assertEqual(diagnostic['point_count'],40)
            np.testing.assert_allclose(executor._candidate_target_cloud.points_world_m,
                                       points+transform[:3,3])
            self.assertIn('actual_camera_calibration',diagnostic['source'])

    def test_band_first_core_uses_priority_groups_instead_of_additive_cost(self):
        executor=self.executor();executor.candidate_contact_ranking=True
        original_solve=executor._solve.side_effect
        def solve(point,rotation,seed=None,**kwargs):
            q,diag=original_solve(point,rotation,seed,**kwargs)
            diag['orientation_error_deg']=10. if point[0]>.26 else 0.
            return q,diag
        executor._solve.side_effect=solve
        executor._contact_band_diagnostic=Mock(side_effect=lambda point,rotation:{
            'enabled':True,'status':'scored','penalty_deg_equivalent':3. if point[0]<.26 else 0.,
            'observed_median_y_offset_m':.02 if point[0]<.26 else 0.})
        candidates=[]
        for i,x in enumerate((.25,.3)):
            pose=np.eye(4);pose[:3,3]=[x,-.4,.1]
            candidates.append({'candidate_id':f'g{i}','T_world_grasp':pose.tolist(),'depth_m':0.,'width_m':.075})
        with patch.dict(sys.modules,{'isaaclab.utils.math':SimpleNamespace(quat_apply_inverse=lambda q,v:v)}):
            executor.candidate_contact_policy='soft8'
            self.assertEqual(executor._plan_repaired(candidates,np.eye(3))[1],0)
            executor.candidate_contact_policy='band_first'
            selected=executor._plan_repaired(candidates,np.eye(3))
        self.assertEqual(selected[1],1)
        self.assertEqual(executor.candidate_contact_events[-1]['selected_priority'],[0,10.1])

    def test_full_closure_envelope_adapts_once_within_original_total_limit(self):
        executor=self.executor();executor.arm_tracking_feedback=True
        executor._planning_closure_envelope.side_effect=[
            {'valid':False,'required_vertical_correction_m':.003},
            {'valid':True,'required_vertical_correction_m':0.},
            {'valid':False,'required_vertical_correction_m':.009}]
        pose=np.eye(4);pose[:3,3]=[.25,-.4,.1]
        candidate={'candidate_id':'net','T_world_grasp':pose.tolist(),'depth_m':0.,'width_m':.075}
        with patch.dict(sys.modules,{'isaaclab.utils.math':SimpleNamespace(quat_apply_inverse=lambda q,v:v)}):
            selected=executor._plan_repaired([candidate],np.eye(3))
        self.assertIsNotNone(selected)
        self.assertAlmostEqual(selected[3][2]-.1,.0031)
        self.assertLessEqual(selected[3][2]-.1,.008)
        np.testing.assert_array_equal(candidate['T_world_grasp'],pose)
        rejected=[e for e in executor.ik_attempts if e.get('rejected')]
        self.assertEqual(rejected[-1]['rejected'],'closure_envelope_vertical_correction_exceeds_limit')
        self.assertGreater(rejected[-1]['required_total_vertical_correction_m'],.008)


if __name__=='__main__':unittest.main()
