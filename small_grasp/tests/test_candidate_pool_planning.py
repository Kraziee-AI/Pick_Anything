"""CPU control-flow checks for bounded candidate retention and tail fallback."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from small_grasp.tests import test_servo_motion


class CandidatePoolPlanningTests(unittest.TestCase):
    def executor(self):
        executor = test_servo_motion.ServoMotionTests.executor(self)
        executor.robot.data.joint_pos[0, executor.grip] = 1.4
        executor._pad_midpoint = Mock(return_value=(np.zeros(3), np.zeros(3)))
        executor._grasp_axes_in_body = Mock(return_value=np.eye(3))
        executor.target_centre, executor.destination, executor.contact_frame = None, None, 'open'
        def switch(opening):
            executor.open_joint_rad = opening
        executor._switch_opening_tool = Mock(side_effect=switch)
        def move(target, phase, steps):
            self.assertEqual(phase, 'open_adjust')
            executor.robot.data.joint_pos.copy_(target)
            return True
        executor._move = Mock(side_effect=move)
        return executor

    def candidates(self, count):
        result = []
        for index in range(count):
            pose = np.eye(4)
            pose[:3, 3] = [.3, -.4, .2]
            result.append({'candidate_id': f'g{index*7:06d}', 'raw_index': index*7,
                           'rank': index, 'width_m': .075, 'depth_m': 0., 'T_world_grasp': pose})
        return result

    def test_primary_narrow_success_prevents_all_tail_attempts(self):
        executor = self.executor()
        candidates = self.candidates(100)
        selected = (0., 3, candidates[3], np.zeros(3), None, None, np.eye(3), False)
        calls = []
        def plan(got, axes, **kwargs):
            self.assertIs(got, candidates)
            self.assertEqual(kwargs, {})
            calls.append(executor.open_joint_rad)
            executor.ik_attempts.append({'candidate': 3, 'rejected': 'grasp_ik'})
            return selected if executor.open_joint_rad == .8 else None
        executor._plan_repaired = Mock(side_effect=plan)
        self.assertIs(executor._plan_servo_openings(candidates), selected)
        self.assertEqual(calls, [1.4, 1., .8])
        self.assertEqual([s['candidate_stage'] for s in executor.opening_plan_stages], ['primary']*3)
        self.assertTrue(all(s['candidate_index_stop'] == 20 for s in executor.opening_plan_stages))
        self.assertTrue(all(a['raw_index'] == 21 and a['candidate_rank'] == 3 for a in executor.ik_attempts))

    def test_all_primary_openings_fail_before_tail_with_global_cache_keys(self):
        executor = self.executor()
        candidates = self.candidates(100)
        selected = (0., 99, candidates[99], np.zeros(3), None, None, np.eye(3), True)
        calls = []
        def plan(got, axes, **kwargs):
            tail = bool(kwargs)
            calls.append(('tail' if tail else 'primary', executor.open_joint_rad))
            index = 99 if tail else 0
            if tail:
                self.assertEqual(kwargs, {'candidate_start': 20, 'candidate_stop': 100})
                self.assertEqual(executor.servo_routes, {})
                self.assertEqual(executor.servo_descents, {})
                executor.servo_routes[(99, True)] = 'route99'
                executor.servo_descents[(99, True)] = 'descent99'
            else:
                executor.servo_routes[(0, False)] = 'stale-primary-route'
            executor.ik_attempts.append({'candidate': index, 'accepted': tail, 'symmetric_flip': tail})
            return selected if tail else None
        executor._plan_repaired = Mock(side_effect=plan)
        self.assertIs(executor._plan_servo_openings(candidates), selected)
        self.assertEqual(calls, [('primary', 1.4), ('primary', 1.), ('primary', .8), ('tail', 1.4)])
        self.assertEqual(executor.servo_routes, {(99, True): 'route99'})
        self.assertEqual(executor.servo_descents, {(99, True): 'descent99'})
        attempt = executor.ik_attempts[-1]
        self.assertEqual((attempt['candidate'], attempt['candidate_rank'], attempt['raw_index']), (99, 99, 693))
        self.assertEqual((attempt['candidate_stage'], attempt['opening_stage_index']), ('tail', 3))

    def test_explicit_twenty_never_creates_tail_and_opening_failure_stops(self):
        executor = self.executor()
        executor._plan_repaired = Mock(return_value=None)
        self.assertIsNone(executor._plan_servo_openings(self.candidates(20)))
        self.assertEqual(executor._plan_repaired.call_count, 3)
        self.assertTrue(all(not c.kwargs for c in executor._plan_repaired.call_args_list))
        executor = self.executor()
        executor._plan_repaired = Mock(return_value=None)
        executor._move = Mock(return_value=False)
        self.assertIsNone(executor._plan_servo_openings(self.candidates(100)))
        executor._plan_repaired.assert_called_once()
        self.assertEqual(executor.opening_plan_stages[-1]['outcome'], 'opening_adjust_terminated')

    def test_real_planner_tail_uses_original_mesh_gate_and_keeps_index_99(self):
        executor = self.executor()
        candidates = self.candidates(100)
        for candidate in candidates[21:99]:
            candidate['width_m'] = 0.
        candidates[20]['T_world_grasp'][0, 3] = .25
        def solve(point, rotation, seed=None, **kwargs):
            q = torch.zeros((1, 6)); q[0, 0] = float(point[0])
            return q, {'valid': True, 'achieved_rotation': np.eye(3).tolist(),
                       'orientation_error_deg': 0., 'achieved_pocket_world_m': np.asarray(point).tolist()}
        executor._solve = Mock(side_effect=solve)
        executor._mesh_at_q = Mock(side_effect=lambda q: {'reason': 'observed_box_mesh'} if q[0, 0] < .26 else None)
        executor._plan_cartesian_descent = Mock(side_effect=lambda start, seed, point, rotation: ([(point, seed)], {'valid': True}))
        executor._move = Mock(side_effect=AssertionError('Planning must not approach or close'))
        math_module = SimpleNamespace(quat_apply_inverse=lambda quat, vector: vector)
        with patch.dict(sys.modules, {'isaaclab.utils.math': math_module}):
            plan = executor._plan_repaired(candidates, np.eye(3), candidate_start=20, candidate_stop=100)
        self.assertEqual(plan[1], 99)
        self.assertIs(plan[2], candidates[99])
        self.assertEqual({a['candidate'] for a in executor.ik_attempts}, set(range(20, 100)))
        self.assertTrue(all(a.get('rejected') == 'observed_box_mesh' for a in executor.ik_attempts if a['candidate'] == 20))
        self.assertEqual(set(executor.servo_descents), {(99, False), (99, True)})
        self.assertIn((99, False), executor.servo_routes)
        self.assertGreater(executor._mesh_path.call_count, 0)
        executor._move.assert_not_called()
        executor.scorer.assert_not_called()

    def test_tail_selection_execution_failure_does_not_plan_second_grasp(self):
        import tempfile
        from pathlib import Path
        executor = self.executor()
        executor.env.terminal_result = None
        executor.runtime_collision = None
        executor.closure = 'first_contact'
        executor.collision_evidence, executor.mesh_evidence = {}, {}
        executor.trace, executor.close_ik_attempts, executor.close_control_events = [], [], []
        executor.lift_success = executor.executed = False
        executor.max_rise = 0.
        candidates = self.candidates(100)
        index = 41
        candidate = candidates[index]
        point = candidate['T_world_grasp'][:3, 3]
        q = torch.zeros((1, 6))
        selected = (0., index, candidate, point, q.clone(), q.clone(), np.eye(3), False)
        executor.ik_attempts = [{'candidate': index, 'candidate_id': candidate['candidate_id'], 'accepted': True,
            'candidate_stage': 'tail', 'symmetric_flip': False, 'network_pocket_world_m': point.tolist(),
            'executed_pocket_offset_m': [0., 0., 0.], 'opening_stage_index': 3}]
        executor.servo_routes[(index, False)] = (q.clone(), point.copy())
        executor.servo_descents[(index, False)] = [(point.copy(), q.clone())]
        executor._plan_servo_openings = Mock(return_value=selected)
        executor._move = Mock(return_value=True)
        executor._arrive = Mock(return_value=True)
        executor._cartesian_descend = Mock(return_value=True)
        executor._close_centered = Mock(return_value=q.clone())
        def stop(target, phase):
            executor.env.tracking = False
            return False
        executor._step = Mock(side_effect=stop)
        cache = Path(__file__).resolve().parents[1] / 'cache'
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            executor.directory = Path(temporary)
            result = executor.execute(candidates, 'single')
        executor._plan_servo_openings.assert_called_once()
        executor._cartesian_descend.assert_called_once()
        executor._close_centered.assert_called_once()
        executor._switch_opening_tool.assert_not_called()
        self.assertFalse(result['grasp_lift_success'])
        self.assertEqual(result['chosen']['candidate'], 41)
        self.assertEqual(result['chosen']['raw_index'], 287)
        self.assertEqual(result['chosen']['candidate_stage'], 'tail')
        self.assertEqual(result['execution_config']['candidate_pool_count'], 100)
        self.assertEqual(result['execution_config']['primary_candidate_limit'], 20)


if __name__ == '__main__':
    unittest.main()
