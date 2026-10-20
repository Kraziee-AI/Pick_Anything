"""CPU regression checks for camera grasp confirmation and task isolation."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import torch

import pick_anything.tasks  # noqa: F401 -- register camera tasks
from isaaclab.envs import ManagerBasedRLEnv
from pick_anything.tasks.pick_apple.camera_env import CameraPickEnv, StagedPickState
from pick_anything.tasks.pick_apple.camera_mdp import CameraApproachProgress
from pick_anything.tasks.pick_apple.camera_progress import CameraStageProgress
from pick_anything.tasks.pick_apple.stages import (
    ConfirmedGraspStageProgress, PickStage, STAGE_NAMES, StageProgress,
)


class CameraGraspTests(unittest.TestCase):
    @staticmethod
    def conditions(n=1):
        return {name: torch.zeros(n, dtype=torch.bool) for name in STAGE_NAMES}

    def test_pregrasp_contact_does_not_block_alignment_or_approach(self):
        fixed, legacy = ConfirmedGraspStageProgress(1, "cpu"), StageProgress(1, "cpu")
        for progress in (fixed, legacy):
            progress.stage[:] = PickStage.PREGRASP
            progress.first_step[:] = False
            for step in range(1, 5):
                cond = self.conditions()
                cond["GRASP"][:] = True
                progress.update(torch.tensor([step]), cond, {"GRASP": 3})
        self.assertFalse(fixed.grasped.item())
        self.assertTrue(legacy.grasped.item())
        self.assertEqual(fixed.confirm_count[0, PickStage.GRASP].item(), 1)

        reward = object.__new__(CameraApproachProgress)
        reward._prev_dist, reward._has_prev = torch.tensor([0.04]), torch.tensor([True])
        state = SimpleNamespace(progress=fixed, orange_pos=torch.zeros(1, 3),
                                pocket_pos=torch.tensor([[0.03, 0., 0.]]), update=lambda env: None)
        self.assertAlmostEqual(reward(SimpleNamespace(staged_pick_state=state)).item(), 1.0, places=5)

        for progress in (fixed, legacy):
            cond = self.conditions()
            cond["PREGRASP"] = ~progress.grasped
            progress.update(torch.tensor([5]), cond, {"PREGRASP": 1, "GRASP": 3})
        self.assertEqual(fixed.stage.item(), PickStage.GRASP)
        self.assertEqual(legacy.stage.item(), PickStage.PREGRASP)
        self.assertTrue(fixed.just_completed[0, PickStage.PREGRASP].item())
        self.assertFalse(fixed.grasped.item())

    def test_grasp_requires_fresh_consecutive_contact_and_pays_once(self):
        progress = ConfirmedGraspStageProgress(1, "cpu")
        progress.stage[:] = PickStage.PREGRASP
        progress.first_step[:] = False
        cond = self.conditions()
        cond["GRASP"][:] = True
        for step in range(1, 8):
            progress.update(torch.tensor([step]), cond, {"GRASP": 3})
        cond["PREGRASP"][:] = True
        cond["GRASP"][:] = False
        progress.update(torch.tensor([8]), cond, {"PREGRASP": 1, "GRASP": 3})
        for step, contact in enumerate([True, True, False, True, True, True], 9):
            cond = self.conditions()
            cond["GRASP"][:] = contact
            progress.update(torch.tensor([step]), cond, {"GRASP": 3})
            self.assertEqual(progress.grasped.item(), step == 14)
            self.assertEqual(progress.stage.item(), PickStage.LIFT if step == 14 else PickStage.GRASP)
        self.assertTrue(progress.just_completed[0, PickStage.GRASP].item())
        progress.update(torch.tensor([15]), cond, {"GRASP": 3})
        self.assertTrue(progress.grasped.item())
        self.assertFalse(progress.just_completed[0, PickStage.GRASP].item())
        reward = object.__new__(CameraApproachProgress)
        reward._prev_dist, reward._has_prev = torch.tensor([0.04]), torch.tensor([True])
        state = SimpleNamespace(progress=progress, orange_pos=torch.zeros(1, 3),
                                pocket_pos=torch.tensor([[0.03, 0., 0.]]), update=lambda env: None)
        self.assertEqual(reward(SimpleNamespace(staged_pick_state=state)).item(), 0.0)

    def test_first_step_bank_seeds_and_partial_reset(self):
        progress = ConfirmedGraspStageProgress(4, "cpu")
        progress.reset(stage_seed=torch.tensor([0, 1, 2, 3]))
        self.assertEqual(progress.grasped.tolist(), [False, False, True, True])
        cond = self.conditions(4)
        for values in cond.values():
            values[:] = True
        progress.update(torch.ones(4, dtype=torch.long), cond, {"GRASP": 3, "LIFT": 4})
        self.assertFalse(progress.just_completed.any().item())
        self.assertEqual(progress.stage.tolist(), [0, 1, 2, 3])
        progress.reset(torch.tensor([2]), stage_seed=torch.tensor([0]))
        self.assertEqual(progress.grasped.tolist(), [False, False, False, True])
        self.assertEqual(progress.stage.tolist(), [0, 1, 0, 3])
        self.assertTrue(progress.first_step[2].item())

    def test_installs_before_managers_and_does_not_patch_shared_state(self):
        state = SimpleNamespace(progress=StageProgress(2, "cpu"))
        legacy = StageProgress(2, "cpu")
        env = object.__new__(CameraPickEnv)
        env.__dict__.update(scene=SimpleNamespace(num_envs=2), sim=SimpleNamespace(device="cpu"))
        seen = []
        with patch.object(StagedPickState, "get", return_value=state), patch.object(
            ManagerBasedRLEnv, "load_managers",
            lambda instance: seen.append(isinstance(state.progress, ConfirmedGraspStageProgress)),
        ):
            env.load_managers()
        self.assertEqual(seen, [True])
        self.assertIs(type(legacy), StageProgress)

    def test_camera_and_filtered_entry_points_share_the_fix(self):
        expected = "pick_anything.tasks.pick_apple.camera_env:CameraPickEnv"
        for task in ("Pick-Apple-SO101-Staged-Camera-v8", "Pick-Apple-SO101-Staged-Camera-Filtered-v0"):
            self.assertEqual(gym.spec(task).entry_point, expected)
        for task in ("Pick-Apple-SO101-Staged-v0", "Pick-Apple-SO101-Staged-Camera-Distill-v10"):
            self.assertEqual(gym.spec(task).entry_point, "isaaclab.envs:ManagerBasedRLEnv")

    def test_existing_physical_success_rule_is_preserved(self):
        for cls in (StageProgress, ConfirmedGraspStageProgress):
            progress = cls(1, "cpu")
            progress.stage[:] = PickStage.PREGRASP
            progress.first_step[:] = False
            cond = self.conditions()
            cond["RELEASE"][:] = True
            for step in range(1, 11):
                progress.update(torch.tensor([step]), cond, {})
                self.assertEqual(progress.success.item(), step == 10)
            self.assertEqual(progress.time_to_success.item(), 10)

    def test_camera_already_held_object_advances_without_reopening(self):
        p = CameraStageProgress(1, "cpu")
        p.stage[:] = PickStage.PREGRASP
        p.first_step[:] = False
        cond = self.conditions();cond['GRASP'][:] = True
        for step in range(1,6):
            p.update(torch.tensor([step]),cond,{'PREGRASP':1,'GRASP':3})
            self.assertEqual(p.stage.item(), 1 if step<3 else (2 if step<5 else 3))
            self.assertEqual(p.grasped.item(),step==5)
        self.assertTrue(p.just_completed[0,2].item())
        self.assertFalse(cond['PREGRASP'].item())  # caller's conditions are not modified

    def test_camera_early_contact_break_and_first_step_do_not_unlock(self):
        p=CameraStageProgress(2,'cpu');p.stage[:]=1
        cond=self.conditions(2);cond['GRASP'][:]=True
        p.update(torch.ones(2,dtype=torch.long),cond,{'GRASP':3})
        self.assertEqual(p.pregrasp_contact_count.tolist(),[0,0])
        for step,contact in enumerate([True,True,False,True,True],2):
            cond['GRASP'][:]=contact
            p.update(torch.full((2,),step),cond,{'GRASP':3})
        self.assertEqual(p.stage.tolist(),[1,1])
        p.reset(torch.tensor([0]),stage_seed=torch.tensor([3]))
        self.assertEqual(p.pregrasp_contact_count.tolist(),[0,2])
        self.assertEqual(p.grasped.tolist(),[True,False])

    def test_home_reset_restores_only_home_gripper_and_target(self):
        calls=[]
        robot=SimpleNamespace(data=SimpleNamespace(default_joint_pos=torch.tensor([[0.,0.2555],[0.,0.2555]])))
        for method in ('write_joint_position_to_sim_index','write_joint_velocity_to_sim_index','set_joint_position_target_index'):
            setattr(robot,method,lambda _method=method,**kwargs:calls.append((_method,kwargs)))
        class Scene(dict):num_envs=2
        env=object.__new__(CameraPickEnv)
        env.__dict__.update(scene=Scene(robot=robot),sim=SimpleNamespace(device='cpu'),
                            staged_pick_state=SimpleNamespace(birth_stage=torch.tensor([0,4]),gripper_id=1))
        with patch.object(ManagerBasedRLEnv,'_reset_idx',lambda self,ids:None):
            env._reset_idx(torch.tensor([0,1]))
        self.assertEqual(len(calls),3)
        for name,kwargs in calls:
            self.assertEqual(kwargs['env_ids'].tolist(),[0])
            self.assertEqual(kwargs['joint_ids'],[1])
            value=kwargs.get('position',kwargs.get('velocity',kwargs.get('target')))
            self.assertAlmostEqual(value.item(),0 if 'velocity' in name else 0.2555,places=5)

    def test_same_control_step_cannot_supply_extra_confirmation(self):
        p=CameraStageProgress(1,'cpu');p.stage[:]=1;p.first_step[:]=False
        cond=self.conditions();cond['GRASP'][:]=True
        for step in range(1,6):
            p.update(torch.tensor([step]),cond,{'PREGRASP':1,'GRASP':3})
            before={name:value.clone() for name,value in vars(p).items() if isinstance(value,torch.Tensor)}
            p.update(torch.tensor([step]),cond,{'PREGRASP':1,'GRASP':3})
            for name,value in before.items():torch.testing.assert_close(getattr(p,name),value)
        self.assertTrue(p.grasped.item())
        self.assertEqual(p.stage.item(),3)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
