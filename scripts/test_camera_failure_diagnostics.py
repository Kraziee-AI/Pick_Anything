"""Regression check: per-step camera metrics and terminal classification survive reset."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import torch
from scripts.diagnose_camera_failures import CameraFailureDiagnostics


class CameraDiagnosticsTests(unittest.TestCase):
    def test_first_episode_conditions_release_snapshot_and_frames(self):
        pos = torch.tensor([[0., 0., 0.13]])
        p = NS(stage=torch.tensor([1]), confirm_count=torch.zeros(1, 8, dtype=torch.long),
               stable_count=torch.tensor([0]), grasped=torch.tensor([False]), lifted=torch.tensor([True]),
               over_box_latched=torch.tensor([False]), in_box_latched=torch.tensor([True]))
        state = NS(progress=p, orange_pos=pos, pocket_pos=pos + torch.tensor([[0.02, 0., 0.]]),
                   orange_vel=torch.zeros(1, 3), claw_force=torch.tensor([3.]), orange_force=torch.tensor([0.3]),
                   gripper_pos=torch.tensor([0.4]), orange_local=torch.tensor([[0., 0., 0.028]]),
                   box_pos=torch.tensor([[0., 0., 0.1]]), release_clearance_m=0.055)
        class Scene(dict):
            env_origins = torch.zeros(1, 3)
            sensors = {'wrist_camera': NS(cfg=NS(height=12, width=16),
                                         data=NS(output={'rgb':torch.zeros(1, 12, 16, 3, dtype=torch.uint8)}))}
        scene = Scene(robot=NS(data=NS(joint_pos=torch.zeros(1,6),joint_pos_target=torch.ones(1,6))),
                      orange=NS(data=NS(root_pos_w=pos)))
        env = NS(num_envs=1, device='cpu', scene=scene, staged_pick_state=state,
                 episode_length_buf=torch.tensor([1]), action_manager=NS(action=torch.zeros(1,6)),
                 reward_manager=NS(_episode_sums={'approach_progress':torch.tensor([2.5])}),
                 termination_manager=NS(get_term=lambda name:torch.tensor([name=='time_out'])))
        out=Path('outputs/camera_ppo/failure_diagnosis');out.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out) as directory:
            d=CameraFailureDiagnostics(directory,1,1/30,10)
            d.start_episodes(env,[0])
            for step in (1,1,2,3):
                env.episode_length_buf[:]=step;d.capture(env)
            env.episode_length_buf[:]=4;d.capture(env,[0],terminal=True);d.end_episodes(env,[0])
            env.episode_length_buf[:]=1;d.start_episodes(env,[0]);d.capture(env)
            d.finish({'episodes':1},42)
            report=json.loads((Path(directory)/'summary.json').read_text())
            ep=report['episodes'][0]
            self.assertEqual(ep['steps'],4)
            self.assertEqual(ep['near_but_not_open_steps'],4)
            self.assertEqual(ep['max_gripper_when_near_rad'],float(torch.tensor(0.4)))
            self.assertEqual(ep['release_inside_steps'],4)
            self.assertEqual(ep['release_ready_steps'],0)
            self.assertEqual(ep['failure_category'],'inside_box_but_gripper_not_clear')
            self.assertEqual(ep['reward_totals']['approach_progress'],2.5)
            self.assertTrue(ep['terminal_snapshot']['release_inside'])
            self.assertFalse(ep['terminal_snapshot']['release_clear'])
            self.assertTrue(list((Path(directory)/'frames').glob('*terminal_previous_obs.png')))


if __name__=='__main__':
    unittest.main()
