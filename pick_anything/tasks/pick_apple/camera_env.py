"""Camera PPO environment with confirmed grasp milestones."""

from isaaclab.envs import ManagerBasedRLEnv

from .staged_mdp import StagedPickState
from .camera_progress import CameraStageProgress
from .camera_mdp import _tensor
import torch


class CameraPickEnv(ManagerBasedRLEnv):
    def load_managers(self):
        # Install before observation terms first read progress. Shared rewards
        # and reset events keep accessing this environment's own state object.
        state = StagedPickState.get(self)
        state.progress = CameraStageProgress(self.num_envs, self.device)
        super().load_managers()
        print(f"[CAMERA] grasp milestone: {state.progress.grasp_semantics}", flush=True)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Shared resets restore arm joints and reset-bank rows, but HOME leaves
        # the previous gripper position/target behind. Match the first episode
        # by resetting only HOME grippers to the existing default reference.
        state = self.staged_pick_state
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        home_ids = ids[state.birth_stage[ids] == 0]
        if home_ids.numel():
            robot = self.scene["robot"]
            joint_ids = [state.gripper_id]
            positions = _tensor(robot.data.default_joint_pos)[home_ids][:, joint_ids]
            robot.write_joint_position_to_sim_index(position=positions, env_ids=home_ids, joint_ids=joint_ids)
            robot.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(positions), env_ids=home_ids, joint_ids=joint_ids,
            )
            robot.set_joint_position_target_index(target=positions, env_ids=home_ids, joint_ids=joint_ids)
