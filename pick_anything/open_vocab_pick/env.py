"""Count each environment's first HOME episode before automatic reset."""

import json

import torch
from isaaclab.envs import ManagerBasedRLEnv
from .mdp import TargetState


class OpenVocabPickEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        self._seen = None
        self._results = []
        super().__init__(cfg, **kwargs)

    def _reset_idx(self, env_ids):
        if self.cfg.evaluate_first_episode:
            if self._seen is None:
                self._seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            ids = ids[(self.episode_length_buf[ids] > 0) & ~self._seen[ids]]
            state = TargetState.get(self)
            for idx in ids.tolist():
                causes = {name: bool(self.termination_manager.get_term(name)[idx])
                          for name in ("success", "time_out", "dropped_or_invalid")}
                self._results.append({"env_id": idx, "target_key": self.cfg.object_keys[int(state.target_index[idx])],
                                      "grasped": bool(state.grasped[idx]), "rise_m": float(state.rise[idx]),
                                      "max_rise_m": float(state.max_rise[idx]),
                                      "wrong_object_lifted": bool(state.wrong_lift_ever[idx]), **causes})
                self._seen[idx] = True
            if len(self._results) == self.num_envs:
                params = self.cfg.observations.rgb_mask.image.params
                result = {"episodes": self.num_envs, "birth_policy": "HOME", "mask_backend": params["backend"],
                          "instruction": params["instruction"], "caption": self.cfg.grounding_caption,
                          "image_history": params["image_history"],
                          "wrong_object_lift_rate": sum(row["wrong_object_lifted"] for row in self._results) / self.num_envs,
                          "success_rate": sum(row["success"] for row in self._results) / self.num_envs,
                          "results": self._results}
                print("OPEN_VOCAB_PICK_EVAL_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
                raise SystemExit(0)
        super()._reset_idx(env_ids)
