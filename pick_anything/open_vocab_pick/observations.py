"""Reuse the binary four-channel observation contract with target-aware GT."""

import torch

from pick_anything.grounded_pick.language import goal_caption
from pick_anything.grounded_pick.masks import quat_xyzw_matrix, sphere_boxes
from pick_anything.grounded_pick.observations import GroundedRGBMask, tensor
from pick_anything.tabletop.catalog import OBJECTS
from .geometry import HALF_EXTENTS, oriented_boxes


class OpenVocabRGBMask(GroundedRGBMask):
    def __init__(self, cfg, env):
        self.caption = env.cfg.grounding_caption
        super().__init__(cfg, env)

    def _validate_instruction(self, instruction):
        goal_caption(instruction, self.caption)

    def _predict(self, raw, instruction):
        # Only language and RGB reach the detector; no selected simulator ID.
        return self.detector.predict(raw, instruction, caption=self.caption)

    def _dump(self, raw, mask, env, directory, maximum):
        import json
        from pick_anything.grounded_pick.grounding import workspace_output

        before = self._dump_count
        super()._dump(raw, mask, env, directory, maximum)
        if self._dump_count == before:
            return
        path = workspace_output(directory) / f"{before:04d}_step_{int(env.common_step_counter):06d}.json"
        metadata = json.loads(path.read_text())
        metadata["caption"] = goal_caption(metadata["instruction"], self.caption)
        metadata["scoring_target_key"] = env.cfg.target_key
        if metadata["backend"] == "gt":
            from .mdp import TargetState
            key = env.cfg.object_keys[int(TargetState.get(env).target_index[0])]
            metadata["training_target_key"] = key
            metadata["caption"] = OBJECTS[key].caption
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    def _gt_boxes(self, env, changed, data, radius):
        from .mdp import TargetState

        state = TargetState.get(env)
        indices = state.target_index[changed]
        centers, quats, half = [], [], []
        for key in env.cfg.object_keys:
            obj = OBJECTS[key]
            asset = env.scene[key]
            quat = tensor(asset.data.root_quat_w)[changed]
            offset = quat.new_tensor([0, 0, obj.height / 2 - obj.origin_to_bottom])
            center = tensor(asset.data.root_pos_w)[changed] + torch.einsum("nij,j->ni", quat_xyzw_matrix(quat), offset)
            centers.append(center)
            quats.append(quat)
            half.append(quat.new_tensor(HALF_EXTENTS[key]).expand(len(changed), 3))
        rows = torch.arange(len(changed), device=indices.device)
        centers = torch.stack(centers, 1)[rows, indices]
        quats = torch.stack(quats, 1)[rows, indices]
        half = torch.stack(half, 1)[rows, indices]
        camera_args = (tensor(data.pos_w)[changed], tensor(data.quat_w_ros)[changed],
                       tensor(data.intrinsic_matrices)[changed])
        boxes, valid = oriented_boxes(centers, quats, half, *camera_args)
        if "orange" in env.cfg.object_keys:
            orange = indices == env.cfg.object_keys.index("orange")
            sphere, sphere_valid = sphere_boxes(centers, *camera_args, radius)
            boxes[orange], valid[orange] = sphere[orange], sphere_valid[orange]
        return boxes, valid
