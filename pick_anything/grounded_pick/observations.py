"""Four-channel goal-conditioned vision; detector never accesses scene state."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from isaaclab.managers import ManagerTermBase

from .grounding import GroundingDino, workspace_output
from .masks import boxes_to_masks, orange_caption, sphere_boxes


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


class GroundedRGBMask(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        self._validate_instruction(params["instruction"])
        if params["backend"] not in ("gt", "dino"):
            raise ValueError("mask backend must be gt or dino")
        self.detector = None
        if params["backend"] == "dino":
            self.detector = GroundingDino(
                model_name=params["model_name"], device=params["grounding_device"] or env.device,
                cache_dir=params["cache_dir"], local_files_only=params["local_files_only"],
                box_threshold=params["box_threshold"], text_threshold=params["text_threshold"],
            )
        self._frame = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._boxes = torch.zeros((env.num_envs, 4), device=env.device)
        self._valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._confidence = torch.zeros(env.num_envs, device=env.device)
        self._exposure = torch.ones((env.num_envs, 1, 1, 1), device=env.device)
        self._white_balance = torch.ones((env.num_envs, 3, 1, 1), device=env.device)
        self._dump_count = 0
        self._history = None
        self._history_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._history_frame = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.reset()

    def _validate_instruction(self, instruction):
        orange_caption(instruction)

    def _gt_boxes(self, env, changed, data, radius):
        center = tensor(env.scene["orange"].data.root_pos_w)[changed]
        return sphere_boxes(
            center, tensor(data.pos_w)[changed], tensor(data.quat_w_ros)[changed],
            tensor(data.intrinsic_matrices)[changed], radius,
        )

    def _predict(self, raw, instruction):
        return self.detector.predict(raw, instruction)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._frame[ids] = -1
        self._boxes[ids] = 0
        self._valid[ids] = False
        self._confidence[ids] = 0
        self._history_initialized[ids] = False
        self._history_frame[ids] = -1
        self._exposure[ids] = torch.empty_like(self._exposure[ids]).uniform_(0.8, 1.2)
        self._white_balance[ids] = torch.empty_like(self._white_balance[ids]).uniform_(0.9, 1.1)

    def __call__(
        self, env, sensor_cfg, backend, instruction, model_name, grounding_device,
        cache_dir, local_files_only, box_threshold, text_threshold, radius,
        policy_height, policy_width, box_jitter, mask_dropout, dump_dir, dump_every, dump_max,
        image_history=1,
    ):
        camera = env.scene.sensors[sensor_cfg.name]
        data = camera.data  # update RGB and camera pose together before reading either
        raw = tensor(data.output["rgb"])[..., :3]
        frames = tensor(camera.frame)
        changed_mask = frames != self._frame
        if getattr(self, "_last_instruction", instruction) != instruction:
            self._validate_instruction(instruction)
            changed_mask = torch.ones_like(changed_mask)
            # History for a different target must not leak into the new goal.
            if image_history > 1:
                self._history_initialized[:] = False
        self._last_instruction = instruction
        if backend == "dino" and getattr(env.cfg, "evaluate_first_episode", False):
            # Autoreset environments whose scored episode is over still render.
            # Do not spend detector inference on those unscored later episodes.
            completed = getattr(env, "_evaluation_seen", None)
            if completed is None:
                completed = getattr(env, "_seen", None)
            if completed is not None:
                changed_mask &= ~completed
        changed = changed_mask.nonzero(as_tuple=False).flatten()
        training = env.cfg.observations.rgb_mask.enable_corruption
        if changed.numel():
            if backend == "gt":
                # This is the only branch allowed to read orange coordinates.
                boxes, valid = self._gt_boxes(env, changed, data, radius)
                confidence = valid.float()
                if training:
                    half = (boxes[:, 2:] - boxes[:, :2]) / 2
                    mid = (boxes[:, :2] + boxes[:, 2:]) / 2
                    mid += torch.empty_like(mid).uniform_(-box_jitter, box_jitter) * half
                    half *= torch.empty_like(half).uniform_(1 - box_jitter, 1 + box_jitter)
                    boxes = torch.cat([mid - half, mid + half], -1)
                    valid &= torch.rand_like(confidence) >= mask_dropout
            else:
                boxes, valid, confidence = self._predict(raw[changed], instruction)
            self._boxes[changed], self._valid[changed] = boxes, valid
            self._confidence[changed] = confidence
            self._frame[changed] = frames[changed]
        height, width = raw.shape[1:3]
        mask = boxes_to_masks(self._boxes, self._valid, height, width)
        rgb = raw.permute(0, 3, 1, 2).contiguous().float() / 255.0
        if training:
            rgb = (rgb * self._exposure * self._white_balance).clamp(0, 1)
        # Bilinear RGB and nearest mask preserve the binary mask contract.
        rgb = F.interpolate(rgb, (policy_height, policy_width), mode="bilinear", align_corners=False)
        policy_mask = F.interpolate(mask, (policy_height, policy_width), mode="nearest")
        env.grounding_diagnostics = {
            "backend": backend, "instruction": instruction, "boxes": self._boxes,
            "detected": self._valid, "confidence": self._confidence,
        }
        if dump_dir and changed.numel() and int(env.common_step_counter) % dump_every == 0:
            self._dump(raw, mask, env, dump_dir, dump_max)
        return self._stack_image(torch.cat([rgb, policy_mask], dim=1), frames, image_history)

    def _stack_image(self, image, frames, image_history):
        """Stack aligned RGB/mask frames once per render; reset rows independently."""
        if image_history not in (1, 3):
            raise ValueError("image_history must be 1 or 3")
        if image_history == 1:
            return image
        shape = (image.shape[0], image_history, *image.shape[1:])
        if self._history is None:
            self._history = image.new_zeros(shape)
        elif tuple(self._history.shape) != shape:
            raise ValueError("Image history shape cannot change while an environment is running")
        initial = ~self._history_initialized
        fresh = (frames != self._history_frame) & ~initial
        # clone handles overlap while shifting; no history changes on duplicate reads.
        self._history[fresh, :-1] = self._history[fresh, 1:].clone()
        self._history[fresh, -1] = image[fresh]
        self._history[initial] = image[initial, None].expand(-1, image_history, -1, -1, -1)
        self._history_initialized[:] = True
        self._history_frame.copy_(frames)
        return self._history.reshape(image.shape[0], 4 * image_history, *image.shape[2:]).clone()

    def _dump(self, raw, mask, env, directory, maximum):
        if self._dump_count >= maximum:
            return
        import json
        from PIL import Image

        output = workspace_output(directory)
        output.mkdir(parents=True, exist_ok=True)
        stem = f"{self._dump_count:04d}_step_{int(env.common_step_counter):06d}"
        Image.fromarray(raw[0].detach().cpu().numpy()).save(output / f"{stem}_rgb.png")
        Image.fromarray((mask[0, 0] * 255).byte().cpu().numpy()).save(output / f"{stem}_mask.png")
        metadata = {"backend": env.grounding_diagnostics["backend"],
                    "instruction": env.grounding_diagnostics["instruction"],
                    "box": self._boxes[0].tolist(), "detected": bool(self._valid[0]),
                    "confidence": float(self._confidence[0])}
        (output / f"{stem}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        self._dump_count += 1
