"""Passive, single-episode PPO recording through Newton cameras and ffmpeg.

Use as --external_callback scripts.record_rl_demo.install_recorder with play_pick.py.
PICK_RL_DEMO_OUTPUT selects a new MP4 under this repository's outputs/ directory.
Policy weights, action computation, rewards, and termination criteria are unchanged.
The visible-tray presentation asset changes rendered appearance, not tray physics.
"""
import json
import os
from pathlib import Path
import sys


def install_recorder():
    import atexit
    import numpy as np
    from PIL import ImageDraw, ImageFont
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import CameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg
    from rsl_rl.runners import OnPolicyRunner
    from small_grasp.video import EpisodeVideo, OBSERVER_EYE, OBSERVER_TARGET

    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ.get("PICK_RL_DEMO_OUTPUT", root / "outputs/rl_demo/demo.mp4")).resolve()
    if not output.is_relative_to(root / "outputs") or output.exists() or output.suffix != ".mp4":
        raise ValueError("PICK_RL_DEMO_OUTPUT must be a new MP4 inside this repository's outputs/")
    budget = int(os.environ.get("PICK_RL_DEMO_MAX_STEPS", "2700"))
    if budget <= 0:
        raise ValueError("PICK_RL_DEMO_MAX_STEPS must be positive")
    state = {"steps": 0, "terminal": None, "recorder": None, "stage": "HOME"}

    def close_on_exit():
        recorder = state["recorder"]
        if recorder is not None and not recorder.closed:
            recorder.finish(None)

    atexit.register(close_on_exit)

    class PPOVideo(EpisodeVideo):
        def _default_status(self):
            return "Camera PPO | wrist RGB + proprioception | stage: " + state["stage"]

        def _result_status(self, result):
            if result is None:
                return "Recording interrupted | no benchmark result"
            if result.get("termination"):
                labels = ", ".join(k for k, v in result["termination"].items() if v)
                return "Original task termination: " + (labels or "unspecified")
            return "Recording duration limit | no terminal result"

        def _compose(self, status=None):
            from PIL import Image
            frame = Image.fromarray(super()._compose(status)).copy()
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 1280, 43), fill=(18, 22, 28))
            draw.text((14, 9), f"SO101 Camera PPO | seed {self.seed} | sim {self.steps*self.step_dt:.2f}s",
                      font=ImageFont.truetype("DejaVuSans.ttf", 21), fill="white")
            return np.asarray(frame)

    original_init = ManagerBasedRLEnv.__init__
    original_step = ManagerBasedRLEnv.step
    original_reset = ManagerBasedRLEnv._reset_idx

    def initialize(self, cfg, *args, **kwargs):
        if cfg.scene.num_envs != 1 or cfg.events.reset_staged.params["forced_stage"] != 0:
            raise ValueError("Demo requires one environment and the normal HOME playback configuration")
        # The existing visual-only USD references the original tray physics asset.
        # Newton otherwise omits its collision-only panels from the overview.
        visible_tray = root / "small_grasp/assets/cardboard_box_visible.usda"
        if not visible_tray.is_file():
            raise FileNotFoundError(visible_tray)
        cfg.scene.cardboard_box.spawn.usd_path = str(visible_tray)
        cfg.scene.recording_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/DemoRecordingCamera", update_period=0.,
            width=1280, height=720, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=22., clipping_range=(.01, 20.)),
            renderer_cfg=NewtonWarpRendererCfg(),
        )
        original_init(self, cfg, *args, **kwargs)
        observer = self.scene.sensors["recording_camera"]
        origin = self.scene.env_origins[0]
        observer.set_world_poses_from_view(
            torch.tensor([OBSERVER_EYE], device=self.device) + origin,
            torch.tensor([OBSERVER_TARGET], device=self.device) + origin,
        )

        def frames():
            def rgb(camera):
                return camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            return rgb(observer), rgb(self.scene.sensors["wrist_camera"])

        state["recorder"] = PPOVideo(output, frames, self.step_dt, 1, 1, cfg.seed, fps=15.)

    def reset(self, ids):
        if state["recorder"] is not None and bool((self.episode_length_buf[ids] > 0).any()):
            keys = ("success", "time_out", "staged_orange_dropped", "staged_box_tipped", "staged_abnormal_robot")
            state["terminal"] = {"termination": {
                k: bool(self.termination_manager.get_term(k)[0]) for k in keys
            }}
            # Capture the final pre-reset state; do not change or postpone reset.
            state["recorder"].after_step(terminal=True)
        return original_reset(self, ids)

    def step(self, action):
        result = original_step(self, action)
        state["steps"] += 1
        stages = ("HOME", "PREGRASP", "GRASP", "LIFT", "STABILIZE", "TRANSPORT", "PLACE", "RELEASE")
        state["stage"] = stages[int(self.staged_pick_state.progress.stage[0])]
        if state["terminal"] is None:
            state["recorder"].after_step()
        if state["terminal"] is not None or state["steps"] >= budget:
            result_metadata = state["terminal"] or {"duration_limit": True}
            metadata = state["recorder"].finish(result_metadata)
            metadata.update({"route": "camera_ppo", "benchmark": False, "original_result": result_metadata,
                             "capture_backend": "passive Newton camera + ffmpeg; no Kit viewport capture",
                             "visual_asset": "small_grasp/assets/cardboard_box_visible.usda; original tray physics referenced unchanged"})
            output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            print("PPO_DEMO_COMPLETE=" + json.dumps(metadata), flush=True)
            raise SystemExit(0)
        return result

    ManagerBasedRLEnv.__init__ = initialize
    ManagerBasedRLEnv.step = step
    ManagerBasedRLEnv._reset_idx = reset
    # Playback should not overwrite deployment exports beside the selected checkpoint.
    OnPolicyRunner.export_policy_to_jit = lambda *args, **kwargs: None
    OnPolicyRunner.export_policy_to_onnx = lambda *args, **kwargs: None
    return sys.argv[1:]
