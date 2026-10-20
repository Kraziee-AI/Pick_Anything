"""Passive RGB recording. No video observations are supplied to the controller."""
import json
import math
from pathlib import Path
import shutil
import subprocess


VIDEO_WIDTH, VIDEO_HEIGHT = 1280, 720
OBSERVER_EYE = (.9, -1.05, .8)
OBSERVER_TARGET = (.35, -.46, .12)


class EpisodeVideo:
    """Stream sampled simulation frames to H.264 without retaining raw frames."""

    def __init__(self, path, supplier, step_dt, trial, episodes, seed, fps=15.):
        if not math.isfinite(fps) or fps <= 0 or fps > 1 / step_dt + 1e-6:
            raise ValueError('Video FPS must be positive and not exceed control frequency')
        self.path, self.supplier = Path(path), supplier
        if self.path.exists():
            raise FileExistsError(self.path)
        executable = shutil.which('ffmpeg')
        if executable is None:
            raise RuntimeError('ffmpeg is required for --record-video')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stride = max(1, round(1 / (fps * step_dt)))
        self.fps, self.step_dt = 1 / (self.stride * step_dt), float(step_dt)
        self.trial, self.episodes, self.seed = trial, episodes, seed
        self.steps, self.frames, self.last, self.closed = 0, 0, None, False
        self.metadata = None
        self.log_path = self.path.with_suffix('.ffmpeg.log')
        self.log = self.log_path.open('wb')
        command = [executable, '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                   '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{VIDEO_WIDTH}x{VIDEO_HEIGHT}',
                   '-r', str(self.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                   '-preset', 'veryfast', '-crf', '23', '-threads', '2',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.path)]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self.log)
        except BaseException:
            self.log.close()
            raise

    def after_step(self, terminal=False):
        if self.closed:
            return
        self.steps += 1
        if terminal or (self.steps - 1) % self.stride == 0:
            self.last = self.supplier()
            self._write(self._compose())

    def _compose(self, status=None):
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        overview, wrist = self.last
        overview, wrist = np.asarray(overview), np.asarray(wrist)
        if overview.shape != (VIDEO_HEIGHT, VIDEO_WIDTH, 3) or overview.dtype != np.uint8:
            raise ValueError(f'Unexpected observer frame {overview.shape}/{overview.dtype}')
        frame = Image.fromarray(overview).copy()
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype('DejaVuSans.ttf', 21)
        draw.rectangle((0, 0, VIDEO_WIDTH, 43), fill=(18, 22, 28))
        draw.text((14, 9), f'Trial {self.trial:02d}/{self.episodes} | seed {self.seed} | '
                  f'SO101 servo | sim {self.steps * self.step_dt:.2f}s', font=font, fill='white')
        inset = Image.fromarray(wrist[..., :3].astype(np.uint8)).resize((320, 240))
        frame.paste(inset, (VIDEO_WIDTH - 334, 80))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((VIDEO_WIDTH - 336, 78, VIDEO_WIDTH - 12, 322), outline='white', width=2)
        draw.text((VIDEO_WIDTH - 334, 49), 'Wrist RGB', font=font, fill='white', stroke_width=1,
                  stroke_fill='black')
        draw.rectangle((0, VIDEO_HEIGHT - 42, VIDEO_WIDTH, VIDEO_HEIGHT), fill=(18, 22, 28))
        draw.text((14, VIDEO_HEIGHT - 33), status or 'DINO + SAM + RGB-D + GraspNet | single-orange simulation',
                  font=font, fill='white')
        return np.asarray(frame)

    def _write(self, frame):
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(f'Video encoder failed; inspect {self.log_path}') from error
        self.frames += 1

    def finish(self, result=None):
        if self.closed:
            return self.metadata
        try:
            if self.last is not None:
                status = ('INTERRUPTED' if result is None else
                          f'GRASP: {"SUCCESS" if result.get("grasp_lift_success") else "FAIL"} | '
                          f'TASK: {"SUCCESS" if result.get("success") else "FAIL"} | '
                          f'{result.get("failure_stage") or "completed"}')
                end_frame = self._compose(status)
                for _ in range(round(self.fps)):
                    self._write(end_frame)
        finally:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            code = self.process.wait()
            self.log.close()
            self.closed = True
        if code or not self.frames:
            raise RuntimeError(f'Video encoding failed ({code}); inspect {self.log_path}')
        self.metadata = {'path': str(self.path.resolve()), 'fps': self.fps, 'frames': self.frames,
                         'duration_s': self.frames / self.fps, 'control_steps': self.steps,
                         'sample_stride': self.stride, 'width': VIDEO_WIDTH, 'height': VIDEO_HEIGHT,
                         'codec': 'h264', 'pixel_format': 'yuv420p', 'result_hold_s': 1.,
                         'observer_eye': list(OBSERVER_EYE), 'observer_target': list(OBSERVER_TARGET),
                         'timing': 'simulation time; inference/planning wall-time waits omitted',
                         'role': 'passive observer; never supplied to perception or control',
                         'completed': result is not None}
        self.path.with_suffix('.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
        return self.metadata
