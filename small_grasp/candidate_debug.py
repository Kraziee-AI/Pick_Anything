"""Frame-bound, passive inspection of actual exported GraspNet SE(3) poses."""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import validate_transform


def project(point, intrinsics):
    point = np.asarray(point, dtype=float)
    if not np.isfinite(point).all() or point[2] <= 1e-6:
        return None
    return np.array([point[0] / point[2] * intrinsics['fx'] + intrinsics['cx'],
                     point[1] / point[2] * intrinsics['fy'] + intrinsics['cy']])


def execution_summary(candidates, execution=None):
    execution = execution or {}
    chosen = execution.get('chosen')
    summary = {}
    for index, candidate in enumerate(candidates):
        attempts = [a for a in execution.get('ik_attempts', [])
                    if a.get('candidate_id') == candidate['candidate_id']
                    or a.get('candidate') == index]
        reasons = sorted({a['rejected'] for a in attempts if a.get('rejected')})
        selected = bool(chosen and chosen.get('candidate_id') == candidate['candidate_id'])
        accepted = any(a.get('accepted') for a in attempts)
        status = ('selected' if selected else 'plan_accepted' if accepted else
                  'plans_rejected' if attempts and all(a.get('rejected') for a in attempts) else
                  'attempted' if attempts else 'not_checked')
        summary[candidate['candidate_id']] = {'status': status, 'reasons': reasons}
    return {'candidates': summary, 'chosen': chosen,
            'finished': execution is not None and 'robot_executed' in execution}


class CandidateSnapshot:
    def __init__(self, spec):
        self.spec = spec
        directory = Path(spec['proposals_dir'])
        result = json.loads((directory / 'result.json').read_text())
        self.candidates = result['candidates']
        self.rgb = Image.open(directory / 'rgb.png').convert('RGB')
        self.mask = np.asarray(Image.open(directory / 'target_mask.png')) > 0
        self.intrinsics = json.loads(Path(spec['intrinsics_path']).read_text())
        if self.mask.shape != (self.rgb.height, self.rgb.width):
            raise ValueError('Candidate RGB/mask resolution mismatch')
        self.poses = [validate_transform(c['T_camera_grasp']) for c in self.candidates]
        self.cache = {}
        self.default = next((i for i, c in enumerate(self.candidates)
                             if spec['execution']['candidates'].get(c['candidate_id'], {}).get('status') == 'selected'), 0)

    def data(self):
        rows = []
        for c, pose in zip(self.candidates, self.poses):
            world = c.get('T_world_grasp')
            rows.append({'candidate_id': c['candidate_id'], 'rank': c['rank'],
                         'score': c['score'], 'width_m': c['width_m'],
                         'position_camera_m': pose[:3, 3].tolist(),
                         'approach_camera': pose[:3, 0].tolist(),
                         'closing_camera': pose[:3, 1].tolist(),
                         'up_camera': pose[:3, 2].tolist(),
                         'T_camera_grasp': pose.tolist(), 'T_world_grasp': world,
                         **self.spec['execution']['candidates'].get(c['candidate_id'], {})})
        return {'trial': self.spec['trial'], 'source': self.spec['proposals_dir'],
                'default_index': self.default, 'candidates': rows,
                'execution_finished': self.spec['execution']['finished'],
                'chosen_execution': self.spec['execution']['chosen'],
                'note': 'Frozen network source RGB; network pose before SO101 tool adaptation; score is not success probability'}

    def jpeg(self, index=None):
        index = self.default if index is None else index
        if index not in self.cache:
            stream = io.BytesIO()
            self.render(index).save(stream, format='JPEG')
            self.cache[index] = stream.getvalue()
        return self.cache[index]

    def render(self, index=None):
        board = Image.new('RGB', (1280, 800), '#101824')
        font = ImageFont.truetype('DejaVuSans.ttf', 18)
        draw = ImageDraw.Draw(board)
        if not self.candidates:
            draw.text((20, 30), 'No exported candidates. Inspect result.json rejection counts.', font=font, fill='white')
            return board
        index = self.default if index is None else index
        if not 0 <= index < len(self.candidates):
            raise ValueError('Candidate index out of range')
        c, pose = self.candidates[index], self.poses[index]
        rotation, centre = pose[:3, :3], pose[:3, 3]
        uv = project(centre, self.intrinsics)
        pixels = np.asarray(self.rgb).copy()
        pixels[self.mask] = (.75 * pixels[self.mask] + .25 * np.array([40, 210, 80])).astype(np.uint8)
        source = Image.fromarray(pixels)
        full = source.resize((640, 480))
        zoom_centre = uv if uv is not None else np.array([self.rgb.width / 2, self.rgb.height / 2])
        span = max(64., min(self.rgb.width, self.rgb.height) * .36)
        x0, y0 = zoom_centre - span / 2
        crop = (int(x0), int(y0), int(x0 + span), int(y0 + span))
        zoom = source.crop(crop).resize((640, 480))
        board.paste(full, (0, 48)); board.paste(zoom, (640, 48))
        draw = ImageDraw.Draw(board)

        def xy(point, panel):
            point = project(point, self.intrinsics)
            if point is None:
                return None
            if panel == 0:
                return tuple(point * [640 / self.rgb.width, 480 / self.rgb.height] + [0, 48])
            return tuple((point - crop[:2]) * [640 / (crop[2] - crop[0]), 480 / (crop[3] - crop[1])] + [640, 48])

        # Draw each panel independently so off-screen geometry cannot cross the divider.
        for panel in (0, 1):
            layer = Image.new('RGBA', (1280, 800))
            pen = ImageDraw.Draw(layer)

            def line(a, b, color, arrow=False, width=3):
                a, b = xy(a, panel), xy(b, panel)
                if a is None or b is None:
                    return
                pen.line([a, b], fill=color, width=width)
                delta = np.array(b) - a
                length = np.linalg.norm(delta)
                if arrow and length > 3:
                    along = delta / length
                    normal = np.array([-along[1], along[0]])
                    tip = np.array(b)
                    pen.polygon([b, tuple(tip - along * 10 + normal * 5),
                                 tuple(tip - along * 10 - normal * 5)], fill=color)

            if panel == 0:
                ids = sorted(set(list(range(min(10, len(self.candidates)))) + [index]))
                for i in ids:
                    p = xy(self.poses[i][:3, 3], panel)
                    if p:
                        pen.ellipse((p[0]-3, p[1]-3, p[0]+3, p[1]+3), fill='#c0c9d8')
                        pen.text((p[0]+4, p[1]+4), str(i+1), fill='white', stroke_width=1, stroke_fill='black')
            half_width = c['width_m'] / 2
            left, right = centre - rotation[:, 1] * half_width, centre + rotation[:, 1] * half_width
            back = -rotation[:, 0] * max(.025, c.get('depth_m', .03))
            for a, b in ((left, left + back), (right, right + back), (left + back, right + back), (left, right)):
                line(a, b, '#ffffff')
            line(centre - rotation[:, 0] * .045, centre, '#ff5555', arrow=True, width=4)
            line(centre, centre + rotation[:, 1] * .03, '#55ef88', arrow=True)
            line(centre, centre + rotation[:, 2] * .03, '#559aff', arrow=True)
            p = xy(centre, panel)
            if p:
                pen.ellipse((p[0]-5, p[1]-5, p[0]+5, p[1]+5), fill='#ff65e5', outline='white')
            box = (640 * panel, 48, 640 * (panel + 1), 528)
            board.paste(layer.crop(box), box[:2], layer.crop(box))
        draw = ImageDraw.Draw(board)
        state = self.spec['execution']['candidates'].get(c['candidate_id'], {})
        lines = [f'NETWORK pose #{index+1}/{len(self.candidates)} | {c["candidate_id"]} | {state.get("status", "not_checked")}',
                 'Original inference RGB (frozen)                       |                       Selected candidate zoom',
                 f'Centre camera [m]: {np.round(centre, 4).tolist()} | gap {c["width_m"]*1000:.1f} mm | score {c["score"]:.4f}',
                 f'Red +X approach: {np.round(rotation[:, 0], 4).tolist()}',
                 f'Green +Y closing: {np.round(rotation[:, 1], 4).tolist()}',
                 f'Blue +Z up: {np.round(rotation[:, 2], 4).tolist()}',
                 'White: parallel-jaw model | magenta: 3D centre | score is NOT success probability',
                 'Network pose before SO101 tool adaptation; this is NOT the current wrist view.',
                 'Rejected plan attempts: ' + (', '.join(state.get('reasons', [])) or 'none recorded')]
        for text, y in zip(lines, [8, 535, 567, 599, 631, 663, 695, 727, 759]):
            draw.text((12, y), text, font=font, fill='white')
        return board
