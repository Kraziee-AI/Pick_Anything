"""PIL diagnostic overlay of projected 3-D gripper centres, jaws and approach."""
import numpy as np
from PIL import Image, ImageDraw


def save_overlay(rgb, mask, candidates, intrinsics, destination):
    pixels = np.asarray(rgb).copy()
    pixels[mask] = (0.7 * pixels[mask] + 0.3 * np.array([40, 210, 80])).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)

    def project(point):
        if point[2] <= 0:
            return None
        return (float(point[0] / point[2] * intrinsics['fx'] + intrinsics['cx']),
                float(point[1] / point[2] * intrinsics['fy'] + intrinsics['cy']))

    for index, candidate in reversed(list(enumerate(candidates[:10]))):
        matrix = np.asarray(candidate['T_camera_grasp'])
        rotation, centre = matrix[:3, :3], matrix[:3, 3]
        left = project(centre - rotation[:, 1] * candidate['width_m'] / 2)
        right = project(centre + rotation[:, 1] * candidate['width_m'] / 2)
        origin, approach = project(centre), project(centre - rotation[:, 0] * 0.04)
        color = 'yellow' if index == 0 else 'cyan'
        if left and right:
            draw.line([left, right], fill=color, width=3 if index == 0 else 1)
            draw.ellipse((left[0] - 2, left[1] - 2, left[0] + 2, left[1] + 2), fill=color)
            draw.ellipse((right[0] - 2, right[1] - 2, right[0] + 2, right[1] + 2), fill=color)
        if origin and approach:
            draw.line([origin, approach], fill='red', width=2)
        if origin:
            # Keep the image legible when several proposals share one small target.
            # The one-based label maps to candidate.rank + 1 in result.json.
            draw.text((origin[0] + 4, origin[1] + 4), str(index + 1),
                      fill=color, stroke_width=1, stroke_fill='black')
    draw.rectangle((0, 0, image.width, 23), fill='black')
    draw.text((6, 6), f'{len(candidates)} proposals | camera frame | red: approach | robot not executed', fill='white')
    image.save(destination)
