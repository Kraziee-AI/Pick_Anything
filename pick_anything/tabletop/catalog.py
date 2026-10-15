"""CPU-only object metadata, single-target language mapping and safe spawn layouts.

Dimensions are metres. New procedural assets have an origin at their bottom;
the existing orange retains its original centre origin and scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_TOP_Z = 0.041
ROBOT_BASE_XY = (0.35, -0.64)
DEFAULT_OBJECTS = ("orange", "yellow_duck", "blue_cube")
BOX_KEY = "cardboard_box"
BOX_ASSET = PROJECT_ROOT / "assets/objects/cardboard_box/cardboard_box.usd"
BOX_HALF_SIZE_M = 0.08
BOX_SPAWN_Z = TABLE_TOP_Z + 0.005


@dataclass(frozen=True)
class TabletopObject:
    key: str
    label: str
    prim_name: str
    asset: Path
    caption: str
    aliases: tuple[str, ...]
    scale: float
    footprint_radius: float  # conservative yaw-independent XY bounding circle
    origin_to_bottom: float
    height: float

    @property
    def spawn_z(self) -> float:
        return TABLE_TOP_Z + self.origin_to_bottom + 0.004


def _asset(name: str, suffix: str = "usda") -> Path:
    return PROJECT_ROOT / "assets" / "objects" / name / f"{name}.{suffix}"


OBJECTS = {
    "orange": TabletopObject(
        "orange", "橘子", "Orange", _asset("orange", "usd"), "orange fruit.",
        ("橘子", "橙子", "orange fruit", "orange"), 0.775, 0.041, 0.035682 * 0.775, 0.05530,
    ),
    "yellow_duck": TabletopObject(
        "yellow_duck", "小黄鸭", "YellowDuck", _asset("yellow_duck"), "yellow rubber duck.",
        ("小黄鸭", "黄鸭", "鸭子", "yellow rubber duck", "rubber duck", "yellow duck", "duck"),
        1.0, 0.047, 0.0, 0.062,
    ),
    "blue_cube": TabletopObject(
        "blue_cube", "蓝色方块", "BlueCube", _asset("blue_cube"), "blue cube.",
        ("蓝色方块", "蓝方块", "方块", "立方体", "blue cube", "cube", "block"),
        1.0, 0.036, 0.0, 0.05,
    ),
    "red_cylinder": TabletopObject(
        "red_cylinder", "红色圆柱", "RedCylinder", _asset("red_cylinder"), "red cylinder.",
        ("红色圆柱", "红圆柱", "圆柱", "red cylinder", "cylinder"),
        1.0, 0.021, 0.0, 0.06,
    ),
}


def object_keys(keys) -> tuple[str, ...]:
    keys = tuple(keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("物体列表不能为空或重复。")
    unknown = set(keys) - OBJECTS.keys()
    if unknown:
        raise ValueError(f"未知物体：{sorted(unknown)}；可选：{list(OBJECTS)}")
    return keys


def resolve_target(instruction: str, available=DEFAULT_OBJECTS) -> TabletopObject:
    """Resolve one known object; this is vocabulary mapping, not visual detection.

    Reject ambiguous commands instead of silently picking an object. English
    aliases use word boundaries, so 'duckling' and 'orangeish' do not match.
    """
    keys = object_keys(available)
    text = instruction.strip().lower()
    hits = []
    for obj in OBJECTS.values():
        if any((alias in text if not alias.isascii() else
                re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text))
               for alias in obj.aliases):
            hits.append(obj)
    if len(hits) != 1:
        raise ValueError("指令必须明确指定一个已知物体，例如“抓小黄鸭”或“抓蓝色方块”。")
    target = hits[0]
    if target.key not in keys:
        raise ValueError(f"目标 {target.label} 不在当前场景中。")
    return target


def sample_layout(keys=DEFAULT_OBJECTS, seed: int | None = None, gap: float = 0.012,
                  include_box: bool = True):
    """Bounded rejection sampling in the existing arm's reachable table sector.

    Reserve each object's full XY bounding circle plus a gap, including random
    yaw. Start over if a partial layout jams; never fall back to overlapping
    placements. The original axis-aligned tray is retained by default; objects
    stay outside its whole footprint, rather than spawning inside the tray.
    """
    keys = object_keys(keys)
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("gap must be finite and nonnegative")
    rng = random.Random(seed)
    for _ in range(100):
        layout = {}
        if include_box:
            for _ in range(300):
                r = math.sqrt(rng.uniform(0.16**2, 0.24**2))
                angle = math.radians(rng.uniform(-30, 30))
                bx = ROBOT_BASE_XY[0] + r * math.sin(angle)
                by = ROBOT_BASE_XY[1] + r * math.cos(angle)
                if 0.14 <= bx <= 0.56 and -0.51 <= by <= -0.39:
                    layout[BOX_KEY] = {
                        "position": (bx, by, BOX_SPAWN_Z),
                        "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
                    }
                    break
            else:
                continue
        for key in sorted(keys, key=lambda k: OBJECTS[k].footprint_radius, reverse=True):
            obj = OBJECTS[key]
            radius = obj.footprint_radius
            for _ in range(300):
                r = math.sqrt(rng.uniform(0.14**2, 0.29**2))
                angle = math.radians(rng.uniform(-55, 55))
                x = ROBOT_BASE_XY[0] + r * math.sin(angle)
                y = ROBOT_BASE_XY[1] + r * math.cos(angle)
                # Table bounds from the existing scene, with 1cm edge clearance.
                if not (-0.135 + radius <= x <= 0.845 - radius and
                        -0.667 + radius <= y <= -0.237 - radius):
                    continue
                if include_box:
                    # Distance to the tray's filled XY rectangle. Reserving its
                    # interior keeps objects out of the box at episode birth.
                    dx = max(abs(x - bx) - BOX_HALF_SIZE_M, 0.0)
                    dy = max(abs(y - by) - BOX_HALF_SIZE_M, 0.0)
                    if math.hypot(dx, dy) < radius + gap:
                        continue
                if any(math.hypot(x - p["position"][0], y - p["position"][1]) <
                       radius + OBJECTS[other].footprint_radius + gap
                       for other, p in layout.items() if other != BOX_KEY):
                    continue
                yaw = rng.uniform(-math.pi, math.pi)
                layout[key] = {
                    "position": (x, y, obj.spawn_z),
                    "orientation_xyzw": (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
                }
                break
            else:
                break
        if len(layout) == len(keys) + int(include_box):
            ordered = (BOX_KEY, *keys) if include_box else keys
            return {key: layout[key] for key in ordered}
    raise ValueError("无法在可达桌面区域放下这些物体及间距，请减少物体数量或 gap。")
