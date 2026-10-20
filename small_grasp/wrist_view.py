"""Opt-in wrist optics for seeing both SO101 jaws; source USD stays untouched."""
from functools import partial
import math


PROFILES = ('legacy', 'open-jaws')
OPEN_JAW_YAW_DEG = -30.
OPEN_JAW_PITCH_DEG = 8.
OPEN_JAW_FOCAL_SCALE = .75  # Relative to the existing closure-view spawn.
OPEN_JAW_RESOLUTION = (384, 288)


def spawn_open_jaw_camera(prim_path, cfg, translation=None, orientation=None,
                          *, base_spawn, **kwargs):
    """Chain the established robot spawn, then override optics before sensor init."""
    from pxr import UsdGeom
    prim = base_spawn(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    camera = prim.GetStage().GetPrimAtPath(f'{prim_path}/gripper/wowrobo_2MP_camera')
    if not camera.IsA(UsdGeom.Camera):
        raise RuntimeError('SO101 wrist camera missing; cannot apply open-jaws profile')
    orient = next((op for op in UsdGeom.Xformable(camera).GetOrderedXformOps()
                   if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if orient is None:
        raise RuntimeError('SO101 camera has no orient op')
    angle = math.radians(OPEN_JAW_YAW_DEG) / 2
    # USD looks along -Z: negative local Y yaw turns toward the opening jaw (+X).
    value = orient.Get()
    yaw = type(value)(math.cos(angle), type(value.GetImaginary())(0., math.sin(angle), 0.))
    pitch = math.radians(OPEN_JAW_PITCH_DEG) / 2
    tilt = type(value)(math.cos(pitch), type(value.GetImaginary())(math.sin(pitch), 0., 0.))
    orient.Set((value * yaw * tilt).GetNormalized())
    focal = UsdGeom.Camera(camera).GetFocalLengthAttr()
    focal.Set(focal.Get() * OPEN_JAW_FOCAL_SCALE)
    print(f'[OPEN-JAW-CAMERA] yaw={OPEN_JAW_YAW_DEG}deg pitch={OPEN_JAW_PITCH_DEG}deg '
          f'focal={focal.Get()}mm', flush=True)
    return prim


def configure_wrist_view(cfg, profile):
    if profile not in PROFILES:
        raise ValueError(f'Unknown wrist camera profile: {profile}')
    if profile == 'legacy':
        return
    cfg.scene.robot.spawn = cfg.scene.robot.spawn.replace(
        func=partial(spawn_open_jaw_camera, base_spawn=cfg.scene.robot.spawn.func))
    cfg.scene.wrist_camera.width, cfg.scene.wrist_camera.height = OPEN_JAW_RESOLUTION
