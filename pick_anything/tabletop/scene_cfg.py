"""Independent scene factory. No mutations to single-orange training configs."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from pick_anything.env_cfg import PickSceneCfg
from .catalog import BOX_KEY, DEFAULT_OBJECTS, OBJECTS, object_keys, sample_layout


@configclass
class MultiObjectSceneCfg(PickSceneCfg):
    # Keep the original tray and table; this independent scene does not use the
    # orange-only reward managers or reset bank.
    orange = None
    yellow_duck = None
    blue_cube = None
    red_cylinder = None
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg(), collision_group=-1,
    )


def make_scene_cfg(keys=DEFAULT_OBJECTS, seed=None, num_envs=1, include_box=True):
    keys = object_keys(keys)
    layout = sample_layout(keys, seed, include_box=include_box)
    cfg = MultiObjectSceneCfg(num_envs=num_envs, env_spacing=2.5)
    if include_box:
        cfg.cardboard_box.init_state.pos = layout[BOX_KEY]["position"]
        cfg.cardboard_box.init_state.rot = layout[BOX_KEY]["orientation_xyzw"]
        cfg.cardboard_box.spawn.activate_contact_sensors = True
        cfg.cardboard_box.spawn.semantic_tags = [("class", BOX_KEY)]
    else:
        cfg.cardboard_box = None
    for key in keys:
        obj = OBJECTS[key]
        if not obj.asset.is_file():
            raise FileNotFoundError(f"缺少资产 {obj.asset}；运行 scripts/make_tabletop_assets.py")
        setattr(cfg, key, RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{obj.prim_name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(obj.asset), scale=(obj.scale,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.3, angular_damping=2.0),
                activate_contact_sensors=True,
                semantic_tags=[("class", key)],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=layout[key]["position"], rot=layout[key]["orientation_xyzw"],
            ),
        ))
    return cfg
