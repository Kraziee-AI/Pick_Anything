"""Generate portable MJCF physics from an actual Isaac/Newton runtime export.

Run export_isaac_teacher.py first, then this script. Deployment needs only the
resulting XML/meshes, not Isaac, USD, the cache or the exported binary model.
The template supplies visuals whose unscaled meshes match the Isaac USD asset;
it is never used as physics ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from env import ASSETS_DIR, JOINT_NAMES, SCAN_HOME_ARM_JOINTS


def fmt(values):
    return " ".join(f"{float(v):.16g}" for v in np.asarray(values).ravel())


def matrix(pos, quat):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quat, scalar_first=True).as_matrix()
    result[:3, 3] = pos
    return result


def pose(element, xf):
    element.set("pos", fmt(xf[:3, 3]))
    element.set("quat", fmt(Rotation.from_matrix(xf[:3, :3]).as_quat(scalar_first=True)))


# Isaac 资产相机（so101-camera_mount payload authored，USD gripper 系）与
# 闭包口径 pitch：闭包只给相机 prim 右乘 pitch，支架网格不动。
ASSET_CAMERA_POS = (-0.0018862803672875517, 0.05226449046420151, -0.05853182757774217)
ASSET_CAMERA_QUAT = (0.953716950748227, -0.3007057995042728, 0.0, 0.0)
CLOSURE_PITCH_DEG = -12.0
def mount_camera_visual(robot):
    """USD gripper 局部网格原位挂载，与原尺寸夹爪直接贴合。"""
    visual = robot.find(".//body[@name='gripper']/geom[@name='wrist_camera_visual']")
    visual.set("type", "mesh")
    visual.set("mesh", "usd_camera_housing")
    visual.attrib.pop("size", None)  # 清掉历史 box 残留
    assets = robot.find("asset")
    if assets.find("mesh[@name='usd_camera_housing']") is None:
        assets.insert(0, ET.Element("mesh", name="usd_camera_housing",
                                    file="usd_camera_housing_gripusd.obj"))
    pose(visual, np.eye(4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path,
                        default=ASSETS_DIR.parent / "reference/runtime_unclipped32/isaac_model.mjb")
    args = parser.parse_args()
    source = mujoco.MjModel.from_binary_path(str(args.runtime))
    initial = args.runtime.parent / "initial_states.npz"
    origin = np.load(initial)["origins"][0] if initial.is_file() else np.zeros(3)
    contacts = args.runtime.parent / "contacts_0.npz"
    exported = np.load(contacts)
    kf_values = np.unique(exported["shape_material_kf"])
    if len(kf_values) != 1:
        raise ValueError("This converter requires homogeneous shape_material_kf")
    kf = float(kf_values[0])
    template = ASSETS_DIR / "so101_visual_template.xml"
    robot_tree = ET.parse(template)
    robot = robot_tree.getroot()
    scene_tree = ET.parse(ASSETS_DIR / "scene_visual_template.xml")
    scene = scene_tree.getroot()
    assets = robot.find("asset")
    geom_names = {0: "floor", 1: "table"}
    # Keep the Onshape base frame for its visual meshes. Other links use the
    # actual solver frames (including the wrist calibration + sensor payload).
    base_xf = matrix([0.3292091, -0.6557608, 0.0424817], [2**-.5, 0, 0, 2**-.5])
    conversion = np.linalg.inv(base_xf) @ matrix(source.body_pos[1] - origin, source.body_quat[1])
    bodies = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1",
              "orange", "cardboard_box"]
    for bid, name in enumerate(bodies, 1):
        tree = robot if bid <= 7 else scene
        body = tree.find(f".//body[@name='{name}']")
        xf = matrix(source.body_pos[bid], source.body_quat[bid])
        if bid in (1, 8, 9):
            xf[:3, 3] -= origin
        if bid == 1:
            xf = base_xf
        elif bid == 2:
            xf = conversion @ xf
        pose(body, xf)
        inertia = body.find("inertial")
        inertia.attrib.clear()
        com = matrix(source.body_ipos[bid], source.body_iquat[bid])
        if bid == 1:
            com = conversion @ com
        pose(inertia, com)
        inertia.set("mass", fmt(source.body_mass[bid]))
        inertia.set("diaginertia", fmt(source.body_inertia[bid]))
        for geom in list(body.findall("geom")):
            if bid <= 7 and geom.get("class") == "collision":
                body.remove(geom)
            elif bid == 8 and geom.get("name") != "orange_visual":
                # The source orange mesh is visual only; use the exported hull
                # for collision while retaining the textured render mesh.
                body.remove(geom)
            elif bid == 9:
                # Keep the cardboard panels for rendering, but replace their
                # approximate collision with the exported solver geometry.
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "2")
        for gid in np.flatnonzero(source.geom_bodyid == bid):
            geom_xf = matrix(source.geom_pos[gid], source.geom_quat[gid])
            if bid == 1:
                geom_xf = conversion @ geom_xf
            mesh_id = int(source.geom_dataid[gid])
            geom_name = f"isaac_collision_{name}_{gid}"
            geom_names[int(gid)] = geom_name
            attrs = {"name": geom_name, "group": "3", "contype": str(source.geom_contype[gid]),
                     "conaffinity": str(source.geom_conaffinity[gid])}
            if source.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                start = source.mesh_vertadr[mesh_id]
                vertices = source.mesh_vert[start:start + source.mesh_vertnum[mesh_id]]
                ET.SubElement(assets, "mesh", name=geom_name, vertex=fmt(vertices))
                attrs.update(type="mesh", mesh=geom_name)
            else:
                assert source.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
                attrs.update(type="box", size=fmt(source.geom_size[gid]), material="cardboard_mat")
            geom = ET.SubElement(body, "geom", **attrs)
            pose(geom, geom_xf)
            for field in ("friction", "solref", "solimp", "margin", "gap", "condim"):
                geom.set(field, fmt(getattr(source, "geom_" + field)[gid]))
        if bid > 7:
            joint = body.find("joint")
            if joint is not None:
                joint.set("damping", "0")
    # All six actuators share the joint's summed force clamp, with independent
    # position and velocity drives, just as Newton's MuJoCo conversion does.
    actuator = robot.find("actuator")
    actuator.clear()
    for j, name in enumerate(JOINT_NAMES):
        joint = robot.find(f".//joint[@name='{name}']")
        for field, value in (("range", source.jnt_range[j]), ("solreflimit", source.jnt_solref[j]),
                             ("solimplimit", source.jnt_solimp[j]), ("solreffriction", source.dof_solref[j]),
                             ("solimpfriction", source.dof_solimp[j]), ("armature", source.dof_armature[j]),
                             ("damping", source.dof_damping[j]), ("frictionloss", source.dof_frictionloss[j]),
                             ("actuatorfrcrange", source.jnt_actfrcrange[j])):
            joint.set(field, fmt(value))
        joint.set("actuatorfrclimited", "true")
        ET.SubElement(actuator, "position", name=name, joint=name,
                      kp=fmt(source.actuator_gainprm[2*j, 0]), kv="0",
                      ctrllimited="false", forcelimited="false")
        ET.SubElement(actuator, "velocity", name=name + "_velocity", joint=name,
                      kv=fmt(source.actuator_gainprm[2*j+1, 0]),
                      ctrllimited="false", forcelimited="false")
    # The Newton gripper frame matches the USD gripper frame. Keep the camera
    # on its authored optical pose; the closure adjustment rotates only camera.
    pitch_q = Rotation.from_euler("x", CLOSURE_PITCH_DEG, degrees=True).as_quat(scalar_first=True)
    cam_local = matrix(ASSET_CAMERA_POS, ASSET_CAMERA_QUAT) @ matrix([0, 0, 0], pitch_q)
    pose(robot.find(".//camera[@name='wrist_camera']"), cam_local)
    mount_camera_visual(robot)
    option = scene.find("option")
    for key in ("timestep", "iterations", "ls_iterations", "impratio", "tolerance", "ls_tolerance"):
        option.set(key, fmt(getattr(source.opt, key)))
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    # Isaac disables MuJoCo's collision pass because Newton supplies a contact
    # manifold. Native MuJoCo needs multiccd to obtain a comparable manifold.
    flag.set("multiccd", "enable")
    option.set("ccd_iterations", "35")
    for name, gid in (("floor", 0), ("table", 1)):
        geom = scene.find(f".//geom[@name='{name}']")
        position = source.geom_pos[gid] - origin if gid == 1 else source.geom_pos[gid]
        pose(geom, matrix(position, source.geom_quat[gid]))
        for field in ("contype", "conaffinity", "friction", "solref", "solimp", "margin", "gap", "condim"):
            geom.set(field, fmt(getattr(source, "geom_" + field)[gid]))
    # Newton's elliptic-cone adapter supplies a separate tangential solref
    # derived from kf. Geom friction/normal solref alone omit this behavior.
    contact = ET.SubElement(scene, "contact")
    for g1 in range(source.ngeom):
        for g2 in range(g1 + 1, source.ngeom):
            b1, b2 = source.geom_bodyid[[g1, g2]]
            if source.body_weldid[b1] == source.body_weldid[b2]:
                continue
            if source.body_dofnum[b1] == 0 and source.body_dofnum[b2] == 0:
                continue
            if not ((source.geom_contype[g1] & source.geom_conaffinity[g2]) or
                    (source.geom_contype[g2] & source.geom_conaffinity[g1])):
                continue
            solimp = (source.geom_solimp[g1] + source.geom_solimp[g2]) / 2
            invw = source.body_invweight0[b1, 0] + source.body_invweight0[b2, 0]
            denom = kf * invw * ((1 - solimp[1]) / source.opt.impratio + solimp[1])
            mu = np.maximum(source.geom_friction[g1], source.geom_friction[g2])
            ET.SubElement(contact, "pair", geom1=geom_names[g1], geom2=geom_names[g2],
                          condim=str(max(source.geom_condim[g1], source.geom_condim[g2])),
                          friction=fmt(mu[[0, 0, 1, 2, 2]]), solimp=fmt(solimp),
                          solref=fmt((source.geom_solref[g1] + source.geom_solref[g2]) / 2),
                          solreffriction=fmt([2 / denom, 1]) if denom > 0 else "0 0",
                          margin=fmt(source.geom_margin[g1] + source.geom_margin[g2]),
                          gap=fmt(source.geom_gap[g1] + source.geom_gap[g2]))
    ctrl = np.zeros(source.nu)
    ctrl[::2] = SCAN_HOME_ARM_JOINTS
    scene.find("keyframe/key").set("ctrl", fmt(ctrl))
    robot.insert(0, ET.Comment(" Physics copied from Isaac runtime by build_calibrated_robot.py. "))
    for tree, destination in ((robot_tree, ASSETS_DIR / "so101_new_calib.xml"),
                              (scene_tree, ASSETS_DIR / "scene_pick.xml")):
        ET.indent(tree, space="  ")
        tree.write(destination, encoding="unicode", xml_declaration=True)
    result = mujoco.MjModel.from_xml_path(str(ASSETS_DIR / "scene_pick.xml"))
    metadata = {"source": str(args.runtime.relative_to(ASSETS_DIR.parent)),
                "source_sha256": hashlib.sha256(args.runtime.read_bytes()).hexdigest(),
                "source_world_origin": origin.tolist(), "shape_material_kf": kf,
                "task": "Pick-Apple-SO101-Staged-TeacherFine-v1", "action_clip": None,
                "policy_hz": 30, "target_hz": 120, "solver_hz": 240,
                "contact_history_hz": 30,
                "kp": source.actuator_gainprm[::2, 0].tolist(),
                "kd": source.actuator_gainprm[1::2, 0].tolist(),
                "damping": source.dof_damping[:6].tolist(),
                "frictionloss": source.dof_frictionloss[:6].tolist()}
    (ASSETS_DIR / "calibration_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Generated {result.nu} drives, {result.ngeom} geoms from {args.runtime}")


if __name__ == "__main__":
    main()
