"""离线提取 Isaac SO101 相机支架网格（USD gripper 系 OBJ，不启动 Isaac）。

背景（2026-10-08"模仿 Isaac 相机"）：MuJoCo 相机视觉模块需要复刻 Isaac 里
`gripper/visuals/camera_mount`（SO-101 camera mount payload：3D 打印支架 +
camera_cover/lens/rings/pcb 等 10 件）的真实几何。数据源 = Isaac 本地资产缓存
（nucleus 拉取后落在 /tmp/https/...，IsaacLab 加载日志可证），authorored 层内
层级即真相（静子树，无物理覆盖问题）。

支架网格保持 USD gripper 局部系并原位挂载；闭包 pitch 只作用于相机 prim。

输出：
* mujoco_sim2sim/assets/meshes/usd_camera_housing_gripusd.obj —— 合并网格
  （单个 OBJ 对象，件名写在注释中；USD gripper 局部系）
* photos/camera_housing/manifest.json —— 每件包围盒/顶点数、A/P 参数与校验

用法：conda activate isaaclab_61 && python scripts/extract_camera_housing_usd.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(
    "/tmp/https/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets"
    "/Isaac/6.1/Isaac/Robots_Multiphysics/RobotStudio/so101_new_calib_SysID"
    "/payloads/so101-camera_mount/payloads/base.usda"
)
# 资产相机（base.usda authored，USD gripper 局部系；quat 为 usda 的 (w,x,y,z)）
ASSET_CAM_POS = (-0.0018862803672875517, 0.05226449046420151, -0.05853182757774217)
ASSET_CAM_QUAT = (0.953716950748227, -0.3007057995042728, 0.0, 0.0)
# 闭包口径 pitch（camera_env_cfg.CLOSURE_VIEW_PITCH_DEG，绕相机局部 +X 右乘）
CLOSURE_PITCH_DEG = -12.0
PIECE_PARENT = "/so101_new_calib/gripper/visuals/camera_mount"


def gf_to_np(m):
    """Gf Matrix4d 是行向量惯例（v' = v·M，平移在末行）——转置 3x3 块
    转成列向量惯例 (R @ v + t)。"""
    return (
        np.array([[m[j][i] for j in range(3)] for i in range(3)]),
        np.array([m[3][j] for j in range(3)]),
    )


def mesh_local_from_geometries(geo_stage, prim_path: str):
    """geometries.usd 里 /Geometries/<件> 下找 Mesh，返回 (4x4 局部阵, points, faces)。"""
    gprim = geo_stage.GetPrimAtPath(prim_path)
    assert gprim.IsValid(), f"geometries 里不存在: {prim_path}"
    mesh = None
    for p in Usd.PrimRange(gprim):
        if p.GetTypeName() == "Mesh":
            mesh = p
            break
    assert mesh is not None, f"{prim_path} 下无 Mesh"
    xf = UsdGeom.XformCache()
    m = np.eye(4)
    Rm, tm = gf_to_np(xf.GetLocalToWorldTransform(mesh))
    m[:3, :3], m[:3, 3] = Rm, tm
    pts = mesh.GetAttribute("points").Get() or []
    idx = mesh.GetAttribute("faceVertexIndices").Get() or []
    counts = mesh.GetAttribute("faceVertexCounts").Get() or []
    faces, base = [], 0
    for c in counts:
        faces.append([int(i) for i in idx[base:base + c]])
        base += c
    return m, np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64), faces


def child_reference_target(prim) -> tuple[str, str]:
    """读 prim 的 prepend references -> (assetPath, primPath)。"""
    refs = prim.GetMetadata("references")
    items = refs.GetAddedOrExplicitItems() if refs else []
    assert items, f"{prim.GetPath()} 无 references"
    return items[0].assetPath, items[0].primPath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-obj", type=Path,
                        default=PROJECT_ROOT / "mujoco_sim2sim/assets/meshes/usd_camera_housing_gripusd.obj")
    parser.add_argument("--out-manifest", type=Path,
                        default=PROJECT_ROOT / "photos/camera_housing/manifest.json")
    args = parser.parse_args()

    stage = Usd.Stage.Open(str(CACHE))
    assert stage is not None, f"无法打开缓存资产: {CACHE}"
    parent = stage.GetPrimAtPath(PIECE_PARENT)
    assert parent.IsValid(), f"camera_mount 层级不存在: {PIECE_PARENT}"
    root = stage.GetPrimAtPath("/so101_new_calib")
    xf = UsdGeom.XformCache()
    Rr, tr = gf_to_np(xf.GetLocalToWorldTransform(root))
    assert np.allclose(Rr, np.eye(3)) and np.allclose(tr, 0), "资产根不在原点（世界系≠gripper 局部系）"

    pieces = []
    # base.usda 孩子是 instanced prim（PrimRange 不下钻 prototype），逐孩子解引用：
    # base 孩子 xform → instances.usda（无 xform，恒等）→ geometries.usd 网格局部。
    # LocalToWorld 含 gripper prim 的场景摆位（translate+orient），须剥掉得
    # USD gripper 局部系（相机 authored 值所在系）。
    grip_prim = stage.GetPrimAtPath("/so101_new_calib/gripper")
    Rg, tg = gf_to_np(xf.GetLocalToWorldTransform(grip_prim))
    inv_grip = np.eye(4)
    inv_grip[:3, :3] = Rg.T
    inv_grip[:3, 3] = -Rg.T @ tg
    inst_stage = Usd.Stage.Open(str(CACHE.parent / "instances.usda"))
    geo_stage = Usd.Stage.Open(str(CACHE.parent / "geometries.usd"))
    for child in parent.GetChildren():
        asset, ref_path = child_reference_target(child)
        assert asset.endswith("instances.usda"), f"{child.GetPath()} 引用了非 instances 资产: {asset}"
        inst_ref = child_reference_target(inst_stage.GetPrimAtPath(ref_path))
        assert inst_ref[0].endswith("geometries.usd"), f"{ref_path} 引用了非 geometries: {inst_ref[0]}"
        m_mesh, pts_np, faces = mesh_local_from_geometries(geo_stage, inst_ref[1])
        Rm, tm = gf_to_np(xf.GetLocalToWorldTransform(child))
        m_child = np.eye(4)
        m_child[:3, :3], m_child[:3, 3] = Rm, tm
        total = inv_grip @ m_child @ m_mesh
        world = (total[:3, :3] @ pts_np.T).T + total[:3, 3]
        pieces.append({"name": child.GetName(), "nverts": len(pts_np),
                       "_verts": world, "_faces": faces})
        print(f"[piece] {child.GetName():14s} nverts={len(pts_np):5d}"
              f" y=[{world[:,1].min():.4f},{world[:,1].max():.4f}]"
              f" z=[{world[:,2].min():.4f},{world[:,2].max():.4f}]")

    args.out_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_obj, "w") as f:
        f.write("# Isaac SO-101 camera mount（USD 缓存离线提取；USD gripper 局部系）\n")
        f.write("# MJCF 引用：USD gripper 局部系原位挂载\n")
        manifest = []
        offset = 0
        for p in pieces:
            # MuJoCo's OBJ loader only imports faces from the first `o` object
            # for one <mesh>. Keep all pieces in a single object.
            f.write(f"# piece {p['name'].replace('/', '_')}\n")
            for v in p["_verts"]:
                f.write(f"v {v[0]:.10g} {v[1]:.10g} {v[2]:.10g}\n")
            for face in p["_faces"]:
                f.write("f " + " ".join(str(i + offset + 1) for i in face) + "\n")
            manifest.append({"name": p["name"], "nverts": p["nverts"],
                             "min": p["_verts"].min(axis=0).round(4).tolist(),
                             "max": p["_verts"].max(axis=0).round(4).tolist()})
            offset += len(p["_verts"])
    meta = {
        "source": str(CACHE), "piece_parent": PIECE_PARENT,
        "asset_cam_pos": list(ASSET_CAM_POS), "asset_cam_quat_wxyz": list(ASSET_CAM_QUAT),
        "closure_pitch_deg": CLOSURE_PITCH_DEG, "n_pieces": len(pieces), "pieces": manifest,
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(meta, indent=2))
    print(f"[done] {len(pieces)} 件 -> {args.out_obj}")


if __name__ == "__main__":
    main()
