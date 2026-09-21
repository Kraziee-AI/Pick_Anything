"""Build a standalone rigid-object USD from a GraspNet simplified object folder.

Reads the photo-textured visual OBJ (with vt/vn) and the VHACD collision OBJ,
then authors one clean USD:
    <name>.usd
      Xform <name>            (RigidBodyAPI, MassAPI)
        Visuals/Mesh          (st UVs, textured UsdPreviewSurface)
        Collisions/Mesh       (CollisionAPI, invisible)

Usage:
    python scripts/convert_graspnet_object.py --object 012 --name apple --mass 0.025
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GRASPNET_ROOT = Path.home() / "ViewInvariantSE3/OVGNet_full_repro/assets/simplified_objects"

parser = argparse.ArgumentParser(description="Build GraspNet OBJ folder into a rigid-object USD.")
parser.add_argument("--object", required=True, help="GraspNet object index, e.g. 012")
parser.add_argument("--name", required=True, help="Output name, e.g. apple")
parser.add_argument("--mass", type=float, default=0.1, help="Mass in kg")
args_cli = parser.parse_args()


def parse_obj(path: Path):
    """Parse OBJ into per-corner-interpolated mesh data."""
    verts, uvs, norms = [], [], []
    faces = []  # list of [(v_idx, vt_idx, vn_idx), ...]
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "vt":
            uvs.append([float(x) for x in parts[1:3]])
        elif parts[0] == "vn":
            norms.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            face = []
            for token in parts[1:]:
                idx = token.split("/")
                v = int(idx[0]) - 1
                vt = int(idx[1]) - 1 if len(idx) > 1 and idx[1] else v
                vn = int(idx[2]) - 1 if len(idx) > 2 and idx[2] else v
                face.append((v, vt, vn))
            faces.append(face)
    return verts, uvs, norms, faces


def build_usd(out_path: Path, tex_path: Path, mass: float) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    def bbox_extent(verts):
        import numpy as np

        arr = np.asarray(verts, dtype=float)
        return Gf.Vec3f(*arr.min(0)), Gf.Vec3f(*arr.max(0))

    src = GRASPNET_ROOT / args_cli.object
    v_verts, v_uvs, v_norms, v_faces = parse_obj(src / "textured_simplified.obj")
    c_verts, _, _, c_faces = parse_obj(src / "textured_simplified_vhacd.obj")

    stage = Usd.Stage.CreateNew(str(out_path))
    root = UsdGeom.Xform.Define(stage, f"/{args_cli.name}")
    root_prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(mass)

    # --- visual mesh (face-varying UVs/normals) ---
    UsdGeom.Xform.Define(stage, f"/{args_cli.name}/Visuals")
    vmesh = UsdGeom.Mesh.Define(stage, f"/{args_cli.name}/Visuals/Mesh")
    vmesh.CreatePointsAttr([Gf.Vec3f(*v) for v in v_verts])
    vmesh.CreateFaceVertexCountsAttr([len(f) for f in v_faces])
    vmesh.CreateFaceVertexIndicesAttr([c[0] for f in v_faces for c in f])
    vmesh.CreateExtentAttr(bbox_extent(v_verts))
    UsdGeom.PrimvarsAPI(vmesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    ).Set([Gf.Vec2f(*v_uvs[c[1]]) for f in v_faces for c in f])
    vmesh.CreateNormalsAttr([Gf.Vec3f(*v_norms[c[2]]) for f in v_faces for c in f])
    vmesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    vmesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    # --- collision mesh (proxy purpose + invisible) ---
    UsdGeom.Xform.Define(stage, f"/{args_cli.name}/Collisions")
    cmesh = UsdGeom.Mesh.Define(stage, f"/{args_cli.name}/Collisions/Mesh")
    cmesh.CreatePointsAttr([Gf.Vec3f(*v) for v in c_verts])
    cmesh.CreateFaceVertexCountsAttr([len(f) for f in c_faces])
    cmesh.CreateFaceVertexIndicesAttr([c[0] for f in c_faces for c in f])
    cmesh.CreateExtentAttr(bbox_extent(c_verts))
    cmesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    cmesh_prim = cmesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(cmesh_prim)
    mesh_coll = UsdPhysics.MeshCollisionAPI.Apply(cmesh_prim)
    mesh_coll.CreateApproximationAttr("convexHull")  # dynamic bodies need convex approx
    cmesh.CreatePurposeAttr().Set(UsdGeom.Tokens.proxy)
    UsdGeom.Imageable(cmesh_prim).MakeInvisible()

    # --- textured material (MDL OmniPBR). MUST live under the defaultPrim,
    # otherwise the spawn-time reference drops it and the mesh renders gray. ---
    looks_path = f"/{args_cli.name}/Looks"
    UsdGeom.Scope.Define(stage, looks_path)
    mat = UsdShade.Material.Define(stage, f"{looks_path}/FruitMaterial")
    shader = UsdShade.Shader.Define(stage, f"{looks_path}/FruitMaterial/Shader")
    sprim = shader.GetPrim()
    sprim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    mdl_asset = sprim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset)
    mdl_asset.Set(Sdf.AssetPath("OmniPBR.mdl"))
    mdl_asset.SetCustom(False)  # custom info attrs are ignored by hydra
    mdl_sub = sprim.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token)
    mdl_sub.Set("OmniPBR")
    mdl_sub.SetCustom(False)
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(tex_path.resolve()))
    )
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("out"))
    UsdShade.MaterialBindingAPI.Apply(vmesh.GetPrim()).Bind(mat)

    stage.GetRootLayer().defaultPrim = args_cli.name
    stage.GetRootLayer().Save()
    print(
        f"built: {out_path} (visual {len(v_verts)} verts / "
        f"collision {len(c_verts)} verts, mass {mass}kg)"
    )


def main() -> None:
    src = GRASPNET_ROOT / args_cli.object
    if not src.is_dir():
        sys.exit(f"GraspNet object folder not found: {src}")
    out_dir = PROJECT_ROOT / "assets" / "objects" / args_cli.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_src = src / "texture_map.png"
    tex_dst = out_dir / "texture" / "texture_map.png"
    if tex_src.exists():
        tex_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tex_src, tex_dst)
    build_usd(out_dir / f"{args_cli.name}.usd", tex_dst, args_cli.mass)


if __name__ == "__main__":
    main()
