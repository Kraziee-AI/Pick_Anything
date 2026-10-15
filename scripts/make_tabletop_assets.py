"""Author small, local tabletop USD assets on CPU; no simulator or downloads.

Only writes the three new asset directories. Existing fruit/robot/table files
are never opened for writing. Rebuilding assets requires --overwrite.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_asset(key: str, overwrite: bool = False) -> Path:
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    masses = {"yellow_duck": 0.025, "blue_cube": 0.035, "red_cylinder": 0.03}
    if key not in masses:
        raise ValueError(key)
    folder = PROJECT_ROOT / "assets/objects" / key
    path = folder / f"{key}.usda"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} 已存在；显式传 --overwrite 才可重建。")
    folder.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{key}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(masses[key])
    base = root.GetPath()
    UsdGeom.Scope.Define(stage, base.AppendChild("Looks"))

    def material(name, color, roughness=0.6):
        texture = folder / "texture" / f"{name}.png"
        texture.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2), color).save(texture)
        mat = UsdShade.Material.Define(stage, f"{base}/Looks/{name}")
        # Universal fallback plus the texture-based MDL path used by the local
        # fruit assets. Relative texture paths keep these assets relocatable.
        preview = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Preview")
        preview.CreateIdAttr("UsdPreviewSurface")
        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(v / 255 for v in color)))
        preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
        mdl = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/MDL")
        mdl.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        mdl.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        mdl.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"texture/{name}.png"))
        mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
        mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
        return mat

    physics = UsdShade.Material.Define(stage, f"{base}/Looks/Physics")
    api = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    api.CreateStaticFrictionAttr(0.8)
    api.CreateDynamicFrictionAttr(0.65)
    api.CreateRestitutionAttr(0.02)

    def bind(geom, mat, collision=True):
        prim = geom.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
        if collision:
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdShade.MaterialBindingAPI(prim).Bind(physics, materialPurpose="physics")

    def ellipsoid(name, center, radii, mat, collision=True):
        # Convex mesh per component; all parts belong to one root rigid body.
        # Avoid nonuniform sphere scaling in PhysX by baking the mesh points.
        slices, rings = 24, 12
        points = [Gf.Vec3f(center[0], center[1], center[2] + radii[2])]
        normals = [Gf.Vec3f(0, 0, 1)]
        for j in range(1, rings):
            phi = math.pi * j / rings
            for i in range(slices):
                theta = 2 * math.pi * i / slices
                unit = (math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi))
                points.append(Gf.Vec3f(*(center[k] + radii[k] * unit[k] for k in range(3))))
                normal = Gf.Vec3f(*(unit[k] / radii[k] for k in range(3)))
                normals.append(normal.GetNormalized())
        bottom = len(points)
        points.append(Gf.Vec3f(center[0], center[1], center[2] - radii[2]))
        normals.append(Gf.Vec3f(0, 0, -1))
        faces = []
        for i in range(slices):
            nxt = (i + 1) % slices
            faces.append((0, 1 + i, 1 + nxt))
            for j in range(rings - 2):
                a, b = 1 + j * slices + i, 1 + j * slices + nxt
                faces.extend(((a, a + slices, b + slices), (a, b + slices, b)))
            a, b = 1 + (rings - 2) * slices + i, 1 + (rings - 2) * slices + nxt
            faces.append((a, bottom, b))
        mesh = UsdGeom.Mesh.Define(stage, f"{base}/{name}")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
        mesh.CreateNormalsAttr(normals)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr([Gf.Vec3f(*(center[k] - radii[k] for k in range(3))),
                               Gf.Vec3f(*(center[k] + radii[k] for k in range(3)))])
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                             UsdGeom.Tokens.constant).Set([Gf.Vec2f(0.5, 0.5)])
        bind(mesh, mat, collision)
        if collision:
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")

    if key == "yellow_duck":
        yellow = material("Yellow", (255, 211, 25))
        beak = material("Beak", (255, 111, 12))
        black = material("Eyes", (16, 16, 18), 0.35)
        ellipsoid("Body", (0, 0, 0.028), (0.023, 0.027, 0.028), yellow)
        ellipsoid("Head", (0, 0.014, 0.047), (0.015, 0.015, 0.015), yellow)
        ellipsoid("Beak", (0, 0.029, 0.042), (0.011, 0.01, 0.006), beak)
        ellipsoid("Tail", (0, -0.028, 0.032), (0.011, 0.01, 0.01), yellow)
        ellipsoid("WingLeft", (-0.019, -0.004, 0.03), (0.006, 0.015, 0.011), yellow, False)
        ellipsoid("WingRight", (0.019, -0.004, 0.03), (0.006, 0.015, 0.011), yellow, False)
        ellipsoid("EyeLeft", (-0.013, 0.02, 0.051), (0.003, 0.003, 0.003), black, False)
        ellipsoid("EyeRight", (0.013, 0.02, 0.051), (0.003, 0.003, 0.003), black, False)
    elif key == "blue_cube":
        cube = UsdGeom.Cube.Define(stage, f"{base}/Cube")
        cube.CreateSizeAttr(0.05)
        cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.025))
        bind(cube, material("Blue", (32, 110, 235)))
    else:
        cylinder = UsdGeom.Cylinder.Define(stage, f"{base}/Cylinder")
        cylinder.CreateRadiusAttr(0.021)
        cylinder.CreateHeightAttr(0.06)
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        cylinder.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.03))
        bind(cylinder, material("Red", (224, 45, 42)))
    stage.GetRootLayer().Save()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for key in ("yellow_duck", "blue_cube", "red_cylinder"):
        print(build_asset(key, args.overwrite))


if __name__ == "__main__":
    main()
