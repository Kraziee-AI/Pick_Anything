"""Build the open-top cardboard tray USD (five panels, no lid).

16 x 16 x 5 cm outer, 3 mm walls, origin at the center of the bottom face.
Bottom + four wall panels, each its own convex-hull collision mesh so the
open interior stays physically reachable. Same USD conventions as the fruit
converter (rigid body on the defaultPrim root, material under the root).

Usage:
    conda activate isaaclab_61
    python scripts/make_cardboard_box.py
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description="Build the cardboard tray rigid-object USD.")
parser.add_argument("--name", default="cardboard_box", help="Output asset name")
parser.add_argument(
    "--mass", type=float, default=0.03, help="Mass in kg (corrugated board ~170 cm^3)"
)
args_cli = parser.parse_args()

OUTER_HALF_XY = 0.08  # 16 cm square footprint
HEIGHT = 0.05  # 5 cm walls
WALL = 0.003  # 3 mm corrugated board
CARDBOARD_RGB = (165, 125, 85)  # kraft brown (texture pixel values)

# Panel bounds (xmin, xmax, ymin, ymax, zmin, zmax); E/W walls tucked between
# the N/S walls so the collision shells do not overlap at the corners.
H = OUTER_HALF_XY
PANELS = {
    "Bottom": (-H, H, -H, H, 0.0, WALL),
    "WallNorth": (-H, H, H - WALL, H, 0.0, HEIGHT),
    "WallSouth": (-H, H, -H, -H + WALL, 0.0, HEIGHT),
    "WallEast": (H - WALL, H, -H + WALL, H - WALL, 0.0, HEIGHT),
    "WallWest": (-H, -H + WALL, -H + WALL, H - WALL, 0.0, HEIGHT),
}

# Quads for a cuboid, counter-clockwise seen from outside (rightHanded).
FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]


def build_texture(tex_path: Path) -> None:
    """Kraft-paper texture with faint corrugated flute lines (diffuse_color is
    ignored by this Isaac version's OmniPBR, so a texture is the proven path)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    size = 256
    flute = 16  # px per flute period
    phase = np.arange(size) * (2 * np.pi / flute)
    shade = 1.0 - 0.08 * (np.sin(phase)[None, :] + np.sin(phase * 0.5 + 1.3)[:, None]) / 2
    noise = 1.0 + rng.normal(0.0, 0.02, (size, size, 1))
    tex = (np.asarray(CARDBOARD_RGB, dtype=float) / 255.0)[None, None, :] * shade[..., None] * noise
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(tex, 0, 1) * 255).astype(np.uint8)).save(tex_path)


def build_usd(out_path: Path, tex_path: Path, mass: float) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    def cuboid_points(bounds):
        x0, x1, y0, y1, z0, z1 = bounds
        return [
            Gf.Vec3f(x0, y0, z0), Gf.Vec3f(x1, y0, z0), Gf.Vec3f(x1, y1, z0), Gf.Vec3f(x0, y1, z0),
            Gf.Vec3f(x0, y0, z1), Gf.Vec3f(x1, y0, z1), Gf.Vec3f(x1, y1, z1), Gf.Vec3f(x0, y1, z1),
        ]

    stage = Usd.Stage.CreateNew(str(out_path))
    root = UsdGeom.Xform.Define(stage, f"/{args_cli.name}")
    root_prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(mass)

    for name, bounds in PANELS.items():
        mesh = UsdGeom.Mesh.Define(stage, f"/{args_cli.name}/{name}")
        mesh.CreatePointsAttr(cuboid_points(bounds))
        mesh.CreateFaceVertexCountsAttr([4] * len(FACES))
        mesh.CreateFaceVertexIndicesAttr([i for face in FACES for i in face])
        mesh.CreateExtentAttr(
            [Gf.Vec3f(bounds[0], bounds[2], bounds[4]), Gf.Vec3f(bounds[1], bounds[3], bounds[5])]
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        # Convex hull of a cuboid panel is exact; the open top stays reachable.
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")

    # Textured material (MDL OmniPBR); must live under the defaultPrim to survive
    # the spawn-time reference, same as the fruit converter. diffuse_texture is
    # the input this Isaac version honors (diffuse_color is not).
    looks_path = f"/{args_cli.name}/Looks"
    UsdGeom.Scope.Define(stage, looks_path)
    mat = UsdShade.Material.Define(stage, f"{looks_path}/CardboardMaterial")
    shader = UsdShade.Shader.Define(stage, f"{looks_path}/CardboardMaterial/Shader")
    sprim = shader.GetPrim()
    sprim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    mdl_asset = sprim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset)
    mdl_asset.Set(Sdf.AssetPath("OmniPBR.mdl"))
    mdl_asset.SetCustom(False)
    mdl_sub = sprim.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token)
    mdl_sub.Set("OmniPBR")
    mdl_sub.SetCustom(False)
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(tex_path.resolve()))
    )
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("out"))
    for name in PANELS:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"/{args_cli.name}/{name}")).Bind(mat)

    stage.GetRootLayer().defaultPrim = args_cli.name
    stage.GetRootLayer().Save()
    print(
        f"built: {out_path} ({len(PANELS)} panels, "
        f"{OUTER_HALF_XY*2:.2f} x {OUTER_HALF_XY*2:.2f} x {HEIGHT:.2f} m, mass {mass}kg)"
    )


def main() -> None:
    out_dir = PROJECT_ROOT / "assets" / "objects" / args_cli.name
    tex_path = out_dir / "texture" / "cardboard.png"
    build_texture(tex_path)
    build_usd(out_dir / f"{args_cli.name}.usd", tex_path, args_cli.mass)


if __name__ == "__main__":
    main()
