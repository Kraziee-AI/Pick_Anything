"""Add visual-only tray meshes locally; preserve referenced collision and mass."""
from pathlib import Path


def main():
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    root = Path(__file__).resolve().parent
    source = root.parent / 'assets/objects/cardboard_box/cardboard_box.usd'
    destination = root / 'assets/cardboard_box_visible.usda'
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    original = Usd.Stage.Open(str(source))
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    body = UsdGeom.Xform.Define(stage, '/cardboard_box').GetPrim()
    body.GetReferences().AddReference('../../assets/objects/cardboard_box/cardboard_box.usd')
    stage.SetDefaultPrim(body)
    material = UsdShade.Material.Define(stage, '/cardboard_box/VisibleMaterial')
    shader = UsdShade.Shader.Define(stage, '/cardboard_box/VisibleMaterial/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.55, .32, .13))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.85)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    for prim in original.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        source_mesh = UsdGeom.Mesh(prim)
        mesh = UsdGeom.Mesh.Define(stage, '/cardboard_box/Visible_' + prim.GetName())
        for getter in ('Points', 'FaceVertexCounts', 'FaceVertexIndices', 'Extent', 'SubdivisionScheme', 'Orientation'):
            value = getattr(source_mesh, 'Get' + getter + 'Attr')().Get()
            if value is not None:
                getattr(mesh, 'Create' + getter + 'Attr')(value)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(.55, .32, .13)])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    print(destination)


if __name__ == '__main__':
    main()
