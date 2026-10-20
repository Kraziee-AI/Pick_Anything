"""CPU SO101 collision-surface diagnostics, without Isaac or object ground truth.

The measured ray gap is local to the documented pad-reference line. Rotating,
nonparallel jaws have no unique scalar opening. Do not use this diagnostic as a
validated physical grasp-width controller or as an object-diameter estimator.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_USD = Path(__file__).resolve().parents[1] / 'assets/isaac_cache/Isaac/Robots_Multiphysics/RobotStudio/so101_new_calib_SysID/so101_new_calib.usda'
DEFAULT_VARIANTS = {'Robot': 'robot', 'Sensor': 'sensors', 'Physics': 'physics'}
FIXED_REFERENCE_M = np.array([-.01289, -.00022, -.0986])
MOVING_REFERENCE_M = np.array([-.0068, -.0766, .0189])


def _rotation(quaternion):
    return Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()


def _first_hit(origin, direction, triangles):
    """Nearest strictly forward triangle intersection (Moller-Trumbore)."""
    v0 = triangles[:, 0]
    e1, e2 = triangles[:, 1] - v0, triangles[:, 2] - v0
    h = np.cross(np.broadcast_to(direction, e2.shape), e2)
    a = np.einsum('ij,ij->i', e1, h)
    inv = np.zeros_like(a)
    valid = np.abs(a) > 1e-12
    inv[valid] = 1 / a[valid]
    s = origin - v0
    u = inv * np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = inv * (q @ direction)
    distance = inv * np.einsum('ij,ij->i', e2, q)
    valid &= (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8) & (distance > 1e-9)
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise ValueError('Reference ray misses collision meshes; local opening undefined')
    index = int(ids[np.argmin(distance[ids])])
    normal = np.cross(e1[index], e2[index])
    normal /= np.linalg.norm(normal)
    # Mesh winding can vary: orient the normal out of the reference solid.
    if normal @ direction < 0:
        normal = -normal
    return float(distance[index]), index, normal


def _coplanar_patch(triangles, names, index, normal):
    """Return the finite planar collision face containing the hit triangle."""
    selected = triangles[index]
    candidates = triangles[np.array(names) == names[index]]
    offsets = (candidates - selected[0]) @ normal
    coplanar = candidates[np.max(np.abs(offsets), axis=1) < 1e-7]
    vertices = np.unique(np.round(coplanar.reshape(-1, 3), 12), axis=0)
    area = .5 * np.linalg.norm(np.cross(coplanar[:, 1] - coplanar[:, 0],
                                      coplanar[:, 2] - coplanar[:, 0]), axis=1).sum()
    return {'vertices_body_m': vertices.tolist(), 'area_m2': float(area),
            'centroid_body_m': vertices.mean(0).tolist()}


class SO101GripperGeometry:
    """Read actual instanced collision meshes and joint-local transforms from USD."""

    def __init__(self, usd_path=DEFAULT_USD, variants=None):
        """Use project live variants by default; explicit {} reads authored defaults.

        Variant overrides live only in this stage's anonymous session layer.
        No source or referenced asset layer is saved or edited.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        self.usd_path = Path(usd_path).resolve()
        stage = Usd.Stage.Open(str(self.usd_path))
        if not stage:
            raise ValueError(f'Cannot read USD: {self.usd_path}')
        requested = dict(DEFAULT_VARIANTS if variants is None else variants)
        root = stage.GetDefaultPrim()
        if not root:
            raise ValueError(f'USD has no default robot prim: {self.usd_path}')
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for name, selection in requested.items():
                if name not in root.GetVariantSets().GetNames():
                    raise ValueError(f'Unknown robot variant set {name!r}')
                variant = root.GetVariantSet(name)
                if selection not in variant.GetVariantNames():
                    raise ValueError(f'Invalid robot variant {name}={selection!r}; available={variant.GetVariantNames()}')
                if not variant.SetVariantSelection(selection):
                    raise ValueError(f'Cannot select robot variant {name}={selection!r}')
        self.selected_variants = {name: root.GetVariantSet(name).GetVariantSelection()
                                  for name in root.GetVariantSets().GetNames()}
        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        if not np.isclose(self.meters_per_unit, 1.):
            raise ValueError('This calibration requires meter-valued USD geometry')
        joint = next(UsdPhysics.RevoluteJoint(p) for p in stage.Traverse()
                     if p.GetName() == 'gripper' and p.IsA(UsdPhysics.RevoluteJoint))
        self.parent_pivot = np.array(joint.GetLocalPos0Attr().Get())
        self.child_pivot = np.array(joint.GetLocalPos1Attr().Get())
        self.parent_rotation = _rotation(joint.GetLocalRot0Attr().Get())
        self.child_rotation = _rotation(joint.GetLocalRot1Attr().Get())
        self.axis = {'X': np.array([1., 0, 0]), 'Y': np.array([0., 1, 0]),
                     'Z': np.array([0., 0, 1])}[joint.GetAxisAttr().Get()]
        self.meshes = {}
        cache = UsdGeom.XformCache()
        for label, relationship in [('fixed', joint.GetBody0Rel()), ('moving', joint.GetBody1Rel())]:
            body = stage.GetPrimAtPath(relationship.GetTargets()[0])
            body_inverse = cache.GetLocalToWorldTransform(body).GetInverse()
            triangles, names = [], []
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
                    continue
                mesh = UsdGeom.Mesh(prim)
                transform = cache.GetLocalToWorldTransform(prim) * body_inverse
                vertices = np.array([transform.Transform(p) for p in mesh.GetPointsAttr().Get()])
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                offset = 0
                for count in mesh.GetFaceVertexCountsAttr().Get():
                    face = indices[offset:offset + count]
                    for k in range(1, count - 1):
                        triangles.append(vertices[face[[0, k, k + 1]]])
                        names.append(str(prim.GetPath()))
                    offset += count
            if not triangles:
                raise ValueError(f'No collision meshes found for {label}')
            self.meshes[label] = (np.asarray(triangles), names)
        self.evidence = {
            'usd_source': str(self.usd_path), 'requested_variants': requested,
            'selected_variants': self.selected_variants,
            'variant_layer': 'anonymous USD session layer; source assets unchanged',
            'variant_policy': 'project live defaults' if variants is None else 'explicit runtime variants',
            'mesh_paths': {label: sorted(set(names)) for label, (_, names) in self.meshes.items()},
            'mesh_counts': {label: len(set(names)) for label, (_, names) in self.meshes.items()},
            'triangle_counts': {label: len(triangles) for label, (triangles, _) in self.meshes.items()},
        }

    def moving_to_fixed(self, points, joint_rad):
        rotation = self.parent_rotation @ Rotation.from_rotvec(self.axis * joint_rad).as_matrix() @ self.child_rotation.T
        return (np.asarray(points) - self.child_pivot) @ rotation.T + self.parent_pivot

    def measure(self, joint_rad):
        """Intersect each pad's collision surface along its reference-center ray.

        Reports signed gap between first exits from the two reference solids,
        the actual hit triangles, local normals and their opposition error.
        Reference-line location and finite surface support remain explicit.
        """
        fixed = FIXED_REFERENCE_M
        moving = self.moving_to_fixed(MOVING_REFERENCE_M, joint_rad)
        span = float(np.linalg.norm(moving - fixed))
        direction = (moving - fixed) / span
        fixed_triangles, fixed_names = self.meshes['fixed']
        local_triangles, moving_names = self.meshes['moving']
        moving_triangles = self.moving_to_fixed(local_triangles, joint_rad)
        fixed_exit, fi, fn = _first_hit(fixed, direction, fixed_triangles)
        moving_exit, mi, mn = _first_hit(moving, -direction, moving_triangles)
        fixed_surface = fixed + fixed_exit * direction
        moving_surface = moving - moving_exit * direction
        opposition_error = float(np.degrees(np.arccos(np.clip(fn @ -mn, -1., 1.))))
        return {
            'joint_rad': float(joint_rad), 'reference_span_m': span,
            'local_surface_gap_m': span - fixed_exit - moving_exit,
            'fixed_surface_body_m': fixed_surface.tolist(),
            'moving_surface_body_m': moving_surface.tolist(),
            'surface_midpoint_body_m': ((fixed_surface + moving_surface) / 2).tolist(),
            'reference_midpoint_body_m': ((fixed + moving) / 2).tolist(),
            'ray_axis_body': direction.tolist(),
            'fixed_outward_normal_body': fn.tolist(), 'moving_outward_normal_body': mn.tolist(),
            'normal_opposition_error_deg': opposition_error,
            'fixed_ray_incidence_deg': float(np.degrees(np.arccos(np.clip(fn @ direction, -1., 1.)))),
            'moving_ray_incidence_deg': float(np.degrees(np.arccos(np.clip(mn @ -direction, -1., 1.)))),
            'fixed_collision_prim': fixed_names[fi], 'moving_collision_prim': moving_names[mi],
            'fixed_hit_triangle_body_m': fixed_triangles[fi].tolist(),
            'moving_hit_triangle_body_m': moving_triangles[mi].tolist(),
            'fixed_surface_patch': _coplanar_patch(fixed_triangles, fixed_names, fi, fn),
            'moving_surface_patch': _coplanar_patch(moving_triangles, moving_names, mi, mn),
            'semantics': 'local first-exit collision-surface gap on pad-reference ray; not universal grasp width',
        }
