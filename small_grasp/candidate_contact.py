"""Optional soft ranking of feasible grasps against observed target geometry.

No candidate is moved, filtered, generated or declared executable here. Inputs
are real target RGB-D points and a robot-body pose already accepted by planning.
The finite SO101 pad's narrow body-Y support is only a geometric preference;
single-view surface medians are not object centers or proof of two-pad contact.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactBandConfig:
    enabled: bool = False
    policy: str = 'soft8'
    # USD fixed pad face Y[-.00593,.00549]m, moving pad Y[-.00581,.00561]m.
    band_center_body_y_m: float = -.00022
    half_width_m: float = .006
    offset_scale_m: float = .020
    coverage_reference: float = .25
    penalty_weight_deg: float = 8.
    offset_weight: float = .6
    min_points: int = 30
    confidence_points: int = 100

    def __post_init__(self):
        positive = (self.half_width_m, self.offset_scale_m, self.coverage_reference)
        if (self.policy not in ('soft8', 'band_first')
                or not all(np.isfinite(x) and x > 0 for x in positive)
                or not np.isfinite(self.band_center_body_y_m)
                or not np.isfinite(self.penalty_weight_deg) or self.penalty_weight_deg < 0
                or not 0 <= self.offset_weight <= 1
                or self.coverage_reference > 1
                or not isinstance(self.min_points, int) or self.min_points < 1
                or not isinstance(self.confidence_points, int)
                or self.confidence_points < self.min_points):
            raise ValueError('Invalid contact-band ranking configuration')


@dataclass(frozen=True)
class PreparedTargetCloud:
    points_world_m: np.ndarray
    evidence: dict


def prepare_target_cloud(points_world_m, *, voxel_m=.003):
    """One centroid per occupied world voxel, independent of candidate ranking.

    Removes duplicate pixel density bias; never imputes occluded target surfaces.
    The caller must supply only real target-mask depth, transformed by actual
    camera calibration. This function cannot verify upstream provenance.
    """
    if not np.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError('voxel_m must be finite and positive')
    points = np.asarray(points_world_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Target point cloud must be Nx3 world meters')
    finite = np.isfinite(points).all(axis=1)
    valid = points[finite]
    if len(valid):
        _, inverse, counts = np.unique(np.floor(valid / voxel_m), axis=0,
                                       return_inverse=True, return_counts=True)
        centroids = np.zeros((len(counts), 3), dtype=float)
        np.add.at(centroids, inverse, valid)
        centroids /= counts[:, None]
    else:
        centroids = np.empty((0, 3), dtype=float)
    centroids.setflags(write=False)
    return PreparedTargetCloud(centroids, {
        'coordinate_frame': 'world', 'units': 'meters', 'voxel_m': float(voxel_m),
        'input_points': len(points), 'finite_points': int(finite.sum()),
        'nonfinite_removed': int((~finite).sum()), 'voxel_points': len(centroids),
        'source_requirement': 'actual target mask depth and actual camera calibration; no privileged target geometry',
    })


def contact_band_cost(cloud, pocket_world_m, body_rotation_world, tool_offset_body_m,
                      *, config=ContactBandConfig()):
    """Return JSON-safe evidence and a [0, weight] degree-equivalent penalty.

    soft8 adds penalty_deg_equivalent after all existing IK, collision and route
    gates pass; band_first must use contact_priority_key. Missing/insufficient geometry falls back to
    zero penalty with a reason; it must not reject an otherwise feasible plan.
    Disabled returns zero without inspecting any geometry. Stable ties retain
    the existing plan enumeration order. This is not a success probability.
    """
    result = {'enabled': bool(config.enabled), 'policy': config.policy, 'status': 'disabled',
              'penalty_deg_equivalent': 0., 'point_count': 0,
              'observed_median_within_band': False,
              'semantics': 'soft observed-surface contact-band preference; not object center, contact proof or success probability',
              'formula': 'weight*confidence*(offset_weight*offset_term+(1-offset_weight)*coverage_deficit)',
              'penalty_weight_deg': float(config.penalty_weight_deg)}
    if not config.enabled:
        return result
    if cloud is None:
        result['status'] = 'missing_target_cloud'
        return result
    if not isinstance(cloud, PreparedTargetCloud):
        raise TypeError('Use prepare_target_cloud for explicitly world-frame target points')
    result['cloud_preparation'] = dict(cloud.evidence)
    points = cloud.points_world_m
    result['point_count'] = len(points)
    if len(points) < config.min_points:
        result['status'] = 'insufficient_observed_points'
        return result
    try:
        pocket = np.asarray(pocket_world_m, dtype=float)
        rotation = np.asarray(body_rotation_world, dtype=float)
        offset = np.asarray(tool_offset_body_m, dtype=float)
    except (TypeError, ValueError):
        result['status'] = 'invalid_plan_geometry'
        return result
    if (pocket.shape != (3,) or offset.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(pocket).all() or not np.isfinite(offset).all()
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4)):
        result['status'] = 'invalid_plan_geometry'
        return result
    # Row-vector inverse of body-world pose; body origin = pocket - R@offset.
    body_y = ((points - pocket) @ rotation)[:, 1] + offset[1]
    q10, median, q90 = np.quantile(body_y, [.1, .5, .9])
    delta = body_y - config.band_center_body_y_m
    band_count = int((np.abs(delta) <= config.half_width_m).sum())
    coverage = band_count / len(points)
    observed_offset = float(median - config.band_center_body_y_m)
    offset_term = float(np.clip(abs(observed_offset) / config.offset_scale_m, 0, 1))
    deficit = float(1 - np.clip(coverage / config.coverage_reference, 0, 1))
    confidence = float(min(1., len(points) / config.confidence_points))
    penalty = config.penalty_weight_deg * confidence * (
        config.offset_weight * offset_term + (1 - config.offset_weight) * deficit)
    result.update(status='scored', penalty_deg_equivalent=float(penalty),
                  observed_median_within_band=abs(observed_offset) <= config.half_width_m,
                  confidence=confidence, band_center_body_y_m=config.band_center_body_y_m,
                  half_width_m=config.half_width_m, offset_scale_m=config.offset_scale_m,
                  coverage_reference=config.coverage_reference, offset_weight=config.offset_weight,
                  coverage_fraction=float(coverage), band_points=band_count,
                  observed_body_y_q10_q50_q90_m=[float(q10), float(median), float(q90)],
                  observed_median_y_offset_m=observed_offset,
                  normalized_offset_term=offset_term, coverage_deficit_term=deficit)
    return result


def contact_priority_key(base_cost, diagnostic, *, config=ContactBandConfig()):
    """Comparable key for already feasible plans; retains all candidates.

    soft8 preserves the existing additive preference. band_first prioritizes
    scored plans whose observed median lies inside the same finite pad-Y band;
    these retain original base-cost ordering. Other plans retain soft8 ordering,
    so if no plan lies inside the band, the whole pool falls back to soft8.
    This surface-median grouping is not proof of centering or two-pad contact.
    Stable ties must retain the original plan enumeration order.
    """
    base = float(base_cost)
    if not np.isfinite(base):
        raise ValueError('Plan base cost must be finite')
    if not config.enabled:
        return (0, base)
    penalty = float(diagnostic.get('penalty_deg_equivalent', 0.))
    if not np.isfinite(penalty) or not 0 <= penalty <= config.penalty_weight_deg:
        raise ValueError('Contact penalty is outside configured finite bounds')
    if config.policy == 'soft8':
        return (0, base + penalty)
    median = diagnostic.get('observed_median_y_offset_m')
    near = (diagnostic.get('enabled') is True and diagnostic.get('status') == 'scored'
            and median is not None and np.isfinite(median)
            and abs(median) <= config.half_width_m)
    return (0, base) if near else (1, base + penalty)
