"""Optional bounded jaw upkeep using robot measurements only.

The optional servo adapter starts this helper only after contact-stall closure
and applies it only during hold/lift/lift_hold. Aggregate contact cannot identify
the held object or prove that a grasp exists. The force ceiling gates additional
closure; it is not a physical pressure guarantee.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HoldConfig:
    contact_floor_n: float = 2.2
    low_force_n: float = 12.0
    force_ceiling_n: float = 30.0
    consecutive_steps: int = 3
    max_velocity_rad_s: float = 0.15
    max_command_lag_rad: float = 0.035
    closure_rate_rad_s: float = 0.04
    max_extra_closure_rad: float = 0.04
    min_joint_rad: float = 0.05
    max_dt_s: float = 0.05

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Hold configuration must be finite')
        if not (0 <= self.contact_floor_n < self.low_force_n < self.force_ceiling_n):
            raise ValueError('Force thresholds must be increasing')
        if isinstance(self.consecutive_steps, bool) or not isinstance(self.consecutive_steps, int) or self.consecutive_steps < 1:
            raise ValueError('consecutive_steps must be a positive integer')
        if any(v <= 0 for v in (self.max_velocity_rad_s, self.max_command_lag_rad,
                               self.closure_rate_rad_s, self.max_extra_closure_rad, self.max_dt_s)):
            raise ValueError('Hold bounds must be positive')


@dataclass(frozen=True)
class HoldDecision:
    target_rad: float
    reason: str
    extra_closure_rad: float


class BoundedContactHold:
    """Own one jaw target after the closure latch, never after release.

    The caller must copy each returned target into every subsequently issued
    hold/lift joint target, avoiding interpolation that overwrites jaw upkeep.
    Construction and updates take no object pose, target label, held score or
    simulation contact-partner information. Small decreasing angles close the
    SO101 jaw. Invalid samples reset the debounce and retain the prior target.
    """

    def __init__(self, initial_target_rad, config=None):
        self.config = config or HoldConfig()
        self.initial_target = float(initial_target_rad)
        if not math.isfinite(self.initial_target) or self.initial_target < self.config.min_joint_rad:
            raise ValueError('Initial hold target must be finite and within the lower bound')
        self.target = self.initial_target
        self.minimum_target = max(self.config.min_joint_rad,
                                  self.initial_target - self.config.max_extra_closure_rad)
        self.consecutive = 0

    def update(self, force_n, joint_rad, velocity_rad_s, dt_s):
        cfg = self.config
        values = (force_n, joint_rad, velocity_rad_s, dt_s)
        if not all(math.isfinite(v) for v in values) or not 0 < dt_s <= cfg.max_dt_s:
            reason = 'invalid_observation'
        elif force_n < 0 or joint_rad < cfg.min_joint_rad:
            reason = 'invalid_observation'
        elif force_n >= cfg.force_ceiling_n:
            reason = 'force_ceiling'
        elif force_n <= cfg.contact_floor_n:
            reason = 'contact_absent'
        elif force_n >= cfg.low_force_n:
            reason = 'force_sufficient'
        elif abs(velocity_rad_s) > cfg.max_velocity_rad_s:
            reason = 'joint_moving'
        elif abs(joint_rad - self.target) > cfg.max_command_lag_rad:
            reason = 'command_not_settled'
        elif self.target <= self.minimum_target:
            reason = 'closure_budget_exhausted'
        else:
            self.consecutive += 1
            if self.consecutive < cfg.consecutive_steps:
                return self._decision('debounce')
            self.target = max(self.minimum_target, self.target - cfg.closure_rate_rad_s * dt_s)
            return self._decision('bounded_closure')
        self.consecutive = 0
        return self._decision(reason)

    def _decision(self, reason):
        return HoldDecision(self.target, reason, self.initial_target - self.target)
