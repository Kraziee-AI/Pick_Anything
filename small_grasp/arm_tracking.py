"""Bounded robot-joint feedback, independent of target objects and scoring.

This helper proposes a command only. The executor must approve its collision
path before committing it. It does not change the nominal arrival reference.
"""
from dataclasses import dataclass

import numpy as np

# Experimental planning reserve above the original runtime table margin.
# It never increases the allowed network-endpoint vertical correction.
PLANNING_TABLE_BUFFER_M = .008


@dataclass(frozen=True)
class MotionSynchronizationConfig:
    close_alignment_move_steps: int = 20
    close_max_steps: int = 360
    close_advance_steps: int = 55
    close_final_hold_steps: int = 15
    # Same speed/consecutive requirements used by the existing approach gate.
    arm_speed_tolerance_rad_s: float = .05
    consecutive_steps: int = 3


@dataclass(frozen=True)
class ArmTrackingConfig:
    gain: float = .5
    max_bias_rad: float = .025
    max_bias_step_rad: float = .004
    deadband_rad: float = .001
    max_command_distance_rad: float = .15

    def __post_init__(self):
        values = np.array(list(vars(self).values()))
        if (not np.isfinite(values).all() or (values < 0).any() or
                self.gain > 1 or self.max_bias_rad <= 0 or self.max_bias_step_rad <= 0 or
                self.max_command_distance_rad <= 0):
            raise ValueError('Invalid bounded arm tracking configuration')


@dataclass
class ArmTrackingDecision:
    nominal: np.ndarray
    command: np.ndarray
    bias: np.ndarray
    error: np.ndarray
    reason: str
    integral: np.ndarray | None = None
    integral_nominal: np.ndarray | None = None
    stationary_steps: int = 0
    integral_reason: str = 'disabled'
    integral_transfer_used: bool = False

    @property
    def active(self):
        return bool(np.any(self.bias))


@dataclass(frozen=True)
class _AdjacentIntegralTransfer:
    session: int
    from_index: int
    to_index: int
    previous_nominal: tuple
    next_nominal: tuple

class BoundedArmTrackingFeedback:
    def __init__(self, arm_indices, config=None, close_integral=False, integral_transfer='reset'):
        self.arm = np.asarray(arm_indices, dtype=int)
        if (self.arm.ndim != 1 or len(self.arm) != 5 or len(np.unique(self.arm)) != 5 or
                (self.arm < 0).any()):
            raise ValueError('Feedback requires five distinct arm indices')
        self.config = config or ArmTrackingConfig()
        self.bias = np.zeros(5)
        self.close_integral = bool(close_integral)
        if integral_transfer not in ('reset', 'adjacent') or (integral_transfer == 'adjacent' and not close_integral):
            raise ValueError('Adjacent integral transfer requires enabled close integral')
        self.integral_transfer = integral_transfer
        self.integral = np.zeros(5)
        self.integral_nominal = None
        self.stationary_steps = 0
        self._close_session = 0
        self._close_path = None
        self._close_index = 0
        self._pending_transfer = None

    def clear_integral(self):
        """Clear only I and its reference; applied total bias still uses original slew."""
        self.integral[:] = 0
        self.integral_nominal = None
        self.stationary_steps = 0
        self._pending_transfer = None

    def begin_close(self, prepared_path):
        self.clear_integral()
        self._close_session += 1
        self._close_path = prepared_path
        self._close_index = 0

    def end_close(self):
        self.clear_integral()
        self._close_path = None

    def authorize_adjacent_transfer(self, prepared_path, from_index, to_index, previous_nominal, next_nominal):
        """Executor calls only after the original full settle gate advances its path.

        No numeric-nearness inference is allowed. Bind a one-use token to the
        exact preflight path, indices, and last accepted nominal reference.
        """
        previous_nominal, next_nominal = map(lambda x: np.asarray(x, dtype=float), (previous_nominal, next_nominal))
        valid = (self.integral_transfer == 'adjacent' and self.close_integral and
                 prepared_path is self._close_path and self._close_path is not None and
                 from_index == self._close_index and to_index == from_index + 1 and
                 0 <= from_index < to_index < len(prepared_path) and
                 self.integral_nominal is not None and np.array_equal(previous_nominal, self.integral_nominal) and
                 np.array_equal(previous_nominal, prepared_path[from_index]) and
                 np.array_equal(next_nominal, prepared_path[to_index]) and
                 np.isfinite(previous_nominal).all() and np.isfinite(next_nominal).all())
        if not valid:
            self.clear_integral()
            return None
        token = _AdjacentIntegralTransfer(self._close_session, from_index, to_index,
                                          tuple(previous_nominal), tuple(next_nominal))
        self._pending_transfer = token
        return token

    def reset(self):
        self.bias[:] = 0
        self.end_close()

    def propose(self, nominal, actual, limits, tracking=True, *, phase=None, dt_s=None, arm_velocity=None,
                integral_transfer=None):
        token = self._pending_transfer
        self._pending_transfer = None  # one use, including an invalid/rejected observation
        authorized = (token is not None and integral_transfer is token and
                      token.session == self._close_session and self._close_path is not None)
        nominal, actual, limits = (np.asarray(x, dtype=float) for x in (nominal, actual, limits))
        if (nominal.ndim != 1 or actual.shape != nominal.shape or
                limits.shape != (len(nominal), 2) or self.arm.max() >= len(nominal)):
            raise ValueError('Feedback requires matching full joint state and limits')
        zero = np.zeros(5)
        def rejected(reason):
            return ArmTrackingDecision(nominal.copy(), nominal.copy(), zero.copy(), zero.copy(), reason)
        if not all(np.isfinite(x).all() for x in (nominal, actual, limits)) or (limits[:, 0] > limits[:, 1]).any():
            return rejected('nonfinite_state_or_invalid_limits')
        c = self.config
        error = nominal[self.arm] - actual[self.arm]
        dead_error = np.sign(error) * np.maximum(np.abs(error) - c.deadband_rad, 0)
        desired_bias = c.gain * dead_error
        integral, reference, stationary, integral_reason = zero.copy(), None, 0, 'disabled'
        transfer_used = False
        if self.close_integral:
            integral_reason = 'outside_close_phase'
            if tracking and phase == 'close_centered':
                velocity = np.asarray(arm_velocity, dtype=float)
                if (velocity.shape != (5,) or not np.isfinite(velocity).all() or
                        dt_s is None or not np.isfinite(dt_s) or not 0 < dt_s <= .05):
                    return rejected('invalid_close_integral_observation')
                reference = nominal.copy()
                same = self.integral_nominal is not None and np.array_equal(nominal, self.integral_nominal)
                transfer_used = (authorized and np.array_equal(nominal, token.next_nominal) and
                    self.integral_nominal is not None and np.array_equal(self.integral_nominal, token.previous_nominal))
                if transfer_used:
                    same = False  # even duplicate numeric endpoints must debounce at the new path index
                stationary = self.stationary_steps + 1 if same else 0
                integral = self.integral.copy() if same or transfer_used else zero.copy()
                integral_reason = ('adjacent_transfer' if transfer_used else
                                   'nominal_changed' if not same else 'nominal_debounce')
                if stationary >= 3:
                    integral_reason = 'arm_moving'
                    if np.max(np.abs(velocity)) <= .05:
                        increment = .2 * float(dt_s) * dead_error
                        candidate = desired_bias + integral + increment
                        allowed = ((np.abs(candidate) <= c.max_bias_rad) |
                                   (np.sign(increment) != np.sign(desired_bias + integral)))
                        integral += np.where(allowed, increment, 0.)
                        integral = np.clip(integral, -c.max_bias_rad, c.max_bias_rad)
                        integral_reason = 'integrating' if allowed.all() else 'conditional_antiwindup'
                desired_bias += integral
        if not tracking:
            desired_bias[:] = 0  # phase transitions shed only the bounded slew
        desired_bias = np.clip(desired_bias, -c.max_bias_rad, c.max_bias_rad)
        bias = self.bias + np.clip(desired_bias - self.bias, -c.max_bias_step_rad, c.max_bias_step_rad)
        command = nominal.copy()
        command[self.arm] += bias
        if ((command[self.arm] < limits[self.arm, 0]) | (command[self.arm] > limits[self.arm, 1])).any():
            return rejected('compensated_command_outside_soft_limits')
        if np.max(np.abs(command[self.arm] - actual[self.arm])) > c.max_command_distance_rad:
            return rejected('compensated_command_discontinuous')
        return ArmTrackingDecision(nominal.copy(), command, bias, error, 'feedback_proposed',
                                   integral, reference, stationary, integral_reason, transfer_used)

    def commit(self, decision, accepted):
        """Remember only the applied offset; collision-rejected bias never persists."""
        self.bias[:] = decision.bias if accepted else 0
        self.integral[:] = decision.integral if accepted and decision.integral is not None else 0
        self.integral_nominal = decision.integral_nominal.copy() if accepted and decision.integral_nominal is not None else None
        self.stationary_steps = decision.stationary_steps if accepted else 0
        if accepted and decision.integral_transfer_used:
            self._close_index += 1
        if not accepted:
            self._pending_transfer = None
