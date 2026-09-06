"""
Wheel-momentum accumulation, thresholds, and saturation-time analysis.

This module composes -- and does not duplicate -- the verified building
blocks from `geometry.py` (wheel-axis matrix, minimum-norm torque
allocation) and `wheel.py` (per-wheel momentum/speed relations) with a
time-domain disturbance torque callable from `disturbances.py`.

Wheel-space disturbance allocation reuses `geometry.allocate_torque`
directly: tau_w(t) = -A^+ @ tau_d(t), i.e. the disturbance vector is fed
through the SAME minimum-norm pseudoinverse map already verified in M2
for maneuver torque commands. This is a deliberate bookkeeping choice
(see docs/momentum_accumulation_methodology.md) rather than a
rigorous closed-loop-controller derivation of the disturbance-rejection
torque; every engineering conclusion drawn from this milestone (momentum
accumulation MAGNITUDE, worst-wheel identification, threshold-crossing
TIME, and geometry/failure comparisons) is invariant to the overall sign
convention, since disturbance pointing directions are themselves
illustrative assumptions, not rigorously derived.

Milestone 3 does NOT implement any unloading/desaturation control --
saturation is only detected and timed, never corrected.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from .geometry import WheelSetGeometry, allocate_torque, is_torque_allocation_feasible


# ---------------------------------------------------------------------------
# Wheel-space disturbance allocation (reuses geometry.allocate_torque)
# ---------------------------------------------------------------------------

def wheel_torque_history(geometry: WheelSetGeometry, tau_d: Callable[[float], np.ndarray],
                          t: np.ndarray, tau_max: Optional[float] = None) -> np.ndarray:
    """Per-wheel allocated torque history for a disturbance callable tau_d(t).

    tau_w(t) = -A^+ @ tau_d(t)  (reuses `geometry.allocate_torque`).

    If `tau_max` is given, every sample is checked for feasibility; an
    infeasible sample is reported via a printed-free exception rather than
    silently clipped (per M2's feasibility policy) -- callers that expect
    occasional infeasibility should catch `InfeasibleDisturbanceError`.

    Returns
    -------
    (len(t), N) array of wheel torques.
    """
    N = geometry.n_wheels
    out = np.zeros((len(t), N))
    for k, tk in enumerate(t):
        res = allocate_torque(geometry, tau_d(tk))
        if tau_max is not None and not is_torque_allocation_feasible(res.wheel_values, tau_max):
            raise InfeasibleDisturbanceError(
                f"Disturbance at t={tk:.3f}s exceeds wheel torque capability "
                f"under minimum-norm allocation: |tau_w|_max = "
                f"{np.max(np.abs(res.wheel_values)):.4g} N*m > tau_max = {tau_max:.4g} N*m"
            )
        out[k, :] = res.wheel_values
    return out


class InfeasibleDisturbanceError(RuntimeError):
    """Raised when an instantaneous disturbance exceeds wheel torque capability."""


# ---------------------------------------------------------------------------
# Wheel momentum integration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MomentumHistory:
    """Time history of per-wheel momentum under a disturbance torque history."""

    t: np.ndarray            # (M,)
    tau_w: np.ndarray        # (M, N) allocated wheel torque history
    h_w: np.ndarray          # (M, N) integrated wheel momentum history


def integrate_wheel_momentum(geometry: WheelSetGeometry, tau_d: Callable[[float], np.ndarray],
                              t: np.ndarray, h0: Optional[np.ndarray] = None) -> MomentumHistory:
    """Integrate per-wheel momentum h_w(t) = h_w(0) + integral(tau_w dt').

    Uses cumulative trapezoidal integration of the allocated wheel-torque
    history (transparent, verified against the exact constant- and
    sinusoidal-torque analytical solutions in tests/test_momentum.py).

    Parameters
    ----------
    geometry : WheelSetGeometry
    tau_d : callable
        Body-frame disturbance torque history, tau_d(t) -> (3,).
    t : (M,) array_like
        Monotonically increasing sample times [s].
    h0 : (N,) array_like, optional
        Initial per-wheel momentum [N*m*s]. Defaults to zero.
    """
    t = np.asarray(t, dtype=float)
    N = geometry.n_wheels
    h0_vec = np.zeros(N) if h0 is None else np.asarray(h0, dtype=float).reshape(N)

    tau_w = wheel_torque_history(geometry, tau_d, t)
    h_w = np.empty_like(tau_w)
    h_w[0, :] = h0_vec
    for k in range(1, len(t)):
        dt_seg = t[k] - t[k - 1]
        h_w[k, :] = h_w[k - 1, :] + 0.5 * (tau_w[k, :] + tau_w[k - 1, :]) * dt_seg

    return MomentumHistory(t=t, tau_w=tau_w, h_w=h_w)


# ---------------------------------------------------------------------------
# Analytical verification helpers
# ---------------------------------------------------------------------------

def analytical_constant_torque_momentum(h0: float, tau_w: float, t) -> np.ndarray:
    """h(t) = h0 + tau_w * t, for a constant per-wheel torque."""
    t = np.asarray(t, dtype=float)
    return h0 + tau_w * t


def analytical_sinusoidal_momentum_excursion(tau0: float, omega: float, t) -> np.ndarray:
    """Delta h(t) for tau_w(t) = tau0*sin(omega*t), starting from h(0)=0.

        Delta h(t) = (tau0/omega) * (1 - cos(omega*t))

    Bounded in [0, 2*tau0/omega] -- demonstrates that a zero-mean periodic
    wheel torque produces bounded, non-secular momentum.
    """
    _validate_omega = omega
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError(f"omega must be finite and > 0; got {omega!r}")
    t = np.asarray(t, dtype=float)
    return (tau0 / omega) * (1.0 - np.cos(omega * t))


def analytical_constant_torque_saturation_time(h0: float, tau_w: float, H_limit: float) -> Optional[float]:
    """Time for a single wheel under CONSTANT torque to reach +-H_limit.

    Handles sign correctly: a positive tau_w drives h(t) toward +H_limit,
    a negative tau_w drives it toward -H_limit. Returns None if tau_w is
    (numerically) zero -- momentum never changes, so the threshold is
    never reached in finite time.

        t_sat = (H_limit - h0) / tau_w    if tau_w > 0
        t_sat = (-H_limit - h0) / tau_w   if tau_w < 0

    Returns None (rather than a fabricated value) if the requested
    direction is already at/past the limit at t=0 in the direction tau_w
    is NOT driving toward (t_sat would be negative) -- callers should
    treat a negative analytical result as "already saturated at t=0" via
    the numerical `first_threshold_crossing` helper instead.
    """
    if abs(tau_w) < 1e-15:
        return None
    target = H_limit if tau_w > 0 else -H_limit
    t_sat = (target - h0) / tau_w
    return float(t_sat) if t_sat >= 0 else None


# ---------------------------------------------------------------------------
# Threshold crossing / utilization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThresholdCrossing:
    """Result of a numerical threshold-crossing search."""

    reached: bool
    time: Optional[float]
    wheel_index: Optional[int]
    wheel_label: Optional[str]
    momentum_at_threshold: Optional[float]


def first_threshold_crossing(history: MomentumHistory, H_threshold, labels=None) -> ThresholdCrossing:
    """Earliest time at which any |h_w,i(t)| >= H_threshold_i.

    Parameters
    ----------
    history : MomentumHistory
    H_threshold : float or (N,) array_like
        Per-wheel operational threshold [N*m*s]. A scalar is broadcast to
        all wheels.
    labels : sequence of str, optional
        Wheel labels for reporting (defaults to numeric indices).

    Returns
    -------
    ThresholdCrossing
        `reached=False` (and all other fields None) if the threshold is
        never crossed within the simulated horizon -- never a fabricated
        time.
    """
    N = history.h_w.shape[1]
    H_threshold = np.broadcast_to(np.asarray(H_threshold, dtype=float), (N,))
    abs_h = np.abs(history.h_w)
    exceed = abs_h >= H_threshold[np.newaxis, :]
    if not np.any(exceed):
        return ThresholdCrossing(reached=False, time=None, wheel_index=None,
                                  wheel_label=None, momentum_at_threshold=None)

    # Earliest sample index where ANY wheel exceeds its threshold.
    row_any = np.any(exceed, axis=1)
    k = int(np.argmax(row_any))  # first True
    # Among wheels exceeding at that sample, report the one with highest utilization.
    util_at_k = abs_h[k, :] / H_threshold
    i = int(np.argmax(util_at_k))
    label = labels[i] if labels is not None else str(i)
    return ThresholdCrossing(
        reached=True, time=float(history.t[k]), wheel_index=i, wheel_label=label,
        momentum_at_threshold=float(history.h_w[k, i]),
    )


def momentum_utilization(history: MomentumHistory, H_threshold) -> np.ndarray:
    """rho_H(t) = max_i |h_w,i(t)| / H_threshold_i, per time sample -> (M,) array."""
    N = history.h_w.shape[1]
    H_threshold = np.broadcast_to(np.asarray(H_threshold, dtype=float), (N,))
    return np.max(np.abs(history.h_w) / H_threshold[np.newaxis, :], axis=1)


def wheel_speed_history(history: MomentumHistory, J_w) -> np.ndarray:
    """Convert a wheel momentum history to a wheel speed history [rad/s].

    `J_w` may be a scalar (identical rotor inertia for all wheels) or an
    (N,) array.
    """
    N = history.h_w.shape[1]
    J_w = np.broadcast_to(np.asarray(J_w, dtype=float), (N,))
    return history.h_w / J_w[np.newaxis, :]


# ---------------------------------------------------------------------------
# Per-orbit accumulation and mean-torque approximation
# ---------------------------------------------------------------------------

def mean_wheel_torque(geometry: WheelSetGeometry, tau_d: Callable[[float], np.ndarray],
                       T: float, n: int = 2000) -> np.ndarray:
    """Time-mean allocated wheel torque over [0, T] (per-wheel, (N,) array)."""
    t = np.linspace(0.0, T, n)
    tau_w = wheel_torque_history(geometry, tau_d, t)
    return np.trapezoid(tau_w, t, axis=0) / T


def momentum_per_orbit(geometry: WheelSetGeometry, tau_d: Callable[[float], np.ndarray],
                        T_orb: float, h0: Optional[np.ndarray] = None, n: int = 2000) -> np.ndarray:
    """Delta h_w over exactly one orbital period [0, T_orb] (per-wheel, (N,) array)."""
    t = np.linspace(0.0, T_orb, n)
    history = integrate_wheel_momentum(geometry, tau_d, t, h0=h0)
    return history.h_w[-1, :] - history.h_w[0, :]


def secular_momentum_approximation(h0, tau_w_mean, t) -> np.ndarray:
    """h(t) ~= h0 + tau_w_mean * t, the mean-torque secular-accumulation estimate.

    `h0` and `tau_w_mean` are (N,) per-wheel arrays; `t` is a scalar or
    (M,) array of times. Returns an (N,) array if `t` is scalar, else
    (M, N).
    """
    h0 = np.asarray(h0, dtype=float)
    tau_w_mean = np.asarray(tau_w_mean, dtype=float)
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        return h0 + tau_w_mean * float(t)
    return h0[np.newaxis, :] + np.outer(t, tau_w_mean)


# ---------------------------------------------------------------------------
# Body/wheel momentum-conservation consistency check
# ---------------------------------------------------------------------------

def body_wheel_momentum_consistency(geometry: WheelSetGeometry, history: MomentumHistory,
                                     tau_d: Callable[[float], np.ndarray]) -> np.ndarray:
    """Residual between A@h_w(t) and -integral(tau_d dt') (both from t=0).

    Per the sign convention documented in this module and in
    docs/momentum_accumulation_methodology.md, A@Delta h_w(t) should equal
    -integral_0^t tau_d(t') dt' for a full-row-rank geometry (up to the
    null-space component of h_w, which does not appear for a
    minimum-norm allocation with zero initial null-space content).
    Returns the per-sample residual NORM (should be near-zero for a
    full-rank geometry with h_w(0) purely in the row space of A^+, e.g.
    zero initial momentum).
    """
    t = history.t
    N = geometry.n_wheels
    body_accum = np.zeros((len(t), 3))
    for k in range(1, len(t)):
        dt_seg = t[k] - t[k - 1]
        body_accum[k, :] = body_accum[k - 1, :] - 0.5 * (tau_d(t[k]) + tau_d(t[k - 1])) * dt_seg

    wheel_body_frame = (geometry.A @ (history.h_w - history.h_w[0, :]).T).T
    residual = np.linalg.norm(wheel_body_frame - body_accum, axis=1)
    return residual


# ---------------------------------------------------------------------------
# Multi-geometry comparison convenience
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeometryComparisonResult:
    geometry_name: str
    history: MomentumHistory
    max_abs_momentum: float
    worst_wheel_label: str
    utilization_final: float
    crossing: ThresholdCrossing


def compare_geometries(geometries: Dict[str, WheelSetGeometry], tau_d: Callable[[float], np.ndarray],
                        t: np.ndarray, H_threshold, h0: Optional[np.ndarray] = None) -> Dict[str, GeometryComparisonResult]:
    """Integrate and summarize wheel momentum for several geometries under the same disturbance."""
    results = {}
    for name, g in geometries.items():
        h0_g = None if h0 is None else np.asarray(h0, dtype=float)[: g.n_wheels]
        hist = integrate_wheel_momentum(g, tau_d, t, h0=h0_g)
        max_abs = float(np.max(np.abs(hist.h_w)))
        worst_idx = int(np.argmax(np.max(np.abs(hist.h_w), axis=0)))
        util_final = float(momentum_utilization(hist, H_threshold)[-1])
        crossing = first_threshold_crossing(hist, H_threshold, labels=g.labels)
        results[name] = GeometryComparisonResult(
            geometry_name=name, history=hist, max_abs_momentum=max_abs,
            worst_wheel_label=g.labels[worst_idx], utilization_final=util_final,
            crossing=crossing,
        )
    return results
