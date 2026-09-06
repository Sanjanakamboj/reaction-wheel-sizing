"""
Milestone 5 — integrated reaction-wheel sizing across M1-M4.

This module composes -- and does not duplicate -- the verified building
blocks from every prior milestone:

    - `maneuvers.triangular_slew_requirement`         (M1)
    - `geometry.allocate_torque`, `.isotropy_ratio`,
      `.WheelSetGeometry.remove_wheel`                 (M2)
    - `momentum.mean_wheel_torque`,
      `momentum.analytical_constant_torque_saturation_time`,
      `disturbances.CompositeDisturbance`               (M3)
    - `desaturation.analytical_repeat_interval`,
      `desaturation.simulate_dump`                      (M4)
    - `wheel.ReactionWheel`, `sizing.SizingMargins`     (M1)

It adds only what M5 needs: an explicit requirement hierarchy (torque,
momentum, speed, rotor inertia), fault-tolerant and directional-envelope
sizing, momentum-headroom-at-threshold analysis, operational-threshold
validation, a candidate-design feasibility classifier, and a small
deterministic robustness corner-case check.

No commercial hardware is selected here -- every function in this module
returns a REQUIRED capability, never a product recommendation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .spacecraft import Spacecraft
from .maneuvers import triangular_slew_requirement
from .geometry import WheelSetGeometry, allocate_torque, isotropy_ratio
from .wheel import ReactionWheel
from .sizing import SizingMargins
from .momentum import mean_wheel_torque, analytical_constant_torque_saturation_time
from .desaturation import analytical_repeat_interval


# ---------------------------------------------------------------------------
# Adopted maneuver requirement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DesignManeuver:
    """A named, explicit maneuver requirement (angle/duration/axis).

    M1 intentionally included a 45deg/10s "aggressive" case purely to
    demonstrate that torque and momentum are independent sizing
    constraints (docs/wheel_sizing_methodology.md); it was never adopted
    as a mission requirement. M5 must pick an explicit ADOPTED design
    maneuver rather than silently sizing to the harshest illustrative
    example -- see `ADOPTED_MANEUVER` below for the selection and its
    rationale.
    """

    label: str
    theta_deg: float
    T: float
    axis: str  # 'x', 'y', or 'z' -- which principal axis this maneuver is about


# The routine (non-aggressive) M1 maneuver family, largest angle at a
# realistic duration: a 90-degree reorientation in 60 s about the
# worst-inertia axis (Iy). This represents a plausible operational
# slew (e.g. a full-attitude retarget between imaging/comms pointing),
# NOT a stress-test artifact. It is adopted as the DESIGN requirement.
ADOPTED_MANEUVER = DesignManeuver(label="90 deg / 60 s (adopted design maneuver)",
                                  theta_deg=90.0, T=60.0, axis="y")

# Retained for comparison ONLY -- never discarded, never adopted as a
# sizing requirement (M1/M5 rationale: this was a verification/stress
# case, not an operational requirement).
STRESS_MANEUVER = DesignManeuver(label="45 deg / 10 s (M1 stress/verification case)",
                                 theta_deg=45.0, T=10.0, axis="y")


def maneuver_body_vectors(maneuver: DesignManeuver, spacecraft: Spacecraft) -> Tuple[np.ndarray, np.ndarray]:
    """Return (tau_body_vector, delta_H_body_vector) for a DesignManeuver
    about its specified single axis, using M1's exact triangular-slew
    formulas (reused, not duplicated)."""
    axis_index = {"x": 0, "y": 1, "z": 2}[maneuver.axis]
    I = spacecraft.inertia_about(maneuver.axis)
    req = triangular_slew_requirement(np.deg2rad(maneuver.theta_deg), maneuver.T, I)
    tau_vec = np.zeros(3)
    dH_vec = np.zeros(3)
    tau_vec[axis_index] = req.tau_req
    dH_vec[axis_index] = req.H_body_peak
    return tau_vec, dH_vec


# ---------------------------------------------------------------------------
# Torque requirement: nominal + fault-tolerant + directional envelope
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TorqueRequirementResult:
    tau_nominal: float          # worst-wheel torque, nominal geometry
    tau_failure: float          # worst-wheel torque, worst single-wheel-failure case
    tau_failure_penalty_pct: float  # (tau_failure/tau_nominal - 1) * 100
    per_failure: Dict[int, float]   # wheel index -> worst-wheel torque for that failure


def wheel_torque_requirement(geometry: WheelSetGeometry, tau_body: np.ndarray) -> TorqueRequirementResult:
    """Nominal and one-wheel-failure-tolerant worst-wheel torque requirement
    for a commanded body torque vector, reusing `geometry.allocate_torque`
    for both the nominal geometry and every single-wheel-removed subset.

    A geometry with only 3 wheels offers NO one-wheel-failure tolerance at
    all -- removing any wheel leaves only 2, which cannot span 3-axis body
    torque (`WheelSetGeometry` itself enforces N>=3). This is correctly
    represented as `tau_failure = inf` (mission-impossible after a failure),
    not silently skipped or treated as equal to the nominal requirement.
    """
    res_nominal = allocate_torque(geometry, tau_body)
    tau_nominal = float(np.max(np.abs(res_nominal.wheel_values)))

    per_failure = {}
    if geometry.n_wheels - 1 >= 3:
        for idx in range(geometry.n_wheels):
            g_failed = geometry.remove_wheel(idx)
            res_f = allocate_torque(g_failed, tau_body)
            per_failure[idx] = float(np.max(np.abs(res_f.wheel_values)))
        tau_failure = max(per_failure.values())
    else:
        # No surviving subset can span 3-axis torque -- failure is not
        # tolerable at all for this geometry.
        tau_failure = float("inf")

    penalty_pct = (tau_failure / tau_nominal - 1.0) * 100.0 if tau_nominal > 0 else 0.0

    return TorqueRequirementResult(
        tau_nominal=tau_nominal, tau_failure=tau_failure,
        tau_failure_penalty_pct=penalty_pct, per_failure=per_failure,
    )


def directional_envelope_torque_requirement(geometry: WheelSetGeometry, tau_body_req: float,
                                             n_directions: int = 2000) -> float:
    """Minimum per-wheel tau_max such that ANY body-torque direction up to
    magnitude `tau_body_req` is achievable (guarantee, not a single
    trajectory).

    Uses `geometry.isotropy_ratio` at unit wheel-capability (tau_max=1) to
    find the worst-direction capability-per-unit-tau_max, then scales:
    since minimum-norm allocation is linear in the demand, the required
    tau_max is tau_body_req / tau_min_per_unit_capability.
    """
    iso_unit = isotropy_ratio(geometry, tau_max=1.0, n_directions=n_directions)
    tau_min_per_unit = iso_unit["tau_min"]
    if tau_min_per_unit <= 0:
        raise ValueError("Geometry has zero worst-direction capability (rank-deficient?)")
    return tau_body_req / tau_min_per_unit


# ---------------------------------------------------------------------------
# Momentum requirement: maneuver-at-threshold headroom
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadroomResult:
    delta_H_maneuver: float   # worst-wheel momentum excursion for the maneuver
    H_pre_maneuver: float     # assumed pre-maneuver worst-wheel momentum (e.g. H_on)
    H_max: float
    feasible: bool            # |H_pre + delta_H| <= H_max ?
    margin: float             # H_max - |H_pre + delta_H| (positive = feasible with this much headroom left)


def maneuver_at_threshold_headroom(geometry: WheelSetGeometry, dH_body: np.ndarray,
                                    H_pre_maneuver: float, H_max: float) -> HeadroomResult:
    """Check whether the adopted maneuver can be executed with the worst
    wheel already at `H_pre_maneuver` (e.g. right before a scheduled
    dump, H_pre_maneuver = H_on), without exceeding the PHYSICAL wheel
    momentum capacity H_max (not the operational threshold).
    """
    res = allocate_torque(geometry, dH_body)  # reuses the same pseudoinverse map (see momentum.py)
    delta_H = float(np.max(np.abs(res.wheel_values)))
    worst_total = H_pre_maneuver + delta_H
    feasible = worst_total <= H_max
    return HeadroomResult(
        delta_H_maneuver=delta_H, H_pre_maneuver=H_pre_maneuver, H_max=H_max,
        feasible=feasible, margin=H_max - worst_total,
    )


def required_H_max_for_headroom(delta_H_maneuver: float, f_on: float) -> float:
    """Minimum physical H_max such that a maneuver induces at most
    `delta_H_maneuver` starting from the dump-on threshold H_on = f_on*H_max
    without exceeding H_max:

        f_on*H_max + delta_H_maneuver <= H_max
        H_max >= delta_H_maneuver / (1 - f_on)
    """
    if not (0.0 < f_on < 1.0):
        raise ValueError(f"f_on must be in (0,1); got {f_on!r}")
    return delta_H_maneuver / (1.0 - f_on)


def max_allowable_dump_on_fraction(delta_H_maneuver: float, H_max: float) -> float:
    """Maximum f_on such that maneuver headroom remains:

        f_on_max = 1 - delta_H_maneuver/H_max
    """
    if H_max <= 0:
        raise ValueError("H_max must be > 0")
    return 1.0 - delta_H_maneuver / H_max


def max_pre_maneuver_momentum(H_max: float, delta_H_maneuver: float) -> float:
    """H_pre,max = H_max - delta_H_maneuver -- the largest pre-maneuver
    worst-wheel momentum that still allows the maneuver to complete
    without exceeding H_max. Translates directly into an operations rule:
    'maneuver permitted only if worst-wheel utilization < H_pre,max/H_max'.
    """
    return H_max - delta_H_maneuver


# ---------------------------------------------------------------------------
# Rotor inertia / speed trade
# ---------------------------------------------------------------------------

def required_rotor_inertia(H_sized: float, Omega_max: float) -> float:
    """J_w_min = H_sized / Omega_max."""
    if Omega_max <= 0:
        raise ValueError("Omega_max must be > 0")
    return H_sized / Omega_max


def stored_energy(J_w: float, Omega: float) -> float:
    """E_w = 0.5 * J_w * Omega^2."""
    return 0.5 * J_w * Omega**2


def required_wheel_acceleration(tau_sized: float, J_w: float) -> float:
    """Omega_dot_max_req = tau_sized / J_w."""
    if J_w <= 0:
        raise ValueError("J_w must be > 0")
    return tau_sized / J_w


# ---------------------------------------------------------------------------
# Candidate wheel design space and feasibility classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeasibilityResult:
    torque_nominal_ok: bool
    torque_failure_ok: bool
    momentum_ok: bool
    nominal_feasible: bool     # torque_nominal_ok AND momentum_ok
    failure_tolerant_feasible: bool  # torque_failure_ok AND momentum_ok


def classify_candidate(tau_max: float, H_max: float, torque_req: TorqueRequirementResult,
                        H_required: float) -> FeasibilityResult:
    """Classify a candidate (tau_max, H_max) capability point against the
    derived torque/momentum requirements."""
    torque_nominal_ok = tau_max >= torque_req.tau_nominal
    torque_failure_ok = tau_max >= torque_req.tau_failure
    momentum_ok = H_max >= H_required
    return FeasibilityResult(
        torque_nominal_ok=torque_nominal_ok, torque_failure_ok=torque_failure_ok,
        momentum_ok=momentum_ok,
        nominal_feasible=torque_nominal_ok and momentum_ok,
        failure_tolerant_feasible=torque_failure_ok and momentum_ok,
    )


def feasibility_grid(tau_max_range: np.ndarray, H_max_range: np.ndarray,
                      torque_req: TorqueRequirementResult, H_required: float) -> np.ndarray:
    """Return an integer grid (len(H_max_range), len(tau_max_range)) coding:

        0 = infeasible (fails torque AND/OR momentum, even nominally)
        1 = feasible nominally only (fails under one-wheel failure)
        2 = feasible under one-wheel failure too (fully feasible)
    """
    grid = np.zeros((len(H_max_range), len(tau_max_range)), dtype=int)
    for i, H_max in enumerate(H_max_range):
        for j, tau_max in enumerate(tau_max_range):
            fr = classify_candidate(tau_max, H_max, torque_req, H_required)
            if fr.failure_tolerant_feasible:
                grid[i, j] = 2
            elif fr.nominal_feasible:
                grid[i, j] = 1
            else:
                grid[i, j] = 0
    return grid


# ---------------------------------------------------------------------------
# Sensitivity studies (thin wrappers around M1-M4 machinery)
# ---------------------------------------------------------------------------

def maneuver_time_sensitivity(geometry: WheelSetGeometry, spacecraft: Spacecraft, axis: str,
                               theta_deg: float, T_values) -> List[dict]:
    """tau_req/H_req/worst-wheel torque & momentum vs maneuver duration,
    fixed angle -- reuses `maneuvers.triangular_slew_requirement` and
    `geometry.allocate_torque` directly."""
    I = spacecraft.inertia_about(axis)
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    results = []
    for T in T_values:
        req = triangular_slew_requirement(np.deg2rad(theta_deg), T, I)
        tau_vec = np.zeros(3); tau_vec[axis_index] = req.tau_req
        dH_vec = np.zeros(3); dH_vec[axis_index] = req.H_body_peak
        tau_req_result = wheel_torque_requirement(geometry, tau_vec)
        H_req_result = wheel_torque_requirement(geometry, dH_vec)  # reused allocator, treated as momentum vector
        results.append(dict(T=T, tau_req=req.tau_req, H_req=req.H_body_peak,
                             tau_w_nominal=tau_req_result.tau_nominal,
                             tau_w_failure=tau_req_result.tau_failure,
                             h_w_nominal=H_req_result.tau_nominal,
                             h_w_failure=H_req_result.tau_failure))
    return results


def inertia_sensitivity(geometry: WheelSetGeometry, spacecraft: Spacecraft, maneuver: DesignManeuver,
                         scale_factors) -> List[dict]:
    """Uniform spacecraft-inertia scale sensitivity for the adopted maneuver."""
    I_base = spacecraft.inertia_about(maneuver.axis)
    axis_index = {"x": 0, "y": 1, "z": 2}[maneuver.axis]
    results = []
    for scale in scale_factors:
        I_scaled = I_base * scale
        req = triangular_slew_requirement(np.deg2rad(maneuver.theta_deg), maneuver.T, I_scaled)
        tau_vec = np.zeros(3); tau_vec[axis_index] = req.tau_req
        tau_req_result = wheel_torque_requirement(geometry, tau_vec)
        results.append(dict(scale=scale, I=I_scaled, tau_req=req.tau_req,
                             tau_w_nominal=tau_req_result.tau_nominal,
                             tau_w_failure=tau_req_result.tau_failure))
    return results


# ---------------------------------------------------------------------------
# Robustness corner case
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobustCaseResult:
    tau_required: float
    H_required: float
    tau_recommended: float
    H_recommended: float
    torque_feasible: bool
    momentum_feasible: bool
    overall_feasible: bool


def evaluate_robust_corner_case(geometry: WheelSetGeometry, spacecraft: Spacecraft,
                                 maneuver: DesignManeuver, inertia_scale: float,
                                 disturbance_scale: float, H_pre_maneuver: float,
                                 margins: SizingMargins, tau_recommended: float,
                                 H_recommended: float, failed_wheel_index: int = 0) -> RobustCaseResult:
    """Deterministic structured corner case: +inertia_scale spacecraft
    inertia, one wheel failed, adopted maneuver, with selected torque/
    momentum margins applied to the (scaled) raw requirement.
    `disturbance_scale` is accepted for reporting symmetry with the
    disturbance sensitivity study (M3/M4 already show disturbance
    magnitude affects operational cadence, not physical wheel sizing --
    see docs/final_sizing_methodology.md) and does not enter the torque/
    momentum feasibility check directly here.
    """
    g_failed = geometry.remove_wheel(failed_wheel_index)
    I_scaled = spacecraft.inertia_about(maneuver.axis) * inertia_scale
    req = triangular_slew_requirement(np.deg2rad(maneuver.theta_deg), maneuver.T, I_scaled)
    axis_index = {"x": 0, "y": 1, "z": 2}[maneuver.axis]
    tau_vec = np.zeros(3); tau_vec[axis_index] = req.tau_req
    dH_vec = np.zeros(3); dH_vec[axis_index] = req.H_body_peak

    tau_alloc = allocate_torque(g_failed, tau_vec)
    tau_required_raw = float(np.max(np.abs(tau_alloc.wheel_values)))
    tau_required = margins.SF_tau * tau_required_raw

    dH_alloc = allocate_torque(g_failed, dH_vec)
    delta_H_required_raw = float(np.max(np.abs(dH_alloc.wheel_values)))
    # Margin applies to the MANEUVER EXCURSION only, not to the pre-existing
    # threshold H_pre_maneuver (matching required_H_max_for_headroom's
    # convention exactly). Multiplying the whole (H_pre + delta_H) sum by
    # SF_H would compound margin onto H_pre itself -- and since H_pre is
    # typically f_on*H_max (i.e. scales with the very capacity being
    # solved for), that compounding makes the required-H_max equation
    # diverge whenever SF_H*f_on >= 1. This was caught by exactly that
    # divergence during M5 development and fixed here, not assumed.
    H_required = H_pre_maneuver + margins.SF_H * delta_H_required_raw

    torque_feasible = tau_recommended >= tau_required
    momentum_feasible = H_recommended >= H_required

    return RobustCaseResult(
        tau_required=tau_required, H_required=H_required,
        tau_recommended=tau_recommended, H_recommended=H_recommended,
        torque_feasible=torque_feasible, momentum_feasible=momentum_feasible,
        overall_feasible=torque_feasible and momentum_feasible,
    )


# ---------------------------------------------------------------------------
# Final desaturation schedule with the recommended wheel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinalScheduleResult:
    H_max: float
    H_on: float
    H_off: float
    repeat_interval_s: float
    repeat_interval_orbits: float
    worst_wheel_secular_torque: float


def final_desaturation_schedule(geometry: WheelSetGeometry, tau_d, T_orb: float,
                                 H_max: float, f_on: float, f_off: float) -> FinalScheduleResult:
    """Recompute the M3/M4-style repeat interval for a NEW wheel momentum
    capacity H_max, reusing `momentum.mean_wheel_torque` and
    `desaturation.analytical_repeat_interval` unchanged.
    """
    H_on = f_on * H_max
    H_off = f_off * H_max
    tau_w_mean = mean_wheel_torque(geometry, tau_d, T_orb, n=4000)
    worst_idx = int(np.argmax(np.abs(tau_w_mean)))
    t_repeat = analytical_repeat_interval(H_on, H_off, tau_w_mean[worst_idx])
    return FinalScheduleResult(
        H_max=H_max, H_on=H_on, H_off=H_off,
        repeat_interval_s=t_repeat, repeat_interval_orbits=t_repeat / T_orb,
        worst_wheel_secular_torque=tau_w_mean[worst_idx],
    )
