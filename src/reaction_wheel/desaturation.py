"""
Momentum dumping (magnetorquer-based desaturation), hysteresis, and
operational scheduling.

This module composes -- and does not duplicate -- the verified building
blocks from `geometry.py` (wheel-axis matrix, minimum-norm allocation,
null-space basis, per-wheel utilization) and `momentum.py`/`disturbances.py`
(disturbance composition, mean-torque secular estimate, magnetic-dipole
torque). It adds only what M4 needs: a magnetorquer capability model,
dipole-command inversion, a closed-loop unloading law, a hysteresis state
machine, and closed-loop / hybrid momentum-management simulation.

Sign convention -- DERIVED, not assumed
----------------------------------------
M3 established that the wheel-space allocation of ANY external body-frame
torque (disturbance or otherwise) reuses `geometry.allocate_torque`
unchanged: tau_w = -A^+ @ tau_external, and dh_w/dt = tau_w.

For a magnetorquer unloading law to actually DECREASE stored wheel
momentum (rather than blow it up), the desired *external* unloading
torque must be chosen as

    tau_unload_desired = +k_H * H_w_body,      H_w_body = A @ h_w,  k_H > 0 [1/s]

(the OPPOSITE sign from a naive reading of "point the external torque
opposite the stored momentum" -- that naive sign, substituted into the
already-established tau_w = -A^+ tau_external relation, produces a
POSITIVE feedback loop that grows h_w without bound). This was verified
numerically before being adopted here (see
docs/desaturation_methodology.md section 1): for the orthogonal 3-wheel
geometry, tau_unload_desired = +k_H*H_w_body gives dh_w/dt = -k_H*h_w
exactly (clean exponential decay), which is the derived, correct
convention.

A second consequence of this sign convention, confirmed numerically and
used directly for the M4 null-space-redistribution study: for the
redundant 4-wheel geometry, the resulting dh_w/dt = -k_H * (A^+A) h_w
depends only on the ROW-SPACE (body-momentum-observable) component of
h_w -- it is EXACTLY ZERO for any null-space component of h_w. External
magnetorquer unloading can only ever remove the body-observable momentum
A@h_w; it structurally cannot touch a null-space wheel-momentum
imbalance, no matter how it is tuned. This is the precise, derived
statement of "internal redistribution cannot replace external
unloading" (see docs/desaturation_methodology.md).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from .geometry import WheelSetGeometry, allocate_torque, per_wheel_torque_utilization
from .disturbances import magnetic_dipole_torque
from .momentum import mean_wheel_torque


# ---------------------------------------------------------------------------
# Magnetorquer model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Magnetorquer:
    """Representative synthetic magnetorquer capability model (M4).

    Parameters
    ----------
    m_max : float
        Maximum TOTAL dipole magnitude [A*m^2] (isotropic capability --
        the commanded dipole vector is scaled down, preserving direction,
        if its norm exceeds this). Must be > 0.
    m_max_per_axis : float, optional
        If given, an ADDITIONAL independent per-axis dipole limit
        [A*m^2] applied after the isotropic saturation (component-wise
        clip). This changes the commanded direction (a real limitation of
        3-independent-rod hardware); omit it to keep the isotropic,
        direction-preserving model only.
    """

    m_max: float
    m_max_per_axis: Optional[float] = None
    name: str = "representative magnetorquer"

    def __post_init__(self):
        if not np.isfinite(self.m_max) or self.m_max <= 0.0:
            raise ValueError(f"m_max must be finite and > 0, got {self.m_max!r}")
        if self.m_max_per_axis is not None:
            if not np.isfinite(self.m_max_per_axis) or self.m_max_per_axis <= 0.0:
                raise ValueError(
                    f"m_max_per_axis must be finite and > 0, got {self.m_max_per_axis!r}"
                )

    def saturate(self, m_cmd) -> np.ndarray:
        """Saturate a commanded dipole vector to this magnetorquer's capability."""
        m_cmd = np.asarray(m_cmd, dtype=float).reshape(3)
        norm = np.linalg.norm(m_cmd)
        if norm > self.m_max:
            m_cmd = m_cmd * (self.m_max / norm)
        if self.m_max_per_axis is not None:
            m_cmd = np.clip(m_cmd, -self.m_max_per_axis, self.m_max_per_axis)
        return m_cmd

    def max_torque(self, B) -> float:
        """Maximum achievable torque magnitude m_max*|B|, reached when m perp B."""
        B = np.asarray(B, dtype=float).reshape(3)
        return self.m_max * np.linalg.norm(B)


def representative_magnetorquer() -> Magnetorquer:
    """Synthetic representative magnetorquer for M4 (not a commercial product).

    m_max = 20 A*m^2 is representative of a small/medium smallsat-class
    magnetorquer rod set; chosen to give meaningful, but not
    artificially convenient, torque authority against the M3 baseline
    LEO field (~3e-5 T) -- see docs/desaturation_methodology.md for the
    resulting torque-authority numbers.
    """
    return Magnetorquer(m_max=20.0, name="synthetic representative magnetorquer (M4)")


# ---------------------------------------------------------------------------
# Magnetic torque geometry: projection and dipole inversion
# ---------------------------------------------------------------------------

def project_perpendicular_to_field(tau_desired, B) -> np.ndarray:
    """tau_perp = (I - B_hat B_hat^T) @ tau_desired -- the magnetically
    achievable component of a desired torque (perpendicular to B).

    A magnetic dipole can never produce torque parallel to B, since
    (m x B).B = 0 identically for any m.
    """
    tau_desired = np.asarray(tau_desired, dtype=float).reshape(3)
    B = np.asarray(B, dtype=float).reshape(3)
    B_hat = B / np.linalg.norm(B)
    return tau_desired - np.dot(tau_desired, B_hat) * B_hat


def dipole_command_for_torque(tau_perp, B) -> np.ndarray:
    """Minimum-norm dipole command realizing a (already B-perpendicular)
    desired torque:

        m = (B x tau_perp) / |B|^2

    satisfies m x B = tau_perp exactly when tau_perp perp B (verified:
    triple-product identity (BxT)xB = T|B|^2 - B(B.T), and B.T=0 for
    T perp B). If `tau_perp` is not already perpendicular to B, only its
    perpendicular component is realized -- callers should pass the output
    of `project_perpendicular_to_field` for a torque that is not already
    known to be perpendicular.
    """
    tau_perp = np.asarray(tau_perp, dtype=float).reshape(3)
    B = np.asarray(B, dtype=float).reshape(3)
    B_norm_sq = np.dot(B, B)
    return np.cross(B, tau_perp) / B_norm_sq


def unloading_effectiveness(tau_desired, B) -> float:
    """eta_B = |tau_perp| / |tau_desired| -- fraction of a desired torque
    that is magnetically achievable at this instant, given only the field
    geometry (independent of magnetorquer capability/saturation).

    Returns 1.0 for a zero desired torque (nothing needs to be achieved).
    """
    tau_desired = np.asarray(tau_desired, dtype=float).reshape(3)
    norm_desired = np.linalg.norm(tau_desired)
    if norm_desired < 1e-15:
        return 1.0
    tau_perp = project_perpendicular_to_field(tau_desired, B)
    return float(np.linalg.norm(tau_perp) / norm_desired)


# ---------------------------------------------------------------------------
# Unloading command law
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnloadCommand:
    """One instant's magnetorquer unloading command and its realized torque."""

    tau_desired: np.ndarray   # k_H * H_w_body, before B-projection
    tau_perp: np.ndarray      # magnetically achievable component
    m_cmd_raw: np.ndarray     # unsaturated dipole command
    m_cmd: np.ndarray         # saturated (actual) dipole command
    tau_achieved: np.ndarray  # m_cmd x B (the torque actually realized)
    effectiveness: float      # |tau_perp| / |tau_desired| (field-geometry only)
    dipole_utilization: float  # |m_cmd_raw| / m_max (unsaturated demand vs capability)


def compute_unload_command(h_w, geometry: WheelSetGeometry, B, magnetorquer: Magnetorquer,
                            k_H: float) -> UnloadCommand:
    """Compute one instant's magnetorquer unloading command.

    tau_desired = +k_H * H_w_body   (H_w_body = A @ h_w; see module
    docstring for the sign derivation)
    tau_perp    = projection of tau_desired onto the plane perpendicular to B
    m_cmd       = saturate( (B x tau_perp)/|B|^2 )
    tau_achieved = m_cmd x B
    """
    if k_H < 0:
        raise ValueError(f"k_H must be >= 0 (1/s momentum feedback gain), got {k_H!r}")
    h_w = np.asarray(h_w, dtype=float)
    H_w_body = geometry.A @ h_w
    tau_desired = k_H * H_w_body
    tau_perp = project_perpendicular_to_field(tau_desired, B)
    m_cmd_raw = dipole_command_for_torque(tau_perp, B)
    m_cmd = magnetorquer.saturate(m_cmd_raw)
    tau_achieved = magnetic_dipole_torque(m_cmd, B)
    eff = unloading_effectiveness(tau_desired, B)
    dipole_util = float(np.linalg.norm(m_cmd_raw) / magnetorquer.m_max)
    return UnloadCommand(
        tau_desired=tau_desired, tau_perp=tau_perp, m_cmd_raw=m_cmd_raw, m_cmd=m_cmd,
        tau_achieved=tau_achieved, effectiveness=eff, dipole_utilization=dipole_util,
    )


# ---------------------------------------------------------------------------
# Hysteresis thresholds and state machine
# ---------------------------------------------------------------------------

class DesatState(str, Enum):
    ACCUMULATING = "accumulating"
    DESATURATING = "desaturating"


@dataclass(frozen=True)
class HysteresisThresholds:
    """Dump-on / dump-off operational momentum thresholds, per wheel.

    H_off < H_on is enforced (a proper hysteresis band, not a single
    chatter-prone threshold).
    """

    H_on: float
    H_off: float

    def __post_init__(self):
        if not (0.0 < self.H_off < self.H_on):
            raise ValueError(
                f"Hysteresis requires 0 < H_off < H_on; got H_off={self.H_off!r}, "
                f"H_on={self.H_on!r}"
            )


def next_state(current_state: DesatState, max_abs_h: float,
               thresholds: HysteresisThresholds) -> DesatState:
    """Hysteresis state transition based on worst-wheel |h_i|.

    ACCUMULATING -> DESATURATING when max|h_i| >= H_on.
    DESATURATING -> ACCUMULATING when max|h_i| <= H_off.
    Otherwise stays in the current state (this is what prevents chatter:
    a value between H_off and H_on never triggers a transition regardless
    of which side it was approached from).
    """
    if current_state == DesatState.ACCUMULATING and max_abs_h >= thresholds.H_on:
        return DesatState.DESATURATING
    if current_state == DesatState.DESATURATING and max_abs_h <= thresholds.H_off:
        return DesatState.ACCUMULATING
    return current_state


# ---------------------------------------------------------------------------
# Closed-loop dump simulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DumpResult:
    """Time history and summary of one closed-loop desaturation dump."""

    t: np.ndarray             # (M,)
    h_w: np.ndarray           # (M, N)
    tau_w: np.ndarray         # (M, N) total wheel torque history (disturbance + unload)
    m_cmd: np.ndarray         # (M, 3) commanded (saturated) dipole history
    tau_achieved: np.ndarray  # (M, 3) achieved magnetic torque history
    effectiveness: np.ndarray  # (M,) field-geometry effectiveness history
    reached_off: bool
    duration: float
    h_w_initial: np.ndarray
    h_w_final: np.ndarray


def simulate_dump(geometry: WheelSetGeometry, h0, tau_d: Callable[[float], np.ndarray],
                   B_field: Callable[[float], np.ndarray], magnetorquer: Magnetorquer,
                   k_H: float, H_off: float, dt: float, t_max: float,
                   tau_max: Optional[float] = None) -> DumpResult:
    """Closed-loop time-domain simulation of one desaturation dump.

    At each step: compute the unload command from the CURRENT wheel
    momentum (state feedback), combine with the disturbance torque at
    that instant, allocate the total external torque to wheels (reusing
    `geometry.allocate_torque`, the same map used throughout M2/M3), and
    trapezoidally integrate. Terminates as soon as every wheel is at or
    below H_off (never merely because body-space net momentum is small --
    the release criterion is checked per-wheel, per the M4 spec).

    Parameters
    ----------
    tau_max : float, optional
        If given, `tau_w` samples exceeding this are NOT clipped -- the
        raw (possibly infeasible) allocation is still integrated and
        recorded; check `per_wheel_torque_utilization` on the result for
        honest reporting of any excursion above 1.0 (see M4 spec section
        33: "if it does occur, report it honestly. Do not clip silently").
    """
    if dt <= 0 or t_max <= 0:
        raise ValueError("dt and t_max must be > 0")
    h0 = np.asarray(h0, dtype=float)
    N = geometry.n_wheels
    n_steps = int(np.ceil(t_max / dt)) + 1

    t_arr = np.zeros(n_steps)
    h_arr = np.zeros((n_steps, N))
    tau_w_arr = np.zeros((n_steps, N))
    m_cmd_arr = np.zeros((n_steps, 3))
    tau_ach_arr = np.zeros((n_steps, 3))
    eff_arr = np.zeros(n_steps)

    h = h0.copy()
    t = 0.0
    h_arr[0, :] = h
    reached_off = False
    k_final = 0

    for k in range(n_steps):
        B = np.asarray(B_field(t), dtype=float)
        cmd = compute_unload_command(h, geometry, B, magnetorquer, k_H)
        tau_total = tau_d(t) + cmd.tau_achieved
        tau_w = allocate_torque(geometry, tau_total).wheel_values

        t_arr[k] = t
        h_arr[k, :] = h
        tau_w_arr[k, :] = tau_w
        m_cmd_arr[k, :] = cmd.m_cmd
        tau_ach_arr[k, :] = cmd.tau_achieved
        eff_arr[k] = cmd.effectiveness
        k_final = k

        if np.max(np.abs(h)) <= H_off:
            reached_off = True
            break

        # Trapezoidal step using this step's rate (explicit; dt is chosen
        # small relative to 1/k_H and the field period by the caller).
        if k + 1 < n_steps:
            h_next = h + tau_w * dt
            t_next = t + dt
            B_next = np.asarray(B_field(t_next), dtype=float)
            cmd_next = compute_unload_command(h_next, geometry, B_next, magnetorquer, k_H)
            tau_total_next = tau_d(t_next) + cmd_next.tau_achieved
            tau_w_next = allocate_torque(geometry, tau_total_next).wheel_values
            h = h + 0.5 * (tau_w + tau_w_next) * dt
            t = t_next

    kf = k_final + 1
    result = DumpResult(
        t=t_arr[:kf], h_w=h_arr[:kf, :], tau_w=tau_w_arr[:kf, :],
        m_cmd=m_cmd_arr[:kf, :], tau_achieved=tau_ach_arr[:kf, :],
        effectiveness=eff_arr[:kf], reached_off=reached_off,
        duration=t_arr[kf - 1], h_w_initial=h0, h_w_final=h_arr[kf - 1, :],
    )
    return result


# ---------------------------------------------------------------------------
# Analytical repeat-interval estimate
# ---------------------------------------------------------------------------

def analytical_repeat_interval(H_on: float, H_off: float, tau_w_secular: float) -> Optional[float]:
    """T_repeat ~= (H_on - H_off) / |tau_w_secular|, for one wheel under an
    approximately constant secular accumulation torque, dumped from H_off
    back up to H_on. Returns None if tau_w_secular ~= 0 (never re-accumulates)."""
    if H_off >= H_on:
        raise ValueError(f"H_off ({H_off!r}) must be < H_on ({H_on!r})")
    if abs(tau_w_secular) < 1e-18:
        return None
    return (H_on - H_off) / abs(tau_w_secular)


# ---------------------------------------------------------------------------
# Null-space redistribution (analysis only -- does not replace unloading)
# ---------------------------------------------------------------------------

def null_space_redistribute(geometry: WheelSetGeometry, h_w, z) -> np.ndarray:
    """h_w_new = h_w + N @ z. A@h_w_new == A@h_w exactly (verified in
    tests) -- redistributing momentum among wheels via the null space
    never changes the body-frame observable momentum A@h_w, and
    therefore cannot, by itself, reduce total system angular momentum.
    Only an external torque (magnetorquer, thruster, ...) can do that.
    """
    N = geometry.null_space_basis()
    z = np.asarray(z, dtype=float).reshape(N.shape[1])
    return np.asarray(h_w, dtype=float) + N @ z


# ---------------------------------------------------------------------------
# Hybrid long-duration schedule simulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DumpEvent:
    """One completed desaturation-schedule event."""

    dump_number: int
    start_time: float
    start_orbit: float
    limiting_wheel: str
    initial_utilization: float
    duration: float
    final_utilization: float
    h_w_initial: np.ndarray
    h_w_final: np.ndarray
    next_interval: Optional[float]  # time to the NEXT dump's start, if known


def simulate_mission_schedule(geometry: WheelSetGeometry, tau_d: Callable[[float], np.ndarray],
                               tau_w_secular_mean: np.ndarray, B_field: Callable[[float], np.ndarray],
                               magnetorquer: Magnetorquer, k_H: float,
                               thresholds: HysteresisThresholds, T_horizon: float,
                               h0: Optional[np.ndarray] = None, dt_dump: float = 10.0,
                               t_max_dump: float = 20000.0, T_orb: float = 5677.0) -> List[DumpEvent]:
    """Hybrid accumulation/dump mission-schedule simulation.

    ACCUMULATING phase: advanced analytically using the mean secular
    per-wheel torque (`tau_w_secular_mean`, from `momentum.mean_wheel_torque`)
    -- exact for a linear constant-torque accumulation, and previously
    validated against full numerical integration to 0.03% at this scale
    (Milestone 3). This avoids integrating millions of unnecessary fine
    time steps solely to wait out a multi-hundred-orbit accumulation
    interval.

    DESATURATING phase: full closed-loop time-domain simulation via
    `simulate_dump` (state feedback + time-varying field + active
    disturbance), since this is the phase where the interesting
    fast-timescale dynamics actually happen.

    Parameters
    ----------
    T_horizon : float
        Total mission time to simulate [s].
    h0 : (N,) array_like, optional
        Initial wheel momentum. Defaults to zero.

    Returns
    -------
    List[DumpEvent]
    """
    N = geometry.n_wheels
    h = np.zeros(N) if h0 is None else np.asarray(h0, dtype=float).copy()
    t = 0.0
    events: List[DumpEvent] = []
    dump_number = 0
    start_times: List[float] = []

    # Safety cap so a misconfigured (non-accumulating) case cannot loop forever.
    max_events = 10000

    while t < T_horizon and dump_number < max_events:
        max_abs_h = np.max(np.abs(h))
        if max_abs_h < thresholds.H_on:
            # ACCUMULATING: analytically advance to the first wheel reaching H_on.
            t_to_on = []
            for i in range(N):
                tw = tau_w_secular_mean[i]
                if abs(tw) < 1e-18:
                    continue
                target = thresholds.H_on if tw > 0 else -thresholds.H_on
                dt_i = (target - h[i]) / tw
                if dt_i > 1e-9:
                    t_to_on.append(dt_i)
            if not t_to_on:
                break  # momentum never reaches H_on under this secular torque
            dt_accum = min(t_to_on)
            h = h + tau_w_secular_mean * dt_accum
            t = t + dt_accum
            if t >= T_horizon:
                break

        # DESATURATING: full closed-loop simulation from here.
        dump_number += 1
        start_time = t
        start_orbit = t / T_orb
        util_initial = float(np.max(np.abs(h)) / thresholds.H_on)
        worst_idx = int(np.argmax(np.abs(h)))
        worst_label = geometry.labels[worst_idx]

        def tau_d_shifted(tau_local, t0=t):
            return tau_d(t0 + tau_local)

        def B_field_shifted(tau_local, t0=t):
            return B_field(t0 + tau_local)

        dump = simulate_dump(geometry, h, tau_d_shifted, B_field_shifted, magnetorquer,
                              k_H, thresholds.H_off, dt_dump, t_max_dump)
        h = dump.h_w_final
        t = start_time + dump.duration
        util_final = float(np.max(np.abs(h)) / thresholds.H_on)

        events.append(DumpEvent(
            dump_number=dump_number, start_time=start_time, start_orbit=start_orbit,
            limiting_wheel=worst_label, initial_utilization=util_initial,
            duration=dump.duration, final_utilization=util_final,
            h_w_initial=dump.h_w_initial, h_w_final=dump.h_w_final,
            next_interval=None,
        ))
        start_times.append(start_time)

    # Fill in next_interval (time to the NEXT event's start) where known.
    filled_events = []
    for i, ev in enumerate(events):
        next_interval = (start_times[i + 1] - ev.start_time) if i + 1 < len(start_times) else None
        filled_events.append(DumpEvent(
            dump_number=ev.dump_number, start_time=ev.start_time, start_orbit=ev.start_orbit,
            limiting_wheel=ev.limiting_wheel, initial_utilization=ev.initial_utilization,
            duration=ev.duration, final_utilization=ev.final_utilization,
            h_w_initial=ev.h_w_initial, h_w_final=ev.h_w_final, next_interval=next_interval,
        ))
    return filled_events


# ---------------------------------------------------------------------------
# Schedule statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScheduleStatistics:
    n_dumps: int
    total_dump_time: float
    duty_cycle: float
    mean_interval: Optional[float]
    min_interval: Optional[float]
    max_interval: Optional[float]


def schedule_statistics(events: List[DumpEvent], T_horizon: float) -> ScheduleStatistics:
    """Aggregate statistics over a completed mission schedule."""
    n_dumps = len(events)
    total_dump_time = sum(ev.duration for ev in events)
    duty_cycle = total_dump_time / T_horizon if T_horizon > 0 else 0.0
    intervals = [ev.next_interval for ev in events if ev.next_interval is not None]
    mean_interval = float(np.mean(intervals)) if intervals else None
    min_interval = float(np.min(intervals)) if intervals else None
    max_interval = float(np.max(intervals)) if intervals else None
    return ScheduleStatistics(
        n_dumps=n_dumps, total_dump_time=total_dump_time, duty_cycle=duty_cycle,
        mean_interval=mean_interval, min_interval=min_interval, max_interval=max_interval,
    )
