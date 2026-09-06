"""
Reusable maneuver-driven wheel-sizing API.

This module is the single place that answers the sizing questions posed in
the Milestone-1 spec:

    - What torque is required for this maneuver?
    - What momentum excursion is required?
    - What wheel speed does that imply (for a given rotor inertia)?
    - Does a candidate wheel capability satisfy torque? momentum?
    - Which constraint is active (torque-limited / momentum-limited /
      speed-limited)?

It composes `spacecraft.Spacecraft`, `maneuvers.triangular_slew_requirement`,
and `wheel.ReactionWheel`; it does not itself do any plotting or I/O.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .maneuvers import triangular_slew_requirement
from .wheel import ReactionWheel


class ActiveConstraint(str, Enum):
    """Which capability limit is the binding (active) constraint."""

    TORQUE = "torque-limited"
    MOMENTUM = "momentum-limited"
    SPEED = "speed-limited"
    NONE = "within all limits (undetermined driver)"


@dataclass(frozen=True)
class SizingMargins:
    """Separate engineering margin (safety) factors for torque and momentum.

    These are illustrative representative factors, NOT adopted mission
    requirements — see docs/wheel_sizing_methodology.md.
    """

    SF_tau: float = 1.5
    SF_H: float = 1.5

    def __post_init__(self):
        if self.SF_tau <= 0 or self.SF_H <= 0:
            raise ValueError("Margin factors must be strictly positive")


@dataclass(frozen=True)
class ManeuverSizingResult:
    """Structured result of sizing a single-axis rest-to-rest maneuver."""

    axis: str
    theta_deg: float
    T: float
    I: float

    alpha: float                 # rad/s^2
    tau_req: float                # N*m (body-torque convention, magnitude)
    omega_peak: float             # rad/s
    omega_peak_deg_s: float
    H_req: float                  # N*m*s (required wheel momentum excursion magnitude)

    # Rotor-inertia-dependent derived quantities (representative rotor):
    J_w_ref: float
    Omega_req: float              # rad/s, wheel speed excursion for J_w_ref
    Omega_req_rpm: float
    Omega_dot_req: float          # rad/s^2, wheel accel requirement for J_w_ref

    # Margined ("sized") requirements:
    margins: SizingMargins
    tau_sized: float
    H_sized: float

    # Candidate-capability checks (optional; None if no candidate supplied):
    candidate: Optional[ReactionWheel] = None
    rho_tau: Optional[float] = None
    rho_H: Optional[float] = None
    rho_Omega: Optional[float] = None
    active_constraint: ActiveConstraint = ActiveConstraint.NONE


def size_single_axis_maneuver(
    axis: str,
    theta_deg: float,
    T: float,
    I: float,
    J_w_ref: float,
    margins: Optional[SizingMargins] = None,
    candidate: Optional[ReactionWheel] = None,
) -> ManeuverSizingResult:
    """Size the torque/momentum/speed requirement for one rest-to-rest slew.

    Parameters
    ----------
    axis : str
        Label for the spacecraft axis being maneuvered (e.g. 'x','y','z').
    theta_deg : float
        Commanded slew angle [deg].
    T : float
        Maneuver duration [s].
    I : float
        Axis moment of inertia [kg*m^2].
    J_w_ref : float
        Reference/representative rotor inertia [kg*m^2] used to translate
        the required momentum excursion into a required wheel-speed
        excursion (H_req = J_w_ref * Omega_req).
    margins : SizingMargins, optional
        Separate torque/momentum margin factors. Defaults to SizingMargins().
    candidate : ReactionWheel, optional
        If supplied, candidate-capability ratios (rho_tau, rho_H, rho_Omega)
        and the active constraint are computed against this wheel's
        tau_max / H_max / Omega_max.

    Returns
    -------
    ManeuverSizingResult
    """
    import numpy as np

    if margins is None:
        margins = SizingMargins()
    if J_w_ref <= 0 or not np.isfinite(J_w_ref):
        raise ValueError(f"J_w_ref must be finite and > 0, got {J_w_ref!r}")

    theta_rad = np.deg2rad(theta_deg)
    req = triangular_slew_requirement(theta_rad, T, I)

    H_req = abs(req.H_body_peak)
    tau_req = abs(req.tau_req)
    Omega_req = H_req / J_w_ref
    Omega_dot_req = tau_req / J_w_ref

    tau_sized = margins.SF_tau * tau_req
    H_sized = margins.SF_H * H_req

    rho_tau = rho_H = rho_Omega = None
    active = ActiveConstraint.NONE
    if candidate is not None:
        rho_tau = candidate.torque_margin_ratio(tau_req)
        rho_H = candidate.momentum_margin_ratio(H_req)
        rho_Omega = Omega_req / candidate.Omega_max
        active = _determine_active_constraint(rho_tau, rho_H, rho_Omega)

    return ManeuverSizingResult(
        axis=axis, theta_deg=theta_deg, T=T, I=I,
        alpha=req.alpha, tau_req=tau_req,
        omega_peak=req.omega_peak, omega_peak_deg_s=np.rad2deg(req.omega_peak),
        H_req=H_req,
        J_w_ref=J_w_ref, Omega_req=Omega_req,
        Omega_req_rpm=Omega_req * 60.0 / (2.0 * np.pi),
        Omega_dot_req=Omega_dot_req,
        margins=margins, tau_sized=tau_sized, H_sized=H_sized,
        candidate=candidate, rho_tau=rho_tau, rho_H=rho_H, rho_Omega=rho_Omega,
        active_constraint=active,
    )


def _determine_active_constraint(rho_tau: float, rho_H: float, rho_Omega: float) -> ActiveConstraint:
    """The active (binding) constraint is whichever ratio is largest.

    If the largest ratio exceeds 1.0 the candidate wheel is insufficient
    against that constraint; if all ratios are <= 1.0 the wheel satisfies
    all three, and the "active" (most-restrictive / least-margin)
    constraint is still reported for engineering insight.
    """
    ratios = {
        ActiveConstraint.TORQUE: rho_tau,
        ActiveConstraint.MOMENTUM: rho_H,
        ActiveConstraint.SPEED: rho_Omega,
    }
    return max(ratios, key=ratios.get)


def satisfies_torque(tau_req: float, wheel: ReactionWheel) -> bool:
    """Convenience wrapper: does `wheel` satisfy a required torque?"""
    return wheel.satisfies_torque(tau_req)


def satisfies_momentum(H_req: float, wheel: ReactionWheel) -> bool:
    """Convenience wrapper: does `wheel` satisfy a required momentum excursion?"""
    return wheel.satisfies_momentum(H_req)
