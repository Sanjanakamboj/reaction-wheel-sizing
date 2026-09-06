"""
Reaction-wheel mechanics model.

Implements the ideal-wheel relations (see docs/conventions.md, wheel-frame
convention):

    H_w        = J_w * Omega_w                  (stored momentum)
    H_max      = J_w * Omega_max                 (momentum capacity)
    Omega_w    = H_w / J_w                        (speed from momentum)
    Omega_dot  = tau_w / J_w                       (wheel acceleration)

All quantities here are in the WHEEL-frame sign convention: a positive
tau_w spins the wheel toward positive Omega_w, consistent with
H_w = J_w * Omega_w. Converting a required spacecraft body torque/momentum
into a wheel command requires the sign flip documented in
docs/conventions.md section 3.3 and is handled in maneuvers.py / sizing.py,
not here.

This module deliberately does NOT select a commercial/off-the-shelf wheel.
It defines a reusable capability model. A "representative" or "candidate"
wheel is just a `ReactionWheel` instance with illustrative numbers.
"""

from dataclasses import dataclass

import numpy as np

from .constants import rad_s_to_rpm, rpm_to_rad_s


@dataclass(frozen=True)
class ReactionWheel:
    """Ideal reaction-wheel capability model.

    Parameters
    ----------
    J_w : float
        Rotor moment of inertia about the spin axis [kg*m^2]. Must be > 0.
    Omega_max : float
        Maximum wheel speed magnitude [rad/s]. Must be > 0.
    tau_max : float
        Maximum motor/wheel torque magnitude [N*m]. Must be > 0.
    Omega_min : float, optional
        Minimum useful operating speed magnitude [rad/s] (e.g. to avoid a
        stiction/zero-crossing dead zone). Default 0.0 (no floor).
    name : str
        Human-readable label.
    """

    J_w: float
    Omega_max: float
    tau_max: float
    Omega_min: float = 0.0
    name: str = "representative wheel"

    def __post_init__(self):
        if not np.isfinite(self.J_w) or self.J_w <= 0.0:
            raise ValueError(f"J_w must be finite and > 0, got {self.J_w!r}")
        if not np.isfinite(self.Omega_max) or self.Omega_max <= 0.0:
            raise ValueError(f"Omega_max must be finite and > 0, got {self.Omega_max!r}")
        if not np.isfinite(self.tau_max) or self.tau_max <= 0.0:
            raise ValueError(f"tau_max must be finite and > 0, got {self.tau_max!r}")
        if self.Omega_min < 0.0:
            raise ValueError(f"Omega_min must be >= 0, got {self.Omega_min!r}")
        if self.Omega_min >= self.Omega_max:
            raise ValueError(
                f"Omega_min ({self.Omega_min!r}) must be < Omega_max ({self.Omega_max!r})"
            )

    # -- capacity ----------------------------------------------------------

    @property
    def H_max(self) -> float:
        """Maximum stored angular-momentum magnitude [N*m*s]."""
        return self.J_w * self.Omega_max

    @property
    def H_min(self) -> float:
        """Momentum magnitude at the minimum operating speed [N*m*s]."""
        return self.J_w * self.Omega_min

    # -- state relations -----------------------------------------------------

    def momentum(self, Omega_w: float) -> float:
        """Stored wheel angular momentum H_w = J_w * Omega_w [N*m*s]."""
        return self.J_w * Omega_w

    def speed_from_momentum(self, H_w: float) -> float:
        """Wheel speed Omega_w = H_w / J_w [rad/s]."""
        return H_w / self.J_w

    def angular_acceleration(self, tau_w: float) -> float:
        """Wheel angular acceleration Omega_dot = tau_w / J_w [rad/s^2]."""
        return tau_w / self.J_w

    def torque_from_acceleration(self, Omega_dot_w: float) -> float:
        """Wheel torque tau_w = J_w * Omega_dot_w [N*m]."""
        return self.J_w * Omega_dot_w

    # -- rpm convenience -----------------------------------------------------

    def speed_rpm(self, Omega_w: float) -> float:
        """Convert a wheel speed [rad/s] to rpm."""
        return rad_s_to_rpm(Omega_w)

    def speed_from_rpm(self, rpm: float) -> float:
        """Convert a wheel speed [rpm] to rad/s."""
        return rpm_to_rad_s(rpm)

    @property
    def Omega_max_rpm(self) -> float:
        return rad_s_to_rpm(self.Omega_max)

    # -- capability checks ---------------------------------------------------

    def torque_margin_ratio(self, tau_req: float) -> float:
        """rho_tau = |tau_req| / tau_max.

        < 1: capability satisfies requirement; = 1: at limit; > 1: insufficient.
        """
        return abs(tau_req) / self.tau_max

    def momentum_margin_ratio(self, H_req: float) -> float:
        """rho_H = |H_req| / H_max.

        < 1: capability satisfies requirement; = 1: at limit; > 1: insufficient.
        """
        return abs(H_req) / self.H_max

    def satisfies_torque(self, tau_req: float) -> bool:
        return self.torque_margin_ratio(tau_req) <= 1.0

    def satisfies_momentum(self, H_req: float) -> bool:
        return self.momentum_margin_ratio(H_req) <= 1.0


def representative_wheel() -> ReactionWheel:
    """A synthetic/representative wheel used to verify mechanics in M1.

    This is NOT a commercial product selection — see docs/wheel_sizing
    _methodology.md. Values are illustrative, chosen so the representative
    maneuver set (docs conventions: 10-90 deg over 20-60 s) produces
    physically sensible torque/momentum/speed numbers for a smallsat-class
    reaction wheel.
    """
    return ReactionWheel(
        J_w=0.02,                       # kg*m^2, small smallsat-class rotor
        Omega_max=rpm_to_rad_s(6000.0), # ~628 rad/s
        tau_max=0.20,                   # N*m
        Omega_min=0.0,
        name="synthetic representative wheel (M1)",
    )
