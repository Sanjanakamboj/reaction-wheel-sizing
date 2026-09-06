"""
Maneuver torque fundamentals and the symmetric (triangular-rate) rest-to-rest
single-axis slew.

All torques/momenta/rates returned by this module are in BODY-torque
convention (see docs/conventions.md section 5): tau_req is the torque
applied TO the spacecraft body to achieve the commanded kinematics. Where a
function returns the corresponding WHEEL command, it is named accordingly
and the sign flip (section 3.3) is explicit.

Rest-to-rest triangular-rate slew of angle theta in duration T, with
constant acceleration magnitude alpha for the first half and equal
deceleration for the second half:

    theta       = alpha * (T/2)^2
    alpha       = 4*theta / T^2
    tau_req     = I * alpha = 4*I*theta / T^2
    omega_peak  = alpha * (T/2) = 2*theta / T
    H_body_peak = I * omega_peak
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Basic rigid-body torque/momentum/energy relations
# ---------------------------------------------------------------------------

def angular_acceleration_from_torque(tau: float, I: float) -> float:
    """alpha = tau / I  [rad/s^2]."""
    _validate_inertia(I)
    return tau / I


def torque_from_angular_acceleration(alpha: float, I: float) -> float:
    """tau = I * alpha  [N*m]."""
    _validate_inertia(I)
    return I * alpha


def angular_momentum(I: float, omega: float) -> float:
    """H = I * omega  [N*m*s]."""
    _validate_inertia(I)
    return I * omega


def kinetic_energy(I: float, omega: float) -> float:
    """T = 1/2 * I * omega^2  [J]."""
    _validate_inertia(I)
    return 0.5 * I * omega**2


def _validate_inertia(I: float) -> None:
    if not np.isfinite(I) or I <= 0.0:
        raise ValueError(f"Inertia I must be finite and > 0, got {I!r}")


def _validate_duration(T: float) -> None:
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"Maneuver duration T must be finite and > 0, got {T!r}")


# ---------------------------------------------------------------------------
# Symmetric (triangular-rate) rest-to-rest slew — analytical formulas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlewRequirement:
    """Analytical rest-to-rest triangular-rate slew requirement.

    Attributes
    ----------
    theta : float
        Commanded slew angle [rad].
    T : float
        Total maneuver duration [s].
    I : float
        Axis moment of inertia [kg*m^2].
    alpha : float
        Required angular acceleration magnitude [rad/s^2].
    tau_req : float
        Required body torque magnitude [N*m] (body-torque convention).
    omega_peak : float
        Peak body rate magnitude, reached at T/2 [rad/s].
    H_body_peak : float
        Peak spacecraft body angular-momentum magnitude [N*m*s].
    """

    theta: float
    T: float
    I: float
    alpha: float
    tau_req: float
    omega_peak: float
    H_body_peak: float

    @property
    def wheel_torque_cmd(self) -> float:
        """Wheel-side torque command during the accelerate phase (sign-flipped, section 3.3)."""
        return -self.tau_req

    @property
    def delta_H_wheel(self) -> float:
        """Required wheel momentum excursion (sign-flipped from body momentum, section 3.3)."""
        return -self.H_body_peak


def triangular_slew_requirement(theta: float, T: float, I: float) -> SlewRequirement:
    """Compute the rest-to-rest triangular-rate slew requirement.

    Parameters
    ----------
    theta : float
        Commanded slew angle magnitude [rad]. Must be > 0.
    T : float
        Total maneuver duration [s]. Must be > 0.
    I : float
        Axis moment of inertia [kg*m^2]. Must be > 0.

    Returns
    -------
    SlewRequirement
    """
    if not np.isfinite(theta) or theta <= 0.0:
        raise ValueError(f"theta must be finite and > 0, got {theta!r}")
    _validate_duration(T)
    _validate_inertia(I)

    alpha = 4.0 * theta / T**2
    tau_req = I * alpha
    omega_peak = alpha * (T / 2.0)  # == 2*theta/T
    H_body_peak = I * omega_peak

    return SlewRequirement(
        theta=theta, T=T, I=I,
        alpha=alpha, tau_req=tau_req,
        omega_peak=omega_peak, H_body_peak=H_body_peak,
    )


# ---------------------------------------------------------------------------
# Numerical propagation (independent verification path)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManeuverTimeHistory:
    """Time history of a numerically propagated rest-to-rest slew."""

    t: np.ndarray
    theta: np.ndarray       # spacecraft attitude angle [rad]
    omega: np.ndarray       # spacecraft body rate [rad/s]
    H_wheel: np.ndarray     # wheel angular momentum [N*m*s]
    tau_body: np.ndarray    # applied body torque [N*m]


def propagate_rest_to_rest_slew(
    theta_cmd: float, T: float, I: float, n_points: int = 400
) -> ManeuverTimeHistory:
    """Numerically integrate the ideal rest-to-rest slew.

    Dynamics (isolated spacecraft + single reaction wheel, no external
    torque; body torque = -wheel reaction torque, see conventions section
    3.3-3.4):

        theta_dot   = omega
        I*omega_dot = tau_body(t)
        H_wheel_dot = -tau_body(t)

    tau_body(t) is a bang-bang profile: +tau_req for t in [0, T/2),
    -tau_req for t in [T/2, T], where tau_req = I*alpha from the analytical
    triangular-rate solution. This is an independent numerical check of the
    closed-form result, not a controller design.

    Parameters
    ----------
    theta_cmd : float
        Commanded slew angle [rad]. Must be > 0.
    T : float
        Maneuver duration [s]. Must be > 0.
    I : float
        Axis inertia [kg*m^2]. Must be > 0.
    n_points : int
        Number of output sample points.

    Returns
    -------
    ManeuverTimeHistory
    """
    req = triangular_slew_requirement(theta_cmd, T, I)
    tau_req = req.tau_req

    def tau_body_of_t(t):
        return tau_req if t < T / 2.0 else -tau_req

    def rhs(t, y):
        theta, omega, H_wheel = y
        tau_b = tau_body_of_t(t)
        theta_dot = omega
        omega_dot = tau_b / I
        H_wheel_dot = -tau_b
        return [theta_dot, omega_dot, H_wheel_dot]

    t_eval = np.linspace(0.0, T, n_points)
    # Force the solver to respect the torque discontinuity at T/2 exactly.
    sol = solve_ivp(
        rhs, (0.0, T), y0=[0.0, 0.0, 0.0],
        t_eval=t_eval, method="RK45",
        max_step=T / 200.0, rtol=1e-10, atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Numerical maneuver propagation failed: {sol.message}")

    theta_arr, omega_arr, H_wheel_arr = sol.y
    tau_body_arr = np.array([tau_body_of_t(ti) for ti in t_eval])

    return ManeuverTimeHistory(
        t=t_eval, theta=theta_arr, omega=omega_arr,
        H_wheel=H_wheel_arr, tau_body=tau_body_arr,
    )
