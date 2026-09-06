"""
Deterministic body-frame environmental disturbance torque models.

Every disturbance in this module is a callable of the form

    tau_d(t) -> np.ndarray, shape (3,)   [N*m]

expressed in the SAME spacecraft body frame and sign convention frozen in
docs/conventions.md (a positive component follows the right-hand rule
about the corresponding body axis, exactly like the maneuver body torques
of `maneuvers.py`). This module only produces tau_d(t); it does NOT
allocate it to wheels (that composition lives in `momentum.py`, which
reuses `geometry.allocate_torque` rather than duplicating the pseudoinverse
math here) and it does NOT implement any unloading/desaturation control
(Milestone 4 scope).

Disturbance magnitudes here are ILLUSTRATIVE REPRESENTATIVE ENGINEERING
ASSUMPTIONS for a small LEO spacecraft, not a mission-specific
environmental model, except where a value is a standard, cited physical
constant (e.g. solar pressure at 1 AU). See
docs/momentum_accumulation_methodology.md for the full parameter
rationale.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Physical / orbital constants (documented, not hidden magic numbers)
# ---------------------------------------------------------------------------

MU_EARTH = 3.986004418e14        # m^3/s^2, standard Earth gravitational parameter
R_EARTH = 6378137.0              # m, WGS84 equatorial radius
SOLAR_PRESSURE_1AU = 4.56e-6     # N/m^2, standard solar radiation pressure at 1 AU (P = S/c)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _as_torque_vector(value) -> np.ndarray:
    v = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(v)):
        raise ValueError(f"Disturbance torque must be finite; got {value!r}")
    return v


def _validate_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")


# ---------------------------------------------------------------------------
# Orbital timing reference (timing only -- NOT a full orbit propagator)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrbitReference:
    """A representative circular-orbit timing reference.

    Used ONLY to give disturbance periodicity a physically meaningful
    period and to report "momentum per orbit" / "orbits to threshold".
    This is explicitly NOT a full orbit propagator -- no ground track,
    eclipse, or perturbation modeling.

    Parameters
    ----------
    altitude_km : float
        Circular-orbit altitude above the WGS84 equatorial radius [km].
    inclination_rad : float
        Orbit inclination [rad], used only to shape the representative
        magnetic-field direction history (see `dipole_field_body`).
    mu : float
        Gravitational parameter [m^3/s^2]. Defaults to Earth.
    """

    altitude_km: float = 500.0
    inclination_rad: float = np.deg2rad(51.6)  # representative ISS-class inclination
    mu: float = MU_EARTH

    def __post_init__(self):
        _validate_positive("altitude_km", self.altitude_km)
        _validate_positive("mu", self.mu)
        if not np.isfinite(self.inclination_rad):
            raise ValueError("inclination_rad must be finite")

    @property
    def radius_m(self) -> float:
        """Orbital radius [m] = Earth radius + altitude."""
        return R_EARTH + self.altitude_km * 1000.0

    @property
    def period_s(self) -> float:
        """Circular-orbit period T_orb = 2*pi*sqrt(r^3/mu) [s]."""
        r = self.radius_m
        return 2.0 * np.pi * np.sqrt(r**3 / self.mu)

    @property
    def mean_motion(self) -> float:
        """Orbital mean motion n = 2*pi/T_orb [rad/s]."""
        return 2.0 * np.pi / self.period_s

    @property
    def orbital_speed_mps(self) -> float:
        """Circular-orbit speed v = sqrt(mu/r) [m/s]."""
        return np.sqrt(self.mu / self.radius_m)


def leo_orbit_reference(altitude_km: float = 500.0) -> OrbitReference:
    """Representative LEO circular-orbit timing reference (default 500 km)."""
    return OrbitReference(altitude_km=altitude_km)


# ---------------------------------------------------------------------------
# Basic composable time-domain disturbance building blocks
# ---------------------------------------------------------------------------

def zero_disturbance() -> Callable[[float], np.ndarray]:
    """tau_d(t) = 0 for all t."""
    def _f(t: float) -> np.ndarray:
        return np.zeros(3)
    return _f


def constant_disturbance(tau0) -> Callable[[float], np.ndarray]:
    """tau_d(t) = tau0 (constant vector) for all t."""
    tau0 = _as_torque_vector(tau0)
    def _f(t: float) -> np.ndarray:
        return tau0.copy()
    return _f


def sinusoidal_disturbance(amplitude, omega: float, phase: float = 0.0, bias=None) -> Callable[[float], np.ndarray]:
    """tau_d(t) = bias + amplitude * sin(omega*t + phase), elementwise.

    Parameters
    ----------
    amplitude : array_like, shape (3,)
        Per-axis oscillation amplitude [N*m].
    omega : float
        Angular frequency [rad/s]. Must be > 0.
    phase : float
        Phase offset [rad].
    bias : array_like, shape (3,), optional
        Constant (secular) bias added to the oscillation. Defaults to zero
        (pure zero-mean oscillatory disturbance).
    """
    amplitude = _as_torque_vector(amplitude)
    _validate_positive("omega", omega)
    if not np.isfinite(phase):
        raise ValueError(f"phase must be finite; got {phase!r}")
    bias_vec = np.zeros(3) if bias is None else _as_torque_vector(bias)

    def _f(t: float) -> np.ndarray:
        return bias_vec + amplitude * np.sin(omega * t + phase)
    return _f


def callable_disturbance(func: Callable[[float], np.ndarray]) -> Callable[[float], np.ndarray]:
    """Wrap and validate an arbitrary user-defined tau_d(t) callable."""
    def _f(t: float) -> np.ndarray:
        return _as_torque_vector(func(t))
    return _f


class CompositeDisturbance:
    """A sum of named, independently-accessible disturbance components.

    tau_total(t) = sum_j tau_j(t)

    Each component remains separately queryable (`component(name, t)`) so
    downstream analysis can attribute momentum growth to a specific
    physical source (SRP, drag, gravity-gradient, magnetic, ...).
    """

    def __init__(self, components: Optional[Dict[str, Callable[[float], np.ndarray]]] = None):
        self._components: Dict[str, Callable[[float], np.ndarray]] = dict(components or {})

    @property
    def names(self):
        return tuple(self._components.keys())

    def add(self, name: str, func: Callable[[float], np.ndarray]) -> "CompositeDisturbance":
        self._components[name] = func
        return self

    def component(self, name: str, t: float) -> np.ndarray:
        return _as_torque_vector(self._components[name](t))

    def total(self, t: float) -> np.ndarray:
        if not self._components:
            return np.zeros(3)
        return np.sum([self.component(name, t) for name in self._components], axis=0)

    def __call__(self, t: float) -> np.ndarray:
        return self.total(t)

    def mean_component(self, name: str, T: float, n: int = 2000) -> np.ndarray:
        """Time-mean of one component over [0, T] via trapezoidal quadrature."""
        ts = np.linspace(0.0, T, n)
        vals = np.array([self.component(name, t) for t in ts])
        return np.trapezoid(vals, ts, axis=0) / T

    def mean_total(self, T: float, n: int = 2000) -> np.ndarray:
        """Time-mean of the total disturbance over [0, T]."""
        ts = np.linspace(0.0, T, n)
        vals = np.array([self.total(t) for t in ts])
        return np.trapezoid(vals, ts, axis=0) / T

    def peak_total_magnitude(self, T: float, n: int = 2000) -> float:
        """Peak |tau_total(t)| sampled over [0, T]."""
        ts = np.linspace(0.0, T, n)
        vals = np.array([self.total(t) for t in ts])
        return float(np.max(np.linalg.norm(vals, axis=1)))

    def peak_component_magnitude(self, name: str, T: float, n: int = 2000) -> float:
        """Peak |tau_component(t)| sampled over [0, T], for one named component."""
        ts = np.linspace(0.0, T, n)
        vals = np.array([self.component(name, t) for t in ts])
        return float(np.max(np.linalg.norm(vals, axis=1)))


# ---------------------------------------------------------------------------
# Physics-based disturbance-torque magnitude/vector models
# ---------------------------------------------------------------------------

def srp_torque(P_srp: float, C_R: float, A: float, r_cp, u_sun) -> np.ndarray:
    """Solar-radiation-pressure torque about the center of mass.

    F_srp = P_srp * C_R * A  (scalar force magnitude, along u_sun)
    tau_srp = r_cp x F_srp

    Parameters
    ----------
    P_srp : float
        Solar radiation pressure [N/m^2]. Use `SOLAR_PRESSURE_1AU` for a
        standard near-1-AU value (P = solar constant / speed of light).
    C_R : float
        Radiation-pressure/reflectivity coefficient (dimensionless,
        typically ~1.0-2.0; representative flat-plate value used here).
    A : float
        Effective illuminated area [m^2].
    r_cp : array_like, shape (3,)
        Body-frame vector from center of mass to center of pressure [m].
    u_sun : array_like, shape (3,)
        Body-frame unit vector of the incident solar force direction
        (NOT necessarily normalized here -- normalized internally).
    """
    _validate_positive("P_srp", P_srp)
    _validate_positive("C_R", C_R)
    _validate_positive("A", A)
    r_cp = np.asarray(r_cp, dtype=float).reshape(3)
    u_sun = np.asarray(u_sun, dtype=float).reshape(3)
    u_sun = u_sun / np.linalg.norm(u_sun)
    F = P_srp * C_R * A * u_sun
    return np.cross(r_cp, F)


def aero_torque(rho: float, v: float, C_D: float, A: float, r_cp, u_drag) -> np.ndarray:
    """Simplified free-molecular-flow drag torque about the center of mass.

    F_D = 0.5 * rho * v^2 * C_D * A   (scalar magnitude, along u_drag)
    tau_D = r_cp x F_D

    Parameters
    ----------
    rho : float
        Local atmospheric density [kg/m^3]. Strongly altitude- and
        solar-activity-dependent; use a clearly labeled representative
        value (this module does NOT claim mission-lifetime atmospheric
        fidelity -- see docs/momentum_accumulation_methodology.md).
    v : float
        Relative (ram) velocity magnitude [m/s].
    C_D : float
        Drag coefficient (dimensionless; ~2.0-2.5 representative for a
        convex smallsat body in free-molecular flow).
    A : float
        Projected cross-sectional area [m^2].
    r_cp : array_like, shape (3,)
        Body-frame vector from center of mass to center of pressure [m].
    u_drag : array_like, shape (3,)
        Body-frame unit vector of the drag-force direction.
    """
    _validate_positive("rho", rho)
    _validate_positive("v", v)
    _validate_positive("C_D", C_D)
    _validate_positive("A", A)
    r_cp = np.asarray(r_cp, dtype=float).reshape(3)
    u_drag = np.asarray(u_drag, dtype=float).reshape(3)
    u_drag = u_drag / np.linalg.norm(u_drag)
    F = 0.5 * rho * v**2 * C_D * A * u_drag
    return np.cross(r_cp, F)


def gravity_gradient_torque(mu: float, r: float, r_hat_b, I) -> np.ndarray:
    """Rigid-body gravity-gradient torque.

        tau_gg = 3*(mu/r^3) * (r_hat_b x (I @ r_hat_b))

    Parameters
    ----------
    mu : float
        Gravitational parameter [m^3/s^2].
    r : float
        Orbital radius [m].
    r_hat_b : array_like, shape (3,)
        Body-frame unit vector toward the local vertical (nadir/zenith
        direction, as seen in the body frame). Normalized internally.
    I : array_like, shape (3,3)
        Spacecraft inertia tensor [kg*m^2] (e.g. `Spacecraft.inertia_tensor`).

    Notes
    -----
    tau_gg = 0 exactly when r_hat_b is a principal inertia axis (I @
    r_hat_b is then parallel to r_hat_b, so the cross product vanishes) --
    verified in tests/test_disturbances.py.
    """
    _validate_positive("mu", mu)
    _validate_positive("r", r)
    r_hat_b = np.asarray(r_hat_b, dtype=float).reshape(3)
    r_hat_b = r_hat_b / np.linalg.norm(r_hat_b)
    I = np.asarray(I, dtype=float).reshape(3, 3)
    return 3.0 * (mu / r**3) * np.cross(r_hat_b, I @ r_hat_b)


def magnetic_dipole_torque(m_res, B) -> np.ndarray:
    """Residual magnetic-dipole disturbance torque: tau_m = m_res x B.

    Parameters
    ----------
    m_res : array_like, shape (3,)
        Residual spacecraft magnetic dipole moment [A*m^2].
    B : array_like, shape (3,)
        Local geomagnetic field vector, body frame [T].
    """
    m_res = np.asarray(m_res, dtype=float).reshape(3)
    B = np.asarray(B, dtype=float).reshape(3)
    return np.cross(m_res, B)


def dipole_field_body(t: float, orbit: OrbitReference, B0: float = 3.0e-5) -> np.ndarray:
    """Simplified representative body-frame geomagnetic field history.

    NOT a real IGRF/tilted-dipole propagation. Approximates the field
    direction sweeping through the body frame once per orbit (as it would
    for a roughly inertially-pointed spacecraft), with a fixed offset
    along body z set by the orbit inclination -- enough to give the field,
    and therefore the magnetic disturbance torque, a physically motivated
    orbital periodicity for M3's momentum-accumulation purposes.

    Parameters
    ----------
    t : float
        Time [s].
    orbit : OrbitReference
        Orbital timing reference (uses `mean_motion` and `inclination_rad`).
    B0 : float
        Representative LEO field magnitude [T] (~3e-5 T = 30,000 nT is a
        typical low-to-mid-latitude LEO magnitude; treated as a
        representative constant, not a location-specific value).
    """
    n = orbit.mean_motion
    tilt = np.sin(orbit.inclination_rad)
    return B0 * np.array([np.cos(n * t) * tilt, np.sin(n * t) * tilt, np.cos(orbit.inclination_rad)])
