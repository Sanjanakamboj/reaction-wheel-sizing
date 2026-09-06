"""
Multi-wheel geometry, torque/momentum allocation, and capability analysis.

This module extends the single-axis wheel mechanics of `wheel.py` and the
single-axis maneuver sizing of `maneuvers.py`/`sizing.py` to an arbitrary
set of N reaction wheels with body-frame spin-axis unit vectors. It does
NOT redefine or duplicate the ideal-wheel relations (H_w = J_w*Omega_w,
etc.) — those still live in `wheel.py` and are applied per-wheel (scalar,
along each wheel's own spin axis).

Sign convention (derived here, reconciled with docs/conventions.md
section 3.3 rather than assumed):

For a single wheel i with spin-axis unit vector a_i (expressed in the
spacecraft body frame), the motor applies torque tau_w_i to the wheel
rotor about a_i. By Newton's third law the wheel's stator -- bolted to
the spacecraft structure -- feels the equal and opposite reaction
torque, -tau_w_i, directed along the SAME axis a_i (the reaction is an
internal action/reaction pair along the physical spin axis; only the
sign flips, not the axis). Summing the reaction contribution of every
wheel gives the net body-frame torque:

    tau_body = sum_i ( -tau_w_i * a_i ) = -A @ tau_w

where A = [a_1 | a_2 | ... | a_N] in R^(3xN) is the wheel-axis matrix and
tau_w in R^N is the vector of per-wheel motor torques. Setting N=1 and
A=[[1]] (or any single unit axis) recovers exactly the M1 scalar relation
tau_body = -tau_w (docs/conventions.md section 3.3). This is verified as
a regression test (test_geometry.py::test_single_wheel_matches_m1).

The identical relation holds for angular momentum:

    H_body_vec = A @ h_w

where h_w in R^N is the vector of per-wheel scalar stored momenta
(h_w_i = J_w_i * Omega_i, from wheel.py). For an isolated spacecraft +
wheel-set system with no external torque, H_body_vec + A@h_w is
conserved -- the direct 3-vector generalization of the M1 scalar
conservation law H_body + H_w = const.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

_UNIT_TOL = 1e-8


@dataclass(frozen=True)
class WheelSetGeometry:
    """A rigid set of N reaction-wheel spin axes expressed in the body frame.

    Parameters
    ----------
    axes : (3, N) array_like
        Columns are body-frame unit spin-axis vectors a_i for each wheel.
        Each column MUST already be unit-norm to within `_UNIT_TOL` --
        this class deliberately does NOT silently re-normalize
        badly-specified axes (see module docstring / M2 spec section 4).
        Construct with a badly-scaled axis and the constructor raises.
    labels : sequence of str, optional
        Human-readable wheel labels (e.g. "W1", "x-wheel"). Defaults to
        "W1".."WN".
    """

    axes: np.ndarray
    labels: tuple = None

    def __post_init__(self):
        A = np.asarray(self.axes, dtype=float)
        if A.ndim != 2 or A.shape[0] != 3:
            raise ValueError(f"axes must have shape (3, N); got {A.shape}")
        N = A.shape[1]
        if N < 3:
            raise ValueError(
                f"A wheel set needs at least 3 wheels to span 3-axis body "
                f"torque/momentum; got N={N}"
            )
        if not np.all(np.isfinite(A)):
            raise ValueError("axes must be finite; found NaN/Inf")

        norms = np.linalg.norm(A, axis=0)
        if not np.allclose(norms, 1.0, atol=_UNIT_TOL):
            bad = np.where(np.abs(norms - 1.0) > _UNIT_TOL)[0]
            raise ValueError(
                "Every wheel spin axis must be a unit vector (||a_i|| = 1). "
                f"Columns {bad.tolist()} have norms {norms[bad].tolist()}. "
                "Normalize explicitly before constructing WheelSetGeometry -- "
                "this class does not silently renormalize."
            )

        object.__setattr__(self, "axes", A)
        if self.labels is None:
            object.__setattr__(self, "labels", tuple(f"W{i+1}" for i in range(N)))
        else:
            labels = tuple(self.labels)
            if len(labels) != N:
                raise ValueError(f"labels length {len(labels)} != N wheels {N}")
            object.__setattr__(self, "labels", labels)

    # -- basic properties -----------------------------------------------

    @property
    def n_wheels(self) -> int:
        return self.axes.shape[1]

    @property
    def A(self) -> np.ndarray:
        """Alias for the (3, N) wheel-axis matrix."""
        return self.axes

    def rank(self) -> int:
        """Rank of the wheel-axis matrix A."""
        return int(np.linalg.matrix_rank(self.axes))

    def singular_values(self) -> np.ndarray:
        """Singular values of A, descending."""
        return np.linalg.svd(self.axes, compute_uv=False)

    def condition_number(self) -> float:
        """kappa(A) = sigma_max / sigma_min.

        For a rank-deficient A (sigma_min ~ 0) this returns np.inf.
        """
        sv = self.singular_values()
        sigma_max, sigma_min = sv[0], sv[-1]
        if sigma_min < 1e-12:
            return float("inf")
        return float(sigma_max / sigma_min)

    def is_full_row_rank(self) -> bool:
        """True if A can produce torque/momentum along any of the 3 body axes."""
        return self.rank() == 3

    def null_space_dimension(self) -> int:
        """dim N(A) = N - rank(A)."""
        return self.n_wheels - self.rank()

    def null_space_basis(self) -> np.ndarray:
        """Orthonormal basis for the null space of A, shape (N, dim_null).

        Columns z satisfy A @ z ~ 0: a null-space wheel-torque (or
        wheel-momentum) combination produces zero net body effect.
        """
        # Right singular vectors associated with ~zero singular values.
        _, s, Vt = np.linalg.svd(self.axes)
        tol = max(self.axes.shape) * np.finfo(float).eps * (s[0] if len(s) else 1.0)
        rank = int(np.sum(s > max(tol, 1e-10)))
        return Vt[rank:].T  # (N, N-rank)

    def pinv(self) -> np.ndarray:
        """Moore-Penrose pseudoinverse of A, shape (N, 3)."""
        return np.linalg.pinv(self.axes)

    def subset(self, keep_indices: Sequence[int], labels: Optional[Sequence[str]] = None) -> "WheelSetGeometry":
        """Return a new geometry keeping only the given wheel column indices.

        Used for single-wheel-failure analysis: `subset` excluding one
        wheel's index models that wheel having failed.
        """
        keep_indices = list(keep_indices)
        new_axes = self.axes[:, keep_indices]
        new_labels = labels if labels is not None else tuple(self.labels[i] for i in keep_indices)
        return WheelSetGeometry(axes=new_axes, labels=new_labels)

    def remove_wheel(self, index: int) -> "WheelSetGeometry":
        """Return a new geometry with wheel `index` removed (failure model)."""
        keep = [i for i in range(self.n_wheels) if i != index]
        return self.subset(keep)


# ---------------------------------------------------------------------------
# Standard geometries
# ---------------------------------------------------------------------------

def orthogonal_3wheel() -> WheelSetGeometry:
    """Baseline 3-wheel orthogonal (body-axis-aligned) geometry, A = I_3.

    Wheel 1 aligned with +x, wheel 2 with +y, wheel 3 with +z. This is the
    direct multi-wheel generalization of the M1 single-axis wheel and
    must reproduce M1's single-axis torque/momentum requirements exactly
    (see test_geometry.py::test_single_wheel_matches_m1).
    """
    return WheelSetGeometry(axes=np.eye(3), labels=("Wx", "Wy", "Wz"))


def tetrahedral_4wheel() -> WheelSetGeometry:
    """Symmetric 4-wheel tetrahedral (pyramid) redundant geometry.

    Spin axes are proportional to the four alternating-sign unit-cube
    diagonals:

        [ 1,  1,  1]
        [ 1, -1, -1]
        [-1,  1, -1]
        [-1, -1,  1]

    each normalized to unit length (each has raw norm sqrt(3)). This is a
    standard, maximally-symmetric 4-wheel arrangement: any one wheel
    removed leaves the remaining three axes spanning R^3 (verified in
    tests and in scripts/analyze_wheel_geometry.py), because no two of
    these four directions are parallel or coplanar with the third body
    axis missing. The four wheels are geometrically equivalent to one
    another under the tetrahedral symmetry group, so every
    single-wheel-failure case is equivalent up to relabeling/rotation.
    """
    raw = np.array([
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1],
    ], dtype=float)
    axes = raw / np.linalg.norm(raw, axis=0, keepdims=True)
    return WheelSetGeometry(axes=axes, labels=("W1", "W2", "W3", "W4"))


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AllocationResult:
    """Result of allocating a body-frame torque or momentum vector to wheels."""

    demand_body: np.ndarray      # the requested body-frame vector (3,)
    wheel_values: np.ndarray     # allocated per-wheel scalar values (N,), min-norm solution
    reconstruction: np.ndarray   # -A @ wheel_values (torque) or A @ wheel_values (momentum)
    residual_norm: float         # ||reconstruction - demand_body||


def allocate_torque(geometry: WheelSetGeometry, tau_body_cmd) -> AllocationResult:
    """Minimum-norm wheel-torque allocation for a desired body torque.

    Solves tau_body_cmd = -A @ tau_w for the minimum-Euclidean-norm
    tau_w, using the Moore-Penrose pseudoinverse:

        tau_w = -A^+ @ tau_body_cmd

    For a full-row-rank A this reconstructs tau_body_cmd exactly (to
    floating-point precision); for a rank-deficient A it reconstructs
    the projection of tau_body_cmd onto range(A) and `residual_norm`
    will be > 0.
    """
    tau_body_cmd = np.asarray(tau_body_cmd, dtype=float).reshape(3)
    tau_w = -geometry.pinv() @ tau_body_cmd
    recon = -geometry.A @ tau_w
    return AllocationResult(
        demand_body=tau_body_cmd, wheel_values=tau_w,
        reconstruction=recon, residual_norm=float(np.linalg.norm(recon - tau_body_cmd)),
    )


def allocate_momentum(geometry: WheelSetGeometry, delta_H_body_cmd) -> AllocationResult:
    """Minimum-norm wheel-momentum allocation for a required body momentum exchange.

    `delta_H_body_cmd` is the spacecraft BODY momentum change the maneuver
    requires (the 3-vector generalization of M1's `H_body_peak`). For the
    isolated spacecraft+wheel-set system, total momentum is conserved:

        H_body + A @ h_w = const  =>  A @ h_w = -delta_H_body_cmd

    so the minimum-norm required wheel-momentum vector is

        h_w = -A^+ @ delta_H_body_cmd

    exactly mirroring `allocate_torque`'s sign convention. Setting N=1,
    A=[[1]] recovers M1's `SlewRequirement.delta_H_wheel = -H_body_peak`
    identity exactly (see test_geometry.py::test_single_wheel_matches_m1),
    and the orthogonal 3-wheel geometry (A=I_3) reproduces the M1
    per-axis momentum requirement on the matching wheel with the matching
    sign.
    """
    delta_H_body_cmd = np.asarray(delta_H_body_cmd, dtype=float).reshape(3)
    h_w = -geometry.pinv() @ delta_H_body_cmd
    recon = -geometry.A @ h_w
    return AllocationResult(
        demand_body=delta_H_body_cmd, wheel_values=h_w,
        reconstruction=recon, residual_norm=float(np.linalg.norm(recon - delta_H_body_cmd)),
    )


def null_space_alternative_allocation(geometry: WheelSetGeometry, tau_body_cmd, z) -> np.ndarray:
    """Add a null-space combination z to the minimum-norm torque allocation.

    Demonstrates that tau_w = -A^+ @ tau_body_cmd + N @ z produces the
    SAME body torque as the pure minimum-norm solution, for any
    coefficient vector z (dim = null_space_dimension). This spare degree
    of freedom is not used for anything in M2 (no wheel-speed balancing,
    no desaturation) -- it is only demonstrated to exist.
    """
    tau_body_cmd = np.asarray(tau_body_cmd, dtype=float).reshape(3)
    N = geometry.null_space_basis()
    z = np.asarray(z, dtype=float).reshape(N.shape[1])
    base = -geometry.pinv() @ tau_body_cmd
    return base + N @ z


# ---------------------------------------------------------------------------
# Capability: per-wheel utilization and infeasibility detection
# ---------------------------------------------------------------------------

def per_wheel_torque_utilization(wheel_torques, tau_max) -> np.ndarray:
    """rho_tau_i = |tau_w_i| / tau_max_i for each wheel.

    `tau_max` may be a scalar (identical limit for all wheels) or an
    array matching `wheel_torques`.
    """
    wheel_torques = np.asarray(wheel_torques, dtype=float)
    tau_max = np.broadcast_to(np.asarray(tau_max, dtype=float), wheel_torques.shape)
    return np.abs(wheel_torques) / tau_max


def per_wheel_momentum_utilization(wheel_momenta, H_max) -> np.ndarray:
    """rho_H_i = |h_w_i| / H_max_i for each wheel."""
    wheel_momenta = np.asarray(wheel_momenta, dtype=float)
    H_max = np.broadcast_to(np.asarray(H_max, dtype=float), wheel_momenta.shape)
    return np.abs(wheel_momenta) / H_max


def is_torque_allocation_feasible(wheel_torques, tau_max) -> bool:
    """True iff every |tau_w_i| <= tau_max_i for the (min-norm) allocation.

    Per the M2 spec, an infeasible allocation must NEVER be silently
    clipped -- clipping an individual wheel torque changes the net body
    torque away from what was requested. Callers should check this
    function (or read the utilization ratios) and report infeasibility
    explicitly; see `describe_feasibility`.
    """
    return bool(np.all(per_wheel_torque_utilization(wheel_torques, tau_max) <= 1.0 + 1e-9))


def describe_feasibility(wheel_torques, tau_max) -> str:
    """Human-readable feasibility message for a min-norm torque allocation."""
    if is_torque_allocation_feasible(wheel_torques, tau_max):
        return "requested body torque is realizable under the minimum-norm allocation"
    return "requested body torque not realizable under current minimum-norm allocation/capability"


# ---------------------------------------------------------------------------
# Directional torque-capability envelope
# ---------------------------------------------------------------------------

def max_torque_along_direction(geometry: WheelSetGeometry, u_hat, tau_max: float) -> float:
    """Largest scalar lambda such that tau_c = lambda*u_hat is achievable
    with every |tau_w_i| <= tau_max, using the SAME minimum-norm allocation
    direction scaled up (allocation is linear, so scaling the body-torque
    demand by lambda scales the min-norm wheel-torque solution by the same
    lambda). This is the exact analytical solution for unconstrained
    minimum-norm allocation (M2 spec section 17):

        tau_w(u_hat) = -A^+ @ u_hat            (unit-demand allocation)
        lambda_max   = tau_max / max_i |tau_w_i(u_hat)|

    Parameters
    ----------
    geometry : WheelSetGeometry
    u_hat : array_like, shape (3,)
        Unit body-torque direction.
    tau_max : float
        Common per-wheel torque limit (N*m). (All representative wheels
        in this study share one tau_max; passing a scalar keeps the
        analytical scaling argument exact and simple.)
    """
    u_hat = np.asarray(u_hat, dtype=float).reshape(3)
    u_hat = u_hat / np.linalg.norm(u_hat)
    tau_w_unit = -geometry.pinv() @ u_hat
    peak = np.max(np.abs(tau_w_unit))
    if peak < 1e-14:
        return float("inf")
    return float(tau_max / peak)


def directional_capability_envelope(geometry: WheelSetGeometry, tau_max: float, directions) -> np.ndarray:
    """lambda_max(u_hat) evaluated over a set of unit directions.

    Parameters
    ----------
    directions : (M, 3) array_like of unit vectors.

    Returns
    -------
    (M,) array of lambda_max values.
    """
    directions = np.asarray(directions, dtype=float)
    return np.array([max_torque_along_direction(geometry, d, tau_max) for d in directions])


def fibonacci_sphere_directions(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Deterministic, roughly-uniform sample of `n` unit directions on S^2.

    Uses the Fibonacci-sphere construction (deterministic given `n`; the
    `seed` parameter is accepted for API symmetry with other sampling
    utilities but is unused -- the construction has no randomness).
    """
    i = np.arange(0, n, dtype=float)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def isotropy_ratio(geometry: WheelSetGeometry, tau_max: float, n_directions: int = 2000) -> dict:
    """Directional torque-capability isotropy metrics.

    Returns a dict with tau_min, tau_max_capability, eta (isotropy
    ratio = tau_min/tau_max_capability), and the direction achieving
    each extreme.
    """
    directions = fibonacci_sphere_directions(n_directions)
    lambdas = directional_capability_envelope(geometry, tau_max, directions)
    i_min = int(np.argmin(lambdas))
    i_max = int(np.argmax(lambdas))
    tau_min = float(lambdas[i_min])
    tau_max_cap = float(lambdas[i_max])
    return {
        "tau_min": tau_min,
        "tau_max_capability": tau_max_cap,
        "eta": tau_min / tau_max_cap if tau_max_cap > 0 else float("nan"),
        "direction_min": directions[i_min],
        "direction_max": directions[i_max],
    }
