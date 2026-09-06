import numpy as np
import pytest

from reaction_wheel.geometry import (
    WheelSetGeometry,
    orthogonal_3wheel,
    tetrahedral_4wheel,
    allocate_torque,
    allocate_momentum,
    null_space_alternative_allocation,
    per_wheel_torque_utilization,
    per_wheel_momentum_utilization,
    is_torque_allocation_feasible,
    describe_feasibility,
    max_torque_along_direction,
    directional_capability_envelope,
    fibonacci_sphere_directions,
    isotropy_ratio,
)
from reaction_wheel.maneuvers import triangular_slew_requirement
from reaction_wheel.spacecraft import representative_spacecraft
from reaction_wheel.wheel import representative_wheel


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------

def test_orthogonal_geometry_valid():
    g = orthogonal_3wheel()
    assert g.n_wheels == 3
    assert np.allclose(g.A, np.eye(3))
    assert g.rank() == 3


def test_tetrahedral_geometry_valid_and_unit_norm():
    g = tetrahedral_4wheel()
    assert g.n_wheels == 4
    norms = np.linalg.norm(g.A, axis=0)
    assert np.allclose(norms, 1.0)
    assert g.rank() == 3


def test_rejects_non_unit_axes():
    bad_axes = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]).T
    # column 2 has norm 2, not unit
    with pytest.raises(ValueError):
        WheelSetGeometry(axes=bad_axes)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        WheelSetGeometry(axes=np.eye(2))


def test_rejects_too_few_wheels():
    axes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        WheelSetGeometry(axes=axes)


def test_rejects_nan_axes():
    axes = np.eye(3)
    axes[0, 0] = np.nan
    with pytest.raises(ValueError):
        WheelSetGeometry(axes=axes)


def test_labels_default_and_custom():
    g = orthogonal_3wheel()
    assert g.labels == ("Wx", "Wy", "Wz")
    g2 = WheelSetGeometry(axes=np.eye(3), labels=("a", "b", "c"))
    assert g2.labels == ("a", "b", "c")


def test_labels_length_mismatch_raises():
    with pytest.raises(ValueError):
        WheelSetGeometry(axes=np.eye(3), labels=("only_one",))


# ---------------------------------------------------------------------------
# Rank / condition number / null space
# ---------------------------------------------------------------------------

def test_orthogonal_condition_number_is_one():
    g = orthogonal_3wheel()
    assert g.condition_number() == pytest.approx(1.0, rel=1e-9)


def test_tetrahedral_condition_number_finite_and_symmetric():
    g = tetrahedral_4wheel()
    kappa = g.condition_number()
    assert np.isfinite(kappa)
    assert kappa > 0


def test_tetrahedral_null_space_dimension_is_one():
    g = tetrahedral_4wheel()
    assert g.null_space_dimension() == 1


def test_orthogonal_null_space_dimension_is_zero():
    g = orthogonal_3wheel()
    assert g.null_space_dimension() == 0


def test_null_space_basis_maps_to_near_zero_torque():
    g = tetrahedral_4wheel()
    N = g.null_space_basis()
    assert N.shape == (4, 1)
    assert np.allclose(g.A @ N, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Single-wheel-failure rank preservation (tetrahedral)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("failed_idx", [0, 1, 2, 3])
def test_tetrahedral_single_failure_preserves_full_rank(failed_idx):
    g = tetrahedral_4wheel()
    g_failed = g.remove_wheel(failed_idx)
    assert g_failed.n_wheels == 3
    assert g_failed.rank() == 3
    assert g_failed.is_full_row_rank()


# ---------------------------------------------------------------------------
# Torque allocation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau_c", [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.5, -0.3, 0.2],
    [-1.0, 0.4, -0.7],
])
def test_orthogonal_torque_allocation_reconstructs_exactly(tau_c):
    g = orthogonal_3wheel()
    res = allocate_torque(g, tau_c)
    assert res.residual_norm < 1e-10
    assert np.allclose(res.reconstruction, tau_c, atol=1e-10)


@pytest.mark.parametrize("tau_c", [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.5, -0.3, 0.2],
])
def test_tetrahedral_torque_allocation_reconstructs_exactly(tau_c):
    g = tetrahedral_4wheel()
    res = allocate_torque(g, tau_c)
    assert res.residual_norm < 1e-10
    assert np.allclose(res.reconstruction, tau_c, atol=1e-10)


def test_random_seeded_directions_reconstruct():
    rng = np.random.default_rng(42)
    g = tetrahedral_4wheel()
    for _ in range(20):
        tau_c = rng.uniform(-1, 1, size=3)
        res = allocate_torque(g, tau_c)
        assert res.residual_norm < 1e-9


def test_orthogonal_pure_axis_activates_single_wheel():
    g = orthogonal_3wheel()
    res = allocate_torque(g, [0.5, 0.0, 0.0])
    assert res.wheel_values[0] == pytest.approx(-0.5)
    assert res.wheel_values[1] == pytest.approx(0.0, abs=1e-12)
    assert res.wheel_values[2] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Momentum allocation
# ---------------------------------------------------------------------------

def test_momentum_allocation_reconstructs():
    g = tetrahedral_4wheel()
    dH = [0.3, -0.2, 0.5]
    res = allocate_momentum(g, dH)
    assert res.residual_norm < 1e-9
    assert np.allclose(res.reconstruction, dH, atol=1e-9)


def test_orthogonal_momentum_matches_single_axis():
    g = orthogonal_3wheel()
    res = allocate_momentum(g, [0.0, 0.4, 0.0])
    assert res.wheel_values[1] == pytest.approx(-0.4)
    assert res.wheel_values[0] == pytest.approx(0.0, abs=1e-12)
    assert res.wheel_values[2] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# M1 regression: single-wheel / orthogonal geometry must reproduce M1 exactly
# ---------------------------------------------------------------------------

def test_single_wheel_matches_m1_torque_and_momentum_sign():
    """A wheel-set geometry needs >=3 wheels to be constructible (validated
    above), so the N=1 M1 scalar case is reproduced via the orthogonal
    3-wheel geometry's x-wheel column in isolation: for a pure-x demand,
    only Wx is active and must reproduce M1's scalar
    tau_body = -tau_w and Delta H_w = -Delta H_body exactly."""
    g = orthogonal_3wheel()
    tau_c = [0.05, 0.0, 0.0]
    res = allocate_torque(g, tau_c)
    assert res.wheel_values[0] == pytest.approx(-0.05)
    assert np.allclose(res.wheel_values[1:], 0.0, atol=1e-12)

    dH = [0.4, 0.0, 0.0]
    res_H = allocate_momentum(g, dH)
    assert res_H.wheel_values[0] == pytest.approx(-0.4)
    assert np.allclose(res_H.wheel_values[1:], 0.0, atol=1e-12)


def test_orthogonal_reproduces_m1_maneuver_requirement_exactly():
    sc = representative_spacecraft()
    g = orthogonal_3wheel()
    theta_deg, T = 45.0, 10.0  # M1's aggressive case
    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        I = sc.inertia_about(axis_name)
        req = triangular_slew_requirement(np.deg2rad(theta_deg), T, I)

        tau_c = np.zeros(3)
        tau_c[axis_idx] = req.tau_req
        res_tau = allocate_torque(g, tau_c)
        assert abs(res_tau.wheel_values[axis_idx]) == pytest.approx(req.tau_req)
        # all other wheels get zero torque for a pure principal-axis demand
        others = [i for i in range(3) if i != axis_idx]
        assert np.allclose(res_tau.wheel_values[others], 0.0, atol=1e-10)

        dH = np.zeros(3)
        dH[axis_idx] = req.H_body_peak
        res_H = allocate_momentum(g, dH)
        assert abs(res_H.wheel_values[axis_idx]) == pytest.approx(req.H_body_peak)


# ---------------------------------------------------------------------------
# Null-space invariance
# ---------------------------------------------------------------------------

def test_null_space_alternative_produces_same_body_torque():
    g = tetrahedral_4wheel()
    tau_c = [0.3, -0.1, 0.2]
    for z in ([0.0], [1.0], [-2.5], [10.0]):
        tau_w_alt = null_space_alternative_allocation(g, tau_c, z)
        recon = -g.A @ tau_w_alt
        assert np.allclose(recon, tau_c, atol=1e-9)


# ---------------------------------------------------------------------------
# Utilization
# ---------------------------------------------------------------------------

def test_per_wheel_torque_utilization():
    ratios = per_wheel_torque_utilization([0.1, -0.2, 0.05], 0.2)
    assert np.allclose(ratios, [0.5, 1.0, 0.25])


def test_per_wheel_momentum_utilization():
    ratios = per_wheel_momentum_utilization([1.0, -2.0], 4.0)
    assert np.allclose(ratios, [0.25, 0.5])


def test_feasibility_detection():
    assert is_torque_allocation_feasible([0.1, 0.15], 0.2) is True
    assert is_torque_allocation_feasible([0.1, 0.25], 0.2) is False
    assert "not realizable" in describe_feasibility([0.1, 0.25], 0.2)
    assert "realizable" in describe_feasibility([0.1, 0.15], 0.2)


def test_wheel_speed_conversion_after_allocation():
    wheel = representative_wheel()
    g = orthogonal_3wheel()
    res = allocate_momentum(g, [0.0, 1.0, 0.0])
    Omega = wheel.speed_from_momentum(res.wheel_values[1])
    assert Omega == pytest.approx(res.wheel_values[1] / wheel.J_w)


# ---------------------------------------------------------------------------
# Directional capability / isotropy
# ---------------------------------------------------------------------------

def test_max_torque_along_direction_orthogonal_pure_axis():
    g = orthogonal_3wheel()
    tau_max = 0.2
    lam = max_torque_along_direction(g, [1.0, 0.0, 0.0], tau_max)
    assert lam == pytest.approx(tau_max)


def test_max_torque_along_direction_scales_linearly():
    g = tetrahedral_4wheel()
    tau_max = 0.2
    u = np.array([1.0, 0.5, -0.3])
    u_hat = u / np.linalg.norm(u)
    lam = max_torque_along_direction(g, u_hat, tau_max)
    # allocated wheel torques for lam*u_hat must be within tau_max, at least one active
    res = allocate_torque(g, lam * u_hat)
    assert np.max(np.abs(res.wheel_values)) == pytest.approx(tau_max, rel=1e-6)


def test_fibonacci_sphere_directions_are_unit_and_deterministic():
    d1 = fibonacci_sphere_directions(50)
    d2 = fibonacci_sphere_directions(50)
    assert np.allclose(d1, d2)
    norms = np.linalg.norm(d1, axis=1)
    assert np.allclose(norms, 1.0)


def test_directional_capability_envelope_shape():
    g = orthogonal_3wheel()
    dirs = fibonacci_sphere_directions(20)
    env = directional_capability_envelope(g, 0.2, dirs)
    assert env.shape == (20,)
    assert np.all(env > 0)


def test_isotropy_ratio_orthogonal_less_isotropic_than_tetrahedral_or_reported():
    g_orth = orthogonal_3wheel()
    g_tetra = tetrahedral_4wheel()
    iso_orth = isotropy_ratio(g_orth, 0.2, n_directions=500)
    iso_tetra = isotropy_ratio(g_tetra, 0.2, n_directions=500)
    assert 0.0 < iso_orth["eta"] <= 1.0
    assert 0.0 < iso_tetra["eta"] <= 1.0


# ---------------------------------------------------------------------------
# Failure-case capability degradation
# ---------------------------------------------------------------------------

def test_failed_wheel_capability_less_than_or_equal_nominal():
    g = tetrahedral_4wheel()
    tau_max = 0.2
    nominal = isotropy_ratio(g, tau_max, n_directions=500)
    for idx in range(4):
        g_failed = g.remove_wheel(idx)
        failed = isotropy_ratio(g_failed, tau_max, n_directions=500)
        assert failed["tau_min"] <= nominal["tau_min"] + 1e-9


def test_all_tetrahedral_failure_cases_geometrically_equivalent():
    """By tetrahedral symmetry, min-capability after any single-wheel
    failure should be equal across all 4 failure cases (within numerical
    sampling tolerance)."""
    g = tetrahedral_4wheel()
    tau_max = 0.2
    mins = []
    for idx in range(4):
        g_failed = g.remove_wheel(idx)
        result = isotropy_ratio(g_failed, tau_max, n_directions=1000)
        mins.append(result["tau_min"])
    mins = np.array(mins)
    assert np.max(mins) - np.min(mins) < 1e-3 * np.mean(mins)


def test_subset_and_remove_wheel_consistency():
    g = tetrahedral_4wheel()
    removed = g.remove_wheel(1)
    kept_indices = [0, 2, 3]
    subset = g.subset(kept_indices)
    assert np.allclose(removed.A, subset.A)
    assert removed.labels == subset.labels
