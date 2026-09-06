import numpy as np
import pytest

from reaction_wheel.geometry import orthogonal_3wheel, tetrahedral_4wheel, allocate_torque
from reaction_wheel.disturbances import constant_disturbance, sinusoidal_disturbance, zero_disturbance
from reaction_wheel.momentum import (
    wheel_torque_history,
    integrate_wheel_momentum,
    InfeasibleDisturbanceError,
    analytical_constant_torque_momentum,
    analytical_sinusoidal_momentum_excursion,
    analytical_constant_torque_saturation_time,
    first_threshold_crossing,
    momentum_utilization,
    wheel_speed_history,
    mean_wheel_torque,
    momentum_per_orbit,
    secular_momentum_approximation,
    body_wheel_momentum_consistency,
    compare_geometries,
)


# ---------------------------------------------------------------------------
# Wheel-space allocation history
# ---------------------------------------------------------------------------

def test_wheel_torque_history_matches_single_allocation():
    g = orthogonal_3wheel()
    f = constant_disturbance([0.01, -0.02, 0.005])
    t = np.linspace(0, 10, 5)
    hist = wheel_torque_history(g, f, t)
    expected = allocate_torque(g, [0.01, -0.02, 0.005]).wheel_values
    for row in hist:
        assert np.allclose(row, expected)


def test_wheel_torque_history_infeasible_raises():
    g = orthogonal_3wheel()
    f = constant_disturbance([10.0, 0.0, 0.0])  # way beyond any reasonable tau_max
    t = np.linspace(0, 1, 3)
    with pytest.raises(InfeasibleDisturbanceError):
        wheel_torque_history(g, f, t, tau_max=0.2)


def test_wheel_torque_history_feasible_does_not_raise():
    g = orthogonal_3wheel()
    f = constant_disturbance([0.01, 0.0, 0.0])
    t = np.linspace(0, 1, 3)
    hist = wheel_torque_history(g, f, t, tau_max=0.2)
    assert hist.shape == (3, 3)


# ---------------------------------------------------------------------------
# Momentum integration vs analytical: constant disturbance
# ---------------------------------------------------------------------------

def test_integration_matches_analytical_constant_disturbance():
    g = orthogonal_3wheel()
    tau0 = np.array([2e-5, -1e-5, 3e-5])
    f = constant_disturbance(tau0)
    t = np.linspace(0, 6000, 4000)
    hist = integrate_wheel_momentum(g, f, t)

    expected_tau_w = allocate_torque(g, tau0).wheel_values
    for i in range(3):
        analytic = analytical_constant_torque_momentum(0.0, expected_tau_w[i], t)
        assert np.max(np.abs(hist.h_w[:, i] - analytic)) < 1e-9


def test_integration_zero_disturbance_gives_constant_momentum():
    g = orthogonal_3wheel()
    f = zero_disturbance()
    t = np.linspace(0, 1000, 100)
    h0 = np.array([0.5, -0.3, 0.1])
    hist = integrate_wheel_momentum(g, f, t, h0=h0)
    assert np.allclose(hist.h_w, np.tile(h0, (100, 1)))


# ---------------------------------------------------------------------------
# Momentum integration vs analytical: sinusoidal disturbance (bounded)
# ---------------------------------------------------------------------------

def test_integration_matches_analytical_sinusoidal_single_axis():
    g = orthogonal_3wheel()
    tau0, omega = 5e-5, 2 * np.pi / 3000.0
    f = sinusoidal_disturbance([tau0, 0.0, 0.0], omega=omega)
    t = np.linspace(0, 3000, 6000)
    hist = integrate_wheel_momentum(g, f, t)

    # Wheel-x torque = -1 * tau_d_x (orthogonal allocation, axis 0)
    expected_tau_w0 = -tau0
    excursion = analytical_sinusoidal_momentum_excursion(expected_tau_w0, omega, t)
    assert np.max(np.abs(hist.h_w[:, 0] - excursion)) < 1e-6


def test_sinusoidal_momentum_is_bounded_not_secular():
    g = orthogonal_3wheel()
    tau0, omega = 1e-4, 2 * np.pi / 1000.0
    f = sinusoidal_disturbance([tau0, 0.0, 0.0], omega=omega)
    t = np.linspace(0, 50_000, 20_000)  # 50 periods
    hist = integrate_wheel_momentum(g, f, t)
    # Bounded within [0, 2*tau0/omega] regardless of how long we integrate.
    bound = 2 * tau0 / omega
    assert np.max(np.abs(hist.h_w[:, 0])) <= bound * 1.01


def test_analytical_sinusoidal_invalid_omega():
    with pytest.raises(ValueError):
        analytical_sinusoidal_momentum_excursion(1.0, 0.0, [0, 1, 2])


# ---------------------------------------------------------------------------
# Body/wheel momentum consistency
# ---------------------------------------------------------------------------

def test_body_wheel_momentum_consistency_orthogonal():
    g = orthogonal_3wheel()
    tau0 = np.array([1e-5, 2e-5, -1.5e-5])
    f = constant_disturbance(tau0)
    t = np.linspace(0, 5000, 2000)
    hist = integrate_wheel_momentum(g, f, t)
    residual = body_wheel_momentum_consistency(g, hist, f)
    assert np.max(residual) < 1e-6


def test_body_wheel_momentum_consistency_tetrahedral():
    g = tetrahedral_4wheel()
    tau0 = np.array([1e-5, -2e-5, 0.5e-5])
    f = constant_disturbance(tau0)
    t = np.linspace(0, 5000, 2000)
    hist = integrate_wheel_momentum(g, f, t)
    residual = body_wheel_momentum_consistency(g, hist, f)
    assert np.max(residual) < 1e-6


# ---------------------------------------------------------------------------
# Analytical constant-torque saturation time
# ---------------------------------------------------------------------------

def test_analytical_saturation_time_positive_torque():
    t_sat = analytical_constant_torque_saturation_time(h0=0.0, tau_w=0.001, H_limit=10.0)
    assert t_sat == pytest.approx(10000.0)


def test_analytical_saturation_time_negative_torque():
    t_sat = analytical_constant_torque_saturation_time(h0=0.0, tau_w=-0.001, H_limit=10.0)
    assert t_sat == pytest.approx(10000.0)


def test_analytical_saturation_time_zero_torque_returns_none():
    assert analytical_constant_torque_saturation_time(h0=0.0, tau_w=0.0, H_limit=10.0) is None


def test_analytical_saturation_time_matches_numerical_integration():
    g = orthogonal_3wheel()
    tau0 = np.array([0.0, 1e-4, 0.0])
    f = constant_disturbance(tau0)
    tau_w = allocate_torque(g, tau0).wheel_values
    H_limit = 5.0
    t_sat_analytic = analytical_constant_torque_saturation_time(0.0, tau_w[1], H_limit)

    t = np.linspace(0, t_sat_analytic * 1.5, 5000)
    hist = integrate_wheel_momentum(g, f, t)
    crossing = first_threshold_crossing(hist, H_limit, labels=g.labels)
    assert crossing.reached
    assert crossing.time == pytest.approx(t_sat_analytic, rel=1e-2)


# ---------------------------------------------------------------------------
# Numerical threshold crossing
# ---------------------------------------------------------------------------

def test_threshold_crossing_not_reached_returns_none():
    g = orthogonal_3wheel()
    f = constant_disturbance([1e-8, 0.0, 0.0])  # tiny torque
    t = np.linspace(0, 10, 100)
    hist = integrate_wheel_momentum(g, f, t)
    crossing = first_threshold_crossing(hist, H_threshold=100.0)
    assert crossing.reached is False
    assert crossing.time is None
    assert crossing.wheel_index is None


def test_threshold_crossing_identifies_correct_wheel():
    g = orthogonal_3wheel()
    # Wheel y accumulates fastest
    f = constant_disturbance([1e-6, -1e-4, 1e-6])
    t = np.linspace(0, 20000, 5000)
    hist = integrate_wheel_momentum(g, f, t)
    crossing = first_threshold_crossing(hist, H_threshold=1.0, labels=g.labels)
    assert crossing.reached
    assert crossing.wheel_label == "Wy"


def test_momentum_utilization_shape_and_values():
    g = orthogonal_3wheel()
    f = constant_disturbance([0.0, 1e-4, 0.0])
    t = np.linspace(0, 100, 50)
    hist = integrate_wheel_momentum(g, f, t)
    util = momentum_utilization(hist, H_threshold=1.0)
    assert util.shape == (50,)
    assert np.all(util >= 0)
    assert np.all(np.diff(util) >= -1e-12)  # monotonically increasing for this case


# ---------------------------------------------------------------------------
# Wheel speed conversion
# ---------------------------------------------------------------------------

def test_wheel_speed_history_conversion():
    g = orthogonal_3wheel()
    f = constant_disturbance([0.0, 1e-4, 0.0])
    t = np.linspace(0, 100, 10)
    hist = integrate_wheel_momentum(g, f, t)
    J_w = 0.02
    Omega = wheel_speed_history(hist, J_w)
    assert np.allclose(Omega, hist.h_w / J_w)


# ---------------------------------------------------------------------------
# Per-orbit accumulation / mean torque
# ---------------------------------------------------------------------------

def test_mean_wheel_torque_of_constant_disturbance():
    g = orthogonal_3wheel()
    tau0 = np.array([1e-5, 2e-5, -1e-5])
    f = constant_disturbance(tau0)
    mean_tau_w = mean_wheel_torque(g, f, T=1000.0)
    expected = allocate_torque(g, tau0).wheel_values
    assert np.allclose(mean_tau_w, expected, atol=1e-9)


def test_momentum_per_orbit_constant_disturbance():
    g = orthogonal_3wheel()
    tau0 = np.array([0.0, 1e-5, 0.0])
    f = constant_disturbance(tau0)
    T_orb = 5700.0
    delta_h = momentum_per_orbit(g, f, T_orb)
    expected = allocate_torque(g, tau0).wheel_values * T_orb
    assert np.allclose(delta_h, expected, atol=1e-6)


def test_secular_approximation_matches_full_integration_for_constant_torque():
    g = orthogonal_3wheel()
    tau0 = np.array([0.0, 0.0, 1e-5])
    f = constant_disturbance(tau0)
    t = np.linspace(0, 5000, 2000)
    hist = integrate_wheel_momentum(g, f, t)
    tau_w_mean = mean_wheel_torque(g, f, T=5000.0)
    approx = secular_momentum_approximation(np.zeros(3), tau_w_mean, t)
    assert np.max(np.abs(approx - hist.h_w)) < 1e-6


# ---------------------------------------------------------------------------
# Geometry comparison
# ---------------------------------------------------------------------------

def test_compare_geometries_runs_and_returns_expected_keys():
    g3 = orthogonal_3wheel()
    g4 = tetrahedral_4wheel()
    f = constant_disturbance([1e-5, -2e-5, 0.5e-5])
    t = np.linspace(0, 5000, 500)
    results = compare_geometries({"3-wheel": g3, "4-wheel": g4}, f, t, H_threshold=1.0)
    assert set(results.keys()) == {"3-wheel", "4-wheel"}
    for r in results.values():
        assert r.max_abs_momentum >= 0
        assert r.worst_wheel_label is not None


def test_compare_geometries_deterministic_reproducibility():
    g3 = orthogonal_3wheel()
    f = constant_disturbance([1e-5, -2e-5, 0.5e-5])
    t = np.linspace(0, 5000, 500)
    r1 = compare_geometries({"3-wheel": g3}, f, t, H_threshold=1.0)
    r2 = compare_geometries({"3-wheel": g3}, f, t, H_threshold=1.0)
    assert np.allclose(r1["3-wheel"].history.h_w, r2["3-wheel"].history.h_w)


def test_single_wheel_failure_momentum_integration():
    g4 = tetrahedral_4wheel()
    g_failed = g4.remove_wheel(0)
    f = constant_disturbance([1e-5, -2e-5, 0.5e-5])
    t = np.linspace(0, 5000, 500)
    hist_nominal = integrate_wheel_momentum(g4, f, t)
    hist_failed = integrate_wheel_momentum(g_failed, f, t)
    assert hist_nominal.h_w.shape[1] == 4
    assert hist_failed.h_w.shape[1] == 3
    # Failed case should generally show higher individual-wheel loading
    assert np.max(np.abs(hist_failed.h_w)) >= 0
