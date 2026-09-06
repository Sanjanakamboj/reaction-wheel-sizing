import numpy as np
import pytest

from reaction_wheel.maneuvers import (
    angular_acceleration_from_torque,
    torque_from_angular_acceleration,
    angular_momentum,
    kinetic_energy,
    triangular_slew_requirement,
    propagate_rest_to_rest_slew,
)


# ---------------------------------------------------------------------------
# Basic rigid-body relations
# ---------------------------------------------------------------------------

def test_torque_angular_acceleration_are_inverse():
    I = 22.0
    for alpha in [-0.05, 0.0, 0.05]:
        tau = torque_from_angular_acceleration(alpha, I)
        assert angular_acceleration_from_torque(tau, I) == pytest.approx(alpha, rel=1e-12)


def test_angular_momentum_known_value():
    assert angular_momentum(I=20.0, omega=0.1) == pytest.approx(2.0)


def test_kinetic_energy_known_value():
    assert kinetic_energy(I=20.0, omega=1.0) == pytest.approx(10.0)


@pytest.mark.parametrize("I", [0.0, -5.0])
def test_invalid_inertia_raises(I):
    with pytest.raises(ValueError):
        angular_acceleration_from_torque(1.0, I)
    with pytest.raises(ValueError):
        torque_from_angular_acceleration(1.0, I)


# ---------------------------------------------------------------------------
# Triangular-rate rest-to-rest slew — analytical formulas
# ---------------------------------------------------------------------------

def test_triangular_slew_known_case():
    # theta = 0.1 rad, T = 10 s, I = 20 kg*m^2
    theta, T, I = 0.1, 10.0, 20.0
    req = triangular_slew_requirement(theta, T, I)

    alpha_expected = 4 * theta / T**2         # 0.004 rad/s^2
    tau_expected = I * alpha_expected          # 0.08 N*m
    omega_peak_expected = 2 * theta / T        # 0.02 rad/s
    H_peak_expected = I * omega_peak_expected  # 0.4 N*m*s

    assert req.alpha == pytest.approx(alpha_expected)
    assert req.tau_req == pytest.approx(tau_expected)
    assert req.omega_peak == pytest.approx(omega_peak_expected)
    assert req.H_body_peak == pytest.approx(H_peak_expected)


def test_triangular_slew_theta_reconstructed_from_alpha_and_T():
    theta, T, I = 0.3, 25.0, 18.0
    req = triangular_slew_requirement(theta, T, I)
    theta_reconstructed = req.alpha * (T / 2.0) ** 2
    assert theta_reconstructed == pytest.approx(theta, rel=1e-12)


def test_wheel_torque_and_momentum_are_sign_flipped():
    req = triangular_slew_requirement(0.2, 20.0, 18.0)
    assert req.wheel_torque_cmd == pytest.approx(-req.tau_req)
    assert req.delta_H_wheel == pytest.approx(-req.H_body_peak)


@pytest.mark.parametrize("theta,T,I", [
    (0.0, 10.0, 20.0),
    (-0.1, 10.0, 20.0),
    (0.1, 0.0, 20.0),
    (0.1, -10.0, 20.0),
    (0.1, 10.0, 0.0),
    (0.1, 10.0, -20.0),
])
def test_invalid_slew_inputs_raise(theta, T, I):
    with pytest.raises(ValueError):
        triangular_slew_requirement(theta, T, I)


# ---------------------------------------------------------------------------
# Scaling laws: tau_req ~ theta/T^2 ; H_req ~ theta/T
# ---------------------------------------------------------------------------

def test_angle_doubling_at_fixed_time():
    I, T = 20.0, 30.0
    base = triangular_slew_requirement(0.2, T, I)
    doubled = triangular_slew_requirement(0.4, T, I)
    assert doubled.tau_req == pytest.approx(2 * base.tau_req, rel=1e-12)
    assert doubled.H_body_peak == pytest.approx(2 * base.H_body_peak, rel=1e-12)


def test_time_doubling_at_fixed_angle():
    I, theta = 20.0, 0.3
    base = triangular_slew_requirement(theta, 20.0, I)
    doubled = triangular_slew_requirement(theta, 40.0, I)
    # tau scales as 1/T^2 -> quarter; H scales as 1/T -> half
    assert doubled.tau_req == pytest.approx(0.25 * base.tau_req, rel=1e-9)
    assert doubled.H_body_peak == pytest.approx(0.5 * base.H_body_peak, rel=1e-9)


def test_tau_scaling_law_general_ratio():
    I = 22.0
    theta0, T0 = 0.2, 25.0
    for k_theta, k_T in [(1.5, 1.0), (1.0, 2.0), (3.0, 2.0)]:
        r0 = triangular_slew_requirement(theta0, T0, I)
        r1 = triangular_slew_requirement(theta0 * k_theta, T0 * k_T, I)
        expected_tau_ratio = k_theta / k_T**2
        expected_H_ratio = k_theta / k_T
        assert r1.tau_req / r0.tau_req == pytest.approx(expected_tau_ratio, rel=1e-9)
        assert r1.H_body_peak / r0.H_body_peak == pytest.approx(expected_H_ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# Numerical propagation vs analytical solution
# ---------------------------------------------------------------------------

def test_numerical_matches_analytical_final_state():
    theta, T, I = np.deg2rad(30.0), 30.0, 28.0
    hist = propagate_rest_to_rest_slew(theta, T, I, n_points=2000)

    assert hist.theta[-1] == pytest.approx(theta, rel=1e-6)
    assert hist.omega[-1] == pytest.approx(0.0, abs=1e-8)


def test_numerical_matches_analytical_peak_rate_and_momentum():
    theta, T, I = np.deg2rad(60.0), 60.0, 22.0
    req = triangular_slew_requirement(theta, T, I)
    hist = propagate_rest_to_rest_slew(theta, T, I, n_points=4000)

    # Peak |omega| occurs near t = T/2 (midpoint of sampled history).
    mid_idx = np.argmin(np.abs(hist.t - T / 2.0))
    assert abs(hist.omega[mid_idx]) == pytest.approx(req.omega_peak, rel=2e-3)

    peak_H_wheel = np.max(np.abs(hist.H_wheel))
    assert peak_H_wheel == pytest.approx(req.H_body_peak, rel=2e-3)


def test_numerical_isolated_system_momentum_conservation():
    theta, T, I = np.deg2rad(45.0), 40.0, 18.0
    hist = propagate_rest_to_rest_slew(theta, T, I, n_points=3000)

    H_body = I * hist.omega
    H_total = H_body + hist.H_wheel
    # Total angular momentum must stay ~constant (starts at 0, isolated system).
    assert np.max(np.abs(H_total)) < 1e-8


def test_final_wheel_momentum_returns_to_initial_for_isolated_rest_to_rest():
    theta, T, I = np.deg2rad(20.0), 20.0, 20.0
    hist = propagate_rest_to_rest_slew(theta, T, I, n_points=2000)
    # Wheel momentum must return to (near) zero since body ends at rest
    # and total momentum is conserved at zero.
    assert hist.H_wheel[-1] == pytest.approx(0.0, abs=1e-8)
