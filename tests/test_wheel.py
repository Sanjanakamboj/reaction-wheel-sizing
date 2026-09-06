import numpy as np
import pytest

from reaction_wheel.wheel import ReactionWheel, representative_wheel
from reaction_wheel.constants import rpm_to_rad_s


def make_wheel(J_w=0.02, Omega_max=500.0, tau_max=0.2, Omega_min=0.0):
    return ReactionWheel(J_w=J_w, Omega_max=Omega_max, tau_max=tau_max, Omega_min=Omega_min)


def test_representative_wheel_valid():
    w = representative_wheel()
    assert w.J_w > 0 and w.Omega_max > 0 and w.tau_max > 0


def test_momentum_and_speed_from_momentum_are_inverse():
    w = make_wheel()
    for Omega in [-400.0, -1.0, 0.0, 1.0, 400.0]:
        H = w.momentum(Omega)
        assert w.speed_from_momentum(H) == pytest.approx(Omega, rel=1e-12)


def test_H_max_equals_Jw_times_Omega_max():
    w = make_wheel(J_w=0.02, Omega_max=500.0)
    assert w.H_max == pytest.approx(0.02 * 500.0)


def test_angular_acceleration_and_torque_from_acceleration_are_inverse():
    w = make_wheel()
    for tau in [-0.15, 0.0, 0.15]:
        Omega_dot = w.angular_acceleration(tau)
        assert w.torque_from_acceleration(Omega_dot) == pytest.approx(tau, rel=1e-12)


def test_angular_acceleration_known_value():
    w = make_wheel(J_w=0.02, tau_max=0.2)
    # tau = 0.1 N*m -> Omega_dot = 0.1/0.02 = 5 rad/s^2
    assert w.angular_acceleration(0.1) == pytest.approx(5.0)


def test_speed_rpm_roundtrip():
    w = make_wheel()
    for Omega in [0.0, 10.0, 300.0]:
        rpm = w.speed_rpm(Omega)
        assert w.speed_from_rpm(rpm) == pytest.approx(Omega, rel=1e-9)


def test_Omega_max_rpm_consistent_with_constants():
    w = make_wheel(Omega_max=rpm_to_rad_s(6000.0))
    assert w.Omega_max_rpm == pytest.approx(6000.0, rel=1e-9)


def test_torque_margin_ratio():
    w = make_wheel(tau_max=0.2)
    assert w.torque_margin_ratio(0.1) == pytest.approx(0.5)
    assert w.torque_margin_ratio(0.2) == pytest.approx(1.0)
    assert w.torque_margin_ratio(0.4) == pytest.approx(2.0)


def test_momentum_margin_ratio():
    w = make_wheel(J_w=0.02, Omega_max=500.0)  # H_max = 10
    assert w.momentum_margin_ratio(5.0) == pytest.approx(0.5)
    assert w.momentum_margin_ratio(10.0) == pytest.approx(1.0)
    assert w.momentum_margin_ratio(20.0) == pytest.approx(2.0)


def test_satisfies_torque_and_momentum():
    w = make_wheel(J_w=0.02, Omega_max=500.0, tau_max=0.2)  # H_max=10
    assert w.satisfies_torque(0.1) is True
    assert w.satisfies_torque(0.3) is False
    assert w.satisfies_momentum(5.0) is True
    assert w.satisfies_momentum(15.0) is False


@pytest.mark.parametrize("J_w,Omega_max,tau_max", [
    (0.0, 500.0, 0.2),
    (-1.0, 500.0, 0.2),
    (0.02, 0.0, 0.2),
    (0.02, -500.0, 0.2),
    (0.02, 500.0, 0.0),
    (0.02, 500.0, -0.2),
])
def test_invalid_wheel_params_raise(J_w, Omega_max, tau_max):
    with pytest.raises(ValueError):
        ReactionWheel(J_w=J_w, Omega_max=Omega_max, tau_max=tau_max)


def test_invalid_omega_min_raises():
    with pytest.raises(ValueError):
        ReactionWheel(J_w=0.02, Omega_max=500.0, tau_max=0.2, Omega_min=-1.0)
    with pytest.raises(ValueError):
        ReactionWheel(J_w=0.02, Omega_max=500.0, tau_max=0.2, Omega_min=600.0)
