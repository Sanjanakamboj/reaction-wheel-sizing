import numpy as np
import pytest

from reaction_wheel.sizing import (
    size_single_axis_maneuver,
    SizingMargins,
    ActiveConstraint,
    satisfies_torque,
    satisfies_momentum,
)
from reaction_wheel.wheel import ReactionWheel
from reaction_wheel.maneuvers import triangular_slew_requirement


def test_size_single_axis_maneuver_matches_analytical():
    theta_deg, T, I, J_w = 30.0, 30.0, 28.0, 0.02
    result = size_single_axis_maneuver("y", theta_deg, T, I, J_w)

    req = triangular_slew_requirement(np.deg2rad(theta_deg), T, I)
    assert result.tau_req == pytest.approx(abs(req.tau_req))
    assert result.H_req == pytest.approx(abs(req.H_body_peak))
    assert result.omega_peak == pytest.approx(req.omega_peak)


def test_wheel_speed_requirement_consistent_with_momentum_and_Jw():
    result = size_single_axis_maneuver("x", 60.0, 60.0, 18.0, J_w_ref=0.02)
    assert result.Omega_req == pytest.approx(result.H_req / 0.02)
    assert result.Omega_req_rpm == pytest.approx(result.Omega_req * 60.0 / (2 * np.pi))


def test_wheel_acceleration_requirement_integrates_to_speed_requirement():
    """Integrating Omega_dot_req over the accel half-maneuver (T/2) should
    reproduce the required wheel-speed excursion Omega_req (constant
    acceleration over the first half of a triangular-rate slew)."""
    theta_deg, T, I, J_w = 45.0, 40.0, 22.0, 0.015
    result = size_single_axis_maneuver("z", theta_deg, T, I, J_w)

    Omega_from_integration = result.Omega_dot_req * (T / 2.0)
    assert Omega_from_integration == pytest.approx(result.Omega_req, rel=1e-9)


def test_margins_applied_separately():
    margins = SizingMargins(SF_tau=1.5, SF_H=2.0)
    result = size_single_axis_maneuver("x", 30.0, 30.0, 20.0, 0.02, margins=margins)
    assert result.tau_sized == pytest.approx(1.5 * result.tau_req)
    assert result.H_sized == pytest.approx(2.0 * result.H_req)


def test_default_margins_used_when_not_specified():
    result = size_single_axis_maneuver("x", 30.0, 30.0, 20.0, 0.02)
    assert result.margins.SF_tau == pytest.approx(1.5)
    assert result.margins.SF_H == pytest.approx(1.5)


@pytest.mark.parametrize("SF_tau,SF_H", [(0.0, 1.5), (1.5, 0.0), (-1.0, 1.5)])
def test_invalid_margins_raise(SF_tau, SF_H):
    with pytest.raises(ValueError):
        SizingMargins(SF_tau=SF_tau, SF_H=SF_H)


def test_invalid_Jw_ref_raises():
    with pytest.raises(ValueError):
        size_single_axis_maneuver("x", 30.0, 30.0, 20.0, J_w_ref=0.0)
    with pytest.raises(ValueError):
        size_single_axis_maneuver("x", 30.0, 30.0, 20.0, J_w_ref=-0.02)


def test_candidate_capability_ratios_and_active_constraint():
    # Deliberately torque-limited candidate: small tau_max, generous H_max/Omega_max.
    wheel = ReactionWheel(J_w=0.02, Omega_max=1000.0, tau_max=0.01)
    result = size_single_axis_maneuver("x", 30.0, 30.0, 20.0, 0.02, candidate=wheel)

    assert result.rho_tau == pytest.approx(wheel.torque_margin_ratio(result.tau_req))
    assert result.rho_H == pytest.approx(wheel.momentum_margin_ratio(result.H_req))
    assert result.rho_Omega == pytest.approx(result.Omega_req / wheel.Omega_max)
    assert result.rho_tau > 1.0  # insufficient torque capability by construction
    assert result.active_constraint == ActiveConstraint.TORQUE


def test_no_candidate_leaves_ratios_none():
    result = size_single_axis_maneuver("x", 30.0, 30.0, 20.0, 0.02)
    assert result.candidate is None
    assert result.rho_tau is None
    assert result.rho_H is None
    assert result.rho_Omega is None
    assert result.active_constraint == ActiveConstraint.NONE


def test_satisfies_torque_and_momentum_wrappers():
    wheel = ReactionWheel(J_w=0.02, Omega_max=500.0, tau_max=0.2)
    assert satisfies_torque(0.1, wheel) is True
    assert satisfies_torque(0.3, wheel) is False
    assert satisfies_momentum(5.0, wheel) is True
    assert satisfies_momentum(50.0, wheel) is False
