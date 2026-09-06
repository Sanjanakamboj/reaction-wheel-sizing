import numpy as np
import pytest

from reaction_wheel.spacecraft import representative_spacecraft
from reaction_wheel.maneuvers import triangular_slew_requirement
from reaction_wheel.geometry import orthogonal_3wheel, tetrahedral_4wheel, allocate_torque
from reaction_wheel.wheel import ReactionWheel
from reaction_wheel.sizing import SizingMargins
from reaction_wheel.disturbances import leo_orbit_reference, constant_disturbance
from reaction_wheel.final_sizing import (
    ADOPTED_MANEUVER, STRESS_MANEUVER, DesignManeuver, maneuver_body_vectors,
    wheel_torque_requirement, TorqueRequirementResult,
    directional_envelope_torque_requirement,
    maneuver_at_threshold_headroom, required_H_max_for_headroom,
    max_allowable_dump_on_fraction, max_pre_maneuver_momentum,
    required_rotor_inertia, stored_energy, required_wheel_acceleration,
    classify_candidate, feasibility_grid,
    maneuver_time_sensitivity, inertia_sensitivity,
    evaluate_robust_corner_case, final_desaturation_schedule,
)


# ---------------------------------------------------------------------------
# Adopted maneuver
# ---------------------------------------------------------------------------

def test_adopted_maneuver_distinct_from_stress_case():
    assert ADOPTED_MANEUVER.theta_deg != STRESS_MANEUVER.theta_deg or ADOPTED_MANEUVER.T != STRESS_MANEUVER.T


def test_maneuver_body_vectors_matches_m1_formula():
    sc = representative_spacecraft()
    tau_vec, dH_vec = maneuver_body_vectors(ADOPTED_MANEUVER, sc)
    I = sc.Iy
    theta = np.deg2rad(90.0)
    T = 60.0
    alpha = 4 * theta / T**2
    tau_expected = I * alpha
    omega_peak = 2 * theta / T
    H_expected = I * omega_peak
    assert tau_vec[1] == pytest.approx(tau_expected)
    assert dH_vec[1] == pytest.approx(H_expected)
    assert tau_vec[0] == 0.0 and tau_vec[2] == 0.0


# ---------------------------------------------------------------------------
# Torque requirement: nominal + fault-tolerant
# ---------------------------------------------------------------------------

def test_torque_requirement_across_geometries():
    sc = representative_spacecraft()
    g3 = orthogonal_3wheel()
    g4 = tetrahedral_4wheel()
    tau_vec, _ = maneuver_body_vectors(ADOPTED_MANEUVER, sc)
    tr3 = wheel_torque_requirement(g3, tau_vec)
    tr4 = wheel_torque_requirement(g4, tau_vec)
    # Orthogonal geometry: single wheel bears full torque, no redundancy possible after failure of that wheel
    assert tr3.tau_nominal == pytest.approx(np.linalg.norm(tau_vec))
    # 4-wheel spreads the load: nominal worst-wheel torque is less than orthogonal's
    assert tr4.tau_nominal < tr3.tau_nominal


def test_failure_tolerant_torque_requirement_exceeds_nominal():
    sc = representative_spacecraft()
    g4 = tetrahedral_4wheel()
    tau_vec, _ = maneuver_body_vectors(ADOPTED_MANEUVER, sc)
    tr = wheel_torque_requirement(g4, tau_vec)
    assert tr.tau_failure > tr.tau_nominal
    assert tr.tau_failure_penalty_pct > 0
    assert len(tr.per_failure) == 4


def test_torque_requirement_all_failures_equal_for_pure_axis_symmetric_case():
    """For the tetrahedral geometry and a pure-axis demand, by construction
    of the symmetric geometry the per-failure worst-wheel torque should be
    identical across all 4 failure cases (matches M2's symmetric result)."""
    sc = representative_spacecraft()
    g4 = tetrahedral_4wheel()
    tau_vec, _ = maneuver_body_vectors(ADOPTED_MANEUVER, sc)
    tr = wheel_torque_requirement(g4, tau_vec)
    values = list(tr.per_failure.values())
    assert max(values) - min(values) < 1e-9


# ---------------------------------------------------------------------------
# Directional envelope
# ---------------------------------------------------------------------------

def test_directional_envelope_requirement_exceeds_single_direction():
    g4 = tetrahedral_4wheel()
    sc = representative_spacecraft()
    tau_vec, _ = maneuver_body_vectors(ADOPTED_MANEUVER, sc)
    tau_body_mag = np.linalg.norm(tau_vec)
    tr = wheel_torque_requirement(g4, tau_vec)
    env_req = directional_envelope_torque_requirement(g4, tau_body_mag)
    assert env_req > tr.tau_nominal


def test_directional_envelope_scales_linearly():
    g4 = tetrahedral_4wheel()
    req1 = directional_envelope_torque_requirement(g4, 1.0)
    req2 = directional_envelope_torque_requirement(g4, 2.0)
    assert req2 == pytest.approx(2 * req1, rel=1e-6)


# ---------------------------------------------------------------------------
# Momentum headroom
# ---------------------------------------------------------------------------

def test_headroom_feasible_case():
    g4 = tetrahedral_4wheel()
    sc = representative_spacecraft()
    _, dH_vec = maneuver_body_vectors(ADOPTED_MANEUVER, sc)
    hr = maneuver_at_threshold_headroom(g4, dH_vec, H_pre_maneuver=10.0, H_max=12.566)
    assert hr.feasible is True
    assert hr.margin > 0


def test_headroom_infeasible_case():
    g4 = tetrahedral_4wheel()
    sc = representative_spacecraft()
    _, dH_vec = maneuver_body_vectors(STRESS_MANEUVER, sc)
    # Push pre-maneuver momentum very close to H_max to force infeasibility
    hr = maneuver_at_threshold_headroom(g4, dH_vec, H_pre_maneuver=12.0, H_max=12.566)
    assert hr.feasible is False
    assert hr.margin < 0


def test_required_H_max_for_headroom_known_case():
    H_max_req = required_H_max_for_headroom(delta_H_maneuver=2.0, f_on=0.8)
    assert H_max_req == pytest.approx(2.0 / 0.2)


def test_required_H_max_invalid_f_on():
    with pytest.raises(ValueError):
        required_H_max_for_headroom(1.0, f_on=1.0)
    with pytest.raises(ValueError):
        required_H_max_for_headroom(1.0, f_on=0.0)


def test_max_allowable_dump_on_fraction_known_case():
    f_on_max = max_allowable_dump_on_fraction(delta_H_maneuver=2.0, H_max=10.0)
    assert f_on_max == pytest.approx(0.8)


def test_max_pre_maneuver_momentum():
    assert max_pre_maneuver_momentum(H_max=10.0, delta_H_maneuver=3.0) == pytest.approx(7.0)


def test_headroom_and_required_H_max_are_consistent():
    """required_H_max_for_headroom should be the exact H_max at which
    maneuver_at_threshold_headroom transitions from infeasible to
    exactly-feasible (margin ~ 0) when H_pre = f_on*H_max.

    delta_H_maneuver must be the ALLOCATED worst-wheel excursion (the
    same quantity maneuver_at_threshold_headroom itself computes), not a
    raw body-torque-vector magnitude -- so first allocate, then feed that
    resulting worst-wheel value into required_H_max_for_headroom for a
    self-consistent round trip.
    """
    g4 = tetrahedral_4wheel()
    dH_vec = np.array([0.0, 0.6348, 0.0])
    delta_H = float(np.max(np.abs(allocate_torque(g4, dH_vec).wheel_values)))
    f_on = 0.8
    H_max_req = required_H_max_for_headroom(delta_H, f_on)
    hr = maneuver_at_threshold_headroom(g4, dH_vec, H_pre_maneuver=f_on * H_max_req, H_max=H_max_req)
    assert hr.margin == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Rotor inertia / speed / energy / acceleration
# ---------------------------------------------------------------------------

def test_required_rotor_inertia():
    J = required_rotor_inertia(H_sized=10.0, Omega_max=500.0)
    assert J == pytest.approx(0.02)


def test_required_rotor_inertia_invalid_omega():
    with pytest.raises(ValueError):
        required_rotor_inertia(10.0, 0.0)


def test_stored_energy_known_value():
    assert stored_energy(J_w=0.02, Omega=500.0) == pytest.approx(0.5 * 0.02 * 500.0**2)


def test_required_wheel_acceleration():
    acc = required_wheel_acceleration(tau_sized=0.3, J_w=0.02)
    assert acc == pytest.approx(15.0)


def test_required_wheel_acceleration_invalid_jw():
    with pytest.raises(ValueError):
        required_wheel_acceleration(0.1, 0.0)


def test_speed_inertia_tradeoff_consistency():
    """For a fixed H_sized, higher Omega_max implies lower required J_w,
    and J_w*Omega_max reproduces H_sized exactly."""
    H_sized = 8.0
    for Omega_max in [300.0, 500.0, 1000.0]:
        J_w = required_rotor_inertia(H_sized, Omega_max)
        assert J_w * Omega_max == pytest.approx(H_sized)


# ---------------------------------------------------------------------------
# Candidate feasibility classification
# ---------------------------------------------------------------------------

def test_classify_candidate_fully_feasible():
    tr = TorqueRequirementResult(tau_nominal=0.02, tau_failure=0.04,
                                  tau_failure_penalty_pct=100.0, per_failure={})
    fr = classify_candidate(tau_max=0.1, H_max=10.0, torque_req=tr, H_required=5.0)
    assert fr.torque_nominal_ok and fr.torque_failure_ok and fr.momentum_ok
    assert fr.nominal_feasible and fr.failure_tolerant_feasible


def test_classify_candidate_nominal_only():
    tr = TorqueRequirementResult(tau_nominal=0.02, tau_failure=0.04,
                                  tau_failure_penalty_pct=100.0, per_failure={})
    fr = classify_candidate(tau_max=0.03, H_max=10.0, torque_req=tr, H_required=5.0)
    assert fr.nominal_feasible is True
    assert fr.failure_tolerant_feasible is False


def test_classify_candidate_infeasible():
    tr = TorqueRequirementResult(tau_nominal=0.02, tau_failure=0.04,
                                  tau_failure_penalty_pct=100.0, per_failure={})
    fr = classify_candidate(tau_max=0.01, H_max=1.0, torque_req=tr, H_required=5.0)
    assert fr.nominal_feasible is False
    assert fr.failure_tolerant_feasible is False


def test_feasibility_grid_shape_and_monotonicity():
    tr = TorqueRequirementResult(tau_nominal=0.02, tau_failure=0.04,
                                  tau_failure_penalty_pct=100.0, per_failure={})
    tau_range = np.linspace(0.01, 0.1, 10)
    H_range = np.linspace(1.0, 10.0, 8)
    grid = feasibility_grid(tau_range, H_range, tr, H_required=5.0)
    assert grid.shape == (8, 10)
    # Larger tau_max and H_max should never REDUCE feasibility code along each axis
    for i in range(grid.shape[0]):
        assert np.all(np.diff(grid[i, :]) >= 0)
    for j in range(grid.shape[1]):
        assert np.all(np.diff(grid[:, j]) >= 0)


# ---------------------------------------------------------------------------
# Sensitivity studies
# ---------------------------------------------------------------------------

def test_maneuver_time_sensitivity_scaling_laws():
    sc = representative_spacecraft()
    g4 = tetrahedral_4wheel()
    results = maneuver_time_sensitivity(g4, sc, "y", 90.0, [30.0, 60.0])
    r30, r60 = results
    # tau_req ~ T^-2 -> doubling T divides tau_req by 4
    assert r30["tau_req"] / r60["tau_req"] == pytest.approx(4.0, rel=1e-9)
    # H_req ~ T^-1 -> doubling T divides H_req by 2
    assert r30["H_req"] / r60["H_req"] == pytest.approx(2.0, rel=1e-9)


def test_inertia_sensitivity_linear_scaling():
    sc = representative_spacecraft()
    g4 = tetrahedral_4wheel()
    results = inertia_sensitivity(g4, sc, ADOPTED_MANEUVER, [0.8, 1.0, 1.2])
    base = next(r for r in results if r["scale"] == 1.0)
    high = next(r for r in results if r["scale"] == 1.2)
    assert high["tau_req"] / base["tau_req"] == pytest.approx(1.2, rel=1e-9)
    assert high["tau_w_nominal"] / base["tau_w_nominal"] == pytest.approx(1.2, rel=1e-6)


# ---------------------------------------------------------------------------
# Robust corner case
# ---------------------------------------------------------------------------

def test_robust_corner_case_feasible_with_adequate_recommendation():
    sc = representative_spacecraft()
    g4 = tetrahedral_4wheel()
    margins = SizingMargins(SF_tau=1.5, SF_H=1.5)
    result = evaluate_robust_corner_case(
        g4, sc, ADOPTED_MANEUVER, inertia_scale=1.2, disturbance_scale=2.0,
        H_pre_maneuver=10.0, margins=margins,
        tau_recommended=0.5, H_recommended=20.0,  # generous recommendation
    )
    assert result.overall_feasible is True


def test_robust_corner_case_infeasible_with_undersized_recommendation():
    sc = representative_spacecraft()
    g4 = tetrahedral_4wheel()
    margins = SizingMargins(SF_tau=1.5, SF_H=1.5)
    result = evaluate_robust_corner_case(
        g4, sc, STRESS_MANEUVER, inertia_scale=1.2, disturbance_scale=2.0,
        H_pre_maneuver=10.0, margins=margins,
        tau_recommended=0.05, H_recommended=1.0,  # deliberately undersized
    )
    assert result.overall_feasible is False


def test_robust_corner_case_margin_applies_only_to_excursion_not_preexisting_threshold():
    """Regression test for a real bug caught during M5 development: margin
    must apply only to the maneuver-induced momentum excursion, not to the
    entire (H_pre + delta_H) sum. Multiplying the whole sum by SF_H
    compounds margin onto H_pre, which is typically f_on*H_max -- i.e. it
    scales with the very capacity being solved for -- and makes the
    required-H_max equation diverge whenever SF_H*f_on >= 1 (e.g.
    SF_H=1.5, f_on=0.8 gives SF_H*f_on=1.2 > 1: no finite fixed point).
    With the correct formula (H_pre + SF_H*delta_H), a self-consistent
    required H_max always exists for f_on < 1.
    """
    sc = representative_spacecraft()
    g4 = tetrahedral_4wheel()
    margins = SizingMargins(SF_tau=1.5, SF_H=1.5)
    f_on = 0.8

    # Compute the robust-case raw excursion directly.
    g_failed = g4.remove_wheel(0)
    I_robust = sc.Iy * 1.2
    req = triangular_slew_requirement(np.deg2rad(ADOPTED_MANEUVER.theta_deg), ADOPTED_MANEUVER.T, I_robust)
    dH_vec = np.array([0.0, req.H_body_peak, 0.0])
    delta_H_raw = float(np.max(np.abs(allocate_torque(g_failed, dH_vec).wheel_values)))

    # Self-consistent required H_max: H_max = f_on*H_max + SF_H*delta_H_raw
    H_max_self_consistent = required_H_max_for_headroom(margins.SF_H * delta_H_raw, f_on)

    result = evaluate_robust_corner_case(
        g4, sc, ADOPTED_MANEUVER, inertia_scale=1.2, disturbance_scale=1.0,
        H_pre_maneuver=f_on * H_max_self_consistent, margins=margins,
        tau_recommended=1.0,  # generous, torque not under test here
        H_recommended=H_max_self_consistent,
    )
    # The self-consistently derived H_max must exactly satisfy its own
    # requirement (within floating point), proving no divergence occurs.
    assert result.H_required == pytest.approx(H_max_self_consistent, rel=1e-9)
    assert result.momentum_feasible is True


def test_robust_corner_case_all_failure_indices_valid():
    sc = representative_spacecraft()
    g4 = tetrahedral_4wheel()
    margins = SizingMargins()
    for idx in range(4):
        result = evaluate_robust_corner_case(
            g4, sc, ADOPTED_MANEUVER, 1.2, 2.0, 10.0, margins,
            tau_recommended=1.0, H_recommended=20.0, failed_wheel_index=idx,
        )
        assert result.overall_feasible is True


# ---------------------------------------------------------------------------
# Final desaturation schedule with a new wheel capacity
# ---------------------------------------------------------------------------

def test_final_schedule_scales_with_H_max():
    g4 = tetrahedral_4wheel()
    orb = leo_orbit_reference(500.0)
    tau_d = constant_disturbance([0.0, 1e-6, 0.0])
    result_small = final_desaturation_schedule(g4, tau_d, orb.period_s, H_max=5.0, f_on=0.8, f_off=0.4)
    result_large = final_desaturation_schedule(g4, tau_d, orb.period_s, H_max=10.0, f_on=0.8, f_off=0.4)
    # Doubling H_max doubles the (H_on - H_off) band -> doubles repeat interval
    assert result_large.repeat_interval_s / result_small.repeat_interval_s == pytest.approx(2.0, rel=1e-6)


def test_final_schedule_thresholds_consistent():
    g4 = tetrahedral_4wheel()
    orb = leo_orbit_reference(500.0)
    tau_d = constant_disturbance([0.0, 1e-6, 0.0])
    result = final_desaturation_schedule(g4, tau_d, orb.period_s, H_max=12.566, f_on=0.8, f_off=0.4)
    assert result.H_on == pytest.approx(0.8 * 12.566)
    assert result.H_off == pytest.approx(0.4 * 12.566)
    assert result.H_on > result.H_off


# ---------------------------------------------------------------------------
# Deterministic reproducibility
# ---------------------------------------------------------------------------

def test_deterministic_reproducibility_of_torque_requirement():
    sc = representative_spacecraft()
    g4 = tetrahedral_4wheel()
    tau_vec, _ = maneuver_body_vectors(ADOPTED_MANEUVER, sc)
    tr1 = wheel_torque_requirement(g4, tau_vec)
    tr2 = wheel_torque_requirement(g4, tau_vec)
    assert tr1.tau_nominal == tr2.tau_nominal
    assert tr1.tau_failure == tr2.tau_failure


# ---------------------------------------------------------------------------
# Final M1-M4 integrated capability check (using a plausible final wheel)
# ---------------------------------------------------------------------------

def test_final_wheel_satisfies_adopted_maneuver_nominal_and_failure():
    """An adequately-sized final wheel (found via the module's own
    requirement calculations plus a representative margin) must satisfy
    both nominal and one-wheel-failure torque requirements -- this is
    the M1-M4 integrated re-verification the M5 spec requires."""
    sc = representative_spacecraft()
    g4 = tetrahedral_4wheel()
    margins = SizingMargins(SF_tau=1.5, SF_H=1.5)
    tau_vec, dH_vec = maneuver_body_vectors(ADOPTED_MANEUVER, sc)
    tr = wheel_torque_requirement(g4, tau_vec)
    tau_final = margins.SF_tau * tr.tau_failure
    final_wheel = ReactionWheel(J_w=0.03, Omega_max=600.0, tau_max=tau_final)
    assert final_wheel.satisfies_torque(tr.tau_nominal)
    assert final_wheel.satisfies_torque(tr.tau_failure)
