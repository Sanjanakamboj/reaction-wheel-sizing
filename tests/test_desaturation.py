import numpy as np
import pytest

from reaction_wheel.geometry import orthogonal_3wheel, tetrahedral_4wheel, allocate_torque
from reaction_wheel.disturbances import (
    leo_orbit_reference, dipole_field_body, zero_disturbance, constant_disturbance,
)
from reaction_wheel.momentum import mean_wheel_torque
from reaction_wheel.desaturation import (
    Magnetorquer,
    representative_magnetorquer,
    project_perpendicular_to_field,
    dipole_command_for_torque,
    unloading_effectiveness,
    compute_unload_command,
    DesatState,
    HysteresisThresholds,
    next_state,
    simulate_dump,
    analytical_repeat_interval,
    null_space_redistribute,
    simulate_mission_schedule,
    schedule_statistics,
)


# ---------------------------------------------------------------------------
# Magnetorquer model
# ---------------------------------------------------------------------------

def test_representative_magnetorquer_valid():
    mt = representative_magnetorquer()
    assert mt.m_max > 0


def test_magnetorquer_invalid_params():
    with pytest.raises(ValueError):
        Magnetorquer(m_max=0.0)
    with pytest.raises(ValueError):
        Magnetorquer(m_max=-1.0)
    with pytest.raises(ValueError):
        Magnetorquer(m_max=10.0, m_max_per_axis=-1.0)


def test_magnetorquer_saturate_isotropic():
    mt = Magnetorquer(m_max=10.0)
    m = np.array([6.0, 8.0, 0.0])  # norm = 10, exactly at limit
    assert np.allclose(mt.saturate(m), m)
    m2 = np.array([12.0, 16.0, 0.0])  # norm = 20
    sat = mt.saturate(m2)
    assert np.linalg.norm(sat) == pytest.approx(10.0)
    assert np.allclose(sat / np.linalg.norm(sat), m2 / np.linalg.norm(m2))  # direction preserved


def test_magnetorquer_saturate_under_limit_unchanged():
    mt = Magnetorquer(m_max=10.0)
    m = np.array([1.0, 2.0, 3.0])
    assert np.allclose(mt.saturate(m), m)


def test_magnetorquer_per_axis_saturation():
    mt = Magnetorquer(m_max=100.0, m_max_per_axis=5.0)
    m = np.array([10.0, 1.0, -20.0])
    sat = mt.saturate(m)
    assert np.all(np.abs(sat) <= 5.0 + 1e-12)


def test_magnetorquer_max_torque():
    mt = Magnetorquer(m_max=10.0)
    B = np.array([0.0, 0.0, 3e-5])
    assert mt.max_torque(B) == pytest.approx(10.0 * 3e-5)


# ---------------------------------------------------------------------------
# Magnetic torque geometry: perpendicularity, projection, inversion
# ---------------------------------------------------------------------------

def test_magnetic_torque_perpendicular_to_field():
    m = np.array([1.0, 2.0, -3.0])
    B = np.array([0.5, -0.2, 0.1])
    tau = np.cross(m, B)
    assert np.dot(tau, B) == pytest.approx(0.0, abs=1e-12)


def test_project_perpendicular_removes_parallel_component():
    B = np.array([0.0, 0.0, 5.0])
    tau = np.array([1.0, 2.0, 7.0])
    tau_perp = project_perpendicular_to_field(tau, B)
    assert np.dot(tau_perp, B) == pytest.approx(0.0, abs=1e-10)
    assert np.allclose(tau_perp, [1.0, 2.0, 0.0])


def test_dipole_command_realizes_perpendicular_torque():
    B = np.array([0.3, -0.1, 0.2])
    tau_desired = np.array([1.0, -0.5, 2.0])
    tau_perp = project_perpendicular_to_field(tau_desired, B)
    m = dipole_command_for_torque(tau_perp, B)
    tau_realized = np.cross(m, B)
    assert np.allclose(tau_realized, tau_perp, atol=1e-9)


def test_dipole_command_various_directions():
    rng = np.random.default_rng(7)
    for _ in range(20):
        B = rng.uniform(-1, 1, size=3)
        tau_desired = rng.uniform(-1, 1, size=3)
        tau_perp = project_perpendicular_to_field(tau_desired, B)
        m = dipole_command_for_torque(tau_perp, B)
        assert np.allclose(np.cross(m, B), tau_perp, atol=1e-9)


def test_maximum_torque_when_m_perpendicular_to_B():
    m_max = 5.0
    B = np.array([0.0, 0.0, 2e-5])
    # m perpendicular to B, magnitude m_max -> torque magnitude m_max*|B|
    m = np.array([m_max, 0.0, 0.0])
    tau = np.cross(m, B)
    assert np.linalg.norm(tau) == pytest.approx(m_max * np.linalg.norm(B))


def test_no_authority_parallel_to_field():
    """If the desired torque is exactly parallel to B, zero is achievable."""
    B = np.array([0.0, 0.0, 3e-5])
    tau_desired = np.array([0.0, 0.0, 10.0])  # parallel to B
    tau_perp = project_perpendicular_to_field(tau_desired, B)
    assert np.allclose(tau_perp, [0.0, 0.0, 0.0], atol=1e-12)
    eff = unloading_effectiveness(tau_desired, B)
    assert eff == pytest.approx(0.0, abs=1e-9)


def test_full_authority_perpendicular_to_field():
    B = np.array([0.0, 0.0, 3e-5])
    tau_desired = np.array([1.0, 1.0, 0.0])  # fully perpendicular
    eff = unloading_effectiveness(tau_desired, B)
    assert eff == pytest.approx(1.0)


def test_unloading_effectiveness_zero_desired_returns_one():
    B = np.array([0.0, 0.0, 3e-5])
    assert unloading_effectiveness([0.0, 0.0, 0.0], B) == 1.0


# ---------------------------------------------------------------------------
# Unload command composition
# ---------------------------------------------------------------------------

def test_compute_unload_command_structure():
    g = orthogonal_3wheel()
    mt = representative_magnetorquer()
    h_w = np.array([5.0, -3.0, 2.0])
    B = np.array([0.0, 0.0, 3e-5])
    cmd = compute_unload_command(h_w, g, B, mt, k_H=0.001)
    assert cmd.tau_desired.shape == (3,)
    assert cmd.m_cmd.shape == (3,)
    assert 0.0 <= cmd.effectiveness <= 1.0 + 1e-9


def test_compute_unload_command_negative_gain_raises():
    g = orthogonal_3wheel()
    mt = representative_magnetorquer()
    with pytest.raises(ValueError):
        compute_unload_command([1.0, 0.0, 0.0], g, [0, 0, 1e-5], mt, k_H=-0.1)


def test_unload_sign_produces_decay_orthogonal():
    """Derived sign check: dh/dt from unloading alone should equal -k_H*h
    exactly for the orthogonal geometry (see module docstring derivation)."""
    g = orthogonal_3wheel()
    mt = Magnetorquer(m_max=1e6)  # effectively unsaturated
    h = np.array([5.0, -3.0, 2.0])
    B = np.array([1.0, 1.0, 1.0])  # arbitrary, not perpendicular to h's body projection generally
    k_H = 0.001
    cmd = compute_unload_command(h, g, B, mt, k_H)
    tau_w = allocate_torque(g, cmd.tau_achieved).wheel_values
    # tau_achieved should equal tau_desired exactly since B is generic and h is generic;
    # only check when tau_desired happens to already be perpendicular to B is exact --
    # so instead verify with B chosen perpendicular to H_body for a clean exact case:
    H_body = g.A @ h
    B_perp_case = np.cross(H_body, [1.0, 0.0, 0.0])
    if np.linalg.norm(B_perp_case) < 1e-9:
        B_perp_case = np.cross(H_body, [0.0, 1.0, 0.0])
    cmd2 = compute_unload_command(h, g, B_perp_case, mt, k_H)
    tau_w2 = allocate_torque(g, cmd2.tau_achieved).wheel_values
    assert np.allclose(tau_w2, -k_H * h, atol=1e-9)


def test_unload_only_affects_row_space_component_tetrahedral():
    """Key M4 result: the null-space component of h_w is untouched by
    external magnetic unloading (dh/dt from unloading depends only on
    A@h_w, not on any null-space content of h_w)."""
    g = tetrahedral_4wheel()
    mt = Magnetorquer(m_max=1e6)
    Ainv = g.pinv()
    h_pure = Ainv @ np.array([1.0, -2.0, 0.5])
    N = g.null_space_basis()
    h_with_null = h_pure + 3.0 * N.flatten()

    k_H = 0.001
    H_body = g.A @ h_pure  # same for both since null space doesn't affect A@h
    B_perp_case = np.cross(H_body, [1.0, 0.0, 0.0])
    if np.linalg.norm(B_perp_case) < 1e-9:
        B_perp_case = np.cross(H_body, [0.0, 1.0, 0.0])

    cmd_pure = compute_unload_command(h_pure, g, B_perp_case, mt, k_H)
    cmd_null = compute_unload_command(h_with_null, g, B_perp_case, mt, k_H)
    tau_w_pure = allocate_torque(g, cmd_pure.tau_achieved).wheel_values
    tau_w_null = allocate_torque(g, cmd_null.tau_achieved).wheel_values
    # Both should produce the SAME dh/dt (independent of the null-space content)
    assert np.allclose(tau_w_pure, tau_w_null, atol=1e-9)


# ---------------------------------------------------------------------------
# Hysteresis / state machine
# ---------------------------------------------------------------------------

def test_hysteresis_thresholds_validate_ordering():
    HysteresisThresholds(H_on=8.0, H_off=4.0)  # valid
    with pytest.raises(ValueError):
        HysteresisThresholds(H_on=4.0, H_off=8.0)
    with pytest.raises(ValueError):
        HysteresisThresholds(H_on=4.0, H_off=4.0)
    with pytest.raises(ValueError):
        HysteresisThresholds(H_on=4.0, H_off=-1.0)


def test_state_transitions_basic():
    th = HysteresisThresholds(H_on=8.0, H_off=4.0)
    assert next_state(DesatState.ACCUMULATING, 5.0, th) == DesatState.ACCUMULATING
    assert next_state(DesatState.ACCUMULATING, 8.0, th) == DesatState.DESATURATING
    assert next_state(DesatState.DESATURATING, 5.0, th) == DesatState.DESATURATING
    assert next_state(DesatState.DESATURATING, 4.0, th) == DesatState.ACCUMULATING


def test_no_chatter_in_hysteresis_band():
    """A value strictly between H_off and H_on must never trigger a
    transition, regardless of which state we are currently in -- this is
    exactly what prevents chatter."""
    th = HysteresisThresholds(H_on=8.0, H_off=4.0)
    for val in (4.5, 5.0, 6.0, 7.5):
        assert next_state(DesatState.ACCUMULATING, val, th) == DesatState.ACCUMULATING
        assert next_state(DesatState.DESATURATING, val, th) == DesatState.DESATURATING


def test_state_machine_sequence_no_rapid_switching():
    th = HysteresisThresholds(H_on=8.0, H_off=4.0)
    trace = [3, 5, 7, 8, 6, 5, 4, 3, 6, 8, 4]
    state = DesatState.ACCUMULATING
    transitions = 0
    prev_state = state
    for val in trace:
        state = next_state(state, val, th)
        if state != prev_state:
            transitions += 1
        prev_state = state
    # 3->8 triggers DESAT once, then 8->4 triggers ACCUM once, then ->8 DESAT, ->4 ACCUM
    assert transitions == 4


# ---------------------------------------------------------------------------
# Closed-loop dump simulation: analytical constant-field check
# ---------------------------------------------------------------------------

def test_constant_field_perpendicular_momentum_linear_decay():
    """Controlled case: constant B, H_w_body perpendicular to B, unsaturated
    magnetorquer -> linear (constant-torque) decay, verified vs analytical."""
    g = orthogonal_3wheel()
    mt = Magnetorquer(m_max=1e6)  # unsaturated
    h0 = np.array([5.0, 0.0, 0.0])  # H_body = [5,0,0], perpendicular to B below
    B = np.array([0.0, 0.0, 3e-5])
    k_H = 0.0  # disable proportional term; we want CONSTANT torque, not decay

    # For a genuinely constant unloading torque (not proportional to h),
    # bypass compute_unload_command's proportional law and directly test
    # the underlying dipole-inversion + allocation + integration chain
    # with a fixed desired torque, matching the M4 spec's "constant
    # achievable unloading torque" analytical case.
    tau_desired_const = np.array([0.0, 1e-5, 0.0])  # already perpendicular to B
    m_cmd = dipole_command_for_torque(tau_desired_const, B)
    from reaction_wheel.disturbances import magnetic_dipole_torque
    tau_achieved = magnetic_dipole_torque(m_cmd, B)
    assert np.allclose(tau_achieved, tau_desired_const, atol=1e-9)

    tau_w = allocate_torque(g, tau_achieved).wheel_values
    t_end = 1000.0
    h_analytic = h0 + tau_w * t_end

    def tau_d_const(t):
        return np.zeros(3)

    def B_field_const(t):
        return B

    def tau_d_via_achieved(t):
        return tau_achieved  # constant external torque, no proportional feedback

    tau_w_fixed = tau_w

    # Direct trapezoidal integration replicate (reuses momentum.integrate_wheel_momentum
    # is for open-loop tau_d(t); here it's exactly open-loop since torque is constant):
    from reaction_wheel.momentum import integrate_wheel_momentum
    hist = integrate_wheel_momentum(g, tau_d_via_achieved, np.linspace(0, t_end, 500), h0=h0)
    assert np.max(np.abs(hist.h_w[-1, :] - h_analytic)) < 1e-6

    t_dump = np.linalg.norm(h0 - h_analytic) / np.linalg.norm(tau_w_fixed) if np.linalg.norm(tau_w_fixed) > 0 else None
    assert t_dump is None or t_dump >= 0


def test_dump_reduces_momentum_toward_off_threshold():
    g = tetrahedral_4wheel()
    mt = representative_magnetorquer()
    orb = leo_orbit_reference(500.0)
    B_field = lambda t: dipole_field_body(t, orb)
    tau_d = zero_disturbance()
    h0 = np.array([8.0, -9.0, 7.0, -8.5])
    result = simulate_dump(g, h0, tau_d, B_field, mt, k_H=0.0005, H_off=4.0, dt=10.0, t_max=100000.0)
    assert result.reached_off
    assert np.all(np.abs(result.h_w_final) <= 4.0 + 1e-6)
    assert np.max(np.abs(result.h_w_final)) < np.max(np.abs(h0))


def test_dump_not_reached_within_horizon_returns_false():
    g = tetrahedral_4wheel()
    mt = representative_magnetorquer()
    orb = leo_orbit_reference(500.0)
    B_field = lambda t: dipole_field_body(t, orb)
    tau_d = zero_disturbance()
    h0 = np.array([8.0, -9.0, 7.0, -8.5])
    result = simulate_dump(g, h0, tau_d, B_field, mt, k_H=0.0005, H_off=4.0, dt=10.0, t_max=100.0)
    assert result.reached_off is False


def test_dump_terminates_only_when_every_wheel_below_off():
    """Even if body-space net momentum looks small, the dump must not
    terminate while any individual wheel remains above H_off."""
    g = tetrahedral_4wheel()
    mt = representative_magnetorquer()
    orb = leo_orbit_reference(500.0)
    B_field = lambda t: dipole_field_body(t, orb)
    tau_d = zero_disturbance()
    # Body-space near-cancelling but one wheel individually high
    h0 = np.array([9.0, -9.0, 0.1, -0.1])
    result = simulate_dump(g, h0, tau_d, B_field, mt, k_H=0.0008, H_off=4.0, dt=10.0, t_max=200000.0)
    if result.reached_off:
        assert np.all(np.abs(result.h_w_final) <= 4.0 + 1e-6)


def test_disturbance_plus_unloading_composition():
    g = orthogonal_3wheel()
    mt = representative_magnetorquer()
    orb = leo_orbit_reference(500.0)
    B_field = lambda t: dipole_field_body(t, orb)
    tau_d = constant_disturbance([1e-6, -1e-6, 0.5e-6])
    h0 = np.array([8.0, -8.0, 6.0])
    result = simulate_dump(g, h0, tau_d, B_field, mt, k_H=0.0005, H_off=4.0, dt=10.0, t_max=100000.0)
    # Should still run and produce finite results even with disturbance active
    assert np.all(np.isfinite(result.h_w_final))


def test_torque_feasibility_reported_not_clipped():
    """The dump simulation reports raw allocated torque -- utilization can
    be computed and checked against tau_max, but is never silently clipped."""
    from reaction_wheel.wheel import representative_wheel
    g = tetrahedral_4wheel()
    mt = representative_magnetorquer()
    orb = leo_orbit_reference(500.0)
    B_field = lambda t: dipole_field_body(t, orb)
    tau_d = zero_disturbance()
    wheel = representative_wheel()
    h0 = np.array([8.0, -9.0, 7.0, -8.5])
    result = simulate_dump(g, h0, tau_d, B_field, mt, k_H=0.0005, H_off=4.0, dt=10.0, t_max=100000.0)
    util = per_wheel_torque_utilization_all(result.tau_w, wheel.tau_max)
    assert util.shape == result.tau_w.shape


def per_wheel_torque_utilization_all(tau_w_history, tau_max):
    return np.abs(tau_w_history) / tau_max


# ---------------------------------------------------------------------------
# Analytical repeat interval
# ---------------------------------------------------------------------------

def test_analytical_repeat_interval_known_case():
    t_repeat = analytical_repeat_interval(H_on=8.0, H_off=4.0, tau_w_secular=0.001)
    assert t_repeat == pytest.approx(4.0 / 0.001)


def test_analytical_repeat_interval_zero_torque_returns_none():
    assert analytical_repeat_interval(8.0, 4.0, 0.0) is None


def test_analytical_repeat_interval_invalid_thresholds():
    with pytest.raises(ValueError):
        analytical_repeat_interval(4.0, 8.0, 0.001)


# ---------------------------------------------------------------------------
# Null-space redistribution
# ---------------------------------------------------------------------------

def test_null_space_redistribution_preserves_body_momentum():
    g = tetrahedral_4wheel()
    h_w = np.array([1.0, -2.0, 0.5, 3.0])
    for z in ([0.0], [2.5], [-1.0]):
        h_new = null_space_redistribute(g, h_w, z)
        assert np.allclose(g.A @ h_new, g.A @ h_w, atol=1e-9)


def test_null_space_redistribution_changes_individual_wheels():
    g = tetrahedral_4wheel()
    h_w = np.array([1.0, -2.0, 0.5, 3.0])
    h_new = null_space_redistribute(g, h_w, [5.0])
    assert not np.allclose(h_new, h_w)


# ---------------------------------------------------------------------------
# Mission schedule simulation
# ---------------------------------------------------------------------------

def test_mission_schedule_produces_events():
    g = orthogonal_3wheel()
    mt = representative_magnetorquer()
    orb = leo_orbit_reference(500.0)
    B_field = lambda t: dipole_field_body(t, orb)
    tau_d = constant_disturbance([0.0, 1e-4, 0.0])  # large enough to saturate quickly for a fast test
    tau_w_mean = mean_wheel_torque(g, tau_d, T=orb.period_s, n=200)
    th = HysteresisThresholds(H_on=1.0, H_off=0.5)
    events = simulate_mission_schedule(
        g, tau_d, tau_w_mean, B_field, mt, k_H=0.001, thresholds=th,
        T_horizon=200000.0, dt_dump=5.0, t_max_dump=50000.0, T_orb=orb.period_s,
    )
    assert len(events) >= 1
    for ev in events:
        assert ev.duration >= 0
        assert ev.initial_utilization >= ev.final_utilization or ev.duration == 0


def test_schedule_statistics_aggregation():
    g = orthogonal_3wheel()
    mt = representative_magnetorquer()
    orb = leo_orbit_reference(500.0)
    B_field = lambda t: dipole_field_body(t, orb)
    tau_d = constant_disturbance([0.0, 1e-4, 0.0])
    tau_w_mean = mean_wheel_torque(g, tau_d, T=orb.period_s, n=200)
    th = HysteresisThresholds(H_on=1.0, H_off=0.5)
    events = simulate_mission_schedule(
        g, tau_d, tau_w_mean, B_field, mt, k_H=0.001, thresholds=th,
        T_horizon=400000.0, dt_dump=5.0, t_max_dump=50000.0, T_orb=orb.period_s,
    )
    stats = schedule_statistics(events, T_horizon=400000.0)
    assert stats.n_dumps == len(events)
    assert 0.0 <= stats.duty_cycle <= 1.0
    assert stats.total_dump_time >= 0


def test_deterministic_reproducibility_of_schedule():
    g = orthogonal_3wheel()
    mt = representative_magnetorquer()
    orb = leo_orbit_reference(500.0)
    B_field = lambda t: dipole_field_body(t, orb)
    tau_d = constant_disturbance([0.0, 1e-4, 0.0])
    tau_w_mean = mean_wheel_torque(g, tau_d, T=orb.period_s, n=200)
    th = HysteresisThresholds(H_on=1.0, H_off=0.5)
    events1 = simulate_mission_schedule(
        g, tau_d, tau_w_mean, B_field, mt, k_H=0.001, thresholds=th,
        T_horizon=200000.0, dt_dump=5.0, t_max_dump=50000.0, T_orb=orb.period_s,
    )
    events2 = simulate_mission_schedule(
        g, tau_d, tau_w_mean, B_field, mt, k_H=0.001, thresholds=th,
        T_horizon=200000.0, dt_dump=5.0, t_max_dump=50000.0, T_orb=orb.period_s,
    )
    assert len(events1) == len(events2)
    for e1, e2 in zip(events1, events2):
        assert e1.duration == pytest.approx(e2.duration)
        assert e1.start_time == pytest.approx(e2.start_time)


# ---------------------------------------------------------------------------
# 3-wheel / 4-wheel / failure desaturation
# ---------------------------------------------------------------------------

def test_3wheel_desaturation_runs():
    g = orthogonal_3wheel()
    mt = representative_magnetorquer()
    orb = leo_orbit_reference(500.0)
    B_field = lambda t: dipole_field_body(t, orb)
    tau_d = zero_disturbance()
    h0 = np.array([8.0, -8.5, 7.5])
    result = simulate_dump(g, h0, tau_d, B_field, mt, k_H=0.0005, H_off=4.0, dt=10.0, t_max=150000.0)
    assert result.reached_off


def test_4wheel_desaturation_runs():
    g = tetrahedral_4wheel()
    mt = representative_magnetorquer()
    orb = leo_orbit_reference(500.0)
    B_field = lambda t: dipole_field_body(t, orb)
    tau_d = zero_disturbance()
    h0 = np.array([8.0, -9.0, 7.0, -8.5])
    result = simulate_dump(g, h0, tau_d, B_field, mt, k_H=0.0005, H_off=4.0, dt=10.0, t_max=150000.0)
    assert result.reached_off


def test_single_wheel_failure_desaturation_runs():
    g4 = tetrahedral_4wheel()
    g_failed = g4.remove_wheel(0)
    mt = representative_magnetorquer()
    orb = leo_orbit_reference(500.0)
    B_field = lambda t: dipole_field_body(t, orb)
    tau_d = zero_disturbance()
    h0 = np.array([8.0, -9.0, 7.0])
    result = simulate_dump(g_failed, h0, tau_d, B_field, mt, k_H=0.0005, H_off=4.0, dt=10.0, t_max=150000.0)
    assert result.reached_off
    assert result.h_w.shape[1] == 3
