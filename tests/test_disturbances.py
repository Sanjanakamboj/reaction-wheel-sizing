import numpy as np
import pytest

from reaction_wheel.disturbances import (
    zero_disturbance,
    constant_disturbance,
    sinusoidal_disturbance,
    callable_disturbance,
    CompositeDisturbance,
    srp_torque,
    aero_torque,
    gravity_gradient_torque,
    magnetic_dipole_torque,
    dipole_field_body,
    OrbitReference,
    leo_orbit_reference,
    SOLAR_PRESSURE_1AU,
    MU_EARTH,
)
from reaction_wheel.spacecraft import representative_spacecraft


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

def test_zero_disturbance():
    f = zero_disturbance()
    for t in (0.0, 10.0, 1000.0):
        assert np.allclose(f(t), [0.0, 0.0, 0.0])


def test_constant_disturbance():
    f = constant_disturbance([1.0, -2.0, 0.5])
    for t in (0.0, 5.0, 100.0):
        assert np.allclose(f(t), [1.0, -2.0, 0.5])


def test_constant_disturbance_rejects_bad_shape():
    with pytest.raises(ValueError):
        constant_disturbance([1.0, 2.0])


def test_constant_disturbance_rejects_nan():
    with pytest.raises(ValueError):
        constant_disturbance([np.nan, 0.0, 0.0])


def test_sinusoidal_disturbance_zero_mean_by_default():
    f = sinusoidal_disturbance([1.0, 0.0, 0.0], omega=0.1)
    ts = np.linspace(0, 2 * np.pi / 0.1, 500)
    vals = np.array([f(t) for t in ts])
    assert np.mean(vals[:, 0]) == pytest.approx(0.0, abs=1e-3)


def test_sinusoidal_disturbance_with_bias():
    f = sinusoidal_disturbance([1.0, 0.0, 0.0], omega=0.1, bias=[0.5, 0.0, 0.0])
    ts = np.linspace(0, 2 * np.pi / 0.1, 500)
    vals = np.array([f(t) for t in ts])
    assert np.mean(vals[:, 0]) == pytest.approx(0.5, abs=1e-3)


def test_sinusoidal_disturbance_invalid_omega():
    with pytest.raises(ValueError):
        sinusoidal_disturbance([1.0, 0.0, 0.0], omega=0.0)
    with pytest.raises(ValueError):
        sinusoidal_disturbance([1.0, 0.0, 0.0], omega=-1.0)


def test_callable_disturbance_validates_output():
    good = callable_disturbance(lambda t: [t, 0.0, 0.0])
    assert np.allclose(good(2.0), [2.0, 0.0, 0.0])

    bad = callable_disturbance(lambda t: [np.nan, 0.0, 0.0])
    with pytest.raises(ValueError):
        bad(1.0)


# ---------------------------------------------------------------------------
# Composite disturbance
# ---------------------------------------------------------------------------

def test_composite_disturbance_sums_components():
    comp = CompositeDisturbance()
    comp.add("a", constant_disturbance([1.0, 0.0, 0.0]))
    comp.add("b", constant_disturbance([0.0, 2.0, 0.0]))
    assert np.allclose(comp.total(0.0), [1.0, 2.0, 0.0])
    assert np.allclose(comp(0.0), [1.0, 2.0, 0.0])


def test_composite_disturbance_component_access():
    comp = CompositeDisturbance()
    comp.add("a", constant_disturbance([1.0, 0.0, 0.0]))
    comp.add("b", sinusoidal_disturbance([0.0, 1.0, 0.0], omega=1.0))
    assert np.allclose(comp.component("a", 5.0), [1.0, 0.0, 0.0])
    assert comp.names == ("a", "b")


def test_composite_disturbance_empty_is_zero():
    comp = CompositeDisturbance()
    assert np.allclose(comp.total(3.0), [0.0, 0.0, 0.0])


def test_composite_mean_of_constant_equals_itself():
    comp = CompositeDisturbance()
    comp.add("bias", constant_disturbance([2.0, -1.0, 0.5]))
    mean = comp.mean_total(T=100.0)
    assert np.allclose(mean, [2.0, -1.0, 0.5], atol=1e-9)


def test_composite_mean_of_zero_mean_sinusoid_is_near_zero():
    comp = CompositeDisturbance()
    omega = 2 * np.pi / 50.0
    comp.add("osc", sinusoidal_disturbance([3.0, 0.0, 0.0], omega=omega))
    mean = comp.mean_total(T=50.0, n=5000)  # exactly one period
    assert np.allclose(mean, [0.0, 0.0, 0.0], atol=1e-3)


def test_composite_peak_magnitude():
    comp = CompositeDisturbance()
    comp.add("a", constant_disturbance([3.0, 4.0, 0.0]))  # |.|=5
    peak = comp.peak_total_magnitude(T=10.0, n=10)
    assert peak == pytest.approx(5.0)


def test_composite_peak_component_magnitude():
    comp = CompositeDisturbance()
    comp.add("a", constant_disturbance([3.0, 4.0, 0.0]))  # |.|=5
    comp.add("b", constant_disturbance([0.0, 0.0, 1.0]))  # |.|=1
    assert comp.peak_component_magnitude("a", T=10.0, n=10) == pytest.approx(5.0)
    assert comp.peak_component_magnitude("b", T=10.0, n=10) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Physical disturbance models
# ---------------------------------------------------------------------------

def test_srp_torque_known_case():
    # Force along +x, cp offset along +z: tau = r_cp x F = [0,0,d] x [F,0,0] = [0, d*F, 0]
    d = 0.05
    tau = srp_torque(P_srp=SOLAR_PRESSURE_1AU, C_R=1.3, A=2.0, r_cp=[0, 0, d], u_sun=[1, 0, 0])
    F = SOLAR_PRESSURE_1AU * 1.3 * 2.0
    assert np.allclose(tau, [0.0, d * F, 0.0])


def test_srp_torque_invalid_params():
    with pytest.raises(ValueError):
        srp_torque(P_srp=-1.0, C_R=1.3, A=2.0, r_cp=[0, 0, 0.05], u_sun=[1, 0, 0])
    with pytest.raises(ValueError):
        srp_torque(P_srp=SOLAR_PRESSURE_1AU, C_R=0.0, A=2.0, r_cp=[0, 0, 0.05], u_sun=[1, 0, 0])


def test_aero_torque_known_case():
    d = 0.03
    rho, v, C_D, A = 5e-13, 7600.0, 2.2, 2.0
    tau = aero_torque(rho=rho, v=v, C_D=C_D, A=A, r_cp=[0, 0, d], u_drag=[1, 0, 0])
    F = 0.5 * rho * v**2 * C_D * A
    assert np.allclose(tau, [0.0, d * F, 0.0])


def test_aero_torque_invalid_params():
    with pytest.raises(ValueError):
        aero_torque(rho=0.0, v=7600.0, C_D=2.2, A=2.0, r_cp=[0, 0, 0.03], u_drag=[1, 0, 0])


def test_gravity_gradient_zero_when_aligned_with_principal_axis():
    sc = representative_spacecraft()
    I = sc.inertia_tensor
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        tau = gravity_gradient_torque(mu=MU_EARTH, r=7e6, r_hat_b=axis, I=I)
        assert np.allclose(tau, [0.0, 0.0, 0.0], atol=1e-12)


def test_gravity_gradient_nonzero_for_generic_direction():
    sc = representative_spacecraft()
    I = sc.inertia_tensor
    tau = gravity_gradient_torque(mu=MU_EARTH, r=7e6, r_hat_b=[1.0, 1.0, 1.0], I=I)
    assert np.linalg.norm(tau) > 0.0


def test_gravity_gradient_scales_as_inverse_r_cubed():
    sc = representative_spacecraft()
    I = sc.inertia_tensor
    r_hat = [1.0, 1.0, 0.3]
    tau1 = gravity_gradient_torque(mu=MU_EARTH, r=7e6, r_hat_b=r_hat, I=I)
    tau2 = gravity_gradient_torque(mu=MU_EARTH, r=2 * 7e6, r_hat_b=r_hat, I=I)
    ratio = np.linalg.norm(tau1) / np.linalg.norm(tau2)
    assert ratio == pytest.approx(8.0, rel=1e-9)  # (2r)^-3 scaling -> factor of 8


def test_gravity_gradient_invalid_params():
    sc = representative_spacecraft()
    with pytest.raises(ValueError):
        gravity_gradient_torque(mu=-1.0, r=7e6, r_hat_b=[1, 0, 0], I=sc.inertia_tensor)
    with pytest.raises(ValueError):
        gravity_gradient_torque(mu=MU_EARTH, r=0.0, r_hat_b=[1, 0, 0], I=sc.inertia_tensor)


def test_magnetic_dipole_torque_known_case():
    m = [1.0, 0.0, 0.0]
    B = [0.0, 0.0, 1.0]
    tau = magnetic_dipole_torque(m, B)
    assert np.allclose(tau, np.cross(m, B))
    assert np.allclose(tau, [0.0, -1.0, 0.0])


def test_magnetic_dipole_torque_zero_when_parallel():
    tau = magnetic_dipole_torque([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert np.allclose(tau, [0.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Orbit reference
# ---------------------------------------------------------------------------

def test_orbit_reference_period_known_altitude():
    orb = leo_orbit_reference(altitude_km=500.0)
    # Expected ~94.6 min for a 500 km circular LEO orbit
    assert 90.0 * 60 < orb.period_s < 100.0 * 60


def test_orbit_reference_mean_motion_consistent_with_period():
    orb = leo_orbit_reference(500.0)
    assert orb.mean_motion == pytest.approx(2 * np.pi / orb.period_s)


def test_orbit_reference_radius_consistent():
    orb = leo_orbit_reference(500.0)
    assert orb.radius_m == pytest.approx(6378137.0 + 500_000.0)


def test_orbit_reference_invalid_altitude():
    with pytest.raises(ValueError):
        OrbitReference(altitude_km=-100.0)


def test_dipole_field_body_is_finite_and_periodic():
    orb = leo_orbit_reference(500.0)
    B1 = dipole_field_body(0.0, orb)
    B2 = dipole_field_body(orb.period_s, orb)
    assert np.all(np.isfinite(B1))
    assert np.allclose(B1, B2, atol=1e-9)
