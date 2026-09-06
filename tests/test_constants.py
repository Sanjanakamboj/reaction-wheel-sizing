import numpy as np
import pytest

from reaction_wheel.constants import rpm_to_rad_s, rad_s_to_rpm, deg2rad, rad2deg


def test_rpm_to_rad_s_known_value():
    # 60 rpm = 1 rev/s = 2*pi rad/s
    assert rpm_to_rad_s(60.0) == pytest.approx(2 * np.pi, rel=1e-12)


def test_rad_s_to_rpm_known_value():
    assert rad_s_to_rpm(2 * np.pi) == pytest.approx(60.0, rel=1e-12)


def test_rpm_rad_s_roundtrip():
    for rpm in [0.0, 1.0, 500.0, 6000.0, -3000.0]:
        assert rad_s_to_rpm(rpm_to_rad_s(rpm)) == pytest.approx(rpm, rel=1e-12)


def test_deg_rad_roundtrip():
    for deg in [0.0, 10.0, 90.0, 180.0, -45.0]:
        assert rad2deg(deg2rad(deg)) == pytest.approx(deg, rel=1e-12)


def test_deg2rad_known_value():
    assert deg2rad(180.0) == pytest.approx(np.pi, rel=1e-12)
