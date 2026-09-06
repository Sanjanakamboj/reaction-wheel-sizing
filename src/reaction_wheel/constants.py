"""
Physical constants and unit-conversion factors.

All internal computation uses SI units (kg, m, s, rad, rad/s, N*m, N*m*s).
Conversions to "display" units (degrees, rpm) live here as pure, tested
functions so no conversion factor is ever hand-typed elsewhere.
"""

import numpy as np

TWO_PI = 2.0 * np.pi

# ---------------------------------------------------------------------------
# Angle conversions
# ---------------------------------------------------------------------------

def deg2rad(angle_deg):
    """Convert angle from degrees to radians."""
    return np.deg2rad(angle_deg)


def rad2deg(angle_rad):
    """Convert angle from radians to degrees."""
    return np.rad2deg(angle_rad)


# ---------------------------------------------------------------------------
# Angular rate conversions
# ---------------------------------------------------------------------------

def rpm_to_rad_s(rpm):
    """Convert rotational speed from rpm to rad/s.

    Omega [rad/s] = rpm * (2*pi rad / rev) * (1 rev/min) * (1 min / 60 s)
    """
    return rpm * TWO_PI / 60.0


def rad_s_to_rpm(omega_rad_s):
    """Convert rotational speed from rad/s to rpm."""
    return omega_rad_s * 60.0 / TWO_PI
