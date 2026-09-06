"""
reaction_wheel — Reaction-wheel sizing and momentum-management toolkit.

Milestone 1 scope: single-axis wheel mechanics, rest-to-rest maneuver
torque/momentum sizing, and independent analytical/numerical verification.

Milestone 2 scope: multi-wheel geometry, torque/momentum allocation,
worst-wheel loading, directional torque-capability envelopes, and
single-wheel-failure redundancy analysis (see `geometry.py`).

Milestone 3 scope: environmental disturbance torque models, wheel-space
disturbance allocation, wheel momentum accumulation, operational
momentum thresholds, and saturation/desaturation-time analysis (see
`disturbances.py` and `momentum.py`). Momentum dumping/desaturation
control is NOT implemented yet.

See docs/conventions.md for the frozen sign/unit conventions used
throughout this package.
"""

from . import constants, spacecraft, wheel, maneuvers, sizing, geometry, disturbances, momentum

__all__ = [
    "constants", "spacecraft", "wheel", "maneuvers", "sizing", "geometry",
    "disturbances", "momentum",
]

__version__ = "0.1.0"
