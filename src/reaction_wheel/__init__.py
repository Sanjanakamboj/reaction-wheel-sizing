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
`disturbances.py` and `momentum.py`).

Milestone 4 scope: magnetorquer-based momentum dumping, hysteresis
(dump-on/dump-off) state machine, closed-loop and hybrid long-duration
desaturation-schedule simulation, and the external-unloading-vs-
internal-null-space-redistribution distinction (see `desaturation.py`).
Commercial hardware selection is NOT implemented yet.

See docs/conventions.md for the frozen sign/unit conventions used
throughout this package.
"""

from . import constants, spacecraft, wheel, maneuvers, sizing, geometry, disturbances, momentum, desaturation

__all__ = [
    "constants", "spacecraft", "wheel", "maneuvers", "sizing", "geometry",
    "disturbances", "momentum", "desaturation",
]

__version__ = "0.1.0"
