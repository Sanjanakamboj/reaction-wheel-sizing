"""
reaction_wheel — Reaction-wheel sizing and momentum-management toolkit.

Milestone 1 scope: single-axis wheel mechanics, rest-to-rest maneuver
torque/momentum sizing, and independent analytical/numerical verification.

Milestone 2 scope: multi-wheel geometry, torque/momentum allocation,
worst-wheel loading, directional torque-capability envelopes, and
single-wheel-failure redundancy analysis (see `geometry.py`).

See docs/conventions.md for the frozen sign/unit conventions used
throughout this package.
"""

from . import constants, spacecraft, wheel, maneuvers, sizing, geometry

__all__ = ["constants", "spacecraft", "wheel", "maneuvers", "sizing", "geometry"]

__version__ = "0.1.0"
