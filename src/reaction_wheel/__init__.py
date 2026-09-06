"""
reaction_wheel — Reaction-wheel sizing and momentum-management toolkit.

Milestone 1 scope: single-axis wheel mechanics, rest-to-rest maneuver
torque/momentum sizing, and independent analytical/numerical verification.

See docs/conventions.md for the frozen sign/unit conventions used
throughout this package.
"""

from . import constants, spacecraft, wheel, maneuvers, sizing

__all__ = ["constants", "spacecraft", "wheel", "maneuvers", "sizing"]

__version__ = "0.1.0"
