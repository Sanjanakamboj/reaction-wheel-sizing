"""
Representative rigid-spacecraft model.

The spacecraft is defined by a diagonal (principal-axis) inertia tensor.
Values are ILLUSTRATIVE ENGINEERING ASSUMPTIONS representative of a
small/medium-class spacecraft (roughly a 100-300 kg smallsat with modest
solar-array extension), NOT a specific flown vehicle. They exist to give
the Milestone-1 maneuver sizing study reasonable, order-of-magnitude
correct numbers.

See docs/conventions.md for frame/sign conventions.
"""

from dataclasses import dataclass

import numpy as np

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Spacecraft:
    """Rigid spacecraft with diagonal principal-axis inertia tensor.

    Parameters
    ----------
    Ix, Iy, Iz : float
        Principal moments of inertia about body x, y, z axes [kg*m^2].
        Must all be strictly positive (physically realizable rigid body).
    name : str
        Human-readable label for reporting.
    """

    Ix: float
    Iy: float
    Iz: float
    name: str = "Representative smallsat (illustrative)"

    def __post_init__(self):
        for axis, value in zip(AXES, (self.Ix, self.Iy, self.Iz)):
            if not np.isfinite(value):
                raise ValueError(f"Inertia I{axis} must be finite, got {value!r}")
            if value <= 0.0:
                raise ValueError(
                    f"Inertia I{axis} must be strictly positive for a physically "
                    f"realizable rigid body; got {value!r}"
                )

    @property
    def inertia_tensor(self) -> np.ndarray:
        """3x3 diagonal inertia tensor [kg*m^2]."""
        return np.diag([self.Ix, self.Iy, self.Iz])

    def inertia_about(self, axis: str) -> float:
        """Return the principal moment of inertia about a named axis ('x','y','z')."""
        axis = axis.lower()
        if axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}, got {axis!r}")
        return getattr(self, f"I{axis}")

    def is_positive_definite(self) -> bool:
        """True if the inertia tensor is symmetric positive definite."""
        I = self.inertia_tensor
        if not np.allclose(I, I.T):
            return False
        eigvals = np.linalg.eigvalsh(I)
        return bool(np.all(eigvals > 0.0))


def representative_spacecraft() -> Spacecraft:
    """Return the Milestone-1 representative spacecraft.

    Illustrative engineering assumptions, not a specific spacecraft:
    a ~180 kg smallsat-class bus with an asymmetric inertia tensor
    reflecting a body with one deployed solar-array/appendage axis
    (larger Iy) and a more compact stowed axis (Iz).
    """
    return Spacecraft(Ix=18.0, Iy=28.0, Iz=22.0)
