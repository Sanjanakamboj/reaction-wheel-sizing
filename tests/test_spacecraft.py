import numpy as np
import pytest

from reaction_wheel.spacecraft import Spacecraft, representative_spacecraft


def test_representative_spacecraft_valid():
    sc = representative_spacecraft()
    assert sc.Ix > 0 and sc.Iy > 0 and sc.Iz > 0
    assert sc.is_positive_definite()


def test_inertia_tensor_is_diagonal_and_symmetric():
    sc = Spacecraft(Ix=10.0, Iy=20.0, Iz=15.0)
    I = sc.inertia_tensor
    assert I.shape == (3, 3)
    assert np.allclose(I, np.diag([10.0, 20.0, 15.0]))
    assert np.allclose(I, I.T)


def test_inertia_about_axis():
    sc = Spacecraft(Ix=10.0, Iy=20.0, Iz=15.0)
    assert sc.inertia_about("x") == 10.0
    assert sc.inertia_about("y") == 20.0
    assert sc.inertia_about("Z") == 15.0


def test_inertia_about_invalid_axis_raises():
    sc = Spacecraft(Ix=10.0, Iy=20.0, Iz=15.0)
    with pytest.raises(ValueError):
        sc.inertia_about("w")


@pytest.mark.parametrize("Ix,Iy,Iz", [
    (0.0, 20.0, 15.0),
    (-1.0, 20.0, 15.0),
    (10.0, 0.0, 15.0),
    (10.0, 20.0, -5.0),
])
def test_invalid_inertia_raises(Ix, Iy, Iz):
    with pytest.raises(ValueError):
        Spacecraft(Ix=Ix, Iy=Iy, Iz=Iz)


def test_invalid_inertia_nan_raises():
    with pytest.raises(ValueError):
        Spacecraft(Ix=float("nan"), Iy=20.0, Iz=15.0)


def test_is_positive_definite_true_for_valid_diagonal():
    sc = Spacecraft(Ix=10.0, Iy=20.0, Iz=15.0)
    assert sc.is_positive_definite() is True
