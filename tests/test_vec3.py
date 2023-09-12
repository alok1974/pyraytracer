import random

import numpy as np

from pyraytracer.vec3 import Vec3


def test_to_np_array() -> None:
    x = random.random()
    y = random.random()
    z = random.random()

    v = Vec3(x=x, y=y, z=z)
    np_arr = v.to_np_array()
    expected_arr = np.array([x, y, z])

    assert np.allclose(np_arr, expected_arr, atol=1e-4)


def test_from_np_array() -> None:
    x = random.random()
    y = random.random()
    z = random.random()

    np_arr = np.array([x, y, z])
    v = Vec3.from_np_array(arr=np_arr)

    expected_v = Vec3.from_tuple(values=(x, y, z))
    assert expected_v == v
