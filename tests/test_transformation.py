import random
import time
from typing import Tuple

import numpy as np
import pytest

from pyraytracer.transformation import Transformation
from pyraytracer.vec3 import Vec3


@pytest.fixture(autouse=True)
def log_time(request):
    start_time = time.time()
    yield
    duration = time.time() - start_time
    test_name = request.node.name
    print(f"\nTime taken for {test_name}: {duration:.4f} seconds")


def generate_random_srt() -> Tuple[Vec3, Vec3, Vec3]:
    t_x = random.randint(-5000, 5000)
    t_y = random.randint(-5000, 5000)
    t_z = random.randint(-5000, 5000)
    translation = Vec3.from_tuple(values=(t_x, t_y, t_z))

    r_x = random.randint(-1000, 1000)
    r_y = random.randint(-1000, 1000)
    r_z = random.randint(-1000, 1000)
    rotation = Vec3.from_tuple(values=(r_x, r_y, r_z))

    s_x = random.randint(1, 100)
    s_y = random.randint(1, 100)
    s_z = random.randint(1, 100)
    scale = Vec3.from_tuple(values=(s_x, s_y, s_z))

    return translation, rotation, scale


@pytest.mark.parametrize("translation, rotation, scale", [generate_random_srt() for _ in range(500)])
def test_srt(translation: Vec3, rotation: Vec3, scale: Vec3) -> None:
    t1 = Transformation()
    t1.translation = translation
    t1.rotation = rotation
    t1.scale = scale

    assert translation == t1.translation
    assert scale == t1.scale

    res_rotation = t1.rotation
    t2 = Transformation()
    t2.translation = translation
    t2.scale = scale
    t2.rotation = res_rotation
    assert np.allclose(t1.matrix, t2.matrix, atol=1e-4), f'{t1.matrix}\n{t2.matrix}'


@pytest.mark.parametrize("translation, rotation, scale", [generate_random_srt() for _ in range(500)])
def test_from_srt(translation: Vec3, rotation: Vec3, scale: Vec3) -> None:
    t1 = Transformation()
    t1.translation = translation
    t1.rotation = rotation
    t1.scale = scale

    t2 = Transformation.from_srt(srt=(scale, rotation, translation))

    assert t1 == t2


@pytest.mark.parametrize("translation, rotation, scale", [generate_random_srt() for _ in range(500)])
def test_to_srt(translation: Vec3, rotation: Vec3, scale: Vec3) -> None:
    t1 = Transformation()
    t1.translation = translation
    t1.rotation = rotation
    t1.scale = scale

    r_scale, r_rotation, r_translation = t1.to_srt()

    assert r_scale == scale
    assert r_translation == translation

    t2 = Transformation()
    t2.translation = r_translation
    t2.rotation = r_rotation
    t2.scale = r_scale

    assert t1 == t2


def test_local_to_global() -> None:
    local_point = Vec3.from_tuple(values=(0, 1, 0))
    local_point_xfo = Transformation.from_srt(
        srt=(
            Vec3.from_tuple(values=(1, 1, 1)),
            Vec3.from_tuple(values=(0, 0, 0)),
            local_point,
        ),
    )

    xfo = Transformation.from_srt(
        srt=(
            Vec3.from_tuple(values=(1, 1, 1)),
            Vec3.from_tuple(values=(0, 0, 0)),
            Vec3.from_tuple(values=(0, -1, 0)),
        ),
    )

    global_point_xfo = xfo @ local_point_xfo
    expected_global_point = Vec3.from_tuple(values=(0, 0, 0))
    assert global_point_xfo.translation == expected_global_point


def test_global_to_local() -> None:
    global_point = Vec3.from_tuple(values=(0, 1, 0))
    global_point_xfo = Transformation.from_srt(
        srt=(
            Vec3.from_tuple(values=(1, 1, 1)),
            Vec3.from_tuple(values=(0, 0, 0)),
            global_point,
        ),
    )

    xfo = Transformation.from_srt(
        srt=(
            Vec3.from_tuple(values=(1, 1, 1)),
            Vec3.from_tuple(values=(0, 0, 0)),
            Vec3.from_tuple(values=(0, -1, 0)),
        ),
    )

    local_point_xfo = global_point_xfo @ xfo.inverse
    expected_global_point = Vec3.from_tuple(values=(0, 2, 0))
    assert local_point_xfo.translation == expected_global_point
