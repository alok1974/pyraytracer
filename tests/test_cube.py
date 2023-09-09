from pyraytracer.cube import Cube
from pyraytracer.vec3 import Vec3


def test_cube_hit_data() -> None:
    cube = Cube(
        name='cube',
        center=Vec3.from_tuple(values=(0, 0, 0)),
        rotation=Vec3.from_tuple(values=(0, 0, 0)),
        scale=Vec3.from_tuple(values=(1, 1, 1)),
    )

    ray = Vec3.from_tuple(values=(0, 0, 1))
    ray_origin = Vec3(x=0, y=0, z=-2.5)
    hit_data = cube.get_hit_data(ray=ray, ray_origin=ray_origin)

    assert hit_data.t == 2.0
    assert hit_data.point == Vec3(x=0, y=0, z=-0.5)
    assert hit_data.normal == Vec3(x=0, y=0, z=-1.0)
