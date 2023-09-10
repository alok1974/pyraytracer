from pyraytracer.sphere import Sphere
from pyraytracer.vec3 import Vec3


def test_sphere_hit_data() -> None:
    sphere = Sphere(
        name='sphere',
        center=Vec3.from_tuple(values=(0, 0, 0)),
        rotation=Vec3.from_tuple(values=(0, 0, 0)),
        scale=Vec3.from_tuple(values=(2, 2, 2)),
    )

    ray = Vec3.from_tuple(values=(0, 0, 1))
    ray_origin = Vec3(x=0, y=0, z=-2.5)
    hit_data = sphere.get_hit_data(ray=ray, ray_origin=ray_origin)

    print(f'{hit_data=}')

    assert hit_data.t == 0.25
    assert hit_data.point == Vec3(x=0, y=0, z=-2.0)
    assert hit_data.normal == Vec3(x=0, y=0, z=-1.0)
