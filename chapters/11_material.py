from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from PIL import Image


def random_range(a: float, b: float) -> float:
    if b < a:
        raise RuntimeError('For random range b > a is required!')

    return a + (b - a) * random.random()


class Vec3:
    def __init__(self, x: float, y: float, z: float) -> None:
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def z(self) -> float:
        return self._z

    def __add__(self, other: Vec3) -> Vec3:
        return self.__class__(
            self._x + other.x,
            self._y + other.y,
            self._z + other.z
        )

    def __sub__(self, other: Vec3) -> Vec3:
        return self.__class__(
            self._x - other.x,
            self._y - other.y,
            self._z - other.z
        )

    def __mul__(self, other: Any) -> Vec3:
        if not isinstance(other, float):
            raise ValueError(f"Unsupported type for multiplication: {type(other).__name__}")

        return self.__class__(
            self._x * other,
            self._y * other,
            self._z * other,
        )

    def __rmul__(self, other: Any) -> Vec3:
        return self.__mul__(other)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}({self._x}, {self._y}, {self._z})>'

    @classmethod
    def dot(cls, v1: Vec3, v2: Vec3) -> float:
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

    def length(self) -> float:
        return np.sqrt(self.dot(self, self))

    @classmethod
    def normalize(cls, v: Vec3) -> Vec3:
        v_len = cls.length(v)
        if v_len == 0:
            return Vec3(x=0, y=0, z=0)
        else:
            return Vec3(x=v.x/v_len, y=v.y/v_len, z=v.z/v_len)


class Color:
    def __init__(self, r: int, g: int, b: int) -> None:
        self._r = self._clamp(r)
        self._g = self._clamp(g)
        self._b = self._clamp(b)

    @property
    def r(self) -> int:
        return self._r

    @property
    def g(self) -> int:
        return self._g

    @property
    def b(self) -> int:
        return self._b

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self._r, self._g, self._b)

    def __add__(self, other: Any) -> Color:
        if isinstance(other, int):
            return self.__class__(
                r=self._clamp(self._r + other),
                g=self._clamp(self._g + other),
                b=self._clamp(self._b + other),
            )
        elif isinstance(other, Color):
            return self.__class__(
                r=self._clamp(self._r + other.r),
                g=self._clamp(self._g + other.g),
                b=self._clamp(self._b + other.b),
            )
        else:
            raise ValueError(f"Unsupported type for multiplication: {type(other).__name__}")

    def __sub__(self, other: Any) -> Color:
        if isinstance(other, int):
            return self.__class__(
                r=self._clamp(self._r - other),
                g=self._clamp(self._g - other),
                b=self._clamp(self._b - other),
            )
        elif isinstance(other, Color):
            return self.__class__(
                r=self._clamp(self._r - other.r),
                g=self._clamp(self._g - other.g),
                b=self._clamp(self._b - other.b),
            )
        else:
            raise ValueError(f"Unsupported type for multiplication: {type(other).__name__}")

    def __mul__(self, other: Any) -> Color:
        if isinstance(other, (float, int)):
            return self.__class__(
                r=self._clamp(self._r * other),
                g=self._clamp(self._g * other),
                b=self._clamp(self._b * other),
            )
        elif isinstance(other, Color):
            # Element-wise product
            return self.__class__(
                r=self._clamp(self._r * other.r),
                g=self._clamp(self._g * other.g),
                b=self._clamp(self._b * other.b),
            )
        else:
            raise ValueError(f"Unsupported type for multiplication: {type(other).__name__}")

    def __rmul__(self, other: Any) -> Color:
        return self.__mul__(other)

    def _clamp(self, val: Union[int, float]) -> int:
        val = int(val)
        if val < 0:
            return 0
        elif val > 255:
            return 255
        else:
            return val

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}({self._r}, {self._g}, {self._b})>'


class RenderImage:
    def __init__(self, width: int, height: int, bg_color: Optional[Color] = None) -> None:
        self._width: int = width
        self._height: int = height
        self._aspect: float = self._width / self._height
        self._bg_color = bg_color or Color(0, 0, 0)

        self._image: Image.Image = Image.new("RGB", (self._width, self._height), self._bg_color.to_tuple())
        self._pixels = self._image.load()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def aspect(self) -> float:
        return self._aspect

    def add_pixel_color(self, x: int, y: int, color: Color) -> None:
        self._pixels[x, y] = color.to_tuple()

    def save(self, output_path: Union[Path, str]) -> None:
        self._image.save(output_path)


class Camera:
    def __init__(self, center: Vec3, fov: float) -> None:
        self._center = center
        self._fov = fov

    @property
    def center(self) -> Vec3:
        return self._center

    @property
    def fov(self) -> float:
        return self._fov


class Material:
    def __init__(
            self, color: Color, is_constant: bool = False,
            random_color_range: Optional[Tuple[float, float]] = None,
    ) -> None:

        self._color = color
        self._is_constant = is_constant

        self._validate_random_color_range(random_color_range=random_color_range)
        self._random_color_range = random_color_range

    @property
    def color(self) -> Color:
        return self._color

    @property
    def is_constant(self) -> bool:
        return self._is_constant

    @property
    def is_color_randomized(self) -> bool:
        return bool(self._random_color_range)

    @property
    def random_color_range(self) -> Optional[Tuple[float, float]]:
        return self._random_color_range

    def _validate_random_color_range(self, random_color_range: Optional[Tuple[float, float]]) -> None:
        if random_color_range is not None:
            a, b = random_color_range
            if b < a:
                raise RuntimeError(f'Invalid random_color_range (a, b)={a, b}! b > a is required.')


class Sphere:
    def __init__(self, center: Vec3, radius: float, material: Material) -> None:
        self._center = center
        self._radius = radius
        self._material = material

    @property
    def center(self) -> Vec3:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def material(self) -> Material:
        return self._material


class PointLight:
    def __init__(self, center: Vec3, color: Color = Color(0, 0, 0), intensity: float = 1.0) -> None:
        self._center = center
        self._intensity = intensity
        self._color = color

    @property
    def center(self) -> Vec3:
        return self._center

    @property
    def intensity(self) -> float:
        return self._intensity

    @property
    def color(self) -> Color:
        return self._color


class Renderer:
    def __init__(
            self, image_width: int, image_height: int, camera: Camera,
            sphere: Sphere, light: PointLight, bg_color: Optional[Color] = None
    ) -> None:

        self._image_width = image_width
        self._image_height = image_height
        self._bg_color = bg_color or Color(0, 0, 0)
        self._camera = camera
        self._sphere = sphere
        self._light = light

        self._image = RenderImage(
            width=image_width,
            height=image_height,
            bg_color=bg_color,
        )

    @property
    def image(self) -> RenderImage:
        return self._image

    def render(self) -> None:
        for y in range(self._image.height):
            for x in range(self._image.width):
                pixel_pos = self._get_pixel_pos(x=x, y=y)
                ray_dir = Vec3.normalize(pixel_pos - self._camera.center)

                intersection = self._get_intersection_point(ray_dir=ray_dir)
                if intersection is None:
                    continue

                pixel_color = self._get_pixel_color(intersection_point=intersection)
                self._image.add_pixel_color(x=x, y=y, color=pixel_color)

    def _get_intersection_point(self, ray_dir: Vec3) -> Optional[Vec3]:
        oc = self._camera.center - self._sphere.center
        a = Vec3.dot(ray_dir, ray_dir)
        b = 2 * Vec3.dot(oc, ray_dir)
        c = Vec3.dot(oc, oc) - (self._sphere.radius * self._sphere.radius)
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        t = (-b - np.sqrt(discriminant)) / (2 * a)

        if t < 0:
            return None
        else:
            return self._camera.center + (ray_dir * t)

    def _lambertian_factor(self, intersection_point: Vec3) -> float:
        normal = Vec3.normalize(intersection_point - self._sphere.center)
        to_light = Vec3.normalize(self._light.center - intersection_point)

        # Lambertian Cosine model
        return float(max(0, Vec3.dot(to_light, normal)))

    def _get_pixel_color(self, intersection_point: Vec3) -> Color:
        diffuse = self._sphere.material.color
        if self._sphere.material.is_constant:
            return diffuse

        if self._sphere.material.is_color_randomized and self._sphere.material.random_color_range:
            a, b = self._sphere.material.random_color_range
            diffuse = Color(
                r=int(diffuse.r * random_range(a, b)),
                g=int(diffuse.g * random_range(a, b)),
                b=int(diffuse.b * random_range(a, b)),
            )

        l_factor = self._lambertian_factor(intersection_point=intersection_point)
        reflectance = diffuse * l_factor

        # Lerp of light and diffuse color with t = 0.5
        final_color = (0.5 * reflectance) + (0.5 * self._light.color)
        return final_color * self._light.intensity

    def _get_pixel_pos(self, x: int, y: int) -> Vec3:
        px = (2 * (x + 0.5) / self._image_width - 1) * self._image.aspect * self._camera.fov
        py = (1 - 2 * (y + 0.5) / self._image_height) * self._camera.fov
        return Vec3(x=px, y=py, z=self._camera.center.z + 1)


def main(width: int, height: int, output_path: Union[Path, str]) -> None:
    camera = Camera(center=Vec3(0, 0, -1), fov=0.26)
    mat = Material(color=Color(255, 0, 0), is_constant=False, random_color_range=None)
    sphere = Sphere(center=Vec3(0, 0, 5), radius=1, material=mat)
    light = PointLight(center=Vec3(0, 5, -10), color=Color(0, 0, 0), intensity=1)
    renderer = Renderer(
        image_width=width,
        image_height=height,
        camera=camera,
        sphere=sphere,
        light=light,
        bg_color=None,
    )
    renderer.render()
    renderer.image.save(output_path=str(output_path))


if __name__ == '__main__':
    image_name = f'{Path(__file__).stem}.png'
    output_path = Path(Path(__file__).parent.parent.parent, 'image', image_name)
    main(width=300, height=170, output_path=output_path)
