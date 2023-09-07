from __future__ import annotations

import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


def random_range(a: float, b: float) -> float:
    if b < a:
        raise RuntimeError('For random range b > a is required!')

    return a + (b - a) * random.random()


@contextmanager
def progress_bar_context(bar_length=50):
    # Setup action: print a newline
    print()
    yield lambda percentage: print_progress_bar(percentage, bar_length)
    # Teardown action: print a newline and flush
    print('\n')
    sys.stdout.flush()


def print_progress_bar(percentage, bar_length=50):
    # Calculate the number of blocks the progress bar should have
    blocks = int((percentage * bar_length) / 100)

    # Build the progress bar
    progress = '█' * blocks
    spaces = ' ' * (bar_length - blocks)

    # Print the progress bar
    print(f'\r{progress}{spaces} {percentage:.2f}%', end='')
    sys.stdout.flush()


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

    def save(self, output_path: Union[Path, str], antialias_samples: float = 1.0) -> None:
        if antialias_samples > 1.0:
            self._image = self._image.resize(
                (
                    int(self._width / antialias_samples),
                    int(self._height / antialias_samples),
                ),
                resample=Image.LANCZOS,
            )

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

        self._includes: List[int] = []
        self._excludes: List[int] = []

    @property
    def center(self) -> Vec3:
        return self._center

    @property
    def intensity(self) -> float:
        return self._intensity

    @property
    def color(self) -> Color:
        return self._color

    def add_include(self, sphere: Sphere) -> None:
        if id(sphere) in self._excludes:
            raise RuntimeError(f'This object {sphere} is already added to excludes.')

        self._includes.append(id(sphere))

    def is_included(self, sphere: Sphere) -> bool:
        if not self._includes and not self.is_excluded(sphere=sphere):
            return True

        return id(sphere) in self._includes

    def add_exclude(self, sphere: Sphere) -> None:
        if id(sphere) in self._includes:
            raise RuntimeError(f'This object {sphere} is already added to the includes.')

        self._excludes.append(id(sphere))

    def is_excluded(self, sphere: Sphere) -> bool:
        return id(sphere) in self._excludes


class Scene:
    def __init__(self) -> None:
        self._camera: Camera
        self._lights: List[PointLight] = []
        self._spheres: List[Sphere] = []

    @property
    def camera(self) -> Camera:
        return self._camera

    @camera.setter
    def camera(self, camera: Camera) -> None:
        self._camera = camera

    @property
    def lights(self) -> List[PointLight]:
        return self._lights

    def add_light(self, light: PointLight) -> None:
        self._lights.append(light)

    @property
    def spheres(self) -> List[Sphere]:
        return self._spheres

    def add_sphere(self, sphere: Sphere) -> None:
        self._spheres.append(sphere)


@dataclass(frozen=True)
class RenderSettings:
    output_path: Union[str, Path]
    image_width: int
    image_height: int
    aa_samples: float = 1.0
    bg_color: Optional[Color] = Color(0, 0, 0)
    add_circular_noise: bool = False
    circular_noise_scale: float = 0.4
    circular_noise_color_min: int = 165
    circular_noise_color_max: int = 255
    circular_noise_size_min: float = 0.002
    circular_noise_size_max: float = 0.0035
    add_pixel_noise: bool = False
    pixel_noise_scale: float = 0.4
    pixel_noise_color_min: int = 165
    pixel_noise_color_max: int = 255


class Renderer:
    def __init__(self, scene: Scene, render_settings: RenderSettings) -> None:
        self._scene = scene
        self._settings = render_settings

        aa_samples = self._settings.aa_samples
        if aa_samples < 1.0:
            raise RuntimeError(f'aa_samples={aa_samples} cannot be less that 1.0')

        self._aa_samples: float = aa_samples

        self._image_width: int = int(self._settings.image_width * self._aa_samples)
        self._image_height: int = int(self._settings.image_height * self._aa_samples)
        self._bg_color: Color = self._settings.bg_color or Color(0, 0, 0)

        self._camera: Camera = self._scene.camera
        self._spheres: List[Sphere] = self._scene.spheres
        self._lights: List[PointLight] = self._scene.lights

        self._image = RenderImage(
            width=self._image_width,
            height=self._image_height,
            bg_color=self._bg_color,
        )

        self._random_arr: List[Tuple[float, float, Color, float]] = []

    def save_image(self) -> None:
        output_path = self._settings.output_path
        self._image.save(output_path=output_path, antialias_samples=self._aa_samples)

    def render(self) -> None:
        with progress_bar_context() as update_bar:
            counter = 1
            for x, y in product(range(self._image_width), range(self._image_height)):
                pixel_pos = self._get_pixel_pos(x=x, y=y)
                ray_dir = Vec3.normalize(pixel_pos - self._camera.center)
                pixel_color = self._raytrace(ray_dir=ray_dir)

                # If ray did not hit an object, get a circle with random color and size
                if pixel_color is None and self._settings.add_circular_noise:
                    pixel_color = self._circular_noise(
                        x=x,
                        y=y,
                        x_pos=pixel_pos.x,
                        y_pos=pixel_pos.y,
                    )

                # Add pixel noise
                if pixel_color is None and self._settings.add_pixel_noise:
                    pixel_color = self._pixel_noise(x=x, y=y)

                if pixel_color is not None:
                    self._image.add_pixel_color(x=x, y=y, color=pixel_color)

                percentage = (counter / (self._image_width * self._image_height)) * 100
                update_bar(percentage=percentage)
                counter += 1

    def _circular_noise(self, x: int, y: int, x_pos: float, y_pos: float) -> Optional[Color]:
        scale = self._settings.circular_noise_scale
        is_random_pixel = self._pick_random_pixel(x=x, y=y, scale=scale)
        not_border_pixel = self._is_not_border_pixel(x=x, y=y)

        if is_random_pixel and not_border_pixel:
            self._update_random_arr(x_pos=x_pos, y_pos=y_pos)

        for s_x, s_y, s_color, s_size in self._random_arr:
            # The equation of circle with center [x + s/2, y - s/2] and radius
            # as s/2
            if (x_pos - (s_x + (s_size / 2)))**2 + (y_pos - (s_y - (s_size / 2)))**2 <= (s_size / 2)**2:
                return s_color

        return None

    def _pixel_noise(self, x: int, y: int) -> Optional[Color]:
        scale = self._settings.pixel_noise_scale
        is_random_pixel = self._pick_random_pixel(x=x, y=y, scale=scale)
        not_border_pixel = self._is_not_border_pixel(x=x, y=y)

        if is_random_pixel and not_border_pixel:
            color_min = self._settings.pixel_noise_color_min
            color_max = self._settings.pixel_noise_color_max
            return self._random_color(color_min=color_min, color_max=color_max)

        return None

    def _pick_random_pixel(self, x: int, y: int, scale: float) -> bool:
        max_range_x = max(int(scale * self._image_width), 2)
        max_range_y = max(int(scale * self._image_height), 2)
        random_x = x % random.choice(range(1, max_range_x)) == 0
        random_y = y % random.choice(range(1, max_range_y)) == 0
        is_random_pixel = random_x and random_y

        return is_random_pixel

    def _is_not_border_pixel(self, x: int, y: int) -> bool:
        not_border_x = 0 < x < self._image_width - 1
        not_border_y = 0 < y < self._image_height - 1
        not_border_pixel = not_border_x and not_border_y

        return not_border_pixel

    def _update_random_arr(self, x_pos, y_pos) -> None:
        color_min = self._settings.circular_noise_color_min
        color_max = self._settings.circular_noise_color_max
        r_color = self._random_color(color_min=color_min, color_max=color_max)

        size_min = self._settings.circular_noise_size_min
        size_max = self._settings.circular_noise_size_max
        r_size = random_range(size_min, size_max)

        self._random_arr.append((x_pos, y_pos, r_color, r_size))

    def _random_color(self, color_min: int, color_max: int) -> Color:
        return Color(
            random.randint(color_min, color_max),
            random.randint(color_min, color_max),
            random.randint(color_min, color_max),
        )

    def _raytrace(self, ray_dir: Vec3) -> Optional[Color]:
        min_t = float('inf')
        closest_sphere = None
        for sphere in self._spheres:
            t = self._get_t(sphere=sphere, ray_dir=ray_dir)
            if t is not None and t < min_t:
                min_t = t
                closest_sphere = sphere

        if closest_sphere is not None:
            intersection = self._camera.center + (ray_dir * min_t)

            return self._get_pixel_color(
                sphere=closest_sphere,
                intersection=intersection
            )
        else:
            return None

    def _get_t(self, sphere: Sphere, ray_dir: Vec3) -> Optional[float]:
        oc = self._camera.center - sphere.center
        a = Vec3.dot(ray_dir, ray_dir)
        b = 2 * Vec3.dot(oc, ray_dir)
        c = Vec3.dot(oc, oc) - (sphere.radius * sphere.radius)
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        t = (-b - np.sqrt(discriminant)) / (2 * a)

        if t < 0:
            return None
        else:
            return t

    def _get_pixel_color(self, sphere: Sphere, intersection: Vec3) -> Optional[Color]:
        diffuse = sphere.material.color
        if sphere.material.is_constant:
            return diffuse

        if sphere.material.is_color_randomized and sphere.material.random_color_range:
            a, b = sphere.material.random_color_range
            diffuse = Color(
                r=int(diffuse.r * random_range(a, b)),
                g=int(diffuse.g * random_range(a, b)),
                b=int(diffuse.b * random_range(a, b)),
            )

        final_color = Color(0, 0, 0)
        for light in self._lights:
            # Check light linking
            if light.is_excluded(sphere=sphere):
                continue

            elif light.is_included(sphere=sphere):

                l_factor = self._lambertian_factor(
                    light=light,
                    sphere=sphere,
                    intersection=intersection
                )

                reflectance = diffuse * l_factor

                # Lerp of light and diffuse color with t = 0.5
                final_color += ((0.5 * reflectance) + (0.5 * light.color)) * light.intensity

        return final_color

    def _lambertian_factor(self, light: PointLight, sphere: Sphere, intersection: Vec3) -> float:
        normal = Vec3.normalize(intersection - sphere.center)
        to_light = Vec3.normalize(light.center - intersection)

        # Lambertian Cosine model
        return float(max(0, Vec3.dot(to_light, normal)))

    def _get_pixel_pos(self, x: int, y: int) -> Vec3:
        px = (2 * (x + 0.5) / self._image_width - 1) * self._image.aspect * self._camera.fov
        py = (1 - 2 * (y + 0.5) / self._image_height) * self._camera.fov
        return Vec3(x=px, y=py, z=self._camera.center.z + 1)


def create_scene() -> Scene:
    # Materials
    red_diffuse = Material(color=Color(255, 0, 0))
    green_diffuse = Material(color=Color(0, 255, 0), random_color_range=(0.7, 1.0))
    blue_diffuse = Material(color=Color(0, 0, 255))

    # Spheres
    sphere_1 = Sphere(center=Vec3(1, 0, 8), radius=1.5, material=red_diffuse)
    sphere_2 = Sphere(center=Vec3(-1, 0, 7), radius=1, material=green_diffuse)
    sphere_3 = Sphere(center=Vec3(-0.5, 0.5, 6), radius=0.5, material=blue_diffuse)
    sphere_4 = Sphere(center=Vec3(-3, -0.5, 8), radius=0.5, material=blue_diffuse)

    # Lights
    key = PointLight(center=Vec3(-5, 5, -10), intensity=2)
    rim = PointLight(center=Vec3(4, 10, 20), intensity=30)
    fill = PointLight(center=Vec3(3, -2, 4), color=Color(0, 100, 255), intensity=0.4)

    # Light linking
    key.add_include(sphere=sphere_2)
    rim.add_include(sphere=sphere_1)
    rim.add_include(sphere=sphere_3)
    fill.add_exclude(sphere=sphere_2)

    # Scene
    scene = Scene()
    scene.camera = Camera(center=Vec3(0, 0, -1), fov=0.26)
    scene.add_sphere(sphere=sphere_1)
    scene.add_sphere(sphere=sphere_2)
    scene.add_sphere(sphere=sphere_3)
    scene.add_sphere(sphere=sphere_4)
    scene.add_light(light=key)
    scene.add_light(light=rim)
    scene.add_light(light=fill)

    return scene


def main() -> None:
    scene = create_scene()

    # Render Settings
    image_name = f'{Path(__file__).stem}.png'
    output_path = Path(Path(__file__).parent.parent.parent, 'image', image_name)

    settings = RenderSettings(
        output_path=output_path,
        image_width=300,
        image_height=170,
        aa_samples=1.0,
        bg_color=Color(0, 0, 0),
        add_circular_noise=True,
        circular_noise_scale=0.4,
        circular_noise_size_min=0.002,
        circular_noise_size_max=0.01,
        add_pixel_noise=True,
        pixel_noise_scale=0.15,
    )

    # Renderer
    renderer = Renderer(scene=scene, render_settings=settings)
    renderer.render()
    renderer.save_image()


if __name__ == '__main__':
    main()
