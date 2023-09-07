from __future__ import annotations

import random
import sys
import threading
import tkinter as tk
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageTk


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

    @property
    def image(self) -> Image.Image:
        return self._image

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
            dispersion: float = 0.0,
            tranparency: float = 0.0,
    ) -> None:

        self._validate_params(
            color=color,
            is_constant=is_constant,
            random_color_range=random_color_range,
            dispersion=dispersion,
            tranparency=tranparency,
        )

        self._color = color
        self._is_constant = is_constant
        self._random_color_range = random_color_range
        self._dispersion = dispersion
        self._transparency = tranparency

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

    @property
    def dispersion(self) -> float:
        return self._dispersion

    @property
    def transparency(self) -> float:
        return self._transparency

    def _validate_params(
        self, color: Color, is_constant: bool,
        random_color_range: Optional[Tuple[float, float]],
        dispersion: float, tranparency: float,
    ) -> None:

        if not isinstance(color, Color):
            raise RuntimeError('color needs to be of type <Color>')

        if not isinstance(is_constant, bool):
            raise RuntimeError('is_contant needs to be of type <bool>')

        if random_color_range is not None:
            a, b = random_color_range

            if not isinstance(a, float) or not isinstance(b, float):
                raise RuntimeError('random_color_range needs to be a list of two floats')

            if b < a:
                raise RuntimeError(f'Invalid random_color_range (a, b)={a, b}! b > a is required.')

        if not isinstance(dispersion, float):
            raise RuntimeError('dispersion needs to be of type(float)')

        if dispersion < 0.0 or dispersion > 1.0:
            raise RuntimeError('dispersion needs to be in the range 0.0 - 1.0')

        if not isinstance(tranparency, float):
            raise RuntimeError('tranparency needs to be of type(float)')

        if tranparency < 0.0 or tranparency > 1.0:
            raise RuntimeError('tranparency needs to be in the range 0.0 - 1.0')


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


class InteractiveRenderer:
    def __init__(self, image: Image.Image, close_after_rendering: bool = False) -> None:
        # Create the main tkinter window
        self._image = image
        self._close_after_rendering = close_after_rendering

        self._root = tk.Tk()
        self._root.title("Interactive Renderer")

        self._root.protocol("WM_DELETE_WINDOW", self._disable_close)

        # Convert the PIL image to a PhotoImage and display it
        self._tk_image = ImageTk.PhotoImage(self._image)

        # Use a Canvas instead of a Label to display the image
        self._canvas = tk.Canvas(self._root, width=self._image.width, height=self._image.height)
        self._canvas.pack()
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_image)

        self._root.after(1, self._center_window)

    def _center_window(self) -> None:
        # Get the screen width and height
        screen_width = self._root.winfo_screenwidth()
        screen_height = self._root.winfo_screenheight()

        # Get the window width and height
        window_width = self._root.winfo_width()
        window_height = self._root.winfo_height()

        # Calculate the x and y coordinates to center the window
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)

        # Set the window's position
        self._root.geometry(f'{window_width}x{window_height}+{x}+{y}')

    def _disable_close(self):
        ...

    def update_pixels(self, new_image: Image.Image) -> None:
        # Replace the current image with the new one
        self._image = new_image

        # Convert the updated PIL image to a PhotoImage and update the display
        self._tk_image = ImageTk.PhotoImage(self._image)

        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_image)

    def run(self) -> None:
        self._root.mainloop()

    def on_rendering_complete(self) -> None:
        # Re-enable the close button
        self._root.protocol("WM_DELETE_WINDOW", self._root.destroy)

        # Automatically close the window
        if self._close_after_rendering:
            self._root.destroy()
            sys.exit()


@dataclass(frozen=True)
class RenderSettings:
    # Image Settings
    output_path: Union[str, Path]
    image_width: int
    image_height: int
    aa_samples: float = 1.0
    bg_color: Color = Color(0, 0, 0)

    # Circular Noise Settings
    add_circular_noise: bool = False
    circular_noise_scale: float = 0.4
    circular_noise_color_min: int = 165
    circular_noise_color_max: int = 255
    circular_noise_size_min: float = 0.002
    circular_noise_size_max: float = 0.0035

    # Tail Noise Settings
    add_tail_noise: bool = False
    tail_noise_scale: float = 1.5
    tail_noise_color_min: int = 200
    tail_noise_color_max: int = 255
    tail_noise_size_min: float = 0.002
    tail_noise_size_max: float = 0.008
    tail_noise_width: float = 18
    tail_noise_angle_min: float = -10
    tail_noise_angle_max: float = 10

    # Pixel Noise Settings
    add_pixel_noise: bool = False
    pixel_noise_scale: float = 0.4
    pixel_noise_color_min: int = 165
    pixel_noise_color_max: int = 255

    # Interactive Render Settings
    interactive_render: bool = False
    close_interactive_render: bool = False


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

        self._random_circular_arr: List[Tuple[Vec3, float, Color]] = []
        self._random_tail_arr: List[Tuple[float, Vec3, float, Vec3, Vec3, Vec3, Vec3, Color, float]] = []

        if self._settings.interactive_render:
            self._ir = InteractiveRenderer(
                image=self._image.image,
                close_after_rendering=self._settings.close_interactive_render,
            )

    def save_image(self) -> None:
        output_path = self._settings.output_path
        self._image.save(output_path=output_path, antialias_samples=self._aa_samples)

    def run(self) -> None:
        if self._settings.interactive_render:
            # Start the rendering loop in a separate thread
            rendering_thread = threading.Thread(target=self._render)
            rendering_thread.daemon = True
            rendering_thread.start()

            # Start interactive render window
            self._ir.run()
        else:
            self._render()

    def _render(self) -> None:
        with progress_bar_context() as update_bar:
            counter = 1
            for y, x in product(range(self._image_height), range(self._image_width)):
                if self._camera is None:
                    continue

                pixel_pos = self._get_pixel_pos(x=x, y=y)
                ray_dir = Vec3.normalize(pixel_pos - self._camera.center)
                pixel_color = self._raytrace(ray_dir=ray_dir)

                # If ray did not hit an object, get a circle
                # with random color and size
                pixel_color = self._add_circular_noise(
                        pixel_color=pixel_color,
                        x=x,
                        y=y,
                        x_pos=pixel_pos.x,
                        y_pos=pixel_pos.y,
                )

                # Add tail noise (a circle with a tail)
                pixel_color = self._add_tail_noise(
                    pixel_color=pixel_color,
                    x=x,
                    y=y,
                    x_pos=pixel_pos.x,
                    y_pos=pixel_pos.y,
                )

                # Add pixel noise
                pixel_color = self._add_pixel_noise(
                    pixel_color=pixel_color,
                    x=x,
                    y=y,
                )

                if pixel_color is not None:
                    self._image.add_pixel_color(x=x, y=y, color=pixel_color)

                percentage = (counter / (self._image_width * self._image_height)) * 100
                update_bar(percentage=percentage)
                counter += 1

                if self._settings.interactive_render and percentage % 2 == 0:
                    self._ir.update_pixels(self._image.image)

        if self._settings.interactive_render:
            self._ir.on_rendering_complete()

    def _add_circular_noise(
            self, pixel_color: Optional[Color], x: int, y: int,
            x_pos: float, y_pos: float,
    ) -> Optional[Color]:

        if pixel_color is None and self._settings.add_circular_noise:
            return self._circular_noise(
                x=x,
                y=y,
                x_pos=x_pos,
                y_pos=y_pos,
            )
        else:
            return pixel_color

    def _circular_noise(self, x: int, y: int, x_pos: float, y_pos: float) -> Optional[Color]:
        scale = self._settings.circular_noise_scale
        is_random_pixel = self._pick_random_pixel(x=x, y=y, scale=scale)
        not_border_pixel = self._is_not_border_pixel(x=x, y=y)

        if is_random_pixel and not_border_pixel:
            self._update_random_circular_arr(x_pos=x_pos, y_pos=y_pos)

        for center, radius, s_color in self._random_circular_arr:
            in_circle = self._point_in_circle(
                point=Vec3(x=x_pos, y=y_pos, z=0),
                center=center,
                radius=radius,
            )
            random_in = random.random() < 0.95
            if in_circle and random_in:
                return s_color

        return None

    def _add_tail_noise(
            self, pixel_color: Optional[Color], x: int, y: int,
            x_pos: float, y_pos: float,
    ) -> Optional[Color]:

        if pixel_color is None and self._settings.add_tail_noise:
            return self._tail_noise(
                x=x,
                y=y,
                x_pos=x_pos,
                y_pos=y_pos,
            )
        else:
            return pixel_color

    def _tail_noise(self, x: int, y: int, x_pos: float, y_pos: float) -> Optional[Color]:
        scale = self._settings.tail_noise_scale
        is_random_pixel = self._pick_random_pixel(x=x, y=y, scale=scale)
        not_border_pixel = self._is_not_border_pixel(x=x, y=y)

        if is_random_pixel and not_border_pixel:
            self._update_random_tail_arr(x_pos=x_pos, y_pos=y_pos)

        for s_x, c, r, q1, q2, q3, q4, s_color, angle in self._random_tail_arr:
            p = Vec3(x=x_pos, y=y_pos, z=0)
            in_quad = self._point_in_quad(point=p, a=q1, b=q2, c=q3, d=q4)
            in_circle = self._point_in_circle(point=p, center=c, radius=r)
            lerp_factor = self._fade_to_tail(
                x=x_pos,
                s_x=s_x,
                r=r,
                angle=angle,
                width=self._settings.tail_noise_width,
            )
            random_in = random.random() < lerp_factor

            if (in_circle or in_quad) and random_in:
                return lerp_factor * s_color + (1 - lerp_factor) * self._settings.bg_color

        return None

    def _fade_to_tail(self, x: float, s_x: float, r: float, width: float, angle: float) -> float:
        value = 1 - ((x - s_x) / (width * r))
        angle_rad = np.radians(angle)
        return value * np.cos(angle_rad)

    def _rotate_point(self, rot_center: Vec3, rot_degrees: float, to_rotate: Vec3) -> Vec3:
        to_rotate_arr = np.array([to_rotate.x, to_rotate.y])
        rot_center_arr = np.array([rot_center.x, rot_center.y])
        theta_rad = np.radians(rot_degrees)

        rotation_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)]
        ])

        to_rotate_translated = to_rotate_arr - rot_center_arr
        translated_rotated = rotation_matrix @ to_rotate_translated
        rotated = translated_rotated + rot_center_arr

        return Vec3(x=rotated[0], y=rotated[1], z=0)

    def _point_in_circle(self, point: Vec3, center: Vec3, radius: float) -> bool:
        return (point.x - center.x) ** 2 + (point.y - center.y) ** 2 <= radius ** 2

    def _point_in_quad(self, point: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> bool:
        # Calculate the area of the quadrilateral ABCD
        quad_area = self._triangle_area(a, b, c) + self._triangle_area(a, c, d)

        # Calculate the areas of the triangles formed by point P
        area_sum = (
            self._triangle_area(point, a, b)
            + self._triangle_area(point, b, c)
            + self._triangle_area(point, c, d)
            + self._triangle_area(point, d, a)
        )

        # The point is inside/on the quadrilateral if the areas match
        return abs(quad_area - area_sum) < 1e-7  # using a small threshold to handle floating point imprecisions

    def _triangle_area(self, p1: Vec3, p2: Vec3, p3: Vec3) -> float:
        return 0.5 * abs(p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

    def _add_pixel_noise(self, pixel_color: Optional[Color], x: int, y: int) -> Optional[Color]:
        if pixel_color is None and self._settings.add_pixel_noise:
            return self._pixel_noise(x=x, y=y)
        else:
            return pixel_color

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

    def _update_random_circular_arr(self, x_pos, y_pos) -> None:
        color_min = self._settings.circular_noise_color_min
        color_max = self._settings.circular_noise_color_max
        r_color = self._random_color(color_min=color_min, color_max=color_max)

        size_min = self._settings.circular_noise_size_min
        size_max = self._settings.circular_noise_size_max
        r = random_range(size_min, size_max) / 2
        c = Vec3(x=x_pos + r, y=y_pos - r, z=0)
        self._random_circular_arr.append((c, r, r_color))

    def _update_random_tail_arr(self, x_pos, y_pos) -> None:
        color_min = self._settings.tail_noise_color_min
        color_max = self._settings.tail_noise_color_max
        r_color = self._random_color(color_min=color_min, color_max=color_max)

        size_min = self._settings.tail_noise_size_min
        size_max = self._settings.tail_noise_size_max
        tail_width = self._settings.tail_noise_width

        r = random_range(size_min, size_max) / 2
        c = Vec3(x_pos + r, y_pos - (2 * r), 0)
        q1 = Vec3(x_pos + r, y_pos - r, 0)
        q2 = Vec3(x_pos + (tail_width * r), y_pos, 0)
        q3 = Vec3(x_pos + (tail_width * r), y_pos - (4 * r), 0)
        q4 = Vec3(x_pos + r, y_pos - (3 * r), 0)

        # Randomly rotate points
        angle_min = self._settings.tail_noise_angle_min
        angle_max = self._settings.tail_noise_angle_max
        angle = random_range(angle_min, angle_max)
        q1 = self._rotate_point(rot_center=c, rot_degrees=angle, to_rotate=q1)
        q2 = self._rotate_point(rot_center=c, rot_degrees=angle, to_rotate=q2)
        q3 = self._rotate_point(rot_center=c, rot_degrees=angle, to_rotate=q3)
        q4 = self._rotate_point(rot_center=c, rot_degrees=angle, to_rotate=q4)

        # Shift all points down below y_pos so that we can render them when
        # we hit x_pos, y_pos
        if q2.y > y_pos:
            delta = q2.y - y_pos
            c = Vec3(x=c.x, y=c.y - delta, z=0)
            q1 = Vec3(x=q1.x, y=q1.y - delta, z=0)
            q2 = Vec3(x=q2.x, y=q2.y - delta, z=0)
            q3 = Vec3(x=q3.x, y=q3.y - delta, z=0)
            q4 = Vec3(x=q4.x, y=q4.y - delta, z=0)

        self._random_tail_arr.append((x_pos, c, r, q1, q2, q3, q4, r_color, angle))

    def _random_color(self, color_min: int, color_max: int) -> Color:
        return Color(
            random.randint(color_min, color_max),
            random.randint(color_min, color_max),
            random.randint(color_min, color_max),
        )

    def _raytrace(self, ray_dir: Vec3) -> Optional[Color]:
        spheres_with_t = []
        last_color = self._settings.bg_color
        for sphere in self._spheres:
            t = self._get_t(sphere=sphere, ray_dir=ray_dir)
            if t is not None:
                spheres_with_t.append((t, sphere))

        sorted_spheres = sorted(spheres_with_t)
        if not sorted_spheres:
            return None

        if len(sorted_spheres) > 1:
            last_before_closest_t, last_before_closest_sphere = sorted_spheres[1]
            last_before_closest_intersection = self._camera.center + (ray_dir * last_before_closest_t)
            last_color = self._get_pixel_color(
                sphere=last_before_closest_sphere,
                intersection=last_before_closest_intersection,
                last_color=last_color,
            ) or last_color

        min_t = sorted_spheres[0][0]
        intersection = self._camera.center + (ray_dir * min_t)
        return self._get_pixel_color(
            sphere=sorted_spheres[0][1],
            intersection=intersection,
            last_color=last_color,
        )

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

    def _get_pixel_color(self, sphere: Sphere, intersection: Vec3, last_color: Color) -> Optional[Color]:
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

        if sphere.material.transparency > 0.0:
            t_factor = sphere.material.transparency
            final_color = t_factor * last_color + (1 - t_factor) * final_color

        if sphere.material.dispersion > 0.0:
            if random.random() < sphere.material.dispersion:
                return last_color
            else:
                return final_color

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
    green_diffuse = Material(color=Color(0, 255, 0), random_color_range=(0.7, 1.0), tranparency=0.5)
    blue_diffuse = Material(color=Color(255, 255, 255), tranparency=0.2)
    pink = Material(color=Color(255, 192, 203), dispersion=0.3)

    # Spheres
    sphere_1 = Sphere(center=Vec3(1, 0, 8), radius=1.5, material=red_diffuse)
    sphere_2 = Sphere(center=Vec3(-1, 0, 7), radius=1, material=green_diffuse)
    sphere_3 = Sphere(center=Vec3(-0.5, 0.5, 6), radius=0.5, material=blue_diffuse)
    sphere_4 = Sphere(center=Vec3(-3, -0.5, 8), radius=0.5, material=pink)

    # Lights
    key = PointLight(center=Vec3(-5, 5, -10), intensity=2)
    rim = PointLight(center=Vec3(4, 10, 20), intensity=5)
    fill = PointLight(center=Vec3(3, -2, 4), color=Color(0, 100, 255), intensity=0.4)

    # Light linking
    key.add_include(sphere=sphere_2)
    key.add_include(sphere=sphere_4)
    rim.add_include(sphere=sphere_1)
    rim.add_include(sphere=sphere_3)
    rim.add_include(sphere=sphere_2)
    fill.add_exclude(sphere=sphere_2)

    # Scene
    scene = Scene()
    scene.camera = Camera(center=Vec3(0, 0, -1), fov=0.26)
    scene.add_sphere(sphere=sphere_4)
    scene.add_sphere(sphere=sphere_1)
    scene.add_sphere(sphere=sphere_2)
    scene.add_sphere(sphere=sphere_3)
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
        # Image Settings
        output_path=output_path,
        image_width=300,
        image_height=170,
        aa_samples=1.0,
        bg_color=Color(100, 100, 100),

        # Circular Noise Settings
        add_circular_noise=True,
        circular_noise_scale=0.43,
        circular_noise_color_min=80,
        circular_noise_color_max=255,
        circular_noise_size_min=0.002,
        circular_noise_size_max=0.01,

        # Tail Noise Settings
        add_tail_noise=True,
        tail_noise_scale=1.5,
        tail_noise_color_min=200,
        tail_noise_color_max=255,
        tail_noise_size_min=0.002,
        tail_noise_size_max=0.008,
        tail_noise_width=50,
        tail_noise_angle_min=-10,
        tail_noise_angle_max=10,

        # Pixel Noise Settings
        add_pixel_noise=True,
        pixel_noise_scale=0.1,

        # Interactive Render Settings
        interactive_render=True,
        close_interactive_render=True,
    )

    # Renderer
    renderer = Renderer(scene=scene, render_settings=settings)
    renderer.run()
    renderer.save_image()


if __name__ == '__main__':
    main()
