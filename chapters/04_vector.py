from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image


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
        self._r = r
        self._g = g
        self._b = b

    @property
    def r(self):
        return self._r

    @property
    def g(self):
        return self._g

    @property
    def b(self):
        return self._b

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self._r, self._g, self._b)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}({self._r}, {self._g}, {self._b})>'


class RenderImage:
    def __init__(self, width: int, height: int, bg_color: Optional[Color] = None) -> None:
        self._width: int = width
        self._height: int = height
        self._aspect: float = self._width / self._height
        self._bg_color: Color = bg_color or Color(0, 0, 0)
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


def get_pixel_color(x: int, y: int, width: int, height: int, gradient: bool) -> Color:
    if gradient:
        return Color(
            r=int(255 * (x / width)),
            g=int(255 * (y / height)),
            b=0,
        )
    else:
        return Color(
            r=random.choice(range(255)),
            g=random.choice(range(255)),
            b=random.choice(range(255)),
        )


def render(image: RenderImage) -> RenderImage:
    for y in range(image.height):
        for x in range(image.width):
            color = get_pixel_color(
                x=x,
                y=y,
                width=image.width,
                height=image.height,
                gradient=True,
            )
            image.add_pixel_color(x=x, y=y, color=color)

    return image


def main(width: int, height: int, output_path: Union[Path, str]) -> None:
    image = RenderImage(width=width, height=height)
    rendered_image = render(image=image)
    rendered_image.save(output_path=str(output_path))


if __name__ == '__main__':
    image_name = f'{Path(__file__).stem}.png'
    output_path = Path(Path(__file__).parent.parent.parent, 'image', image_name)
    main(width=300, height=170, output_path=output_path)
