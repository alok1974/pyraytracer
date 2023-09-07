from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .color import Color


class RenderImage:
    def __init__(self, width: int, height: int, bg_color: Optional[Color] = None) -> None:
        self._width: int = width
        self._height: int = height
        self._aspect: float = self._width / self._height
        self._bg_color = bg_color or Color(r=0, g=0, b=0)

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
