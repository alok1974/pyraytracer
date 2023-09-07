import random
from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image


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
