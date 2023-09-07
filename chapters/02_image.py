from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image


class RenderImage:
    def __init__(self, width: int, height: int, bg_color: Optional[Tuple[int, int, int]] = None) -> None:
        self._width: int = width
        self._height: int = height
        self._aspect: float = self._width / self._height
        self._bg_color = bg_color or (0, 0, 0)
        self._image: Image.Image = Image.new("RGB", (self._width, self._height), self._bg_color)

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

    def add_pixel_color(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        self._pixels[x, y] = color

    def save(self, output_path: Union[Path, str]) -> None:
        self._image.save(output_path)


def render(image: RenderImage) -> RenderImage:
    for y in range(image.height):
        for x in range(image.width):
            color = (255, 0, 0)
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
