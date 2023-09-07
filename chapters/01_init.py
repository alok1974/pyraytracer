from pathlib import Path
from typing import Union

from PIL import Image


def main(width: int, height: int, output_path: Union[Path, str]) -> None:
    image = Image.new("RGB", (width, height))
    image.save(str(output_path))


if __name__ == '__main__':
    image_name = f'{Path(__file__).stem}.png'
    output_path = Path(Path(__file__).parent.parent.parent, 'image', image_name)
    main(width=300, height=170, output_path=output_path)
