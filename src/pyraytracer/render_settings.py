from pathlib import Path
from typing import Union

from pydantic import BaseModel

from .color import Color


class RenderSettings(BaseModel):
    # Image Settings
    output_path: Union[str, Path]
    image_width: int
    image_height: int
    aa_samples: float = 1.0
    bg_color: Color = Color(r=10, g=10, b=10)

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
    tail_noise_width: float = 24
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
