from typing import Optional, Tuple

from pydantic import BaseModel

from .color import Color


class Material(BaseModel):
    name: str
    color: Color
    is_constant: bool = False
    random_color_range: Optional[Tuple[float, float]] = None
    dispersion: float = 0.0
    transparency: float = 0.0
