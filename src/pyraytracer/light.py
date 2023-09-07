from typing import List

from pydantic import BaseModel

from .color import Color
from .hittable import Hittable
from .vec3 import Vec3


class PointLight(BaseModel):
    name: str
    center: Vec3
    color: Color = Color(r=0, b=0, g=0)
    intensity: float = 1.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # Init pydanctic BaseModel
        self._includes: List[int] = []
        self._excludes: List[int] = []

    def add_include(self, hittable: Hittable) -> None:
        if id(hittable) in self._excludes:
            raise RuntimeError(f'This object {hittable} is already added to excludes.')

        self._includes.append(id(hittable))

    def add_includes(self, hittables: List[Hittable]) -> None:
        for hittable in hittables:
            self.add_include(hittable=hittable)

    def is_included(self, hittable: Hittable) -> bool:
        if not self._includes and not self.is_excluded(hittable=hittable):
            return True

        return id(hittable) in self._includes

    def add_exclude(self, hittable: Hittable) -> None:
        if id(hittable) in self._includes:
            raise RuntimeError(f'This object {hittable} is already added to the includes.')

        self._excludes.append(id(hittable))

    def add_excludes(self, hittables: List[Hittable]) -> None:
        for hittable in hittables:
            self.add_exclude(hittable=hittable)

    def is_excluded(self, hittable: Hittable) -> bool:
        return id(hittable) in self._excludes
