from typing import List

from .camera import Camera
from .hittable import Hittable
from .light import PointLight


class Scene:
    def __init__(self) -> None:
        self._camera: Camera
        self._lights: List[PointLight] = []
        self._hittables: List[Hittable] = []

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

    def add_lights(self, lights: List[PointLight]) -> None:
        for light in lights:
            self.add_light(light=light)

    @property
    def hittables(self) -> List[Hittable]:
        return self._hittables

    def add_hittable(self, hittable: Hittable) -> None:
        self._hittables.append(hittable)

    def add_hittables(self, hittables: List[Hittable]) -> None:
        for hittable in hittables:
            self.add_hittable(hittable=hittable)
