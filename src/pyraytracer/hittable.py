from abc import ABC, abstractmethod, abstractproperty
from typing import Optional

from pydantic import BaseModel

from .material import Material
from .vec3 import Vec3


class HitData(BaseModel):
    t: Optional[float]
    point: Optional[Vec3]
    normal: Optional[Vec3]


class Hittable(ABC):
    @abstractproperty
    def center(self) -> Vec3:
        raise NotImplementedError

    @abstractproperty
    def material(self) -> Material:
        raise NotImplementedError

    def get_hit_data(self, ray: Vec3, ray_origin: Vec3) -> HitData:
        t: Optional[float] = self.get_t(ray=ray, ray_origin=ray_origin)

        point: Optional[Vec3] = None
        normal: Optional[Vec3] = None
        if t is not None:
            point = ray_origin + (ray * t)
            normal = self.get_normal(hit_point=point)

        return HitData(t=t, point=point, normal=normal)

    @abstractmethod
    def get_t(self, ray: Vec3, ray_origin: Vec3) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def get_normal(self, hit_point: Vec3) -> Vec3:
        raise NotImplementedError
