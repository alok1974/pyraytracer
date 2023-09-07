from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from .color import Color
from .hittable import Hittable
from .material import Material
from .vec3 import Vec3


class Sphere(BaseModel, Hittable):
    name: str
    sphere_center: Vec3 = Field(alias='center')  # For internal use only
    radius: float
    sphere_material: Material = Field(
        default=Material(name='gray', color=Color(r=127, b=127, g=127)),
        alias='material',
    )  # For internal use only

    @property
    def center(self) -> Vec3:
        return self.sphere_center

    @property
    def material(self) -> Material:
        return self.sphere_material

    def get_t(self, ray: Vec3, ray_origin: Vec3) -> Optional[float]:
        oc = ray_origin - self.center
        a = Vec3.dot(ray, ray)
        b = 2 * Vec3.dot(oc, ray)
        c = Vec3.dot(oc, oc) - (self.radius * self.radius)
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        t = (-b - np.sqrt(discriminant)) / (2 * a)

        if t < 0:
            return None
        else:
            return t

    def get_normal(self, hit_point: Vec3) -> Vec3:
        return hit_point - self.center
