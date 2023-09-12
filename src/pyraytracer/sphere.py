from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .color import Color
from .hittable import Hittable
from .material import Material
from .transform import Transform
from .vec3 import Vec3


class Sphere(BaseModel, Hittable):
    name: str
    sphere_center: Vec3 = Field(alias='center', default=Vec3(x=0, y=0, z=0))
    sphere_scale: Vec3 = Field(alias='scale', default=Vec3(x=1, y=1, z=1))
    sphere_rotation: Vec3 = Field(alias='rotation', default=Vec3(x=0, y=0, z=0))
    sphere_material: Material = Field(
        default=Material(name='gray', color=Color(r=127, b=127, g=127)),
        alias='material',
    )  # For internal use only

    _transform: Transform = PrivateAttr(default=Transform())
    _radius_and_center: Tuple[Optional[float], Optional[Vec3]] = PrivateAttr(
        default=(
            None,
            None
        ),
    )

    @property
    def center(self) -> Vec3:
        return self.sphere_center

    @property
    def scale(self) -> Vec3:
        return self.sphere_scale

    @property
    def rotation(self) -> Vec3:
        return self.sphere_rotation

    @property
    def material(self) -> Material:
        return self.sphere_material

    @property
    def transform(self) -> Transform:
        return self._transform

    @property
    def radius_and_center(self):
        radius, center = self._radius_and_center
        if radius is None or center is None:
            self._radius_and_center = self._get_radius_center()

        return self._radius_and_center

    @model_validator(mode='after')  # type: ignore
    def init_transform(self):
        self._transform.scale = self.scale
        self._transform.translation = self.center
        self._transform.rotation = self.rotation

    def get_t(self, ray: Vec3, ray_origin: Vec3) -> Optional[float]:
        radius, center = self.radius_and_center
        oc = ray_origin - center
        a = Vec3.dot(ray, ray)
        b = 2 * Vec3.dot(oc, ray)
        c = Vec3.dot(oc, oc) - (radius * radius)
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        t = (-b - np.sqrt(discriminant)) / (2 * a)

        if t < 0:
            return None
        else:
            return t

    def get_normal(self, hit_point: Vec3) -> Vec3:
        _, center = self.radius_and_center
        normal = hit_point - center
        inv_scale = Vec3(
            x=1 / self.scale.x,
            y=1 / self.scale.y,
            z=1 / self.scale.z,
        )
        adjusted_normal = Vec3.element_wise_product(a=normal, b=inv_scale)

        return Vec3.normalize(v=adjusted_normal)

    def _get_radius_center(self) -> Tuple[float, Vec3]:
        if self._no_transform_needed:
            radius = self.transform.scale.x
            center = self.center
        else:
            radius = 1.0
            center = Vec3(x=0, y=0, z=0)

        return radius, center
