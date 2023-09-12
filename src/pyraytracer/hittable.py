from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Tuple

from pydantic import BaseModel

from .material import Material
from .transform import Transform
from .vec3 import Vec3


class HitData(BaseModel):
    t: Optional[float]
    point: Optional[Vec3]
    normal: Optional[Vec3]


class Hittable(ABC):
    _no_transform_needed: Optional[bool] = None

    @abstractproperty
    def center(self) -> Vec3:
        raise NotImplementedError

    @abstractproperty
    def material(self) -> Material:
        raise NotImplementedError

    @abstractproperty
    def scale(self) -> Vec3:
        raise NotImplementedError

    @abstractproperty
    def transform(self) -> Transform:
        raise NotImplementedError

    @abstractproperty
    def rotation(self) -> Vec3:
        raise NotImplementedError

    @property
    def no_transform_needed(self) -> bool:
        if self._no_transform_needed is None:
            no_rot_no_trans_uni_scale = (
                self.transform.has_uniform_scale
                and self.transform.has_zero_rotation
                and self.transform.has_zero_translation
            )
            self._no_transform_needed = (
                self.transform.is_default
                or no_rot_no_trans_uni_scale
            )

        return self._no_transform_needed

    def get_hit_data(self, ray: Vec3, ray_origin: Vec3) -> HitData:
        ray_t, ray_origin_t = self._ray_global_to_local(
            ray=ray,
            ray_origin=ray_origin,
        )

        t: Optional[float] = self.get_t(ray=ray_t, ray_origin=ray_origin_t)

        point: Optional[Vec3] = None
        normal: Optional[Vec3] = None
        if t is not None:
            point, normal = self._point_normal_local_to_global(
                t=t,
                ray=ray_t,
                ray_origin=ray_origin_t,
            )

        return HitData(t=t, point=point, normal=normal)

    def _ray_global_to_local(
            self,
            ray: Vec3,
            ray_origin: Vec3
    ) -> Tuple[Vec3, Vec3]:

        hittable_transform: Transform = self.transform

        if self.no_transform_needed:
            return ray, ray_origin

        # First tranform ray origin to local space
        ray_origin_t = hittable_transform.global_to_local(point=ray_origin)

        # Next get the point for ray direction in global space
        ray_tip_global = ray_origin + ray

        # Now tranform this global ray tip to local
        ray_tip_local = hittable_transform.global_to_local(point=ray_tip_global)

        # Now get back the ray direction in local
        ray_t = ray_tip_local - ray_origin_t

        # Normalize ray
        ray_t = Vec3.normalize(v=ray_t)

        return ray_t, ray_origin_t

    def _point_normal_local_to_global(
            self, t: float,
            ray: Vec3,
            ray_origin: Vec3
    ) -> Tuple[Vec3, Optional[Vec3]]:

        hittable_transform: Transform = self.transform

        local_point = ray_origin + (ray * t)

        if self.material.is_constant:
            # We do not need normal calculation for constant materials
            local_normal = None
        else:
            local_normal = self.get_normal(hit_point=local_point)

        if self.no_transform_needed:
            return local_point, local_normal

        point = hittable_transform.local_to_global(point=local_point)

        if local_normal is not None:
            local_normal_tip = local_point + local_normal
            global_normal_tip = hittable_transform.local_to_global(point=local_normal_tip)
            normal = global_normal_tip - point
            normal = Vec3.normalize(v=normal)
        else:
            normal = None

        return point, normal

    @abstractmethod
    def get_t(self, ray: Vec3, ray_origin: Vec3) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def get_normal(self, hit_point: Vec3) -> Vec3:
        raise NotImplementedError
