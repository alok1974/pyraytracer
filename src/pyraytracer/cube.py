from enum import Enum, auto, unique
from typing import Dict, Optional, Tuple

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .color import Color
from .hittable import Hittable
from .material import Material
from .transform import Transform
from .vec3 import Vec3


@unique
class CubePlanes(Enum):
    right = auto()
    left = auto()
    top = auto()
    bottom = auto()
    front = auto()
    back = auto()


class Cube(BaseModel, Hittable):
    name: str
    cube_center: Vec3 = Field(alias='center', default=Vec3(x=0, y=0, z=0))
    cube_scale: Vec3 = Field(alias='scale', default=Vec3(x=1, y=1, z=1))
    cube_rotation: Vec3 = Field(alias='rotation', default=Vec3(x=0, y=0, z=0))
    cube_material: Material = Field(
        default=Material(name='gray', color=Color(r=127, b=127, g=127)),
        alias='material',
    )  # For internal use only

    # Private fields
    _transform: Transform = PrivateAttr(default=Transform())
    _planes: Dict[CubePlanes, float] = PrivateAttr(default={})
    _epsilon: float = PrivateAttr(default=1e-6)

    @model_validator(mode='after')  # type: ignore
    def init_transform(self):
        self._transform.scale = self.scale
        self._transform.rotation = self.rotation
        self._transform.translation = self.center

    @property
    def planes(self):
        if not self._planes:
            right, left, top, bottom, front, back = self._get_plane_limits()
            self._planes = {
                CubePlanes.right: right,
                CubePlanes.left: left,
                CubePlanes.top: top,
                CubePlanes.bottom: bottom,
                CubePlanes.front: front,
                CubePlanes.back: back,
            }

        return self._planes

    @property
    def center(self) -> Vec3:
        return self.cube_center

    @property
    def scale(self) -> Vec3:
        return self.cube_scale

    @property
    def rotation(self) -> Vec3:
        return self.cube_rotation

    @property
    def material(self) -> Material:
        return self.cube_material

    @property
    def transform(self) -> Transform:
        return self._transform

    def _calculate_t(
            self,
            plane_name: CubePlanes,
            ray: Vec3,
            ray_origin: Vec3
    ) -> Optional[float]:
        plane_to_component = {
            CubePlanes.right: (ray.x, ray_origin.x),
            CubePlanes.left: (ray.x, ray_origin.x),
            CubePlanes.top: (ray.y, ray_origin.y),
            CubePlanes.bottom: (ray.y, ray_origin.y),
            CubePlanes.front: (ray.z, ray_origin.z),
            CubePlanes.back: (ray.z, ray_origin.z),
        }

        ray_component, ray_origin_component = plane_to_component[plane_name]
        if ray_component == 0:
            return None

        return (self._planes[plane_name] - ray_origin_component) / ray_component

    def get_t(self, ray: Vec3, ray_origin: Vec3) -> Optional[float]:
        # Object Space AABB implementation

        closest_t = float('inf')

        # Iterate through the planes and find intersections
        for plane_name in self.planes:
            t = self._calculate_t(
                plane_name=plane_name,
                ray=ray,
                ray_origin=ray_origin
            )

            if t is None:
                continue

            # Calculate potential intersection point using the ray's equation
            hit_point = Vec3(
                x=ray_origin.x + t * ray.x,
                y=ray_origin.y + t * ray.y,
                z=ray_origin.z + t * ray.z,
            )

            if self._hit_point_in_bounds(hit_point=hit_point):
                # Ensure positive t value and closest intersection
                if t < closest_t and t > 0:
                    closest_t = t

        return closest_t if closest_t != float('inf') else None

    def get_normal(self, hit_point: Vec3) -> Vec3:
        left = self.planes[CubePlanes.left]
        right = self.planes[CubePlanes.right]
        top = self.planes[CubePlanes.top]
        bottom = self.planes[CubePlanes.bottom]
        front = self.planes[CubePlanes.front]
        back = self.planes[CubePlanes.right]

        # Check x-planes
        if abs(hit_point.x - right) < self._epsilon:
            return Vec3(x=1, y=0, z=0)

        if abs(hit_point.x - left) < self._epsilon:
            return Vec3(x=-1, y=0, z=0)

        # Check y-planes
        if abs(hit_point.y - top) < self._epsilon:
            return Vec3(x=0, y=1, z=0)

        if abs(hit_point.y - bottom) < self._epsilon:
            return Vec3(x=0, y=-1, z=0)

        # Check z-planes
        if abs(hit_point.z - front) < self._epsilon:
            return Vec3(x=0, y=0, z=-1)

        if abs(hit_point.z - back) < self._epsilon:
            return Vec3(x=0, y=0, z=1)

        # If none of the above cases match (which shouldn't happen),
        # return a default normal
        return Vec3(x=0, y=1, z=0)

    def _get_plane_limits(
            self
    ) -> Tuple[float, float, float, float, float, float]:
        if self._no_transform_needed:
            right = self.center.x + self.scale.x / 2
            left = self.center.x - self.scale.x / 2
            top = self.center.y + self.scale.y / 2
            bottom = self.center.y - self.scale.y / 2
            front = self.center.z - self.scale.z / 2
            back = self.center.z + self.scale.z / 2
        else:
            right = 1 / 2
            left = -1 / 2
            top = 1 / 2
            bottom = -1 / 2
            front = -1 / 2
            back = 1 / 2

        return right, left, top, bottom, front, back

    def _hit_point_in_bounds(self, hit_point: Vec3) -> bool:
        e = self._epsilon
        left = self.planes[CubePlanes.left]
        right = self.planes[CubePlanes.right]
        top = self.planes[CubePlanes.top]
        bottom = self.planes[CubePlanes.bottom]
        front = self.planes[CubePlanes.front]
        back = self.planes[CubePlanes.right]

        in_x = left - e <= hit_point.x <= right + e
        in_y = bottom - e <= hit_point.y <= top + e
        in_z = front - e <= hit_point.z <= back + e

        return in_x and in_y and in_z
