from enum import Enum, auto, unique
from typing import Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .color import Color
from .hittable import Hittable
from .material import Material
from .transformation import Transformation
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
    _transform: Transformation = PrivateAttr(default=Transformation())
    _right: float = PrivateAttr()
    _left: float = PrivateAttr()
    _top: float = PrivateAttr()
    _bottom: float = PrivateAttr()
    _front: float = PrivateAttr()
    _back: float = PrivateAttr()
    _planes: Dict[CubePlanes, float] = PrivateAttr()
    _epsilon: float = PrivateAttr(default=1e-6)

    @model_validator(mode='after')  # type: ignore
    def init_transform(self):
        self._transform.scale = self.scale
        self._transform.rotation = self.rotation
        self._transform.translation = self.center

        self._right = self.scale.x / 2
        self._left = -1 * self.scale.x / 2
        self._top = self.scale.y / 2
        self._bottom = -1 * self.scale.y / 2
        self._front = -1 * self.scale.z / 2
        self._back = self.scale.z / 2

        # Planes of the cube using Vec3 representation
        self._planes = {
            CubePlanes.right: self._right,
            CubePlanes.left: self._left,
            CubePlanes.top: self._top,
            CubePlanes.bottom: self._bottom,
            CubePlanes.front: self._front,
            CubePlanes.back: self._back,
        }

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
    def transform(self) -> Transformation:
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
        for plane_name in self._planes:
            t = self._calculate_t(
                plane_name=plane_name,
                ray=ray,
                ray_origin=ray_origin
            )

            if t is None:
                continue

            # Calculate potential intersection point using the ray's equation
            hit = Vec3(
                x=ray_origin.x + t * ray.x,
                y=ray_origin.y + t * ray.y,
                z=ray_origin.z + t * ray.z,
            )

            # Check if the intersection point lies
            # within the boundaries of the cube's face
            e = self._epsilon
            in_x = self._left - e <= hit.x <= self._right + e
            in_y = self._bottom - e <= hit.y <= self._top + e
            in_z = self._front - e <= hit.z <= self._back + e

            if in_x and in_y and in_z:
                # Ensure positive t value and closest intersection
                if t < closest_t and t > 0:
                    closest_t = t

        return closest_t if closest_t != float('inf') else None

    def get_normal(self, hit_point: Vec3) -> Vec3:
        # Check x-planes
        if abs(hit_point.x - self._right) < self._epsilon:
            return Vec3(x=1, y=0, z=0)

        if abs(hit_point.x - self._left) < self._epsilon:
            return Vec3(x=-1, y=0, z=0)

        # Check y-planes
        if abs(hit_point.y - self._top) < self._epsilon:
            return Vec3(x=0, y=1, z=0)

        if abs(hit_point.y - self._bottom) < self._epsilon:
            return Vec3(x=0, y=-1, z=0)

        # Check z-planes
        if abs(hit_point.z - self._front) < self._epsilon:
            return Vec3(x=0, y=0, z=-1)

        if abs(hit_point.z - self._back) < self._epsilon:
            return Vec3(x=0, y=0, z=1)

        # If none of the above cases match (which shouldn't happen),
        # return a default normal
        return Vec3(x=0, y=1, z=0)
