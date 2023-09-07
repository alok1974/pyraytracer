from typing import Optional, Tuple

from pydantic import BaseModel, Field

from .color import Color
from .hittable import Hittable
from .material import Material
from .vec3 import Vec3


class Cube(BaseModel, Hittable):
    name: str
    cube_center: Vec3 = Field(alias='center')  # For internal use only
    side: float
    cube_material: Material = Field(
        default=Material(name='gray', color=Color(r=127, b=127, g=127)),
        alias='material',
    )  # For internal use only

    @property
    def center(self) -> Vec3:
        return self.cube_center

    @property
    def material(self) -> Material:
        return self.cube_material

    def get_t(self, ray: Vec3, ray_origin: Vec3) -> Optional[float]:
        t_min, t_max = self._compute_t_min_max(ray.x, ray_origin.x, self.center.x)
        if t_min is None or t_max is None:
            return None

        ty_min, ty_max = self._compute_t_min_max(ray.y, ray_origin.y, self.center.y)
        if ty_min is None or ty_max is None:
            return None

        if (t_min > ty_max) or (ty_min > t_max):
            return None

        t_min = max(t_min, ty_min)
        t_max = min(t_max, ty_max)

        tz_min, tz_max = self._compute_t_min_max(ray.z, ray_origin.z, self.center.z)
        if tz_min is None or tz_max is None:
            return None

        if (t_min > tz_max) or (tz_min > t_max):
            return None

        t_min = max(t_min, tz_min)

        if t_min < 0:
            return None

        return t_min

    def _compute_t_min_max(
            self,
            ray_value: float,
            ray_origin_value: float,
            center_value: float
    ) -> Tuple[Optional[float], Optional[float]]:

        if ray_value == 0:
            if ray_origin_value < center_value - self.side / 2 or ray_origin_value > center_value + self.side / 2:
                return None, None
            return float('-inf'), float('inf')
        t_min = (center_value - self.side / 2 - ray_origin_value) / ray_value
        t_max = (center_value + self.side / 2 - ray_origin_value) / ray_value

        if t_min > t_max:
            t_min, t_max = t_max, t_min

        return t_min, t_max

    def get_normal(self, hit_point: Vec3) -> Vec3:
        # A small value to handle potential floating-point inaccuracies
        epsilon = 0.0001

        # Check x-planes
        if abs(hit_point.x - self.center.x - self.side / 2) < epsilon:
            return Vec3(x=1, y=0, z=0)

        if abs(hit_point.x - self.center.x + self.side / 2) < epsilon:
            return Vec3(x=-1, y=0, z=0)

        # Check y-planes
        if abs(hit_point.y - self.center.y - self.side / 2) < epsilon:
            return Vec3(x=0, y=1, z=0)

        if abs(hit_point.y - self.center.y + self.side / 2) < epsilon:
            return Vec3(x=0, y=-1, z=0)

        # Check z-planes
        if abs(hit_point.z - self.center.z - self.side / 2) < epsilon:
            return Vec3(x=0, y=0, z=1)
        if abs(hit_point.z - self.center.z + self.side / 2) < epsilon:
            return Vec3(x=0, y=0, z=-1)

        # If none of the above cases match (which shouldn't happen),
        # return a default normal
        return Vec3(x=0, y=1, z=0)
