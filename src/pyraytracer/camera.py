import numpy as np
from pydantic import BaseModel, PrivateAttr, model_validator

from .vec3 import Vec3


class Camera(BaseModel):
    name: str
    center: Vec3 = Vec3(x=0, y=0, z=-1)
    fov: float = 0.26
    roll: float = 0
    look_at: Vec3 = Vec3(x=0, y=0, z=1)
    _up: Vec3 = PrivateAttr(Vec3(x=0, y=1, z=0))
    _forward: Vec3 = PrivateAttr()
    _right: Vec3 = PrivateAttr()
    _upward: Vec3 = PrivateAttr()

    @property
    def forward(self) -> Vec3:
        return self._forward

    @property
    def right(self) -> Vec3:
        return self._right

    @property
    def upward(self) -> Vec3:
        return self._upward

    @model_validator(mode='after')  # type: ignore
    def _calulate_base_vectors(self) -> None:
        self._forward = Vec3.normalize(v=self.look_at - self.center)

        # Rotate the up vector based on the roll angle
        rotated_up = self._rotate_vector_around_axis(
            vector=self._up,
            axis=self._forward,
            angle=self.roll
        )

        self._right = Vec3.normalize(v=Vec3.cross(a=self._forward, b=rotated_up))
        self._upward = Vec3.normalize(v=Vec3.cross(a=self._right, b=self._forward))

    def _rotate_vector_around_axis(self, vector: Vec3, axis: Vec3, angle: float) -> Vec3:
        angle_rad = np.deg2rad(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        dot_product = Vec3.dot(v1=vector, v2=axis)

        rotated_vector = (
            vector * cos_angle
            + Vec3.cross(a=axis, b=vector) * sin_angle
            + axis * (dot_product * (1.0 - cos_angle))
        )
        return rotated_vector
