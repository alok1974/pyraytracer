from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from pydantic import BaseModel, PrivateAttr

from .vec3 import Vec3


class Transformation(BaseModel):
    _scale: Vec3 = PrivateAttr(default=Vec3(x=1, y=1, z=1))
    _rotation: Vec3 = PrivateAttr(default=Vec3(x=0, y=0, z=0))
    _translation: Vec3 = PrivateAttr(default=Vec3(x=0, y=0, z=0))
    _epsilon: float = PrivateAttr(default=1e-4)
    _data: np.ndarray[Any, np.dtype[np.floating[Any]]] = PrivateAttr(default=np.eye(4))

    @classmethod
    def from_srt(cls, srt: Tuple[Vec3, Vec3, Vec3]):
        t = cls()
        t.scale = srt[0]
        t.rotation = srt[1]
        t.translation = srt[2]

        return t

    def to_srt(self) -> Tuple[Vec3, Vec3, Vec3]:
        return self.scale, self.rotation, self.translation

    @property
    def is_default(self) -> bool:
        return np.allclose(self.matrix, np.eye(4), atol=self._epsilon)

    @property
    def translation(self) -> Vec3:
        tx, ty, tz = self._data[3, 0], self._data[3, 1], self._data[3, 2]
        return Vec3(x=tx, y=ty, z=tz)

    @translation.setter
    def translation(self, value: Vec3) -> None:
        self._translation = value
        self._set_srt()

    @property
    def rotation(self) -> Vec3:
        # Reference: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf

        # Get the rotation matrix
        R = self._data.copy()[:3, :3]

        # Normalize it
        R[0, :] /= np.linalg.norm(R[0, :])
        R[1, :] /= np.linalg.norm(R[1, :])
        R[2, :] /= np.linalg.norm(R[2, :])

        # Check for gimbal lock
        if np.isclose(R[2, 0], 1.0):
            # Pitch is -90 degrees, handle gimbal lock
            r_z = 0
            r_y = -1 * np.pi / 2
            r_x = np.arctan2(-R[0, 1], -R[0, 2])
        elif np.isclose(R[2, 0], -1.0):
            # Pitch is 90 degrees, handle gimbal lock
            r_z = 0
            r_y = np.pi / 2
            r_x = np.arctan2(R[0, 1], R[0, 2])
        else:
            r_y = -1 * np.arcsin(R[2, 0])
            cos_ry = np.cos(r_y)
            r_x = np.arctan2(R[2, 1] / cos_ry, R[2, 2] / cos_ry)
            r_z = np.arctan2(R[1, 0] / cos_ry, R[0, 0] / cos_ry)

        # Convert angles from radians to degrees
        x = round(np.degrees(r_x), 5)
        y = round(np.degrees(r_y), 5)
        z = round(np.degrees(r_z), 5)

        return Vec3.from_tuple(values=(x, y, z))

    @rotation.setter
    def rotation(self, value: Vec3) -> None:
        self._rotation = value
        self._set_srt()

    @property
    def scale(self) -> Vec3:
        sx = np.linalg.norm(self._data[0, :])
        sy = np.linalg.norm(self._data[1, :])
        sz = np.linalg.norm(self._data[2, :])

        x, y, z = [float(round(v, 5)) for v in (sx, sy, sz)]
        return Vec3.from_tuple(values=(x, y, z))

    @scale.setter
    def scale(self, value: Vec3) -> None:

        if any([v < self._epsilon for v in value.to_tuple()]):
            raise ValueError(
                f'scale values {value} cannot be less than {self._epsilon}'
            )
        self._scale = value
        self._set_srt()

    @property
    def matrix(self) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        return self._data

    @property
    def inverse(self) -> Transformation:
        inverse_data = np.linalg.inv(self._data)
        t = self.__class__()
        t._data = inverse_data
        return t

    def global_to_local(self, point: Vec3) -> Vec3:
        # Tranform global ray direction to cube object space
        global_point = Transformation.from_srt(
            srt=(
                Vec3.from_tuple(values=(1, 1, 1)),  # scale
                Vec3.from_tuple(values=(1, 1, 1)),  # rotation
                Vec3.from_tuple(values=point.to_tuple()),  # translation
            )
        )
        local_point = global_point @ self.inverse

        return local_point.translation

    def local_to_global(self, point: Vec3) -> Vec3:
        # Tranform global ray direction to cube object space
        point_local = Transformation.from_srt(
            srt=(
                Vec3.from_tuple(values=(1, 1, 1)),  # scale
                Vec3.from_tuple(values=(1, 1, 1)),  # rotation
                Vec3.from_tuple(values=point.to_tuple()),  # translation
            )
        )
        point_global = point_local @ self

        return point_global.translation

    def _set_srt(self) -> None:
        self._data = np.eye(4)

        # Set Translation
        self._data[3, 0] = self._translation.x
        self._data[3, 1] = self._translation.y
        self._data[3, 2] = self._translation.z

        # Convert angles from degrees to radians
        x = np.radians(self._rotation.x)
        y = np.radians(self._rotation.y)
        z = np.radians(self._rotation.z)

        # Set Rotation
        # Reference: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
        self._data[0, 0] = np.cos(y) * np.cos(z)
        self._data[0, 1] = (
            np.sin(x) * np.sin(y) * np.cos(z)
            - np.cos(x) * np.sin(z)
        )
        self._data[0, 2] = (
            np.cos(x) * np.sin(y) * np.cos(z)
            + np.sin(x) * np.sin(z)
        )
        self._data[1, 0] = np.cos(y) * np.sin(z)
        self._data[1, 1] = (
            np.sin(x) * np.sin(y) * np.sin(z)
            + np.cos(x) * np.cos(z)
        )
        self._data[1, 2] = (
            np.cos(x) * np.sin(y) * np.sin(z)
            - np.sin(x) * np.cos(z)
        )
        self._data[2, 0] = -1 * np.sin(y)
        self._data[2, 1] = np.sin(x) * np.cos(y)
        self._data[2, 2] = np.cos(x) * np.cos(y)

        # Set Scaling
        self._data[0, :-1] *= self._scale.x
        self._data[1, :-1] *= self._scale.y
        self._data[2, :-1] *= self._scale.z

    def __matmul__(self, other: Any) -> Transformation:
        if not isinstance(other, self.__class__):
            raise ValueError(f'Expecting type {self.__class__.__name__} got {type(other)}')

        # Overloads the @ operator for matrix multiplication
        result = self.matrix @ other.matrix

        resulting_transformation = self.__class__()
        resulting_transformation._data = result

        return resulting_transformation

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            raise ValueError(f'Expecting type {self.__class__.__name__} got {type(other)}')

        return np.allclose(self.matrix, other.matrix, atol=self._epsilon)

    def __neq__(self, other: Any) -> bool:
        return self.matrix != other.matrix

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(s={self.scale}, '
            f'r={self.rotation}, t={self.translation})'
        )
