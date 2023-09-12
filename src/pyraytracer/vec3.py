from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np
from pydantic import BaseModel


class Vec3(BaseModel):
    x: float
    y: float
    z: float

    @classmethod
    def from_tuple(
        cls,
        values: Tuple[Union[int, float], Union[int, float], Union[int, float]]
    ) -> Vec3:
        x, y, z = [float(v) for v in values]
        return cls(x=x, y=y, z=z)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def from_np_array(cls, arr: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> Vec3:
        x, y, z = arr
        return cls(x=x, y=y, z=z)

    def to_np_array(self) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        return np.array(self.to_tuple())

    def __add__(self, other: Vec3) -> Vec3:
        return self.__class__(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z
        )

    def __sub__(self, other: Vec3) -> Vec3:
        return self.__class__(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z
        )

    def __mul__(self, other: Any) -> Vec3:
        if not isinstance(other, (float, int)):
            raise ValueError(f"Unsupported type for multiplication: {type(other).__name__}")

        return self.__class__(
            x=self.x * other,
            y=self.y * other,
            z=self.z * other,
        )

    def __truediv__(self, other: Any) -> Vec3:
        if not isinstance(other, (float, int)):
            raise ValueError(f"Unsupported type for multiplication: {type(other).__name__}")

        return self.__class__(
            x=self.x / other,
            y=self.y / other,
            z=self.z / other,
        )

    def __rmul__(self, other: Any) -> Vec3:
        return self.__mul__(other)

    @classmethod
    def dot(cls, v1: Vec3, v2: Vec3) -> float:
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

    def length(self) -> float:
        return np.sqrt(self.dot(self, self))

    @classmethod
    def normalize(cls, v: Vec3) -> Vec3:
        v_len = cls.length(v)
        if v_len == 0:
            return Vec3(x=0, y=0, z=0)
        else:
            return Vec3(x=v.x/v_len, y=v.y/v_len, z=v.z/v_len)

    @classmethod
    def element_wise_product(cls, a: Vec3, b: Vec3) -> Vec3:
        return Vec3(
            x=a.x * b.x,
            y=a.y * b.y,
            z=a.z * b.z
        )

    @classmethod
    def cross(cls, a: Vec3, b: Vec3) -> Vec3:
        x = a.y * b.z - a.z * b.y
        y = a.z * b.x - a.x * b.z
        z = a.x * b.y - a.y * b.x
        return Vec3(x=x, y=y, z=z)
