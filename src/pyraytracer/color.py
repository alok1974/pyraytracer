from __future__ import annotations

from typing import Any, Tuple

from pydantic import BaseModel


class Color(BaseModel):
    r: int
    g: int
    b: int

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)

    def __add__(self, other: Any) -> Color:
        if isinstance(other, int):
            return self.__class__(
                r=self.r + other,
                g=self.g + other,
                b=self.b + other,
            )
        elif isinstance(other, Color):
            return self.__class__(
                r=self.r + other.r,
                g=self.g + other.g,
                b=self.b + other.b,
            )
        else:
            raise ValueError(f"Unsupported type for multiplication: {type(other).__name__}")

    def __sub__(self, other: Any) -> Color:
        if isinstance(other, int):
            return self.__class__(
                r=self.r - other,
                g=self.g - other,
                b=self.b - other,
            )
        elif isinstance(other, Color):
            return self.__class__(
                r=self.r - other.r,
                g=self.g - other.g,
                b=self.b - other.b,
            )
        else:
            raise ValueError(f"Unsupported type for multiplication: {type(other).__name__}")

    def __mul__(self, other: Any) -> Color:
        if isinstance(other, (float, int)):
            return self.__class__(
                r=int(self.r * other),
                g=int(self.g * other),
                b=int(self.b * other),
            )
        elif isinstance(other, Color):
            # Element-wise product
            return self.__class__(
                r=int(self.r * other.r),
                g=int(self.g * other.g),
                b=int(self.b * other.b),
            )
        else:
            raise ValueError(f"Unsupported type for multiplication: {type(other).__name__}")

    def __rmul__(self, other: Any) -> Color:
        return self.__mul__(other)
