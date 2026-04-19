from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

Coordinates: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Move:
    """Represents a regular move (row, col) or a pass move."""

    row: int | None = None
    col: int | None = None

    def __post_init__(self) -> None:
        row_is_none = self.row is None
        col_is_none = self.col is None

        if row_is_none != col_is_none:
            raise ValueError("row and col must both be defined or both be None for pass.")

        if self.row is not None and self.row < 0:
            raise ValueError("row must be non-negative.")

        if self.col is not None and self.col < 0:
            raise ValueError("col must be non-negative.")

    @property
    def is_pass(self) -> bool:
        return self.row is None

    def as_tuple(self) -> Coordinates | None:
        if self.is_pass:
            return None
        return (int(self.row), int(self.col))

    @classmethod
    def from_coords(cls, row: int, col: int) -> "Move":
        return cls(row=row, col=col)

    @classmethod
    def pass_turn(cls) -> "Move":
        return cls()
