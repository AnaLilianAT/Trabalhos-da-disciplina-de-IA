from __future__ import annotations

from typing import Final, TypeAlias

BLACK: Final[int] = 1
WHITE: Final[int] = -1
EMPTY: Final[int] = 0

Player: TypeAlias = int
CellValue: TypeAlias = int
Position: TypeAlias = tuple[int, int]

MIN_BOARD_SIZE: Final[int] = 4
DEFAULT_BOARD_SIZE: Final[int] = 6
COMMON_BOARD_SIZES: Final[tuple[int, ...]] = (4, 6, 8)
DEFAULT_ALLOWED_SIZES: Final[tuple[int, ...] | None] = None

DIRECTIONS: Final[tuple[Position, ...]] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def opponent(player: Player) -> Player:
    """Return opposite player value (1 -> -1, -1 -> 1)."""
    if player not in (BLACK, WHITE):
        raise ValueError(f"Invalid player value: {player}")
    return -player
