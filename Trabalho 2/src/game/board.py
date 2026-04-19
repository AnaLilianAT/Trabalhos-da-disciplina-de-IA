from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TYPE_CHECKING

from src.utils.constants import BLACK, EMPTY, MIN_BOARD_SIZE, WHITE, Position

if TYPE_CHECKING:
    from src.game.config import GameConfig

Grid = list[list[int]]


@dataclass(slots=True)
class Board:
    """Mutable board representation; game state handles cloning when needed."""

    size: int
    grid: Grid

    def __post_init__(self) -> None:
        _validate_board_size(self.size)

        if len(self.grid) != self.size:
            raise ValueError("Board row count must match board size.")

        if any(len(row) != self.size for row in self.grid):
            raise ValueError("All board rows must have length equal to board size.")

        if any(cell not in (BLACK, WHITE, EMPTY) for row in self.grid for cell in row):
            raise ValueError("Board cells must be one of BLACK(1), WHITE(-1), EMPTY(0).")

    @classmethod
    def create_empty(cls, size: int) -> "Board":
        _validate_board_size(size)
        return cls(size=size, grid=[[EMPTY for _ in range(size)] for _ in range(size)])

    @classmethod
    def create_initial(cls, config: GameConfig) -> "Board":
        board = cls.create_empty(config.board_size)
        middle = board.size // 2

        board.grid[middle - 1][middle - 1] = WHITE
        board.grid[middle][middle] = WHITE
        board.grid[middle - 1][middle] = BLACK
        board.grid[middle][middle - 1] = BLACK

        return board

    def copy(self) -> "Board":
        return Board(size=self.size, grid=[row[:] for row in self.grid])

    def is_inside(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def get(self, row: int, col: int) -> int:
        if not self.is_inside(row, col):
            raise IndexError(f"Position out of bounds: ({row}, {col})")
        return self.grid[row][col]

    def set(self, row: int, col: int, value: int) -> None:
        if not self.is_inside(row, col):
            raise IndexError(f"Position out of bounds: ({row}, {col})")

        if value not in (BLACK, WHITE, EMPTY):
            raise ValueError("Cell value must be BLACK(1), WHITE(-1) or EMPTY(0).")

        self.grid[row][col] = value

    def iter_positions(self) -> Iterator[Position]:
        for row in range(self.size):
            for col in range(self.size):
                yield (row, col)

    def count_discs(self) -> dict[int, int]:
        counts = {BLACK: 0, WHITE: 0, EMPTY: 0}

        for row in self.grid:
            for cell in row:
                if cell not in counts:
                    counts[cell] = 0
                counts[cell] += 1

        return counts

    def count_player_discs(self, player: int) -> int:
        if player not in (BLACK, WHITE):
            raise ValueError("Player must be BLACK(1) or WHITE(-1).")

        counts = self.count_discs()
        return counts[player]


def _validate_board_size(size: int) -> None:
    if not isinstance(size, int):
        raise TypeError("Board size must be an integer.")

    if size < MIN_BOARD_SIZE:
        raise ValueError(f"Board size must be >= {MIN_BOARD_SIZE}.")

    if size % 2 != 0:
        raise ValueError("Board size must be even.")
