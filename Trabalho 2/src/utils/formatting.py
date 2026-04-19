from __future__ import annotations

from src.game.board import Board
from src.game.move import Move
from src.utils.constants import BLACK, EMPTY, WHITE


def format_board(board: Board) -> str:
    """Return a plain text board representation for debugging and logs."""
    header = "    " + " ".join(f"{col:2d}" for col in range(board.size))
    separator = "   +" + "--" * board.size + "-+"
    lines = [header, separator]

    for row_index, row in enumerate(board.grid):
        symbols = " ".join(_cell_symbol(cell) for cell in row)
        lines.append(f"{row_index:2d} | {symbols} |")

    lines.append(separator)
    lines.append("Legend: B=BLACK, W=WHITE, .=EMPTY")

    return "\n".join(lines)


def format_move(move: Move) -> str:
    if move.is_pass:
        return "pass"
    return f"({move.row}, {move.col})"


def _cell_symbol(value: int) -> str:
    if value == BLACK:
        return "B"
    if value == WHITE:
        return "W"
    if value == EMPTY:
        return "."
    return "?"
