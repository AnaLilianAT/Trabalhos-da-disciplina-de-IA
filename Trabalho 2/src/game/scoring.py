from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from src.game.board import Board
from src.utils.constants import BLACK, EMPTY, Player, WHITE

ScoringMode: TypeAlias = Literal["standard", "weighted"]

VALID_SCORING_MODES: tuple[ScoringMode, ...] = ("standard", "weighted")


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """Positional weights used by weighted scoring mode."""

    corner: int = 4
    edge: int = 2
    inner: int = 1

    def __post_init__(self) -> None:
        if self.corner <= 0 or self.edge <= 0 or self.inner <= 0:
            raise ValueError("Score weights must be positive integers.")


def validate_scoring_mode(mode: str) -> None:
    if mode not in VALID_SCORING_MODES:
        valid = ", ".join(VALID_SCORING_MODES)
        raise ValueError(f"Invalid scoring_mode '{mode}'. Valid options: {valid}.")


def position_weight(
    row: int,
    col: int,
    board_size: int,
    score_weights: ScoreWeights,
) -> int:
    """Return positional weight according to corner/edge/internal location."""
    last = board_size - 1

    if row in (0, last) and col in (0, last):
        return score_weights.corner

    if row in (0, last) or col in (0, last):
        return score_weights.edge

    return score_weights.inner


def compute_scores(
    board: Board,
    scoring_mode: ScoringMode = "standard",
    score_weights: ScoreWeights | None = None,
) -> dict[Player, int]:
    """Compute scores for BLACK and WHITE according to selected mode."""
    validate_scoring_mode(scoring_mode)
    weights = score_weights or ScoreWeights()

    black_score = 0
    white_score = 0

    for row, col in board.iter_positions():
        cell = board.get(row, col)
        if cell == EMPTY:
            continue

        if scoring_mode == "standard":
            delta = 1
        else:
            delta = position_weight(row, col, board.size, weights)

        if cell == BLACK:
            black_score += delta
        elif cell == WHITE:
            white_score += delta

    return {BLACK: black_score, WHITE: white_score}


def determine_winner_from_scores(black_score: int, white_score: int) -> Player | None:
    if black_score > white_score:
        return BLACK
    if white_score > black_score:
        return WHITE
    return None


def get_winner_for_board(
    board: Board,
    scoring_mode: ScoringMode = "standard",
    score_weights: ScoreWeights | None = None,
) -> Player | None:
    scores = compute_scores(board, scoring_mode=scoring_mode, score_weights=score_weights)
    return determine_winner_from_scores(scores[BLACK], scores[WHITE])
