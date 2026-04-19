from __future__ import annotations

from dataclasses import dataclass, field, replace

from src.game.scoring import ScoreWeights, ScoringMode, validate_scoring_mode
from src.utils.constants import (
    COMMON_BOARD_SIZES,
    DEFAULT_ALLOWED_SIZES,
    DEFAULT_BOARD_SIZE,
    MIN_BOARD_SIZE,
)


def validate_board_size(
    size: int,
    allowed_sizes: tuple[int, ...] | None = DEFAULT_ALLOWED_SIZES,
) -> None:
    """Validate board size constraints for Othello variants."""
    if not isinstance(size, int):
        raise TypeError("Board size must be an integer.")

    if size < MIN_BOARD_SIZE:
        raise ValueError(f"Board size must be >= {MIN_BOARD_SIZE}.")

    if size % 2 != 0:
        raise ValueError("Board size must be even.")

    if allowed_sizes is not None:
        for allowed_size in allowed_sizes:
            if allowed_size < MIN_BOARD_SIZE or allowed_size % 2 != 0:
                raise ValueError(
                    f"allowed_sizes must contain even values >= {MIN_BOARD_SIZE}."
                )

    if allowed_sizes is not None and size not in allowed_sizes:
        raise ValueError(
            f"Board size {size} is not in allowed sizes {allowed_sizes}."
        )


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Global game configuration used by state, rules and experiments."""

    board_size: int = DEFAULT_BOARD_SIZE
    max_consecutive_passes: int = 2
    allowed_board_sizes: tuple[int, ...] | None = DEFAULT_ALLOWED_SIZES
    scoring_mode: ScoringMode = "standard"
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)

    def __post_init__(self) -> None:
        validate_board_size(self.board_size, self.allowed_board_sizes)
        validate_scoring_mode(self.scoring_mode)

        if self.max_consecutive_passes <= 0:
            raise ValueError("max_consecutive_passes must be positive.")

    def with_board_size(self, size: int) -> "GameConfig":
        return replace(self, board_size=size)

    def with_scoring_mode(self, mode: ScoringMode) -> "GameConfig":
        return replace(self, scoring_mode=mode)

    @property
    def suggested_sizes(self) -> tuple[int, ...]:
        """Return common board sizes to use in experiments/documentation."""
        return COMMON_BOARD_SIZES
