from __future__ import annotations

from dataclasses import dataclass

from src.utils.constants import BLACK, WHITE


@dataclass(frozen=True, slots=True)
class MatchMetrics:
    winner: int | None
    turns: int
    black_discs: int
    white_discs: int
    black_score: int
    white_score: int
    scoring_mode: str
    total_passes: int
    consecutive_passes_at_end: int
    termination_reason: str
    board_size: int
    move_times_seconds: tuple[float, ...] = ()
    total_time_seconds: float = 0.0
    average_time_per_move_seconds: float = 0.0
    black_move_count: int = 0
    white_move_count: int = 0
    black_total_move_time_seconds: float = 0.0
    white_total_move_time_seconds: float = 0.0
    black_nodes_expanded_total: int = 0
    white_nodes_expanded_total: int = 0
    black_alpha_beta_prunes_total: int = 0
    white_alpha_beta_prunes_total: int = 0
    black_mcts_simulations_total: int = 0
    white_mcts_simulations_total: int = 0
    black_mcts_nodes_created_total: int = 0
    white_mcts_nodes_created_total: int = 0

    @property
    def final_score(self) -> tuple[int, int]:
        return (self.black_score, self.white_score)

    @property
    def black_average_move_time_seconds(self) -> float:
        if self.black_move_count == 0:
            return 0.0
        return self.black_total_move_time_seconds / float(self.black_move_count)

    @property
    def white_average_move_time_seconds(self) -> float:
        if self.white_move_count == 0:
            return 0.0
        return self.white_total_move_time_seconds / float(self.white_move_count)


@dataclass(slots=True)
class TournamentMetrics:
    total_matches: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0

    def add_match(self, result: MatchMetrics) -> None:
        self.total_matches += 1

        if result.winner == BLACK:
            self.black_wins += 1
        elif result.winner == WHITE:
            self.white_wins += 1
        else:
            self.draws += 1
