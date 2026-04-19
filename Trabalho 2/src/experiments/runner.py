from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
from time import perf_counter
from typing import Literal

from src.agents.base import Agent
from src.agents.mcts_agent import MCTSAgent
from src.agents.minimax_agent import MinimaxAgent
from src.evaluation.heuristic import get_evaluation_function
from src.experiments.metrics import MatchMetrics
from src.game import rules
from src.game.config import GameConfig
from src.game.move import Move
from src.game.scoring import ScoringMode, validate_scoring_mode
from src.game.state import GameState
from src.utils.constants import BLACK, EMPTY, WHITE, opponent
from src.utils.formatting import format_board, format_move


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    max_turns: int = 1_000
    strict_legal_moves: bool = True
    verbose: bool = False
    seed: int | None = None


AgentKind = Literal["minimax", "mcts"]


@dataclass(frozen=True, slots=True)
class AgentExperimentConfig:
    """Serializable configuration used to instantiate an agent for experiments."""

    kind: AgentKind
    label: str
    minimax_depth: int | None = None
    minimax_evaluation: str | None = None
    minimax_use_move_ordering: bool = True
    mcts_num_simulations: int | None = None
    mcts_exploration_constant: float = 1.41421356237
    mcts_rollout_policy: str = "random"
    mcts_rollout_heuristic_name: str = "lightweight"
    mcts_rollout_topk: int = 3
    mcts_time_limit_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.kind == "minimax":
            if self.minimax_depth is None or self.minimax_depth <= 0:
                raise ValueError("Minimax config requires minimax_depth > 0.")
            if self.minimax_evaluation is None:
                raise ValueError("Minimax config requires minimax_evaluation.")
            return

        if self.kind == "mcts":
            if self.mcts_num_simulations is None or self.mcts_num_simulations <= 0:
                raise ValueError("MCTS config requires mcts_num_simulations > 0.")
            if self.mcts_exploration_constant < 0:
                raise ValueError("mcts_exploration_constant must be non-negative.")
            if self.mcts_rollout_topk <= 0:
                raise ValueError("mcts_rollout_topk must be positive.")
            if self.mcts_time_limit_seconds is not None and self.mcts_time_limit_seconds <= 0:
                raise ValueError("mcts_time_limit_seconds must be positive when provided.")
            return

        raise ValueError(f"Unsupported agent kind: {self.kind}")

    @staticmethod
    def minimax(
        depth: int,
        evaluation_name: str = "balanced",
        use_move_ordering: bool = True,
        label: str | None = None,
    ) -> "AgentExperimentConfig":
        resolved_label = label or f"Minimax[d={depth}|eval={evaluation_name}]"
        return AgentExperimentConfig(
            kind="minimax",
            label=resolved_label,
            minimax_depth=depth,
            minimax_evaluation=evaluation_name,
            minimax_use_move_ordering=use_move_ordering,
        )

    @staticmethod
    def mcts(
        num_simulations: int,
        exploration_constant: float = 1.41421356237,
        rollout_policy: str = "random",
        rollout_heuristic_name: str = "lightweight",
        rollout_topk: int = 3,
        time_limit_seconds: float | None = None,
        label: str | None = None,
    ) -> "AgentExperimentConfig":
        resolved_label = label or (
            "MCTS["
            f"sim={num_simulations}|"
            f"rollout={rollout_policy}|"
            f"heur={rollout_heuristic_name}|"
            f"c={exploration_constant:.3f}"
            "]"
        )
        return AgentExperimentConfig(
            kind="mcts",
            label=resolved_label,
            mcts_num_simulations=num_simulations,
            mcts_exploration_constant=exploration_constant,
            mcts_rollout_policy=rollout_policy,
            mcts_rollout_heuristic_name=rollout_heuristic_name,
            mcts_rollout_topk=rollout_topk,
            mcts_time_limit_seconds=time_limit_seconds,
        )

    def build_agent(self) -> Agent:
        if self.kind == "minimax":
            evaluation_name = self.minimax_evaluation or "balanced"
            evaluation_function = get_evaluation_function(evaluation_name)
            return MinimaxAgent(
                max_depth=int(self.minimax_depth or 4),
                evaluation_function=evaluation_function,
                use_move_ordering=self.minimax_use_move_ordering,
                name=self.label,
            )

        return MCTSAgent(
            num_simulations=int(self.mcts_num_simulations or 300),
            time_limit_seconds=self.mcts_time_limit_seconds,
            exploration_constant=self.mcts_exploration_constant,
            rollout_policy=self.mcts_rollout_policy,
            rollout_heuristic_name=self.mcts_rollout_heuristic_name,
            rollout_topk=self.mcts_rollout_topk,
            name=self.label,
        )

    def to_flat_dict(self, prefix: str) -> dict[str, object]:
        return {
            f"{prefix}_kind": self.kind,
            f"{prefix}_label": self.label,
            f"{prefix}_minimax_depth": self.minimax_depth,
            f"{prefix}_minimax_evaluation": self.minimax_evaluation,
            f"{prefix}_minimax_use_move_ordering": self.minimax_use_move_ordering,
            f"{prefix}_mcts_num_simulations": self.mcts_num_simulations,
            f"{prefix}_mcts_exploration_constant": self.mcts_exploration_constant,
            f"{prefix}_mcts_rollout_policy": self.mcts_rollout_policy,
            f"{prefix}_mcts_rollout_heuristic_name": self.mcts_rollout_heuristic_name,
            f"{prefix}_mcts_rollout_topk": self.mcts_rollout_topk,
            f"{prefix}_mcts_time_limit_seconds": self.mcts_time_limit_seconds,
        }


@dataclass(frozen=True, slots=True)
class ExperimentScenario:
    """One reproducible experiment configuration."""

    scenario_name: str
    agent_a: AgentExperimentConfig
    agent_b: AgentExperimentConfig
    board_size: int = 6
    scoring_mode: ScoringMode = "standard"
    matches: int = 8
    alternate_colors: bool = True
    base_seed: int = 42
    max_turns: int = 1_000
    strict_legal_moves: bool = True

    def __post_init__(self) -> None:
        if self.matches <= 0:
            raise ValueError("matches must be positive.")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive.")

        validate_scoring_mode(self.scoring_mode)
        # Validate board size early.
        GameConfig(
            board_size=self.board_size,
            allowed_board_sizes=None,
            scoring_mode=self.scoring_mode,
        )


@dataclass(frozen=True, slots=True)
class ExperimentMatchRecord:
    """Raw result for one match inside a scenario."""

    scenario_name: str
    match_index: int
    seed: int
    board_size: int
    scoring_mode: str
    agent_a_config: AgentExperimentConfig
    agent_b_config: AgentExperimentConfig
    agent_a_color: str
    agent_b_color: str
    winner_label: str
    turns: int
    total_passes: int
    termination_reason: str
    total_time_seconds: float
    average_move_time_seconds: float
    agent_a_move_count: int
    agent_b_move_count: int
    agent_a_total_move_time_seconds: float
    agent_b_total_move_time_seconds: float
    agent_a_score: int
    agent_b_score: int
    agent_a_discs: int
    agent_b_discs: int
    score_margin_for_a: int
    normalized_score_margin_for_a: float
    agent_a_nodes_expanded_total: int
    agent_b_nodes_expanded_total: int
    agent_a_alpha_beta_prunes_total: int
    agent_b_alpha_beta_prunes_total: int
    agent_a_mcts_simulations_total: int
    agent_b_mcts_simulations_total: int
    agent_a_mcts_nodes_created_total: int
    agent_b_mcts_nodes_created_total: int

    @property
    def agent_a_average_move_time_seconds(self) -> float:
        return _safe_ratio(self.agent_a_total_move_time_seconds, self.agent_a_move_count)

    @property
    def agent_b_average_move_time_seconds(self) -> float:
        return _safe_ratio(self.agent_b_total_move_time_seconds, self.agent_b_move_count)

    def to_csv_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "scenario_name": self.scenario_name,
            "match_index": self.match_index,
            "seed": self.seed,
            "board_size": self.board_size,
            "scoring_mode": self.scoring_mode,
            "agent_a_color": self.agent_a_color,
            "agent_b_color": self.agent_b_color,
            "winner_label": self.winner_label,
            "turns": self.turns,
            "total_passes": self.total_passes,
            "termination_reason": self.termination_reason,
            "total_time_seconds": self.total_time_seconds,
            "average_move_time_seconds": self.average_move_time_seconds,
            "agent_a_move_count": self.agent_a_move_count,
            "agent_b_move_count": self.agent_b_move_count,
            "agent_a_total_move_time_seconds": self.agent_a_total_move_time_seconds,
            "agent_b_total_move_time_seconds": self.agent_b_total_move_time_seconds,
            "agent_a_average_move_time_seconds": self.agent_a_average_move_time_seconds,
            "agent_b_average_move_time_seconds": self.agent_b_average_move_time_seconds,
            "agent_a_score": self.agent_a_score,
            "agent_b_score": self.agent_b_score,
            "agent_a_discs": self.agent_a_discs,
            "agent_b_discs": self.agent_b_discs,
            "score_margin_for_a": self.score_margin_for_a,
            "normalized_score_margin_for_a": self.normalized_score_margin_for_a,
            "agent_a_nodes_expanded_total": self.agent_a_nodes_expanded_total,
            "agent_b_nodes_expanded_total": self.agent_b_nodes_expanded_total,
            "agent_a_alpha_beta_prunes_total": self.agent_a_alpha_beta_prunes_total,
            "agent_b_alpha_beta_prunes_total": self.agent_b_alpha_beta_prunes_total,
            "agent_a_mcts_simulations_total": self.agent_a_mcts_simulations_total,
            "agent_b_mcts_simulations_total": self.agent_b_mcts_simulations_total,
            "agent_a_mcts_nodes_created_total": self.agent_a_mcts_nodes_created_total,
            "agent_b_mcts_nodes_created_total": self.agent_b_mcts_nodes_created_total,
        }
        row.update(self.agent_a_config.to_flat_dict("agent_a"))
        row.update(self.agent_b_config.to_flat_dict("agent_b"))
        return row


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    """Aggregated metrics for one scenario."""

    scenario_name: str
    board_size: int
    scoring_mode: str
    matches: int
    agent_a_config: AgentExperimentConfig
    agent_b_config: AgentExperimentConfig
    agent_a_wins: int
    agent_b_wins: int
    draws: int
    agent_a_win_rate: float
    agent_b_win_rate: float
    draw_rate: float
    avg_turns: float
    avg_total_passes: float
    avg_time_per_match_seconds: float
    avg_time_per_move_seconds: float
    agent_a_avg_move_time_seconds: float
    agent_b_avg_move_time_seconds: float
    agent_a_avg_score: float
    agent_b_avg_score: float
    avg_score_margin_for_a: float
    avg_normalized_score_margin_for_a: float
    agent_a_avg_nodes_expanded_per_move: float
    agent_b_avg_nodes_expanded_per_move: float
    agent_a_avg_alpha_beta_prunes_per_move: float
    agent_b_avg_alpha_beta_prunes_per_move: float
    agent_a_avg_mcts_simulations_per_move: float
    agent_b_avg_mcts_simulations_per_move: float
    agent_a_avg_mcts_nodes_created_per_move: float
    agent_b_avg_mcts_nodes_created_per_move: float

    def to_csv_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "scenario_name": self.scenario_name,
            "board_size": self.board_size,
            "scoring_mode": self.scoring_mode,
            "matches": self.matches,
            "agent_a_wins": self.agent_a_wins,
            "agent_b_wins": self.agent_b_wins,
            "draws": self.draws,
            "agent_a_win_rate": self.agent_a_win_rate,
            "agent_b_win_rate": self.agent_b_win_rate,
            "draw_rate": self.draw_rate,
            "avg_turns": self.avg_turns,
            "avg_total_passes": self.avg_total_passes,
            "avg_time_per_match_seconds": self.avg_time_per_match_seconds,
            "avg_time_per_move_seconds": self.avg_time_per_move_seconds,
            "agent_a_avg_move_time_seconds": self.agent_a_avg_move_time_seconds,
            "agent_b_avg_move_time_seconds": self.agent_b_avg_move_time_seconds,
            "agent_a_avg_score": self.agent_a_avg_score,
            "agent_b_avg_score": self.agent_b_avg_score,
            "avg_score_margin_for_a": self.avg_score_margin_for_a,
            "avg_normalized_score_margin_for_a": self.avg_normalized_score_margin_for_a,
            "agent_a_avg_nodes_expanded_per_move": self.agent_a_avg_nodes_expanded_per_move,
            "agent_b_avg_nodes_expanded_per_move": self.agent_b_avg_nodes_expanded_per_move,
            "agent_a_avg_alpha_beta_prunes_per_move": self.agent_a_avg_alpha_beta_prunes_per_move,
            "agent_b_avg_alpha_beta_prunes_per_move": self.agent_b_avg_alpha_beta_prunes_per_move,
            "agent_a_avg_mcts_simulations_per_move": self.agent_a_avg_mcts_simulations_per_move,
            "agent_b_avg_mcts_simulations_per_move": self.agent_b_avg_mcts_simulations_per_move,
            "agent_a_avg_mcts_nodes_created_per_move": self.agent_a_avg_mcts_nodes_created_per_move,
            "agent_b_avg_mcts_nodes_created_per_move": self.agent_b_avg_mcts_nodes_created_per_move,
        }
        row.update(self.agent_a_config.to_flat_dict("agent_a"))
        row.update(self.agent_b_config.to_flat_dict("agent_b"))
        return row


@dataclass(frozen=True, slots=True)
class ExperimentSuiteOutput:
    matches_csv_path: str
    summary_csv_path: str
    generated_plot_paths: tuple[str, ...] = ()


def play_game(
    agent_black: Agent,
    agent_white: Agent,
    initial_state: GameState | None = None,
    verbose: bool = False,
    seed: int | None = None,
    game_config: GameConfig | None = None,
) -> MatchMetrics:
    """Play a complete game between two agents and return final statistics."""
    start_state = _resolve_start_state(initial_state=initial_state, game_config=game_config)
    max_turns = _safe_turn_limit(start_state.board.size)

    return _run_game(
        black_agent=agent_black,
        white_agent=agent_white,
        state=start_state,
        max_turns=max_turns,
        strict_legal_moves=True,
        verbose=verbose,
        seed=seed,
    )


def play_match(
    black_agent: Agent,
    white_agent: Agent,
    game_config: GameConfig | None = None,
    runner_config: RunnerConfig | None = None,
) -> MatchMetrics:
    """Compatibility wrapper for play_game with extended runner settings."""
    resolved_runner_config = runner_config or RunnerConfig()

    start_state = _resolve_start_state(initial_state=None, game_config=game_config)

    return _run_game(
        black_agent=black_agent,
        white_agent=white_agent,
        state=start_state,
        max_turns=resolved_runner_config.max_turns,
        strict_legal_moves=resolved_runner_config.strict_legal_moves,
        verbose=resolved_runner_config.verbose,
        seed=resolved_runner_config.seed,
    )


def _run_game(
    black_agent: Agent,
    white_agent: Agent,
    state: GameState,
    max_turns: int,
    strict_legal_moves: bool,
    verbose: bool,
    seed: int | None,
) -> MatchMetrics:
    if seed is not None:
        random.seed(seed)

    if max_turns <= 0:
        raise ValueError("max_turns must be positive.")

    turns = 0
    total_passes = 0
    move_times: list[float] = []
    move_counts = {BLACK: 0, WHITE: 0}
    move_time_totals = {BLACK: 0.0, WHITE: 0.0}
    search_totals = {
        BLACK: {
            "nodes_expanded": 0,
            "alpha_beta_prunes": 0,
            "mcts_simulations": 0,
            "mcts_nodes_created": 0,
        },
        WHITE: {
            "nodes_expanded": 0,
            "alpha_beta_prunes": 0,
            "mcts_simulations": 0,
            "mcts_nodes_created": 0,
        },
    }
    game_start = perf_counter()

    if verbose:
        print("=== Start Game ===")
        print(format_board(state.board))

    while turns < max_turns and not rules.is_terminal(state):
        current_player = state.current_player
        current_agent = black_agent if current_player == BLACK else white_agent
        legal_moves = rules.get_legal_moves(state)

        start_move_time = perf_counter()
        chosen_move = current_agent.choose_move(state.clone())
        elapsed_seconds = perf_counter() - start_move_time
        move_times.append(elapsed_seconds)
        move_counts[current_player] += 1
        move_time_totals[current_player] += elapsed_seconds
        _accumulate_search_metrics(
            player=current_player,
            chosen_move=chosen_move,
            search_totals=search_totals,
        )

        resolved_move = _resolve_move(
            state=state,
            legal_moves=legal_moves,
            chosen_move=chosen_move,
            strict_legal_moves=strict_legal_moves,
            agent_name=current_agent.name,
        )

        if resolved_move.is_pass:
            total_passes += 1

        state = rules.apply_move(state, resolved_move)
        turns += 1

        if verbose:
            print(
                f"Turn {turns:03d} | Player={_player_label(current_player)} "
                f"| Agent={current_agent.name} | Move={format_move(resolved_move)} "
                f"| dt={elapsed_seconds:.6f}s"
            )
            print(format_board(state.board))

    if turns >= max_turns and not rules.is_terminal(state):
        raise RuntimeError(
            f"Game exceeded max_turns={max_turns}. "
            "Check agents and rule progression to avoid infinite loops."
        )

    total_time = perf_counter() - game_start
    counts = rules.count_pieces(state)
    scores = rules.get_scores(state)
    average_move_time = sum(move_times) / len(move_times) if move_times else 0.0
    termination_reason = _determine_termination_reason(state)

    return MatchMetrics(
        winner=rules.get_winner(state),
        turns=turns,
        black_discs=counts.get(BLACK, 0),
        white_discs=counts.get(WHITE, 0),
        black_score=scores.get(BLACK, 0),
        white_score=scores.get(WHITE, 0),
        scoring_mode=state.config.scoring_mode,
        total_passes=total_passes,
        consecutive_passes_at_end=state.consecutive_passes,
        termination_reason=termination_reason,
        board_size=state.board.size,
        move_times_seconds=tuple(move_times),
        total_time_seconds=total_time,
        average_time_per_move_seconds=average_move_time,
        black_move_count=move_counts[BLACK],
        white_move_count=move_counts[WHITE],
        black_total_move_time_seconds=move_time_totals[BLACK],
        white_total_move_time_seconds=move_time_totals[WHITE],
        black_nodes_expanded_total=search_totals[BLACK]["nodes_expanded"],
        white_nodes_expanded_total=search_totals[WHITE]["nodes_expanded"],
        black_alpha_beta_prunes_total=search_totals[BLACK]["alpha_beta_prunes"],
        white_alpha_beta_prunes_total=search_totals[WHITE]["alpha_beta_prunes"],
        black_mcts_simulations_total=search_totals[BLACK]["mcts_simulations"],
        white_mcts_simulations_total=search_totals[WHITE]["mcts_simulations"],
        black_mcts_nodes_created_total=search_totals[BLACK]["mcts_nodes_created"],
        white_mcts_nodes_created_total=search_totals[WHITE]["mcts_nodes_created"],
    )


def _resolve_start_state(
    initial_state: GameState | None,
    game_config: GameConfig | None,
) -> GameState:
    if initial_state is None:
        return GameState.initial(game_config or GameConfig())

    if game_config is not None and game_config.board_size != initial_state.board.size:
        raise ValueError(
            "game_config.board_size must match initial_state board size when both are provided."
        )

    return initial_state.clone()


def _resolve_move(
    state: GameState,
    legal_moves: list[Move],
    chosen_move: object,
    strict_legal_moves: bool,
    agent_name: str,
) -> Move:
    move = _extract_move(chosen_move)
    if move is None:
        raise TypeError(
            f"Agent '{agent_name}' returned unsupported move type: {type(chosen_move)}"
        )

    if rules.is_legal_move(state, move):
        return move

    if strict_legal_moves:
        legal_str = (
            "pass"
            if not legal_moves
            else ", ".join(format_move(move) for move in legal_moves)
        )
        raise ValueError(
            f"Agent '{agent_name}' chose illegal move {format_move(move)}. "
            f"Legal moves: {legal_str}."
        )

    if not legal_moves:
        return Move.pass_turn()

    return legal_moves[0]


def _extract_move(chosen_move: object) -> Move | None:
    if isinstance(chosen_move, Move):
        return chosen_move

    move_attr = getattr(chosen_move, "move", None)
    if isinstance(move_attr, Move):
        return move_attr

    return None


def _accumulate_search_metrics(
    player: int,
    chosen_move: object,
    search_totals: dict[int, dict[str, int]],
) -> None:
    metrics = getattr(chosen_move, "metrics", None)
    if metrics is None:
        return

    player_totals = search_totals[player]
    player_totals["nodes_expanded"] += _as_non_negative_int(
        getattr(metrics, "nodes_expanded", 0)
    )
    player_totals["alpha_beta_prunes"] += _as_non_negative_int(
        getattr(metrics, "alpha_beta_prunes", 0)
    )
    player_totals["mcts_simulations"] += _as_non_negative_int(
        getattr(metrics, "total_simulations", 0)
    )
    player_totals["mcts_nodes_created"] += _as_non_negative_int(
        getattr(metrics, "nodes_created", 0)
    )


def _as_non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return max(0, int(value))

    return 0


def _safe_turn_limit(board_size: int) -> int:
    # Conservative safety bound to protect automated runs.
    return board_size * board_size * 2


def _player_label(player: int) -> str:
    if player == BLACK:
        return "BLACK"
    if player == WHITE:
        return "WHITE"
    return "UNKNOWN"


def _determine_termination_reason(state: GameState) -> str:
    counts = rules.count_pieces(state)

    if counts.get(EMPTY, 0) == 0:
        return "board_full"

    if state.consecutive_passes >= state.config.max_consecutive_passes:
        return "consecutive_passes"

    current_has_moves = rules.has_legal_move(state, state.current_player)
    other_has_moves = rules.has_legal_move(state, opponent(state.current_player))
    if not current_has_moves and not other_has_moves:
        return "no_moves_both"

    return "unknown"


def run_experiment_scenario(
    scenario: ExperimentScenario,
    verbose_games: bool = False,
) -> tuple[list[ExperimentMatchRecord], ExperimentSummary]:
    """Run multiple matches for one scenario and return raw + aggregated metrics."""
    records: list[ExperimentMatchRecord] = []

    game_config = GameConfig(
        board_size=scenario.board_size,
        allowed_board_sizes=None,
        scoring_mode=scenario.scoring_mode,
    )

    for match_index in range(scenario.matches):
        seed = scenario.base_seed + match_index
        agent_a_is_black = not (scenario.alternate_colors and match_index % 2 == 1)

        black_config = scenario.agent_a if agent_a_is_black else scenario.agent_b
        white_config = scenario.agent_b if agent_a_is_black else scenario.agent_a

        black_agent = black_config.build_agent()
        white_agent = white_config.build_agent()

        match_result = play_match(
            black_agent=black_agent,
            white_agent=white_agent,
            game_config=game_config,
            runner_config=RunnerConfig(
                max_turns=scenario.max_turns,
                strict_legal_moves=scenario.strict_legal_moves,
                verbose=verbose_games,
                seed=seed,
            ),
        )

        records.append(
            _build_experiment_match_record(
                scenario=scenario,
                match_index=match_index,
                seed=seed,
                result=match_result,
                agent_a_is_black=agent_a_is_black,
            )
        )

    return records, summarize_experiment_scenario(scenario=scenario, records=records)


def summarize_experiment_scenario(
    scenario: ExperimentScenario,
    records: list[ExperimentMatchRecord],
) -> ExperimentSummary:
    if not records:
        raise ValueError("records must not be empty.")

    matches = len(records)
    agent_a_wins = sum(1 for row in records if row.winner_label == "A")
    agent_b_wins = sum(1 for row in records if row.winner_label == "B")
    draws = sum(1 for row in records if row.winner_label == "DRAW")

    total_turns = sum(row.turns for row in records)
    total_time = sum(row.total_time_seconds for row in records)
    total_a_moves = sum(row.agent_a_move_count for row in records)
    total_b_moves = sum(row.agent_b_move_count for row in records)

    total_a_move_time = sum(row.agent_a_total_move_time_seconds for row in records)
    total_b_move_time = sum(row.agent_b_total_move_time_seconds for row in records)

    total_a_nodes = sum(row.agent_a_nodes_expanded_total for row in records)
    total_b_nodes = sum(row.agent_b_nodes_expanded_total for row in records)
    total_a_prunes = sum(row.agent_a_alpha_beta_prunes_total for row in records)
    total_b_prunes = sum(row.agent_b_alpha_beta_prunes_total for row in records)
    total_a_sims = sum(row.agent_a_mcts_simulations_total for row in records)
    total_b_sims = sum(row.agent_b_mcts_simulations_total for row in records)
    total_a_mcts_nodes = sum(row.agent_a_mcts_nodes_created_total for row in records)
    total_b_mcts_nodes = sum(row.agent_b_mcts_nodes_created_total for row in records)

    return ExperimentSummary(
        scenario_name=scenario.scenario_name,
        board_size=scenario.board_size,
        scoring_mode=scenario.scoring_mode,
        matches=matches,
        agent_a_config=scenario.agent_a,
        agent_b_config=scenario.agent_b,
        agent_a_wins=agent_a_wins,
        agent_b_wins=agent_b_wins,
        draws=draws,
        agent_a_win_rate=_safe_ratio(agent_a_wins, matches),
        agent_b_win_rate=_safe_ratio(agent_b_wins, matches),
        draw_rate=_safe_ratio(draws, matches),
        avg_turns=_safe_ratio(total_turns, matches),
        avg_total_passes=_safe_ratio(sum(row.total_passes for row in records), matches),
        avg_time_per_match_seconds=_safe_ratio(total_time, matches),
        avg_time_per_move_seconds=_safe_ratio(total_time, total_turns),
        agent_a_avg_move_time_seconds=_safe_ratio(total_a_move_time, total_a_moves),
        agent_b_avg_move_time_seconds=_safe_ratio(total_b_move_time, total_b_moves),
        agent_a_avg_score=_safe_ratio(sum(row.agent_a_score for row in records), matches),
        agent_b_avg_score=_safe_ratio(sum(row.agent_b_score for row in records), matches),
        avg_score_margin_for_a=_safe_ratio(
            sum(row.score_margin_for_a for row in records),
            matches,
        ),
        avg_normalized_score_margin_for_a=_safe_ratio(
            sum(row.normalized_score_margin_for_a for row in records),
            matches,
        ),
        agent_a_avg_nodes_expanded_per_move=_safe_ratio(total_a_nodes, total_a_moves),
        agent_b_avg_nodes_expanded_per_move=_safe_ratio(total_b_nodes, total_b_moves),
        agent_a_avg_alpha_beta_prunes_per_move=_safe_ratio(total_a_prunes, total_a_moves),
        agent_b_avg_alpha_beta_prunes_per_move=_safe_ratio(total_b_prunes, total_b_moves),
        agent_a_avg_mcts_simulations_per_move=_safe_ratio(total_a_sims, total_a_moves),
        agent_b_avg_mcts_simulations_per_move=_safe_ratio(total_b_sims, total_b_moves),
        agent_a_avg_mcts_nodes_created_per_move=_safe_ratio(total_a_mcts_nodes, total_a_moves),
        agent_b_avg_mcts_nodes_created_per_move=_safe_ratio(total_b_mcts_nodes, total_b_moves),
    )


def run_experiment_suite(
    scenarios: list[ExperimentScenario],
    output_dir: str = "results/experiments",
    save_csv: bool = True,
    print_tables: bool = True,
    generate_plots: bool = False,
    verbose_games: bool = False,
) -> tuple[list[ExperimentMatchRecord], list[ExperimentSummary], ExperimentSuiteOutput | None]:
    """Run all scenarios and optionally persist CSVs/plots."""
    if not scenarios:
        raise ValueError("scenarios must not be empty.")

    all_records: list[ExperimentMatchRecord] = []
    summaries: list[ExperimentSummary] = []

    for scenario in scenarios:
        print(
            "[Experiments] Running "
            f"{scenario.scenario_name} "
            f"(board={scenario.board_size}, scoring={scenario.scoring_mode}, matches={scenario.matches})"
        )
        scenario_records, scenario_summary = run_experiment_scenario(
            scenario=scenario,
            verbose_games=verbose_games,
        )
        all_records.extend(scenario_records)
        summaries.append(scenario_summary)

    if print_tables:
        print_experiment_summary_tables(summaries)

    output: ExperimentSuiteOutput | None = None
    if save_csv:
        output = save_experiment_outputs(
            match_records=all_records,
            summaries=summaries,
            output_dir=output_dir,
            generate_plots=generate_plots,
        )
        print(f"[Experiments] Match CSV: {output.matches_csv_path}")
        print(f"[Experiments] Summary CSV: {output.summary_csv_path}")
        if output.generated_plot_paths:
            print("[Experiments] Plots:")
            for path in output.generated_plot_paths:
                print(f"  - {path}")

    return all_records, summaries, output


def save_experiment_outputs(
    match_records: list[ExperimentMatchRecord],
    summaries: list[ExperimentSummary],
    output_dir: str = "results/experiments",
    generate_plots: bool = False,
) -> ExperimentSuiteOutput:
    if not match_records:
        raise ValueError("match_records must not be empty.")
    if not summaries:
        raise ValueError("summaries must not be empty.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    match_csv_path = output_root / f"experiment_matches_{timestamp}.csv"
    summary_csv_path = output_root / f"experiment_summary_{timestamp}.csv"

    _write_csv(
        rows=[record.to_csv_row() for record in match_records],
        output_path=match_csv_path,
    )
    _write_csv(
        rows=[summary.to_csv_row() for summary in summaries],
        output_path=summary_csv_path,
    )

    plot_paths: tuple[str, ...] = ()
    if generate_plots:
        plot_paths = tuple(generate_summary_plots(summaries=summaries, output_dir=output_dir))

    return ExperimentSuiteOutput(
        matches_csv_path=str(match_csv_path),
        summary_csv_path=str(summary_csv_path),
        generated_plot_paths=plot_paths,
    )


def print_experiment_summary_tables(summaries: list[ExperimentSummary]) -> None:
    if not summaries:
        print("[Experiments] No summary rows to print.")
        return

    overview_headers = [
        "Scenario",
        "Board",
        "Mode",
        "N",
        "A win%",
        "B win%",
        "Draw%",
        "Avg move (ms)",
        "Avg game (s)",
        "Avg margin A",
    ]
    overview_rows: list[list[str]] = []
    for summary in summaries:
        overview_rows.append(
            [
                _truncate(summary.scenario_name, 42),
                str(summary.board_size),
                summary.scoring_mode,
                str(summary.matches),
                f"{summary.agent_a_win_rate * 100:.1f}",
                f"{summary.agent_b_win_rate * 100:.1f}",
                f"{summary.draw_rate * 100:.1f}",
                f"{summary.avg_time_per_move_seconds * 1_000:.3f}",
                f"{summary.avg_time_per_match_seconds:.3f}",
                f"{summary.avg_score_margin_for_a:.3f}",
            ]
        )

    cost_headers = [
        "Scenario",
        "A nodes/move",
        "A prunes/move",
        "A sims/move",
        "A mcts-nodes/move",
        "B nodes/move",
        "B prunes/move",
        "B sims/move",
        "B mcts-nodes/move",
    ]
    cost_rows: list[list[str]] = []
    for summary in summaries:
        cost_rows.append(
            [
                _truncate(summary.scenario_name, 42),
                f"{summary.agent_a_avg_nodes_expanded_per_move:.3f}",
                f"{summary.agent_a_avg_alpha_beta_prunes_per_move:.3f}",
                f"{summary.agent_a_avg_mcts_simulations_per_move:.3f}",
                f"{summary.agent_a_avg_mcts_nodes_created_per_move:.3f}",
                f"{summary.agent_b_avg_nodes_expanded_per_move:.3f}",
                f"{summary.agent_b_avg_alpha_beta_prunes_per_move:.3f}",
                f"{summary.agent_b_avg_mcts_simulations_per_move:.3f}",
                f"{summary.agent_b_avg_mcts_nodes_created_per_move:.3f}",
            ]
        )

    print("\n=== Experiment Summary (Outcomes) ===")
    print(_render_table(overview_headers, overview_rows))

    print("\n=== Experiment Summary (Search Cost) ===")
    print(_render_table(cost_headers, cost_rows))


def generate_summary_plots(
    summaries: list[ExperimentSummary],
    output_dir: str,
) -> list[str]:
    """Generate optional matplotlib charts. Returns generated file paths."""
    try:
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - depends on optional package
        print(f"[Experiments] matplotlib unavailable ({error}); skipping plots.")
        return []

    if not summaries:
        return []

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    labels = [_truncate(summary.scenario_name, 24) for summary in summaries]
    x_positions = list(range(len(summaries)))

    a_rates = [summary.agent_a_win_rate * 100 for summary in summaries]
    b_rates = [summary.agent_b_win_rate * 100 for summary in summaries]
    draw_rates = [summary.draw_rate * 100 for summary in summaries]

    width = max(11.0, len(summaries) * 0.55)
    generated_paths: list[str] = []

    plt.figure(figsize=(width, 6))
    plt.bar(x_positions, a_rates, label="Agent A wins")
    plt.bar(x_positions, draw_rates, bottom=a_rates, label="Draw")
    stacked_bottom = [a + draw for a, draw in zip(a_rates, draw_rates)]
    plt.bar(x_positions, b_rates, bottom=stacked_bottom, label="Agent B wins")
    plt.xticks(x_positions, labels, rotation=60, ha="right")
    plt.ylabel("Rate (%)")
    plt.title("Win/Draw Rate by Scenario")
    plt.legend()
    plt.tight_layout()
    win_plot_path = output_root / "plot_win_draw_rates.png"
    plt.savefig(win_plot_path)
    plt.close()
    generated_paths.append(str(win_plot_path))

    plt.figure(figsize=(width, 6))
    avg_match_time = [summary.avg_time_per_match_seconds for summary in summaries]
    avg_move_ms = [summary.avg_time_per_move_seconds * 1_000 for summary in summaries]
    plt.bar(x_positions, avg_match_time, label="Avg game time (s)")
    plt.plot(x_positions, avg_move_ms, color="tab:red", marker="o", label="Avg move time (ms)")
    plt.xticks(x_positions, labels, rotation=60, ha="right")
    plt.title("Computational Cost by Scenario")
    plt.legend()
    plt.tight_layout()
    cost_plot_path = output_root / "plot_computational_cost.png"
    plt.savefig(cost_plot_path)
    plt.close()
    generated_paths.append(str(cost_plot_path))

    plt.figure(figsize=(width, 6))
    quality_values = [summary.avg_normalized_score_margin_for_a for summary in summaries]
    plt.bar(x_positions, quality_values)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x_positions, labels, rotation=60, ha="right")
    plt.ylabel("Average normalized score margin (Agent A)")
    plt.title("Decision Quality Proxy by Scenario")
    plt.tight_layout()
    quality_plot_path = output_root / "plot_quality_margin.png"
    plt.savefig(quality_plot_path)
    plt.close()
    generated_paths.append(str(quality_plot_path))

    return generated_paths


def build_default_experiment_scenarios(
    matches_per_configuration: int = 8,
    base_seed: int = 42,
    board_sizes: tuple[int, ...] = (4, 6, 8),
    scoring_modes: tuple[ScoringMode, ...] = ("standard", "weighted"),
    alternate_colors: bool = True,
    max_turns: int = 1_000,
    strict_legal_moves: bool = True,
) -> list[ExperimentScenario]:
    """Create a complete default experiment suite for Minimax vs MCTS analysis."""
    if matches_per_configuration <= 0:
        raise ValueError("matches_per_configuration must be positive.")
    if not board_sizes:
        raise ValueError("board_sizes must not be empty.")
    if not scoring_modes:
        raise ValueError("scoring_modes must not be empty.")

    normalized_board_sizes = tuple(dict.fromkeys(board_sizes))
    normalized_scoring_modes = tuple(dict.fromkeys(scoring_modes))
    for mode in normalized_scoring_modes:
        validate_scoring_mode(mode)

    main_board_size = 6 if 6 in normalized_board_sizes else normalized_board_sizes[0]

    minimax_d3_balanced = AgentExperimentConfig.minimax(depth=3, evaluation_name="balanced")
    minimax_d4_balanced = AgentExperimentConfig.minimax(depth=4, evaluation_name="balanced")
    minimax_d5_balanced = AgentExperimentConfig.minimax(depth=5, evaluation_name="balanced")
    minimax_d4_aggressive = AgentExperimentConfig.minimax(depth=4, evaluation_name="aggressive")

    mcts_random_100 = AgentExperimentConfig.mcts(num_simulations=100, rollout_policy="random")
    mcts_random_500 = AgentExperimentConfig.mcts(num_simulations=500, rollout_policy="random")
    mcts_random_1000 = AgentExperimentConfig.mcts(num_simulations=1000, rollout_policy="random")
    mcts_rollout_simple_500 = AgentExperimentConfig.mcts(
        num_simulations=500,
        rollout_policy="rollout_simple",
    )
    mcts_rollout_successor_500 = AgentExperimentConfig.mcts(
        num_simulations=500,
        rollout_policy="rollout_successor_eval",
        rollout_heuristic_name="lightweight",
    )
    mcts_rollout_topk_500 = AgentExperimentConfig.mcts(
        num_simulations=500,
        rollout_policy="rollout_topk",
        rollout_heuristic_name="lightweight",
        rollout_topk=3,
    )
    mcts_rollout_successor_balanced = AgentExperimentConfig.mcts(
        num_simulations=500,
        rollout_policy="rollout_successor_eval",
        rollout_heuristic_name="balanced",
    )
    mcts_rollout_successor_aggressive = AgentExperimentConfig.mcts(
        num_simulations=500,
        rollout_policy="rollout_successor_eval",
        rollout_heuristic_name="aggressive",
    )

    scenarios: list[ExperimentScenario] = []
    seed_cursor = base_seed

    def add(
        name: str,
        agent_a: AgentExperimentConfig,
        agent_b: AgentExperimentConfig,
        board_size: int,
        scoring_mode: ScoringMode,
    ) -> None:
        nonlocal seed_cursor
        scenarios.append(
            ExperimentScenario(
                scenario_name=name,
                agent_a=agent_a,
                agent_b=agent_b,
                board_size=board_size,
                scoring_mode=scoring_mode,
                matches=matches_per_configuration,
                alternate_colors=alternate_colors,
                base_seed=seed_cursor,
                max_turns=max_turns,
                strict_legal_moves=strict_legal_moves,
            )
        )
        seed_cursor += 10_000

    for scoring_mode in normalized_scoring_modes:
        # Minimax(balanced) vs MCTS with multiple rollout policies.
        add(
            "minimax_balanced_vs_mcts_random",
            minimax_d4_balanced,
            mcts_random_500,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )
        add(
            "minimax_balanced_vs_mcts_rollout_simple",
            minimax_d4_balanced,
            mcts_rollout_simple_500,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )
        add(
            "minimax_balanced_vs_mcts_rollout_successor_eval",
            minimax_d4_balanced,
            mcts_rollout_successor_500,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )
        add(
            "minimax_balanced_vs_mcts_rollout_topk",
            minimax_d4_balanced,
            mcts_rollout_topk_500,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )

        # Minimax depth comparison.
        add(
            "minimax_d3_vs_d4_balanced",
            minimax_d3_balanced,
            minimax_d4_balanced,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )
        add(
            "minimax_d4_vs_d5_balanced",
            minimax_d4_balanced,
            minimax_d5_balanced,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )
        add(
            "minimax_d3_vs_d5_balanced",
            minimax_d3_balanced,
            minimax_d5_balanced,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )

        # MCTS simulation budget comparison.
        add(
            "mcts_random_100_vs_500",
            mcts_random_100,
            mcts_random_500,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )
        add(
            "mcts_random_500_vs_1000",
            mcts_random_500,
            mcts_random_1000,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )
        add(
            "mcts_random_100_vs_1000",
            mcts_random_100,
            mcts_random_1000,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )

        # Minimax heuristic influence.
        add(
            "minimax_balanced_vs_aggressive",
            minimax_d4_balanced,
            minimax_d4_aggressive,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )

        # Rollout policy comparisons.
        add(
            "mcts_random_vs_rollout_simple",
            mcts_random_500,
            mcts_rollout_simple_500,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )
        add(
            "mcts_rollout_simple_vs_rollout_successor_eval",
            mcts_rollout_simple_500,
            mcts_rollout_successor_500,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )
        add(
            "mcts_rollout_successor_eval_vs_rollout_topk",
            mcts_rollout_successor_500,
            mcts_rollout_topk_500,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )

        # Additional rollout heuristic-name comparison.
        add(
            "mcts_successor_eval_balanced_vs_aggressive",
            mcts_rollout_successor_balanced,
            mcts_rollout_successor_aggressive,
            board_size=main_board_size,
            scoring_mode=scoring_mode,
        )

    board_sweep_mode: ScoringMode = "standard"
    if "standard" not in normalized_scoring_modes:
        board_sweep_mode = normalized_scoring_modes[0]

    for size in normalized_board_sizes:
        add(
            f"board_size_sweep_{size}_minimax_balanced_vs_mcts_rollout_simple",
            minimax_d4_balanced,
            mcts_rollout_simple_500,
            board_size=size,
            scoring_mode=board_sweep_mode,
        )

    return scenarios


def build_and_run_default_experiment_suite(
    matches_per_configuration: int = 8,
    base_seed: int = 42,
    board_sizes: tuple[int, ...] = (4, 6, 8),
    scoring_modes: tuple[ScoringMode, ...] = ("standard", "weighted"),
    alternate_colors: bool = True,
    output_dir: str = "results/experiments",
    generate_plots: bool = False,
    max_turns: int = 1_000,
    strict_legal_moves: bool = True,
    verbose_games: bool = False,
) -> tuple[list[ExperimentMatchRecord], list[ExperimentSummary], ExperimentSuiteOutput | None]:
    scenarios = build_default_experiment_scenarios(
        matches_per_configuration=matches_per_configuration,
        base_seed=base_seed,
        board_sizes=board_sizes,
        scoring_modes=scoring_modes,
        alternate_colors=alternate_colors,
        max_turns=max_turns,
        strict_legal_moves=strict_legal_moves,
    )

    return run_experiment_suite(
        scenarios=scenarios,
        output_dir=output_dir,
        save_csv=True,
        print_tables=True,
        generate_plots=generate_plots,
        verbose_games=verbose_games,
    )


def _build_experiment_match_record(
    scenario: ExperimentScenario,
    match_index: int,
    seed: int,
    result: MatchMetrics,
    agent_a_is_black: bool,
) -> ExperimentMatchRecord:
    agent_a_color = BLACK if agent_a_is_black else WHITE
    agent_b_color = WHITE if agent_a_is_black else BLACK

    agent_a_score = result.black_score if agent_a_is_black else result.white_score
    agent_b_score = result.white_score if agent_a_is_black else result.black_score
    agent_a_discs = result.black_discs if agent_a_is_black else result.white_discs
    agent_b_discs = result.white_discs if agent_a_is_black else result.black_discs

    agent_a_move_count = result.black_move_count if agent_a_is_black else result.white_move_count
    agent_b_move_count = result.white_move_count if agent_a_is_black else result.black_move_count
    agent_a_total_move_time = (
        result.black_total_move_time_seconds
        if agent_a_is_black
        else result.white_total_move_time_seconds
    )
    agent_b_total_move_time = (
        result.white_total_move_time_seconds
        if agent_a_is_black
        else result.black_total_move_time_seconds
    )

    agent_a_nodes_expanded = (
        result.black_nodes_expanded_total if agent_a_is_black else result.white_nodes_expanded_total
    )
    agent_b_nodes_expanded = (
        result.white_nodes_expanded_total if agent_a_is_black else result.black_nodes_expanded_total
    )
    agent_a_prunes = (
        result.black_alpha_beta_prunes_total
        if agent_a_is_black
        else result.white_alpha_beta_prunes_total
    )
    agent_b_prunes = (
        result.white_alpha_beta_prunes_total
        if agent_a_is_black
        else result.black_alpha_beta_prunes_total
    )
    agent_a_sims = (
        result.black_mcts_simulations_total
        if agent_a_is_black
        else result.white_mcts_simulations_total
    )
    agent_b_sims = (
        result.white_mcts_simulations_total
        if agent_a_is_black
        else result.black_mcts_simulations_total
    )
    agent_a_mcts_nodes = (
        result.black_mcts_nodes_created_total
        if agent_a_is_black
        else result.white_mcts_nodes_created_total
    )
    agent_b_mcts_nodes = (
        result.white_mcts_nodes_created_total
        if agent_a_is_black
        else result.black_mcts_nodes_created_total
    )

    if result.winner is None:
        winner_label = "DRAW"
    elif result.winner == agent_a_color:
        winner_label = "A"
    elif result.winner == agent_b_color:
        winner_label = "B"
    else:
        winner_label = "DRAW"

    score_margin = agent_a_score - agent_b_score
    normalized_margin = _safe_ratio(score_margin, max(1, agent_a_score + agent_b_score))

    return ExperimentMatchRecord(
        scenario_name=scenario.scenario_name,
        match_index=match_index,
        seed=seed,
        board_size=scenario.board_size,
        scoring_mode=scenario.scoring_mode,
        agent_a_config=scenario.agent_a,
        agent_b_config=scenario.agent_b,
        agent_a_color=_player_label(agent_a_color),
        agent_b_color=_player_label(agent_b_color),
        winner_label=winner_label,
        turns=result.turns,
        total_passes=result.total_passes,
        termination_reason=result.termination_reason,
        total_time_seconds=result.total_time_seconds,
        average_move_time_seconds=result.average_time_per_move_seconds,
        agent_a_move_count=agent_a_move_count,
        agent_b_move_count=agent_b_move_count,
        agent_a_total_move_time_seconds=agent_a_total_move_time,
        agent_b_total_move_time_seconds=agent_b_total_move_time,
        agent_a_score=agent_a_score,
        agent_b_score=agent_b_score,
        agent_a_discs=agent_a_discs,
        agent_b_discs=agent_b_discs,
        score_margin_for_a=score_margin,
        normalized_score_margin_for_a=normalized_margin,
        agent_a_nodes_expanded_total=agent_a_nodes_expanded,
        agent_b_nodes_expanded_total=agent_b_nodes_expanded,
        agent_a_alpha_beta_prunes_total=agent_a_prunes,
        agent_b_alpha_beta_prunes_total=agent_b_prunes,
        agent_a_mcts_simulations_total=agent_a_sims,
        agent_b_mcts_simulations_total=agent_b_sims,
        agent_a_mcts_nodes_created_total=agent_a_mcts_nodes,
        agent_b_mcts_nodes_created_total=agent_b_mcts_nodes,
    )


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    header_row = " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    separator = "-+-".join("-" * widths[index] for index in range(len(headers)))

    rendered_rows = [header_row, separator]
    for row in rows:
        rendered_rows.append(
            " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )

    return "\n".join(rendered_rows)


def _truncate(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    if max_length <= 3:
        return value[:max_length]
    return value[: max_length - 3] + "..."


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / float(denominator)


def _parse_board_sizes(raw_value: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("board sizes must not be empty.")
    board_sizes: list[int] = []
    for token in tokens:
        board_sizes.append(int(token))
    return tuple(board_sizes)


def _parse_scoring_modes(raw_value: str) -> tuple[ScoringMode, ...]:
    tokens = [token.strip().lower() for token in raw_value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("scoring modes must not be empty.")

    parsed_modes: list[ScoringMode] = []
    for token in tokens:
        validate_scoring_mode(token)
        parsed_modes.append(token)

    return tuple(parsed_modes)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible Minimax/MCTS experiment suites for Othello/Reversi.",
    )
    parser.add_argument(
        "--matches-per-config",
        type=int,
        default=8,
        help="Number of matches per scenario configuration.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed; scenario/match seeds are derived from this value.",
    )
    parser.add_argument(
        "--board-sizes",
        type=str,
        default="4,6,8",
        help="Comma-separated board sizes for board-size sweep (e.g. 4,6,8).",
    )
    parser.add_argument(
        "--scoring-modes",
        type=str,
        default="standard,weighted",
        help="Comma-separated scoring modes (standard, weighted).",
    )
    parser.add_argument(
        "--alternate-colors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Alternate BLACK/WHITE assignment between matches to reduce color bias.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=1_000,
        help="Safety cap for turns per match.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments",
        help="Directory for generated CSV files and optional plots.",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate optional matplotlib charts.",
    )
    parser.add_argument(
        "--verbose-games",
        action="store_true",
        help="Print full board evolution for each match.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    board_sizes = _parse_board_sizes(args.board_sizes)
    scoring_modes = _parse_scoring_modes(args.scoring_modes)

    scenarios = build_default_experiment_scenarios(
        matches_per_configuration=args.matches_per_config,
        base_seed=args.base_seed,
        board_sizes=board_sizes,
        scoring_modes=scoring_modes,
        alternate_colors=args.alternate_colors,
        max_turns=args.max_turns,
        strict_legal_moves=True,
    )

    print(f"[Experiments] Total scenarios: {len(scenarios)}")
    run_experiment_suite(
        scenarios=scenarios,
        output_dir=args.output_dir,
        save_csv=True,
        print_tables=True,
        generate_plots=args.generate_plots,
        verbose_games=args.verbose_games,
    )


if __name__ == "__main__":
    main()
