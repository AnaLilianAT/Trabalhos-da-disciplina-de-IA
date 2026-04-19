from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Protocol, TypeAlias

from src.game import rules
from src.game.move import Move
from src.game.state import GameState
from src.utils.constants import BLACK, EMPTY, Player, Position, WHITE, opponent

StateEvaluationFunction: TypeAlias = Callable[[GameState, Player], float]
RolloutPolicyFunction: TypeAlias = Callable[..., Move]


@dataclass(frozen=True, slots=True)
class HeuristicWeights:
    """Weights used by composite state evaluation heuristics.

    All feature values are normalized to approximately [-1, 1], so these
    weights can be tuned directly without board-size-dependent scaling.
    """

    piece: float = 0.0
    mobility: float = 0.0
    corners: float = 0.0
    edges: float = 0.0
    corner_proximity: float = 0.0


BALANCED_WEIGHTS = HeuristicWeights(
    piece=0.30,
    mobility=0.25,
    corners=0.30,
    edges=0.10,
    corner_proximity=0.05,
)

AGGRESSIVE_WEIGHTS = HeuristicWeights(
    piece=0.10,
    mobility=0.40,
    corners=0.35,
    edges=0.05,
    corner_proximity=0.10,
)

LIGHTWEIGHT_WEIGHTS = HeuristicWeights(
    piece=0.65,
    mobility=0.35,
)


class Heuristic(Protocol):
    def evaluate(self, state: GameState, perspective_player: Player) -> float:
        """Return utility score from perspective_player point of view."""


@dataclass(frozen=True, slots=True)
class WeightedHeuristic:
    """Backward-compatible weighted heuristic based on simple features."""

    piece_count_weight: float = 1.0
    mobility_weight: float = 0.2

    def evaluate(self, state: GameState, perspective_player: Player) -> float:
        piece_score = piece_difference_feature(state, perspective_player)
        mobility_score = mobility_feature(state, perspective_player)

        return self.piece_count_weight * piece_score + self.mobility_weight * mobility_score


def piece_difference_feature(state: GameState, player: Player) -> float:
    """Normalized piece-count difference from player's perspective.

    Formula: (player_pieces - opponent_pieces) / (player_pieces + opponent_pieces)
    """
    _validate_player(player)
    counts = state.board.count_discs()
    player_pieces = counts.get(player, 0)
    opponent_pieces = counts.get(opponent(player), 0)

    return _normalized_difference(player_pieces, opponent_pieces)


def mobility_feature(state: GameState, player: Player) -> float:
    """Normalized legal-move difference from player's perspective.

    Formula: (player_moves - opponent_moves) / (player_moves + opponent_moves)
    """
    _validate_player(player)
    player_moves = len(rules.get_legal_moves_for_player(state, player))
    opponent_moves = len(rules.get_legal_moves_for_player(state, opponent(player)))

    return _normalized_difference(player_moves, opponent_moves)


def corner_control_feature(state: GameState, player: Player) -> float:
    """Corner control score normalized by number of corners.

    Formula: (player_corners - opponent_corners) / total_corners
    """
    _validate_player(player)
    corners = _corner_positions(state.board.size)

    player_corners = 0
    opponent_corners = 0
    for row, col in corners:
        cell = state.board.get(row, col)
        if cell == player:
            player_corners += 1
        elif cell == opponent(player):
            opponent_corners += 1

    return _normalized_difference(
        player_corners,
        opponent_corners,
        normalizer=len(corners),
    )


def edge_control_feature(state: GameState, player: Player) -> float:
    """Edge control score (excluding corners), normalized by edge cells.

    Formula: (player_edges - opponent_edges) / total_edge_cells_excluding_corners
    """
    _validate_player(player)
    edge_positions = _edge_positions(state.board.size)
    if not edge_positions:
        return 0.0

    player_edges = 0
    opponent_edges = 0
    for row, col in edge_positions:
        cell = state.board.get(row, col)
        if cell == player:
            player_edges += 1
        elif cell == opponent(player):
            opponent_edges += 1

    return _normalized_difference(
        player_edges,
        opponent_edges,
        normalizer=len(edge_positions),
    )


def corner_proximity_feature(state: GameState, player: Player) -> float:
    """Approximate corner-risk score around currently empty corners.

    Pieces adjacent to empty corners are often unstable. This feature is positive
    when the opponent has more such risky pieces than the reference player.
    """
    _validate_player(player)
    risk_player = 0
    risk_opponent = 0

    for corner in _corner_positions(state.board.size):
        corner_row, corner_col = corner
        if state.board.get(corner_row, corner_col) != EMPTY:
            continue

        for row, col in _adjacent_to_corner(state.board.size, corner):
            cell = state.board.get(row, col)
            if cell == player:
                risk_player += 1
            elif cell == opponent(player):
                risk_opponent += 1

    return _normalized_difference(risk_opponent, risk_player)


def evaluate_state_balanced(state: GameState, player: Player) -> float:
    """Balanced evaluation for Minimax.

    Weights:
    - piece difference: 0.30
    - mobility: 0.25
    - corner control: 0.30
    - edge control: 0.10
    - corner proximity: 0.05

    Strategy intent:
    - General-purpose heuristic for mixed openings/mid-game where no single
      feature should dominate.
    """
    return _evaluate_with_weights(state, player, BALANCED_WEIGHTS)


def evaluate_state_aggressive(state: GameState, player: Player) -> float:
    """Aggressive evaluation focused on mobility and corners.

    Weights:
    - piece difference: 0.10
    - mobility: 0.40
    - corner control: 0.35
    - edge control: 0.05
    - corner proximity: 0.10

    Strategy intent:
    - Useful when you want to prioritize forcing options and quickly taking
      strategic anchors (corners), even if piece count is temporarily worse.
    """
    return _evaluate_with_weights(state, player, AGGRESSIVE_WEIGHTS)


def evaluate_state_lightweight(state: GameState, player: Player) -> float:
    """Cheap evaluation for fast decision loops.

    Weights:
    - piece difference: 0.65
    - mobility: 0.35

    Strategy intent:
    - Fast to compute and adequate for shallow evaluations or high-throughput
      scenarios where richer features are too expensive.
    """
    return _evaluate_with_weights(state, player, LIGHTWEIGHT_WEIGHTS)


def score_move_rollout_simple(state: GameState, move: Move, player: Player) -> float:
    """Simple rollout move score: corners first, then mobility, then edges."""
    state_for_player = _state_for_player_turn(state, player)
    _validate_legal_move_for_player(state_for_player, move, player)

    if move.is_pass:
        return -1.0

    successor = rules.apply_move(state_for_player, move)
    corner_term = 1.0 if _is_corner_move(move, state_for_player.board.size) else 0.0
    mobility_term = mobility_feature(successor, player)
    edge_term = 1.0 if _is_edge_move(move, state_for_player.board.size) else 0.0

    return 0.65 * corner_term + 0.25 * mobility_term + 0.10 * edge_term


def score_move_rollout_successor_eval(
    state: GameState,
    move: Move,
    player: Player,
    heuristic_name: str = "lightweight",
) -> float:
    """Score move by evaluating its successor state with a lightweight heuristic."""
    state_for_player = _state_for_player_turn(state, player)
    _validate_legal_move_for_player(state_for_player, move, player)

    successor = rules.apply_move(state_for_player, move)
    evaluate_fn = get_evaluation_function(heuristic_name)
    return evaluate_fn(successor, player)


def rank_moves_by_heuristic(
    state: GameState,
    moves: list[Move],
    player: Player,
    heuristic_name: str = "lightweight",
) -> list[tuple[Move, float]]:
    """Rank legal moves by successor-state evaluation in descending score order."""
    if not moves:
        return []

    scored = [
        (
            move,
            score_move_rollout_successor_eval(
                state=state,
                move=move,
                player=player,
                heuristic_name=heuristic_name,
            ),
        )
        for move in moves
    ]

    return sorted(scored, key=lambda item: (-item[1], _move_sort_key(item[0])))


def choose_rollout_move_simple(
    state: GameState,
    moves: list[Move],
    player: Player,
) -> Move:
    """Choose move using simple rollout score (corner > mobility > edge)."""
    if not moves:
        return Move.pass_turn()

    scored = [
        (move, score_move_rollout_simple(state=state, move=move, player=player))
        for move in moves
    ]
    return _best_move_from_scores(scored)


def choose_rollout_move_greedy(
    state: GameState,
    moves: list[Move],
    player: Player,
    heuristic_name: str = "lightweight",
) -> Move:
    """Choose move greedily by evaluating successor states."""
    if not moves:
        return Move.pass_turn()

    ranked = rank_moves_by_heuristic(
        state=state,
        moves=moves,
        player=player,
        heuristic_name=heuristic_name,
    )
    return ranked[0][0]


def choose_rollout_move_topk(
    state: GameState,
    moves: list[Move],
    player: Player,
    k: int = 3,
    heuristic_name: str = "lightweight",
    seed: int | None = None,
    rng: random.Random | None = None,
) -> Move:
    """Choose randomly among top-k heuristic-ranked moves.

    This keeps rollout guidance lightweight while avoiding fully deterministic
    simulations.
    """
    if not moves:
        return Move.pass_turn()

    if k <= 0:
        raise ValueError("k must be positive.")

    ranked = rank_moves_by_heuristic(
        state=state,
        moves=moves,
        player=player,
        heuristic_name=heuristic_name,
    )

    top_k_count = min(k, len(ranked))
    candidates = [move for move, _score in ranked[:top_k_count]]
    if len(candidates) == 1:
        return candidates[0]

    random_source = _resolve_random_source(seed=seed, rng=rng)
    return random_source.choice(candidates)


def get_evaluation_function(name: str) -> StateEvaluationFunction:
    """Return state evaluation function by name.

    Accepted names:
    - balanced
    - aggressive
    - lightweight
    """
    key = name.strip().lower()
    if key not in _EVALUATION_FUNCTIONS:
        valid = ", ".join(sorted(_EVALUATION_FUNCTIONS))
        raise ValueError(f"Unknown evaluation heuristic '{name}'. Valid options: {valid}")

    return _EVALUATION_FUNCTIONS[key]


def get_rollout_policy_function(name: str) -> RolloutPolicyFunction:
    """Return rollout policy function by name.

    Accepted names:
    - rollout_simple
    - rollout_successor_eval
    - rollout_topk
    """
    key = name.strip().lower()
    if key not in _ROLLOUT_POLICY_FUNCTIONS:
        valid = ", ".join(sorted(_ROLLOUT_POLICY_FUNCTIONS))
        raise ValueError(f"Unknown rollout policy '{name}'. Valid options: {valid}")

    return _ROLLOUT_POLICY_FUNCTIONS[key]


def _evaluate_with_weights(
    state: GameState,
    player: Player,
    weights: HeuristicWeights,
) -> float:
    _validate_player(player)
    return (
        weights.piece * piece_difference_feature(state, player)
        + weights.mobility * mobility_feature(state, player)
        + weights.corners * corner_control_feature(state, player)
        + weights.edges * edge_control_feature(state, player)
        + weights.corner_proximity * corner_proximity_feature(state, player)
    )


def _normalized_difference(
    own_value: int,
    opponent_value: int,
    normalizer: int | None = None,
) -> float:
    if normalizer is not None:
        if normalizer <= 0:
            return 0.0
        return max(-1.0, min(1.0, (own_value - opponent_value) / float(normalizer)))

    denominator = own_value + opponent_value
    if denominator == 0:
        return 0.0

    return max(-1.0, min(1.0, (own_value - opponent_value) / float(denominator)))


def _corner_positions(board_size: int) -> tuple[Position, ...]:
    last_index = board_size - 1
    return ((0, 0), (0, last_index), (last_index, 0), (last_index, last_index))


def _edge_positions(board_size: int) -> tuple[Position, ...]:
    if board_size <= 2:
        return ()

    last_index = board_size - 1
    edge_positions: list[Position] = []

    for col in range(1, last_index):
        edge_positions.append((0, col))
        edge_positions.append((last_index, col))

    for row in range(1, last_index):
        edge_positions.append((row, 0))
        edge_positions.append((row, last_index))

    return tuple(edge_positions)


def _adjacent_to_corner(board_size: int, corner: Position) -> tuple[Position, ...]:
    row, col = corner
    row_delta = 1 if row == 0 else -1
    col_delta = 1 if col == 0 else -1

    return (
        (row, col + col_delta),
        (row + row_delta, col),
        (row + row_delta, col + col_delta),
    )


def _state_for_player_turn(state: GameState, player: Player) -> GameState:
    if state.current_player == player:
        return state

    return state.with_updates(current_player=player)


def _validate_legal_move_for_player(state: GameState, move: Move, player: Player) -> None:
    _validate_player(player)

    if not isinstance(move, Move):
        raise TypeError("move must be an instance of Move.")

    if not rules.is_legal_move(state, move):
        raise ValueError(
            f"Move {move} is not legal for player {player} in the provided state."
        )


def _is_corner_move(move: Move, board_size: int) -> bool:
    if move.is_pass:
        return False

    row = int(move.row)
    col = int(move.col)
    return (row, col) in _corner_positions(board_size)


def _is_edge_move(move: Move, board_size: int) -> bool:
    if move.is_pass:
        return False

    if _is_corner_move(move, board_size):
        return False

    row = int(move.row)
    col = int(move.col)
    last_index = board_size - 1
    return row in (0, last_index) or col in (0, last_index)


def _move_sort_key(move: Move) -> tuple[int, int]:
    if move.is_pass:
        return (10**9, 10**9)

    return (int(move.row), int(move.col))


def _best_move_from_scores(scored_moves: list[tuple[Move, float]]) -> Move:
    if not scored_moves:
        return Move.pass_turn()

    ranked = sorted(scored_moves, key=lambda item: (-item[1], _move_sort_key(item[0])))
    return ranked[0][0]


def _resolve_random_source(
    seed: int | None,
    rng: random.Random | None,
) -> random.Random:
    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both.")

    if rng is not None:
        return rng

    if seed is not None:
        return random.Random(seed)

    return random.Random()


def _validate_player(player: Player) -> None:
    if player not in (BLACK, WHITE):
        raise ValueError("player must be BLACK(1) or WHITE(-1).")


_EVALUATION_FUNCTIONS: dict[str, StateEvaluationFunction] = {
    "balanced": evaluate_state_balanced,
    "aggressive": evaluate_state_aggressive,
    "lightweight": evaluate_state_lightweight,
}


_ROLLOUT_POLICY_FUNCTIONS: dict[str, RolloutPolicyFunction] = {
    "rollout_simple": choose_rollout_move_simple,
    "rollout_successor_eval": choose_rollout_move_greedy,
    "rollout_topk": choose_rollout_move_topk,
}
