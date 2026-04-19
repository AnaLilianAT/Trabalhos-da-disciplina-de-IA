from __future__ import annotations

from src.game.board import Board
from src.game.move import Move
from src.game.scoring import compute_scores, determine_winner_from_scores
from src.game.state import GameState
from src.utils.constants import BLACK, DIRECTIONS, EMPTY, Player, Position, WHITE, opponent


def _collect_flips(board: Board, row: int, col: int, player: Player) -> list[Position]:
    if not board.is_inside(row, col):
        return []

    if board.get(row, col) != EMPTY:
        return []

    flips: list[Position] = []
    enemy = opponent(player)

    for delta_row, delta_col in DIRECTIONS:
        current_row = row + delta_row
        current_col = col + delta_col
        direction_flips: list[Position] = []

        while board.is_inside(current_row, current_col) and board.get(
            current_row, current_col
        ) == enemy:
            direction_flips.append((current_row, current_col))
            current_row += delta_row
            current_col += delta_col

        if (
            direction_flips
            and board.is_inside(current_row, current_col)
            and board.get(current_row, current_col) == player
        ):
            flips.extend(direction_flips)

    return flips


def get_legal_moves(state: GameState) -> list[Move]:
    return get_legal_moves_for_player(state, state.current_player)


def get_legal_moves_for_player(state: GameState, player: Player) -> list[Move]:
    _validate_player(player)
    legal_moves: list[Move] = []

    for row, col in state.board.iter_positions():
        if _collect_flips(state.board, row, col, player):
            legal_moves.append(Move.from_coords(row, col))

    return legal_moves


def has_legal_move(state: GameState, player: Player) -> bool:
    _validate_player(player)
    return bool(get_legal_moves_for_player(state, player))


def is_legal_move(state: GameState, move: Move) -> bool:
    if not isinstance(move, Move):
        raise TypeError("move must be an instance of Move.")

    if move.is_pass:
        return not has_legal_move(state, state.current_player)

    position = move.as_tuple()
    if position is None:
        return False

    row, col = position
    return bool(_collect_flips(state.board, row, col, state.current_player))


def apply_move(state: GameState, move: Move) -> GameState:
    if not isinstance(move, Move):
        raise TypeError("move must be an instance of Move.")

    if not is_legal_move(state, move):
        raise ValueError(
            f"Illegal move {move} for player {state.current_player}. "
            "Use Move.pass_turn() only when no legal moves are available."
        )

    next_player = opponent(state.current_player)

    if move.is_pass:
        return state.with_updates(
            current_player=next_player,
            consecutive_passes=state.consecutive_passes + 1,
            append_move=move,
        )

    position = move.as_tuple()
    if position is None:
        raise ValueError("Expected regular move coordinates.")

    row, col = position
    flips = _collect_flips(state.board, row, col, state.current_player)

    next_board = state.board.copy()
    next_board.set(row, col, state.current_player)

    for flip_row, flip_col in flips:
        next_board.set(flip_row, flip_col, state.current_player)

    return state.with_updates(
        board=next_board,
        current_player=next_player,
        consecutive_passes=0,
        append_move=move,
    )


def is_terminal(state: GameState) -> bool:
    if state.consecutive_passes >= state.config.max_consecutive_passes:
        return True

    counts = state.board.count_discs()
    if counts.get(EMPTY, 0) == 0:
        return True

    if has_legal_move(state, state.current_player):
        return False

    return not has_legal_move(state, opponent(state.current_player))


def count_pieces(state: GameState) -> dict[int, int]:
    """Return counts for BLACK, WHITE and EMPTY cells."""
    return state.board.count_discs()


def get_scores(state: GameState) -> dict[int, int]:
    """Return BLACK/WHITE scores according to configured scoring mode."""
    return compute_scores(
        state.board,
        scoring_mode=state.config.scoring_mode,
        score_weights=state.config.score_weights,
    )


def get_winner(state: GameState) -> Player | None:
    scores = get_scores(state)
    return determine_winner_from_scores(scores.get(BLACK, 0), scores.get(WHITE, 0))


def score_difference(state: GameState, perspective_player: Player) -> int:
    _validate_player(perspective_player)
    scores = get_scores(state)
    return scores.get(perspective_player, 0) - scores.get(opponent(perspective_player), 0)


def _validate_player(player: Player) -> None:
    if player not in (BLACK, WHITE):
        raise ValueError("player must be BLACK(1) or WHITE(-1).")
