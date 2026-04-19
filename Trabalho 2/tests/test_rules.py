import unittest

from src.game.board import Board
from src.game import rules
from src.game.config import GameConfig
from src.game.move import Move
from src.game.state import GameState
from src.utils.constants import BLACK, EMPTY, WHITE


def _build_state(
    grid: list[list[int]],
    current_player: int,
    *,
    consecutive_passes: int = 0,
) -> GameState:
    size = len(grid)
    config = GameConfig(board_size=size, allowed_board_sizes=None)
    board = Board(size=size, grid=grid)
    return GameState(
        board=board,
        current_player=current_player,
        consecutive_passes=consecutive_passes,
        config=config,
        history=[],
    )


class TestRules(unittest.TestCase):
    def test_initial_state_has_four_legal_moves_with_expected_positions(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        legal_moves = rules.get_legal_moves(state)
        legal_positions = {move.as_tuple() for move in legal_moves}

        self.assertEqual(
            legal_positions,
            {(1, 2), (2, 1), (3, 4), (4, 3)},
        )

    def test_initial_state_has_four_legal_moves(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        legal_moves = rules.get_legal_moves(state)
        self.assertEqual(len(legal_moves), 4)

    def test_apply_simple_move_flips_single_line(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))

        next_state = rules.apply_move(state, Move.from_coords(1, 2))

        self.assertEqual(next_state.board.get(1, 2), BLACK)
        self.assertEqual(next_state.board.get(2, 2), BLACK)
        self.assertEqual(next_state.current_player, WHITE)

        counts = rules.count_pieces(next_state)
        self.assertEqual(counts[BLACK], 4)
        self.assertEqual(counts[WHITE], 1)

    def test_apply_regular_move_switches_player_and_updates_history(self) -> None:
        state = GameState.initial(GameConfig(board_size=4))
        first_move = rules.get_legal_moves(state)[0]

        next_state = rules.apply_move(state, first_move)

        self.assertEqual(next_state.current_player, WHITE)
        self.assertEqual(next_state.consecutive_passes, 0)
        self.assertEqual(len(next_state.history), 1)

    def test_apply_move_can_flip_in_multiple_directions(self) -> None:
        state = _build_state(
            grid=[
                [EMPTY, WHITE, BLACK, EMPTY],
                [WHITE, WHITE, EMPTY, EMPTY],
                [BLACK, EMPTY, BLACK, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY],
            ],
            current_player=BLACK,
        )

        move = Move.from_coords(0, 0)
        self.assertTrue(rules.is_legal_move(state, move))

        next_state = rules.apply_move(state, move)

        self.assertEqual(next_state.board.get(0, 0), BLACK)
        self.assertEqual(next_state.board.get(0, 1), BLACK)
        self.assertEqual(next_state.board.get(1, 0), BLACK)
        self.assertEqual(next_state.board.get(1, 1), BLACK)

    def test_pass_is_not_legal_when_regular_moves_exist(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        self.assertFalse(rules.is_legal_move(state, Move.pass_turn()))

    def test_pass_is_required_when_player_has_no_legal_moves(self) -> None:
        state = _build_state(
            grid=[
                [BLACK, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, EMPTY],
            ],
            current_player=WHITE,
        )

        self.assertEqual(rules.get_legal_moves(state), [])
        self.assertTrue(rules.is_legal_move(state, Move.pass_turn()))
        self.assertTrue(rules.has_legal_move(state, BLACK))

        next_state = rules.apply_move(state, Move.pass_turn())
        self.assertEqual(next_state.current_player, BLACK)
        self.assertEqual(next_state.consecutive_passes, 1)

    def test_invalid_move_raises_clear_error(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))

        with self.assertRaises(ValueError) as context:
            rules.apply_move(state, Move.from_coords(0, 0))

        self.assertIn("Illegal move", str(context.exception))

    def test_initial_state_is_not_terminal(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        self.assertFalse(rules.is_terminal(state))

    def test_terminal_when_board_is_full(self) -> None:
        state = _build_state(
            grid=[
                [BLACK, BLACK, BLACK, BLACK],
                [BLACK, WHITE, WHITE, BLACK],
                [BLACK, WHITE, WHITE, BLACK],
                [BLACK, BLACK, BLACK, BLACK],
            ],
            current_player=BLACK,
        )

        self.assertTrue(rules.is_terminal(state))

    def test_terminal_when_both_players_have_no_moves(self) -> None:
        state = _build_state(
            grid=[
                [WHITE, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, EMPTY],
            ],
            current_player=BLACK,
        )

        self.assertFalse(rules.has_legal_move(state, BLACK))
        self.assertFalse(rules.has_legal_move(state, WHITE))
        self.assertTrue(rules.is_terminal(state))

    def test_terminal_when_two_consecutive_passes_reached(self) -> None:
        state = _build_state(
            grid=[
                [BLACK, WHITE, EMPTY, EMPTY],
                [EMPTY, BLACK, WHITE, EMPTY],
                [EMPTY, EMPTY, BLACK, WHITE],
                [EMPTY, EMPTY, EMPTY, BLACK],
            ],
            current_player=BLACK,
            consecutive_passes=2,
        )

        self.assertTrue(rules.is_terminal(state))

    def test_winner_is_player_with_more_discs(self) -> None:
        state = _build_state(
            grid=[
                [BLACK, BLACK, BLACK, BLACK],
                [BLACK, BLACK, WHITE, WHITE],
                [BLACK, BLACK, WHITE, WHITE],
                [BLACK, BLACK, BLACK, WHITE],
            ],
            current_player=WHITE,
        )

        self.assertEqual(rules.get_winner(state), BLACK)

    def test_winner_is_none_on_draw(self) -> None:
        state = _build_state(
            grid=[
                [BLACK, BLACK, WHITE, WHITE],
                [BLACK, BLACK, WHITE, WHITE],
                [WHITE, WHITE, BLACK, BLACK],
                [WHITE, WHITE, BLACK, BLACK],
            ],
            current_player=BLACK,
        )

        self.assertIsNone(rules.get_winner(state))


if __name__ == "__main__":
    unittest.main()
