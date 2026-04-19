import unittest

from src.agents.base import Agent
from src.experiments.runner import play_game
from src.game import rules
from src.game.board import Board
from src.game.config import GameConfig
from src.game.move import Move
from src.game.scoring import compute_scores
from src.game.state import GameState
from src.utils.constants import BLACK, WHITE


class _PassAgent(Agent):
    def choose_move(self, state: GameState) -> Move:
        return Move.pass_turn()


class TestScoringExtension(unittest.TestCase):
    def test_weighted_mode_changes_winner_for_same_terminal_board(self) -> None:
        grid = [
            [BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, WHITE],
            [WHITE, WHITE, WHITE, WHITE],
            [BLACK, WHITE, WHITE, WHITE],
        ]

        standard_state = GameState(
            board=Board(size=4, grid=[row[:] for row in grid]),
            current_player=BLACK,
            consecutive_passes=0,
            config=GameConfig(board_size=4, allowed_board_sizes=None, scoring_mode="standard"),
            history=[],
        )
        weighted_state = GameState(
            board=Board(size=4, grid=[row[:] for row in grid]),
            current_player=BLACK,
            consecutive_passes=0,
            config=GameConfig(board_size=4, allowed_board_sizes=None, scoring_mode="weighted"),
            history=[],
        )

        self.assertTrue(rules.is_terminal(standard_state))
        self.assertTrue(rules.is_terminal(weighted_state))

        self.assertIsNone(rules.get_winner(standard_state))
        self.assertEqual(rules.get_winner(weighted_state), BLACK)

        self.assertEqual(rules.get_scores(standard_state), {BLACK: 8, WHITE: 8})
        self.assertEqual(rules.get_scores(weighted_state), {BLACK: 20, WHITE: 16})

    def test_weighted_score_uses_dynamic_positions_on_8x8(self) -> None:
        grid = [[0 for _ in range(8)] for _ in range(8)]
        grid[0][0] = BLACK  # corner => 4
        grid[0][3] = WHITE  # edge => 2
        grid[3][3] = BLACK  # inner => 1
        grid[4][4] = WHITE  # inner => 1

        state = GameState(
            board=Board(size=8, grid=grid),
            current_player=BLACK,
            consecutive_passes=0,
            config=GameConfig(board_size=8, allowed_board_sizes=None, scoring_mode="weighted"),
            history=[],
        )

        self.assertEqual(rules.get_scores(state), {BLACK: 5, WHITE: 3})
        self.assertEqual(rules.score_difference(state, BLACK), 2)

    def test_compute_scores_standard_mode_matches_piece_count(self) -> None:
        grid = [
            [BLACK, WHITE, BLACK, WHITE],
            [WHITE, BLACK, WHITE, BLACK],
            [BLACK, WHITE, BLACK, WHITE],
            [WHITE, BLACK, WHITE, BLACK],
        ]
        board = Board(size=4, grid=grid)

        scores = compute_scores(board, scoring_mode="standard")
        self.assertEqual(scores[BLACK], 8)
        self.assertEqual(scores[WHITE], 8)

    def test_play_game_reports_weighted_scores(self) -> None:
        grid = [
            [BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, WHITE],
            [WHITE, WHITE, WHITE, WHITE],
            [BLACK, WHITE, WHITE, WHITE],
        ]
        weighted_state = GameState(
            board=Board(size=4, grid=[row[:] for row in grid]),
            current_player=BLACK,
            consecutive_passes=0,
            config=GameConfig(board_size=4, allowed_board_sizes=None, scoring_mode="weighted"),
            history=[],
        )

        result = play_game(
            agent_black=_PassAgent("pass-black"),
            agent_white=_PassAgent("pass-white"),
            initial_state=weighted_state,
            verbose=False,
        )

        self.assertEqual(result.scoring_mode, "weighted")
        self.assertEqual(result.black_score, 20)
        self.assertEqual(result.white_score, 16)
        self.assertEqual(result.winner, BLACK)
        self.assertEqual(result.final_score, (20, 16))


if __name__ == "__main__":
    unittest.main()
