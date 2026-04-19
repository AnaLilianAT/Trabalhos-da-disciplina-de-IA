import unittest

from src.agents.minimax_agent import MinimaxAgent, MinimaxDecision
from src.game import rules
from src.game.board import Board
from src.game.config import GameConfig
from src.game.state import GameState
from src.utils.constants import BLACK, EMPTY, WHITE


def _build_state(grid: list[list[int]], current_player: int) -> GameState:
    size = len(grid)
    return GameState(
        board=Board(size=size, grid=grid),
        current_player=current_player,
        consecutive_passes=0,
        config=GameConfig(board_size=size, allowed_board_sizes=None),
        history=[],
    )


class TestMinimaxAgent(unittest.TestCase):
    def test_choose_move_returns_structured_result_and_legal_move(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        agent = MinimaxAgent(max_depth=3, use_move_ordering=True)

        decision = agent.choose_move(state)

        self.assertIsInstance(decision, MinimaxDecision)
        self.assertTrue(rules.is_legal_move(state, decision.move))
        self.assertIsInstance(decision.minimax_value, float)
        self.assertGreater(decision.metrics.nodes_expanded, 0)
        self.assertLessEqual(decision.metrics.max_depth_reached, 3)
        self.assertGreaterEqual(decision.metrics.decision_time_seconds, 0.0)

    def test_terminal_state_returns_pass(self) -> None:
        state = _build_state(
            [
                [BLACK, BLACK, BLACK, BLACK],
                [BLACK, WHITE, WHITE, BLACK],
                [BLACK, WHITE, WHITE, BLACK],
                [BLACK, BLACK, BLACK, BLACK],
            ],
            current_player=BLACK,
        )
        agent = MinimaxAgent(max_depth=4)

        decision = agent.choose_move(state)

        self.assertTrue(rules.is_terminal(state))
        self.assertTrue(decision.move.is_pass)
        self.assertGreater(decision.minimax_value, 0.0)

    def test_pass_turn_when_player_has_no_legal_moves(self) -> None:
        state = _build_state(
            [
                [BLACK, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, EMPTY],
            ],
            current_player=WHITE,
        )
        agent = MinimaxAgent(max_depth=3)

        decision = agent.choose_move(state)

        self.assertFalse(rules.is_terminal(state))
        self.assertEqual(rules.get_legal_moves(state), [])
        self.assertTrue(decision.move.is_pass)

    def test_alpha_beta_pruning_occurs(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        agent = MinimaxAgent(max_depth=4, use_move_ordering=True)

        decision = agent.choose_move(state)

        self.assertGreater(decision.metrics.alpha_beta_prunes, 0)

    def test_depth_variation_changes_search_depth(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        shallow = MinimaxAgent(max_depth=1, use_move_ordering=True)
        deep = MinimaxAgent(max_depth=3, use_move_ordering=True)

        shallow_decision = shallow.choose_move(state)
        deep_decision = deep.choose_move(state)

        self.assertLessEqual(shallow_decision.metrics.max_depth_reached, 1)
        self.assertLessEqual(deep_decision.metrics.max_depth_reached, 3)
        self.assertGreaterEqual(
            deep_decision.metrics.max_depth_reached,
            shallow_decision.metrics.max_depth_reached,
        )
        self.assertGreaterEqual(
            deep_decision.metrics.nodes_expanded,
            shallow_decision.metrics.nodes_expanded,
        )

    def test_works_for_multiple_board_sizes(self) -> None:
        for size in (4, 8):
            with self.subTest(board_size=size):
                state = GameState.initial(GameConfig(board_size=size))
                agent = MinimaxAgent(max_depth=2)

                decision = agent.choose_move(state)

                self.assertTrue(rules.is_legal_move(state, decision.move))
                self.assertLessEqual(decision.metrics.max_depth_reached, 2)


if __name__ == "__main__":
    unittest.main()
