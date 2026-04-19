import unittest

from src.agents.mcts_agent import MCTSAgent, MCTSDecision
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


class TestMCTSAgent(unittest.TestCase):
    def test_tree_construction_basic(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        agent = MCTSAgent(num_simulations=20, rollout_policy="random")

        decision = agent.choose_move(state)

        self.assertIsInstance(decision, MCTSDecision)
        self.assertIsNotNone(agent.last_root)
        self.assertEqual(agent.last_root.visits, decision.metrics.total_simulations)
        self.assertGreater(len(agent.last_root.children), 0)
        self.assertGreater(decision.metrics.nodes_created, 1)

    def test_choose_move_returns_legal_move(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        agent = MCTSAgent(num_simulations=30, rollout_policy="random")

        decision = agent.choose_move(state)

        self.assertTrue(rules.is_legal_move(state, decision.move))
        self.assertEqual(decision.metrics.total_simulations, 30)

    def test_simulations_execute_without_error(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        agent = MCTSAgent(num_simulations=15, rollout_policy="random")

        decision = agent.choose_move(state)

        self.assertGreaterEqual(decision.metrics.average_rollout_depth, 0.0)
        self.assertGreater(decision.metrics.decision_time_seconds, 0.0)
        self.assertEqual(decision.metrics.rollout_policy, "random")

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
        agent = MCTSAgent(num_simulations=10, rollout_policy="random")

        decision = agent.choose_move(state)

        self.assertTrue(rules.is_terminal(state))
        self.assertTrue(decision.move.is_pass)
        self.assertEqual(decision.metrics.total_simulations, 0)

    def test_pass_is_handled_when_player_has_no_regular_move(self) -> None:
        state = _build_state(
            [
                [BLACK, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, EMPTY],
            ],
            current_player=WHITE,
        )
        agent = MCTSAgent(num_simulations=20, rollout_policy="random")

        decision = agent.choose_move(state)

        self.assertEqual(rules.get_legal_moves(state), [])
        self.assertTrue(decision.move.is_pass)

    def test_supports_multiple_board_sizes(self) -> None:
        for size in (4, 8):
            with self.subTest(board_size=size):
                state = GameState.initial(GameConfig(board_size=size))
                agent = MCTSAgent(num_simulations=12, rollout_policy="random")

                decision = agent.choose_move(state)

                self.assertTrue(rules.is_legal_move(state, decision.move))
                self.assertEqual(decision.metrics.total_simulations, 12)

    def test_rollout_random_runs(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        agent = MCTSAgent(num_simulations=1, rollout_policy="random")

        winner, depth = agent.rollout_random(state)

        self.assertIn(winner, (BLACK, WHITE, None))
        self.assertGreaterEqual(depth, 0)

    def test_heuristic_rollout_policies_run(self) -> None:
        for policy in ("rollout_simple", "rollout_successor_eval", "rollout_topk"):
            with self.subTest(rollout_policy=policy):
                state = GameState.initial(GameConfig(board_size=6))
                agent = MCTSAgent(num_simulations=10, rollout_policy=policy)

                decision = agent.choose_move(state)

                self.assertTrue(rules.is_legal_move(state, decision.move))
                self.assertEqual(decision.metrics.rollout_policy, policy)
                self.assertEqual(decision.metrics.total_simulations, 10)

    def test_rollout_policy_can_be_switched_by_parameter(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        random_agent = MCTSAgent(num_simulations=8, rollout_policy="random")
        heuristic_agent = MCTSAgent(num_simulations=8, rollout_policy="rollout_simple")

        random_decision = random_agent.choose_move(state)
        heuristic_decision = heuristic_agent.choose_move(state)

        self.assertEqual(random_decision.metrics.rollout_policy, "random")
        self.assertEqual(heuristic_decision.metrics.rollout_policy, "rollout_simple")
        self.assertTrue(rules.is_legal_move(state, random_decision.move))
        self.assertTrue(rules.is_legal_move(state, heuristic_decision.move))

    def test_time_limit_mode_runs_at_least_one_simulation(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        agent = MCTSAgent(
            num_simulations=10_000,
            time_limit_seconds=0.02,
            rollout_policy="random",
        )

        decision = agent.choose_move(state)

        self.assertGreater(decision.metrics.total_simulations, 0)
        self.assertTrue(rules.is_legal_move(state, decision.move))


if __name__ == "__main__":
    unittest.main()
