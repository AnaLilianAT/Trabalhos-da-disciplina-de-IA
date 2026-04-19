import unittest

from src.agents.base import Agent
from src.agents.mcts_agent import MCTSAgent
from src.agents.minimax_agent import MinimaxAgent
from src.experiments.runner import play_game
from src.game import rules
from src.game.config import GameConfig
from src.game.move import Move
from src.game.state import GameState
from src.utils.constants import BLACK, WHITE


class FirstLegalAgent(Agent):
    """Simple deterministic test agent that always chooses the first legal move."""

    def __init__(self, name: str = "FirstLegalAgent") -> None:
        super().__init__(name=name)
        self.all_moves_were_legal = True

    def choose_move(self, state: GameState) -> Move:
        legal_moves = rules.get_legal_moves(state)
        if not legal_moves:
            chosen = Move.pass_turn()
        else:
            chosen = legal_moves[0]

        self.all_moves_were_legal = self.all_moves_were_legal and rules.is_legal_move(
            state, chosen
        )
        return chosen


class TestPlayGame(unittest.TestCase):
    def test_complete_game_runs_with_valid_agents(self) -> None:
        result = play_game(
            agent_black=MinimaxAgent(name="Minimax-AB"),
            agent_white=MCTSAgent(name="MCTS"),
            game_config=GameConfig(board_size=6),
        )

        self.assertGreater(result.turns, 0)
        self.assertGreaterEqual(result.total_passes, 0)
        self.assertEqual(result.board_size, 6)
        self.assertLessEqual(result.black_discs + result.white_discs, 36)
        self.assertEqual(len(result.move_times_seconds), result.turns)
        self.assertIn(
            result.termination_reason,
            {"board_full", "consecutive_passes", "no_moves_both"},
        )

    def test_end_of_game_statistics_are_consistent(self) -> None:
        result = play_game(
            agent_black=MinimaxAgent(name="Minimax-AB"),
            agent_white=MCTSAgent(name="MCTS"),
            game_config=GameConfig(board_size=6),
        )

        self.assertGreaterEqual(result.black_discs, 0)
        self.assertGreaterEqual(result.white_discs, 0)
        self.assertLessEqual(result.black_discs + result.white_discs, 36)

        if result.termination_reason == "board_full":
            self.assertEqual(result.black_discs + result.white_discs, 36)
        else:
            self.assertLessEqual(result.black_discs + result.white_discs, 36)

        if result.black_discs > result.white_discs:
            self.assertEqual(result.winner, BLACK)
        elif result.white_discs > result.black_discs:
            self.assertEqual(result.winner, WHITE)
        else:
            self.assertIsNone(result.winner)

    def test_chosen_moves_are_legal_for_current_state(self) -> None:
        black_agent = FirstLegalAgent(name="LegalAgent-B")
        white_agent = FirstLegalAgent(name="LegalAgent-W")

        result = play_game(
            agent_black=black_agent,
            agent_white=white_agent,
            game_config=GameConfig(board_size=6),
        )

        self.assertTrue(black_agent.all_moves_were_legal)
        self.assertTrue(white_agent.all_moves_were_legal)
        self.assertLessEqual(result.black_discs + result.white_discs, 36)

    def test_supports_multiple_board_sizes(self) -> None:
        for size in (4, 8):
            with self.subTest(board_size=size):
                result = play_game(
                    agent_black=MinimaxAgent(name="Minimax-AB"),
                    agent_white=MCTSAgent(name="MCTS"),
                    game_config=GameConfig(board_size=size),
                )

                self.assertEqual(result.board_size, size)
                self.assertLessEqual(result.black_discs + result.white_discs, size * size)
                self.assertGreater(result.turns, 0)

    def test_same_seed_produces_reproducible_statistics(self) -> None:
        game_config = GameConfig(board_size=6)

        result_a = play_game(
            agent_black=MinimaxAgent(name="Minimax-AB"),
            agent_white=MCTSAgent(name="MCTS"),
            game_config=game_config,
            seed=123,
        )
        result_b = play_game(
            agent_black=MinimaxAgent(name="Minimax-AB"),
            agent_white=MCTSAgent(name="MCTS"),
            game_config=game_config,
            seed=123,
        )

        self.assertEqual(result_a.final_score, result_b.final_score)
        self.assertEqual(result_a.winner, result_b.winner)
        self.assertEqual(result_a.turns, result_b.turns)
        self.assertEqual(result_a.total_passes, result_b.total_passes)


if __name__ == "__main__":
    unittest.main()
