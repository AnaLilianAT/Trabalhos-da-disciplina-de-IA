import unittest

from src.evaluation.heuristic import (
    choose_rollout_move_greedy,
    choose_rollout_move_simple,
    choose_rollout_move_topk,
    corner_control_feature,
    corner_proximity_feature,
    edge_control_feature,
    evaluate_state_aggressive,
    evaluate_state_balanced,
    evaluate_state_lightweight,
    get_evaluation_function,
    get_rollout_policy_function,
    mobility_feature,
    piece_difference_feature,
    rank_moves_by_heuristic,
    score_move_rollout_simple,
)
from src.game import rules
from src.game.board import Board
from src.game.config import GameConfig
from src.game.move import Move
from src.game.state import GameState
from src.utils.constants import BLACK, EMPTY, WHITE


def _build_state(grid: list[list[int]], current_player: int = BLACK) -> GameState:
    size = len(grid)
    return GameState(
        board=Board(size=size, grid=grid),
        current_player=current_player,
        consecutive_passes=0,
        config=GameConfig(board_size=size, allowed_board_sizes=None),
        history=[],
    )


class TestHeuristicFeatures(unittest.TestCase):
    def test_piece_difference_feature(self) -> None:
        state = _build_state(
            [
                [BLACK, BLACK, WHITE, EMPTY],
                [BLACK, BLACK, EMPTY, EMPTY],
                [WHITE, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY],
            ]
        )

        self.assertAlmostEqual(piece_difference_feature(state, BLACK), 1.0 / 3.0)
        self.assertAlmostEqual(piece_difference_feature(state, WHITE), -1.0 / 3.0)

    def test_mobility_feature_symmetric_on_initial_states(self) -> None:
        for size in (4, 8):
            with self.subTest(board_size=size):
                state = GameState.initial(GameConfig(board_size=size))
                self.assertAlmostEqual(mobility_feature(state, BLACK), 0.0)
                self.assertAlmostEqual(mobility_feature(state, WHITE), 0.0)

    def test_corner_control_feature_dynamic(self) -> None:
        state = _build_state(
            [
                [BLACK, EMPTY, EMPTY, WHITE],
                [EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, BLACK],
            ]
        )

        self.assertAlmostEqual(corner_control_feature(state, BLACK), 0.25)
        self.assertAlmostEqual(corner_control_feature(state, WHITE), -0.25)

    def test_edge_control_feature_dynamic(self) -> None:
        state = _build_state(
            [
                [EMPTY, BLACK, BLACK, EMPTY],
                [BLACK, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, WHITE],
                [EMPTY, EMPTY, BLACK, EMPTY],
            ]
        )

        self.assertAlmostEqual(edge_control_feature(state, BLACK), 0.375)
        self.assertAlmostEqual(edge_control_feature(state, WHITE), -0.375)

    def test_corner_proximity_feature(self) -> None:
        state = _build_state(
            [
                [EMPTY, BLACK, WHITE, EMPTY],
                [BLACK, BLACK, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY, EMPTY],
            ]
        )

        self.assertAlmostEqual(corner_proximity_feature(state, BLACK), -0.5)
        self.assertAlmostEqual(corner_proximity_feature(state, WHITE), 0.5)


class TestCompositeHeuristics(unittest.TestCase):
    def test_composite_heuristics_have_consistent_sign(self) -> None:
        state = _build_state(
            [
                [BLACK, BLACK, BLACK, BLACK],
                [BLACK, BLACK, WHITE, WHITE],
                [BLACK, WHITE, WHITE, WHITE],
                [BLACK, BLACK, BLACK, BLACK],
            ]
        )

        self.assertGreater(evaluate_state_balanced(state, BLACK), 0.0)
        self.assertGreater(evaluate_state_aggressive(state, BLACK), 0.0)
        self.assertGreater(evaluate_state_lightweight(state, BLACK), 0.0)

        self.assertLess(evaluate_state_balanced(state, WHITE), 0.0)
        self.assertLess(evaluate_state_aggressive(state, WHITE), 0.0)
        self.assertLess(evaluate_state_lightweight(state, WHITE), 0.0)

    def test_composite_heuristics_work_in_other_board_size(self) -> None:
        state = GameState.initial(GameConfig(board_size=8))

        self.assertAlmostEqual(evaluate_state_balanced(state, BLACK), 0.0)
        self.assertAlmostEqual(evaluate_state_aggressive(state, BLACK), 0.0)
        self.assertAlmostEqual(evaluate_state_lightweight(state, BLACK), 0.0)


class TestRolloutHeuristics(unittest.TestCase):
    def test_score_move_rollout_simple_prioritizes_corner(self) -> None:
        state = _build_state(
            [
                [EMPTY, BLACK, EMPTY, WHITE],
                [EMPTY, WHITE, EMPTY, WHITE],
                [BLACK, WHITE, WHITE, EMPTY],
                [EMPTY, EMPTY, WHITE, BLACK],
            ],
            current_player=BLACK,
        )

        moves = rules.get_legal_moves(state)
        corner_move = Move.from_coords(0, 0)
        non_corner_move = next(move for move in moves if move != corner_move)

        corner_score = score_move_rollout_simple(state, corner_move, BLACK)
        non_corner_score = score_move_rollout_simple(state, non_corner_move, BLACK)

        self.assertGreater(corner_score, non_corner_score)
        self.assertEqual(choose_rollout_move_simple(state, moves, BLACK), corner_move)

    def test_rank_moves_by_heuristic_returns_descending_order(self) -> None:
        state = _build_state(
            [
                [EMPTY, BLACK, EMPTY, WHITE],
                [EMPTY, WHITE, EMPTY, WHITE],
                [BLACK, WHITE, WHITE, EMPTY],
                [EMPTY, EMPTY, WHITE, BLACK],
            ],
            current_player=BLACK,
        )
        moves = rules.get_legal_moves(state)

        ranked = rank_moves_by_heuristic(state, moves, BLACK, heuristic_name="lightweight")

        self.assertEqual(len(ranked), len(moves))
        scores = [score for _move, score in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_choose_rollout_move_greedy_returns_legal_move(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        moves = rules.get_legal_moves(state)

        chosen = choose_rollout_move_greedy(
            state,
            moves,
            state.current_player,
            heuristic_name="lightweight",
        )

        self.assertTrue(rules.is_legal_move(state, chosen))

    def test_choose_rollout_move_topk_is_seed_reproducible(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        moves = rules.get_legal_moves(state)

        chosen_a = choose_rollout_move_topk(state, moves, state.current_player, k=3, seed=42)
        chosen_b = choose_rollout_move_topk(state, moves, state.current_player, k=3, seed=42)

        self.assertEqual(chosen_a, chosen_b)
        self.assertTrue(rules.is_legal_move(state, chosen_a))

    def test_rollout_policies_work_for_two_board_sizes(self) -> None:
        policy = get_rollout_policy_function("rollout_successor_eval")

        for size in (4, 8):
            with self.subTest(board_size=size):
                state = GameState.initial(GameConfig(board_size=size))
                moves = rules.get_legal_moves(state)
                chosen = policy(state, moves, state.current_player, heuristic_name="lightweight")
                self.assertTrue(rules.is_legal_move(state, chosen))


class TestHeuristicNameInterfaces(unittest.TestCase):
    def test_get_evaluation_function_by_name(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))

        for name in ("balanced", "aggressive", "lightweight"):
            with self.subTest(name=name):
                fn = get_evaluation_function(name)
                self.assertIsInstance(fn(state, BLACK), float)

    def test_get_rollout_policy_function_by_name(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        moves = rules.get_legal_moves(state)

        for name in ("rollout_simple", "rollout_successor_eval", "rollout_topk"):
            with self.subTest(name=name):
                fn = get_rollout_policy_function(name)
                if name == "rollout_topk":
                    chosen = fn(state, moves, state.current_player, k=2, seed=7)
                else:
                    chosen = fn(state, moves, state.current_player)
                self.assertTrue(rules.is_legal_move(state, chosen))

    def test_unknown_heuristic_names_raise(self) -> None:
        with self.assertRaises(ValueError):
            get_evaluation_function("unknown")

        with self.assertRaises(ValueError):
            get_rollout_policy_function("unknown")


if __name__ == "__main__":
    unittest.main()
