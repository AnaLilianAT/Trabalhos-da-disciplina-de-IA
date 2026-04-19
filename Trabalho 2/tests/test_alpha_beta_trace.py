import tempfile
import unittest
from pathlib import Path

from src.experiments.alpha_beta_trace import (
    AlphaBetaTracer,
    build_alpha_beta_markdown,
    load_state_from_text_file,
    render_trace_tree,
    state_from_grid,
    trace_to_graphviz_dot,
    write_alpha_beta_markdown,
    write_graphviz_dot,
)
from src.game import rules
from src.game.config import GameConfig
from src.game.state import GameState
from src.utils.constants import BLACK, EMPTY, WHITE


class TestAlphaBetaTrace(unittest.TestCase):
    def test_trace_chooses_legal_move_and_respects_depth_limit(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        tracer = AlphaBetaTracer(max_depth=2, use_move_ordering=True)

        trace = tracer.trace(state)

        self.assertTrue(rules.is_legal_move(state, trace.chosen_move))
        self.assertGreater(trace.metrics.nodes_expanded, 0)
        self.assertLessEqual(trace.metrics.max_depth_reached, 2)

    def test_depth_configuration_changes_search_effort(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        trace_depth_2 = AlphaBetaTracer(max_depth=2).trace(state)
        trace_depth_3 = AlphaBetaTracer(max_depth=3).trace(state)

        self.assertGreaterEqual(
            trace_depth_3.metrics.max_depth_reached,
            trace_depth_2.metrics.max_depth_reached,
        )
        self.assertGreaterEqual(
            trace_depth_3.metrics.nodes_expanded,
            trace_depth_2.metrics.nodes_expanded,
        )

    def test_trace_handles_pass_when_no_moves(self) -> None:
        state = state_from_grid(
            grid=[
                [BLACK, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, WHITE],
                [WHITE, WHITE, WHITE, EMPTY],
            ],
            current_player=WHITE,
            game_config=GameConfig(board_size=4, allowed_board_sizes=None),
        )
        tracer = AlphaBetaTracer(max_depth=3)

        trace = tracer.trace(state)

        self.assertTrue(trace.chosen_move.is_pass)
        root = trace.nodes[trace.root_node_id]
        self.assertEqual(len(root.considered_moves), 1)
        self.assertTrue(root.considered_moves[0].is_pass)

    def test_trace_registers_alpha_beta_pruning(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        tracer = AlphaBetaTracer(max_depth=4, use_move_ordering=True)

        trace = tracer.trace(state)
        tree_text = render_trace_tree(trace)

        self.assertGreater(trace.metrics.pruning_events, 0)
        self.assertGreater(trace.metrics.pruned_branches, 0)
        self.assertIn("PRUNED", tree_text)

        pruning_nodes = [node for node in trace.nodes.values() if node.pruned_moves]
        self.assertTrue(pruning_nodes)
        self.assertTrue(all(node.prune_reason is not None for node in pruning_nodes))

    def test_trace_runs_on_multiple_board_sizes(self) -> None:
        for board_size in (4, 8):
            with self.subTest(board_size=board_size):
                state = GameState.initial(GameConfig(board_size=board_size))
                trace = AlphaBetaTracer(max_depth=2).trace(state)

                self.assertEqual(trace.board_size, board_size)
                self.assertTrue(rules.is_legal_move(state, trace.chosen_move))

    def test_markdown_and_dot_outputs(self) -> None:
        state = GameState.initial(GameConfig(board_size=6))
        trace = AlphaBetaTracer(max_depth=2).trace(state)

        markdown = build_alpha_beta_markdown(trace)
        dot_text = trace_to_graphviz_dot(trace)

        self.assertIn("Tabuleiro inicial usado", markdown)
        self.assertIn("Podas Alfa-Beta", markdown)
        self.assertIn("Arvore textual da busca", markdown)
        self.assertIn("digraph AlphaBetaTrace", dot_text)

        with tempfile.TemporaryDirectory() as temp_dir:
            md_path = Path(temp_dir) / "trace.md"
            dot_path = Path(temp_dir) / "trace.dot"

            written_md = write_alpha_beta_markdown(trace, str(md_path))
            written_dot = write_graphviz_dot(trace, str(dot_path))

            self.assertTrue(Path(written_md).exists())
            self.assertTrue(Path(written_dot).exists())

    def test_load_state_from_text_file(self) -> None:
        board_text = """
        . . . .
        . W B .
        . B W .
        . . . .
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.txt"
            state_path.write_text(board_text, encoding="utf-8")

            loaded_state = load_state_from_text_file(
                file_path=str(state_path),
                current_player=BLACK,
                game_config=GameConfig(board_size=4, allowed_board_sizes=None),
            )

            self.assertEqual(loaded_state.board.size, 4)
            self.assertEqual(loaded_state.board.get(1, 1), WHITE)
            self.assertEqual(loaded_state.board.get(1, 2), BLACK)


if __name__ == "__main__":
    unittest.main()
