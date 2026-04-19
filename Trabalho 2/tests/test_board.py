import unittest

from src.game.board import Board
from src.game.config import GameConfig
from src.utils.formatting import format_board
from src.utils.constants import BLACK, EMPTY, WHITE


class TestBoard(unittest.TestCase):
    def test_initial_6x6_has_correct_center_pieces(self) -> None:
        board = Board.create_initial(GameConfig(board_size=6))

        self.assertEqual(board.get(2, 2), WHITE)
        self.assertEqual(board.get(3, 3), WHITE)
        self.assertEqual(board.get(2, 3), BLACK)
        self.assertEqual(board.get(3, 2), BLACK)

    def test_initial_8x8_has_correct_center_pieces(self) -> None:
        board = Board.create_initial(GameConfig(board_size=8))

        self.assertEqual(board.get(3, 3), WHITE)
        self.assertEqual(board.get(4, 4), WHITE)
        self.assertEqual(board.get(3, 4), BLACK)
        self.assertEqual(board.get(4, 3), BLACK)

    def test_initial_board_counts_for_multiple_sizes(self) -> None:
        for size in (4, 6, 8):
            board = Board.create_initial(GameConfig(board_size=size))
            counts = board.count_discs()

            self.assertEqual(counts[BLACK], 2)
            self.assertEqual(counts[WHITE], 2)
            self.assertEqual(counts[EMPTY], size * size - 4)

    def test_copy_produces_distinct_board(self) -> None:
        board = Board.create_initial(GameConfig(board_size=6))
        board_copy = board.copy()

        board_copy.set(0, 0, BLACK)

        self.assertNotEqual(board.get(0, 0), board_copy.get(0, 0))

    def test_format_board_contains_coordinates_and_legend(self) -> None:
        board = Board.create_initial(GameConfig(board_size=4))
        rendered = format_board(board)

        self.assertIn("0", rendered)
        self.assertIn("1", rendered)
        self.assertIn("B", rendered)
        self.assertIn("W", rendered)
        self.assertIn("Legend:", rendered)


if __name__ == "__main__":
    unittest.main()
