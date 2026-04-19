import unittest

from src.game.config import GameConfig, validate_board_size


class TestGameConfig(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = GameConfig()
        self.assertEqual(config.board_size, 6)
        self.assertEqual(config.scoring_mode, "standard")

    def test_accepts_common_even_sizes(self) -> None:
        for size in (4, 6, 8):
            config = GameConfig(board_size=size)
            self.assertEqual(config.board_size, size)

    def test_default_validation_accepts_other_even_sizes(self) -> None:
        config = GameConfig(board_size=10)
        self.assertEqual(config.board_size, 10)

    def test_rejects_odd_size(self) -> None:
        with self.assertRaises(ValueError):
            validate_board_size(5)

    def test_rejects_too_small_size(self) -> None:
        with self.assertRaises(ValueError):
            validate_board_size(2)

    def test_rejects_size_outside_allowed_set(self) -> None:
        with self.assertRaises(ValueError):
            GameConfig(board_size=10, allowed_board_sizes=(4, 6, 8))

    def test_allows_custom_even_size_if_allowed_list_is_none(self) -> None:
        config = GameConfig(board_size=10, allowed_board_sizes=None)
        self.assertEqual(config.board_size, 10)

    def test_rejects_invalid_allowed_sizes_definition(self) -> None:
        with self.assertRaises(ValueError):
            GameConfig(board_size=6, allowed_board_sizes=(4, 5, 8))

    def test_accepts_weighted_scoring_mode(self) -> None:
        config = GameConfig(board_size=6, scoring_mode="weighted")
        self.assertEqual(config.scoring_mode, "weighted")

    def test_rejects_invalid_scoring_mode(self) -> None:
        with self.assertRaises(ValueError):
            GameConfig(scoring_mode="invalid-mode")


if __name__ == "__main__":
    unittest.main()
