from __future__ import annotations

import argparse

from src.agents.mcts_agent import MCTSAgent
from src.agents.minimax_agent import MinimaxAgent
from src.experiments.runner import play_game
from src.game.config import GameConfig
from src.utils.constants import BLACK, WHITE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an automated Othello/Reversi match.")
    parser.add_argument(
        "--board-size",
        type=int,
        default=6,
        help="Board size (even number >= 4). Default: 6",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print board and move info every turn.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=("standard", "weighted"),
        default="standard",
        help="Scoring mode: standard (piece count) or weighted (corner/edge/inner).",
    )
    return parser.parse_args()


def main() -> None:
    """Simple entrypoint for running one automatic match between two agents."""
    args = parse_args()
    game_config = GameConfig(board_size=args.board_size, scoring_mode=args.scoring_mode)

    black_agent = MinimaxAgent(name="Minimax-AB")
    white_agent = MCTSAgent(name="MCTS")

    result = play_game(
        agent_black=black_agent,
        agent_white=white_agent,
        verbose=args.verbose,
        seed=args.seed,
        game_config=game_config,
    )

    print("Othello/Reversi - Projeto Base")
    print(f"Board size: {game_config.board_size}x{game_config.board_size}")
    print(f"Scoring mode: {result.scoring_mode}")
    print(f"Winner: {_winner_label(result.winner)}")
    print(f"Final score (B-W): {result.black_score}-{result.white_score}")
    print(f"Final discs (B-W): {result.black_discs}-{result.white_discs}")
    print(f"Turns: {result.turns}")
    print(f"Total passes: {result.total_passes}")
    print(f"Average move time: {result.average_time_per_move_seconds:.6f}s")


def _winner_label(winner: int | None) -> str:
    if winner == BLACK:
        return "BLACK"
    if winner == WHITE:
        return "WHITE"
    return "DRAW"


if __name__ == "__main__":
    main()
