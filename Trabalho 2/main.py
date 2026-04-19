from __future__ import annotations

"""CLI entrypoint for running one configurable Minimax vs MCTS match."""

import argparse

from src.agents.mcts_agent import MCTSAgent
from src.agents.minimax_agent import MinimaxAgent
from src.evaluation.heuristic import get_evaluation_function
from src.experiments.runner import play_game
from src.game.config import GameConfig
from src.utils.constants import BLACK, WHITE


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the match runner."""
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
    parser.add_argument(
        "--minimax-depth",
        type=int,
        default=4,
        help="Minimax search depth. Default: 4",
    )
    parser.add_argument(
        "--minimax-eval",
        choices=("balanced", "aggressive", "lightweight"),
        default="balanced",
        help="Minimax evaluation heuristic.",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=300,
        help="MCTS simulation budget used when no time limit is set. Default: 300",
    )
    parser.add_argument(
        "--mcts-time-limit",
        type=float,
        default=None,
        help="Optional MCTS time budget in seconds (overrides fixed simulation budget behavior).",
    )
    parser.add_argument(
        "--mcts-exploration-constant",
        type=float,
        default=1.41421356237,
        help="UCT exploration constant c. Default: 1.41421356237",
    )
    parser.add_argument(
        "--mcts-rollout-policy",
        choices=("random", "rollout_simple", "rollout_successor_eval", "rollout_topk"),
        default="random",
        help="Rollout policy used by MCTS.",
    )
    parser.add_argument(
        "--mcts-rollout-heuristic",
        choices=("balanced", "aggressive", "lightweight"),
        default="lightweight",
        help="Heuristic used by rollout_successor_eval and rollout_topk policies.",
    )
    parser.add_argument(
        "--mcts-rollout-topk",
        type=int,
        default=3,
        help="Top-k parameter used by rollout_topk policy. Default: 3",
    )
    return parser.parse_args()


def main() -> None:
    """Simple entrypoint for running one automatic match between two agents."""
    args = parse_args()
    game_config = GameConfig(board_size=args.board_size, scoring_mode=args.scoring_mode)

    minimax_eval_fn = get_evaluation_function(args.minimax_eval)
    black_agent = MinimaxAgent(
        max_depth=args.minimax_depth,
        evaluation_function=minimax_eval_fn,
        name="Minimax-AB",
    )
    white_agent = MCTSAgent(
        num_simulations=args.mcts_simulations,
        time_limit_seconds=args.mcts_time_limit,
        exploration_constant=args.mcts_exploration_constant,
        rollout_policy=args.mcts_rollout_policy,
        rollout_heuristic_name=args.mcts_rollout_heuristic,
        rollout_topk=args.mcts_rollout_topk,
        name="MCTS",
    )

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
    print(f"Minimax: depth={args.minimax_depth}, eval={args.minimax_eval}")
    print(
        "MCTS: "
        f"simulations={args.mcts_simulations}, "
        f"time_limit={args.mcts_time_limit}, "
        f"c={args.mcts_exploration_constant}, "
        f"rollout_policy={args.mcts_rollout_policy}, "
        f"rollout_heuristic={args.mcts_rollout_heuristic}, "
        f"rollout_topk={args.mcts_rollout_topk}"
    )
    print(f"Winner: {_winner_label(result.winner)}")
    print(f"Final score (B-W): {result.black_score}-{result.white_score}")
    print(f"Final discs (B-W): {result.black_discs}-{result.white_discs}")
    print(f"Turns: {result.turns}")
    print(f"Total passes: {result.total_passes}")
    print(f"Average move time: {result.average_time_per_move_seconds:.6f}s")


def _winner_label(winner: int | None) -> str:
    """Return a printable label for the winner value."""
    if winner == BLACK:
        return "BLACK"
    if winner == WHITE:
        return "WHITE"
    return "DRAW"


if __name__ == "__main__":
    main()
