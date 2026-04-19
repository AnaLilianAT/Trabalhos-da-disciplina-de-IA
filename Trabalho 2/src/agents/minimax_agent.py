from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from src.agents.base import Agent
from src.evaluation.heuristic import (
    StateEvaluationFunction,
    evaluate_state_balanced,
)
from src.game import rules
from src.game.move import Move
from src.game.state import GameState
from src.utils.constants import Player, opponent


@dataclass(frozen=True, slots=True)
class MinimaxSearchMetrics:
    """Runtime metrics collected during one Minimax decision."""

    nodes_expanded: int
    alpha_beta_prunes: int
    max_depth_reached: int
    decision_time_seconds: float


@dataclass(frozen=True, slots=True)
class MinimaxDecision:
    """Structured decision returned by MinimaxAgent.choose_move."""

    move: Move
    minimax_value: float
    metrics: MinimaxSearchMetrics


class MinimaxAgent(Agent):
    """Depth-limited Minimax agent with Alpha-Beta pruning for Othello/Reversi.

    Search values are always computed from the root player's perspective.
    """

    def __init__(
        self,
        max_depth: int = 4,
        evaluation_function: StateEvaluationFunction | None = None,
        use_move_ordering: bool = True,
        verbose_search: bool = False,
        name: str = "MinimaxAgent",
    ) -> None:
        super().__init__(name=name)

        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")

        self.max_depth = max_depth
        self.evaluation_function = evaluation_function or evaluate_state_balanced
        self.use_move_ordering = use_move_ordering
        self.verbose_search = verbose_search

        self._nodes_expanded = 0
        self._alpha_beta_prunes = 0
        self._max_depth_reached = 0

    def choose_move(self, state: GameState) -> MinimaxDecision:
        """Return best move, minimax value and search metrics for current player."""
        root_player = state.current_player
        start_time = perf_counter()
        self._reset_metrics()

        if rules.is_terminal(state):
            value = self._terminal_value(state, root_player)
            metrics = self._build_metrics(start_time)
            return MinimaxDecision(move=Move.pass_turn(), minimax_value=value, metrics=metrics)

        legal_moves = rules.get_legal_moves(state)
        if not legal_moves:
            # Player must pass if no legal moves.
            pass_state = rules.apply_move(state, Move.pass_turn())
            value = self._min_value(
                state=pass_state,
                depth=1,
                alpha=float("-inf"),
                beta=float("inf"),
                root_player=root_player,
            )
            metrics = self._build_metrics(start_time)
            return MinimaxDecision(move=Move.pass_turn(), minimax_value=value, metrics=metrics)

        moves = self._order_moves(
            state=state,
            legal_moves=legal_moves,
            root_player=root_player,
            maximizing=True,
        )

        best_move = moves[0]
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for move in moves:
            child_state = rules.apply_move(state, move)
            value = self._min_value(
                state=child_state,
                depth=1,
                alpha=alpha,
                beta=beta,
                root_player=root_player,
            )

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)

            if self.verbose_search:
                print(
                    f"[Minimax] root move={move} value={value:.4f} "
                    f"alpha={alpha:.4f} beta={beta:.4f}"
                )

        metrics = self._build_metrics(start_time)
        return MinimaxDecision(move=best_move, minimax_value=best_value, metrics=metrics)

    def _max_value(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        root_player: Player,
    ) -> float:
        self._visit_node(depth)

        if rules.is_terminal(state):
            return self._terminal_value(state, root_player)

        if depth >= self.max_depth:
            return self.evaluation_function(state, root_player)

        legal_moves = rules.get_legal_moves(state)
        if not legal_moves:
            passed_state = rules.apply_move(state, Move.pass_turn())
            return self._min_value(
                state=passed_state,
                depth=depth + 1,
                alpha=alpha,
                beta=beta,
                root_player=root_player,
            )

        ordered_moves = self._order_moves(
            state=state,
            legal_moves=legal_moves,
            root_player=root_player,
            maximizing=True,
        )

        value = float("-inf")
        for move in ordered_moves:
            child_state = rules.apply_move(state, move)
            value = max(
                value,
                self._min_value(
                    state=child_state,
                    depth=depth + 1,
                    alpha=alpha,
                    beta=beta,
                    root_player=root_player,
                ),
            )
            alpha = max(alpha, value)

            # Alpha-Beta prune at MAX node.
            if alpha >= beta:
                self._alpha_beta_prunes += 1
                if self.verbose_search:
                    print(
                        f"[Minimax] prune at MAX depth={depth} alpha={alpha:.4f} beta={beta:.4f}"
                    )
                break

        return value

    def _min_value(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        root_player: Player,
    ) -> float:
        self._visit_node(depth)

        if rules.is_terminal(state):
            return self._terminal_value(state, root_player)

        if depth >= self.max_depth:
            return self.evaluation_function(state, root_player)

        legal_moves = rules.get_legal_moves(state)
        if not legal_moves:
            passed_state = rules.apply_move(state, Move.pass_turn())
            return self._max_value(
                state=passed_state,
                depth=depth + 1,
                alpha=alpha,
                beta=beta,
                root_player=root_player,
            )

        ordered_moves = self._order_moves(
            state=state,
            legal_moves=legal_moves,
            root_player=root_player,
            maximizing=False,
        )

        value = float("inf")
        for move in ordered_moves:
            child_state = rules.apply_move(state, move)
            value = min(
                value,
                self._max_value(
                    state=child_state,
                    depth=depth + 1,
                    alpha=alpha,
                    beta=beta,
                    root_player=root_player,
                ),
            )
            beta = min(beta, value)

            # Alpha-Beta prune at MIN node.
            if alpha >= beta:
                self._alpha_beta_prunes += 1
                if self.verbose_search:
                    print(
                        f"[Minimax] prune at MIN depth={depth} alpha={alpha:.4f} beta={beta:.4f}"
                    )
                break

        return value

    def _order_moves(
        self,
        state: GameState,
        legal_moves: list[Move],
        root_player: Player,
        maximizing: bool,
    ) -> list[Move]:
        if not self.use_move_ordering or len(legal_moves) <= 1:
            return list(legal_moves)

        scored: list[tuple[Move, float]] = []
        for move in legal_moves:
            successor = rules.apply_move(state, move)
            score = self.evaluation_function(successor, root_player)
            scored.append((move, score))

        scored.sort(
            key=lambda item: (-item[1], self._move_key(item[0]))
            if maximizing
            else (item[1], self._move_key(item[0]))
        )

        return [move for move, _score in scored]

    def _terminal_value(self, state: GameState, root_player: Player) -> float:
        scores = rules.get_scores(state)
        own = scores.get(root_player, 0)
        opp = scores.get(opponent(root_player), 0)
        denominator = own + opp

        if denominator == 0:
            return 0.0

        return (own - opp) / float(denominator)

    def _visit_node(self, depth: int) -> None:
        self._nodes_expanded += 1
        if depth > self._max_depth_reached:
            self._max_depth_reached = depth

    def _reset_metrics(self) -> None:
        self._nodes_expanded = 0
        self._alpha_beta_prunes = 0
        self._max_depth_reached = 0

    def _build_metrics(self, start_time: float) -> MinimaxSearchMetrics:
        return MinimaxSearchMetrics(
            nodes_expanded=self._nodes_expanded,
            alpha_beta_prunes=self._alpha_beta_prunes,
            max_depth_reached=self._max_depth_reached,
            decision_time_seconds=perf_counter() - start_time,
        )

    @staticmethod
    def _move_key(move: Move) -> tuple[int, int]:
        if move.is_pass:
            return (10**9, 10**9)

        return (int(move.row), int(move.col))
