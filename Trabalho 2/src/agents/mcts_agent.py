from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from time import perf_counter

from src.agents.base import Agent
from src.evaluation.heuristic import get_rollout_policy_function
from src.game import rules
from src.game.move import Move
from src.game.state import GameState
from src.utils.constants import Player


@dataclass(frozen=True, slots=True)
class MCTSSearchMetrics:
    """Runtime metrics collected during one MCTS decision."""

    total_simulations: int
    decision_time_seconds: float
    nodes_created: int
    average_rollout_depth: float
    rollout_policy: str


@dataclass(frozen=True, slots=True)
class MCTSDecision:
    """Structured decision returned by MCTSAgent.choose_move."""

    move: Move
    estimated_value: float
    metrics: MCTSSearchMetrics


@dataclass(slots=True)
class MCTSNode:
    """Node stored in the Monte Carlo search tree."""

    state: GameState
    parent: MCTSNode | None
    move: Move | None
    statistics_player: Player
    visits: int = 0
    value_sum: float = 0.0
    children: list[MCTSNode] = field(default_factory=list)
    untried_moves: list[Move] = field(default_factory=list)

    @property
    def average_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / float(self.visits)

    def is_terminal(self) -> bool:
        return rules.is_terminal(self.state)


class MCTSAgent(Agent):
    """Monte Carlo Tree Search agent with UCT selection for Othello/Reversi."""

    _ROLLOUT_POLICY_NAMES = {
        "random",
        "rollout_simple",
        "rollout_successor_eval",
        "rollout_topk",
    }

    def __init__(
        self,
        num_simulations: int = 300,
        time_limit_seconds: float | None = None,
        exploration_constant: float = 1.41421356237,
        rollout_policy: str = "random",
        verbose_search: bool = False,
        rollout_heuristic_name: str = "lightweight",
        rollout_topk: int = 3,
        name: str = "MCTSAgent",
    ) -> None:
        super().__init__(name=name)

        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive.")

        if time_limit_seconds is not None and time_limit_seconds <= 0.0:
            raise ValueError("time_limit_seconds must be positive when provided.")

        if exploration_constant < 0.0:
            raise ValueError("exploration_constant must be non-negative.")

        if rollout_policy not in self._ROLLOUT_POLICY_NAMES:
            valid = ", ".join(sorted(self._ROLLOUT_POLICY_NAMES))
            raise ValueError(f"Unknown rollout_policy '{rollout_policy}'. Valid options: {valid}")

        if rollout_topk <= 0:
            raise ValueError("rollout_topk must be positive.")

        self.num_simulations = num_simulations
        self.time_limit_seconds = time_limit_seconds
        self.exploration_constant = exploration_constant
        self.rollout_policy = rollout_policy
        self.verbose_search = verbose_search
        self.rollout_heuristic_name = rollout_heuristic_name
        self.rollout_topk = rollout_topk

        self.last_root: MCTSNode | None = None

        self._nodes_created = 0
        self._rollout_depth_sum = 0
        self._simulations_run = 0

    def choose_move(self, state: GameState) -> MCTSDecision:
        """Return best move from MCTS search using visit-count criterion."""
        start_time = perf_counter()
        self._reset_search_counters()

        root_state = state.clone()
        root_moves = self._available_moves(root_state)

        if rules.is_terminal(root_state):
            metrics = self._build_metrics(start_time)
            return MCTSDecision(move=Move.pass_turn(), estimated_value=0.0, metrics=metrics)

        root = self._make_node(
            state=root_state,
            parent=None,
            move=None,
            statistics_player=root_state.current_player,
        )
        root.untried_moves = list(root_moves)
        self.last_root = root

        if self.time_limit_seconds is None:
            target_simulations = max(1, self.num_simulations)
            for _ in range(target_simulations):
                self._run_single_simulation(root)
        else:
            while self._simulations_run == 0 or (
                perf_counter() - start_time < self.time_limit_seconds
            ):
                self._run_single_simulation(root)

        if not root.children:
            # Fallback should be rare; it protects against unexpected edge cases.
            selected_move = root_moves[0] if root_moves else Move.pass_turn()
            metrics = self._build_metrics(start_time)
            return MCTSDecision(move=selected_move, estimated_value=0.0, metrics=metrics)

        best_child = max(
            root.children,
            key=lambda child: (child.visits, child.average_value, self._move_sort_key(child.move)),
        )

        decision = MCTSDecision(
            move=best_child.move if best_child.move is not None else Move.pass_turn(),
            estimated_value=best_child.average_value,
            metrics=self._build_metrics(start_time),
        )

        if self.verbose_search:
            print(
                "[MCTS] simulations="
                f"{decision.metrics.total_simulations} nodes={decision.metrics.nodes_created} "
                f"avg_rollout_depth={decision.metrics.average_rollout_depth:.2f} "
                f"move={decision.move} value={decision.estimated_value:.4f}"
            )

        return decision

    def selection(self, root: MCTSNode) -> MCTSNode:
        """Selection phase: descend the tree using UCT until expansion candidate."""
        node = root

        while not node.is_terminal():
            if node.untried_moves:
                return node

            if not node.children:
                return node

            node = self._select_child_with_uct(node)

        return node

    def expansion(self, node: MCTSNode) -> MCTSNode:
        """Expansion phase: create one child for an untried move, when possible."""
        if node.is_terminal() or not node.untried_moves:
            return node

        move_index = random.randrange(len(node.untried_moves))
        move = node.untried_moves.pop(move_index)
        next_state = rules.apply_move(node.state, move)

        child = self._make_node(
            state=next_state,
            parent=node,
            move=move,
            statistics_player=node.state.current_player,
        )
        child.untried_moves = self._available_moves(child.state)

        node.children.append(child)
        return child

    def rollout(self, state: GameState) -> tuple[Player | None, int]:
        """Simulation phase: play from state to game end using configured policy."""
        if self.rollout_policy == "random":
            return self.rollout_random(state)

        return self.rollout_with_policy(state)

    def rollout_random(self, state: GameState) -> tuple[Player | None, int]:
        """Pure random rollout policy used as baseline."""
        current_state = state
        depth = 0
        max_steps = self._rollout_step_limit(current_state.board.size)

        while not rules.is_terminal(current_state) and depth < max_steps:
            moves = self._available_moves(current_state)
            if not moves:
                break

            chosen_move = random.choice(moves)
            current_state = rules.apply_move(current_state, chosen_move)
            depth += 1

        return rules.get_winner(current_state), depth

    def rollout_with_policy(self, state: GameState) -> tuple[Player | None, int]:
        """Heuristic-guided rollout using policies from evaluation/heuristic.py."""
        policy_fn = get_rollout_policy_function(self.rollout_policy)
        current_state = state
        depth = 0
        max_steps = self._rollout_step_limit(current_state.board.size)

        while not rules.is_terminal(current_state) and depth < max_steps:
            moves = self._available_moves(current_state)
            if not moves:
                break

            if self.rollout_policy == "rollout_simple":
                chosen_move = policy_fn(current_state, moves, current_state.current_player)
            elif self.rollout_policy == "rollout_successor_eval":
                chosen_move = policy_fn(
                    current_state,
                    moves,
                    current_state.current_player,
                    heuristic_name=self.rollout_heuristic_name,
                )
            elif self.rollout_policy == "rollout_topk":
                chosen_move = policy_fn(
                    current_state,
                    moves,
                    current_state.current_player,
                    k=self.rollout_topk,
                    heuristic_name=self.rollout_heuristic_name,
                    seed=random.randrange(2**31 - 1),
                )
            else:
                raise ValueError(f"Unsupported rollout policy: {self.rollout_policy}")

            current_state = rules.apply_move(current_state, chosen_move)
            depth += 1

        return rules.get_winner(current_state), depth

    def backpropagation(self, node: MCTSNode, winner: Player | None) -> None:
        """Backpropagation phase: update visits and rewards from leaf to root."""
        current: MCTSNode | None = node

        while current is not None:
            current.visits += 1
            current.value_sum += self._reward_for_player(
                winner=winner,
                player=current.statistics_player,
            )
            current = current.parent

    def _run_single_simulation(self, root: MCTSNode) -> None:
        selected = self.selection(root)
        expanded = self.expansion(selected)
        winner, rollout_depth = self.rollout(expanded.state)
        self.backpropagation(expanded, winner)

        self._simulations_run += 1
        self._rollout_depth_sum += rollout_depth

    def _select_child_with_uct(self, node: MCTSNode) -> MCTSNode:
        parent_log = math.log(max(1, node.visits))

        def uct_score(child: MCTSNode) -> float:
            if child.visits == 0:
                return float("inf")

            exploitation = child.average_value
            exploration = self.exploration_constant * math.sqrt(parent_log / child.visits)
            return exploitation + exploration

        return max(
            node.children,
            key=lambda child: (uct_score(child), child.average_value, self._move_sort_key(child.move)),
        )

    def _available_moves(self, state: GameState) -> list[Move]:
        if rules.is_terminal(state):
            return []

        legal_moves = rules.get_legal_moves(state)
        if legal_moves:
            return legal_moves

        return [Move.pass_turn()]

    def _reward_for_player(self, winner: Player | None, player: Player) -> float:
        if winner is None:
            return 0.5

        return 1.0 if winner == player else 0.0

    def _make_node(
        self,
        state: GameState,
        parent: MCTSNode | None,
        move: Move | None,
        statistics_player: Player,
    ) -> MCTSNode:
        node = MCTSNode(
            state=state,
            parent=parent,
            move=move,
            statistics_player=statistics_player,
        )
        self._nodes_created += 1
        return node

    def _reset_search_counters(self) -> None:
        self._nodes_created = 0
        self._rollout_depth_sum = 0
        self._simulations_run = 0

    def _build_metrics(self, start_time: float) -> MCTSSearchMetrics:
        average_depth = (
            self._rollout_depth_sum / float(self._simulations_run)
            if self._simulations_run > 0
            else 0.0
        )

        return MCTSSearchMetrics(
            total_simulations=self._simulations_run,
            decision_time_seconds=perf_counter() - start_time,
            nodes_created=self._nodes_created,
            average_rollout_depth=average_depth,
            rollout_policy=self.rollout_policy,
        )

    @staticmethod
    def _rollout_step_limit(board_size: int) -> int:
        # Conservative upper bound to guarantee rollout termination.
        return board_size * board_size * 4 + 10

    @staticmethod
    def _move_sort_key(move: Move | None) -> tuple[int, int]:
        if move is None or move.is_pass:
            return (10**9, 10**9)

        return (int(move.row), int(move.col))
