from __future__ import annotations

"""Didactic Alpha-Beta trace utilities for Minimax report generation.

This module intentionally keeps tracing logic outside the main Minimax agent.
That way, the production agent remains clean while the report-oriented tracing
can record detailed expansion events, alpha/beta transitions and pruning points.
"""

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from src.evaluation.heuristic import (
    StateEvaluationFunction,
    evaluate_state_balanced,
    get_evaluation_function,
)
from src.game import rules
from src.game.board import Board
from src.game.config import GameConfig
from src.game.move import Move
from src.game.state import GameState
from src.utils.constants import BLACK, EMPTY, Player, WHITE, opponent
from src.utils.formatting import format_board, format_move


@dataclass(frozen=True, slots=True)
class AlphaBetaTraceMetrics:
    nodes_expanded: int
    pruning_events: int
    pruned_branches: int
    max_depth_reached: int
    decision_time_seconds: float


@dataclass(slots=True)
class AlphaBetaTraceNode:
    node_id: int
    parent_id: int | None
    depth: int
    node_type: str
    current_player: Player
    move_from_parent: Move | None
    alpha_in: float
    beta_in: float
    alpha_out: float | None = None
    beta_out: float | None = None
    propagated_value: float | None = None
    leaf_value: float | None = None
    is_terminal: bool = False
    is_depth_cutoff: bool = False
    is_pruned: bool = False
    prune_reason: str | None = None
    considered_moves: list[Move] = field(default_factory=list)
    pruned_moves: list[Move] = field(default_factory=list)
    children_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AlphaBetaTraceResult:
    chosen_move: Move
    chosen_value: float
    root_player: Player
    depth_limit: int
    board_size: int
    evaluation_name: str
    use_move_ordering: bool
    initial_state: GameState
    root_node_id: int
    nodes: dict[int, AlphaBetaTraceNode]
    metrics: AlphaBetaTraceMetrics


class AlphaBetaTracer:
    """Run Minimax + Alpha-Beta and record an explicit search tree trace."""

    def __init__(
        self,
        max_depth: int = 2,
        evaluation_function: StateEvaluationFunction | None = None,
        evaluation_name: str = "balanced",
        use_move_ordering: bool = True,
        verbose: bool = False,
    ) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")

        self.max_depth = max_depth
        self.evaluation_function = evaluation_function or evaluate_state_balanced
        self.evaluation_name = evaluation_name
        self.use_move_ordering = use_move_ordering
        self.verbose = verbose

        self._nodes: dict[int, AlphaBetaTraceNode] = {}
        self._next_node_id = 1
        self._nodes_expanded = 0
        self._pruning_events = 0
        self._pruned_branches = 0
        self._max_depth_reached = 0

    def trace(self, state: GameState) -> AlphaBetaTraceResult:
        self._reset_internal_state()

        root_state = state.clone()
        root_player = root_state.current_player
        start_time = perf_counter()

        root_id = self._create_node(
            parent_id=None,
            depth=0,
            node_type="MAX",
            current_player=root_state.current_player,
            move_from_parent=None,
            alpha=float("-inf"),
            beta=float("inf"),
        )

        root = self._nodes[root_id]

        if rules.is_terminal(root_state):
            root_value = self._terminal_value(root_state, root_player)
            root.is_terminal = True
            root.leaf_value = root_value
            root.propagated_value = root_value
            root.alpha_out = root.alpha_in
            root.beta_out = root.beta_in

            metrics = self._build_metrics(start_time)
            return AlphaBetaTraceResult(
                chosen_move=Move.pass_turn(),
                chosen_value=root_value,
                root_player=root_player,
                depth_limit=self.max_depth,
                board_size=root_state.board.size,
                evaluation_name=self.evaluation_name,
                use_move_ordering=self.use_move_ordering,
                initial_state=root_state,
                root_node_id=root_id,
                nodes=dict(self._nodes),
                metrics=metrics,
            )

        legal_moves = rules.get_legal_moves(root_state)

        if not legal_moves:
            pass_move = Move.pass_turn()
            root.considered_moves = [pass_move]

            pass_state = rules.apply_move(root_state, pass_move)
            value, child_id = self._min_value(
                state=pass_state,
                depth=1,
                alpha=float("-inf"),
                beta=float("inf"),
                root_player=root_player,
                parent_id=root_id,
                move_from_parent=pass_move,
            )

            root.children_ids.append(child_id)
            root.propagated_value = value
            root.alpha_out = value
            root.beta_out = float("inf")

            metrics = self._build_metrics(start_time)
            return AlphaBetaTraceResult(
                chosen_move=pass_move,
                chosen_value=value,
                root_player=root_player,
                depth_limit=self.max_depth,
                board_size=root_state.board.size,
                evaluation_name=self.evaluation_name,
                use_move_ordering=self.use_move_ordering,
                initial_state=root_state,
                root_node_id=root_id,
                nodes=dict(self._nodes),
                metrics=metrics,
            )

        ordered_moves = self._order_moves(
            state=root_state,
            legal_moves=legal_moves,
            root_player=root_player,
            maximizing=True,
        )
        root.considered_moves = list(ordered_moves)

        best_move = ordered_moves[0]
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for move in ordered_moves:
            child_state = rules.apply_move(root_state, move)
            child_value, child_id = self._min_value(
                state=child_state,
                depth=1,
                alpha=alpha,
                beta=beta,
                root_player=root_player,
                parent_id=root_id,
                move_from_parent=move,
            )
            root.children_ids.append(child_id)

            if child_value > best_value:
                best_value = child_value
                best_move = move

            alpha = max(alpha, best_value)

        root.propagated_value = best_value
        root.alpha_out = alpha
        root.beta_out = beta

        metrics = self._build_metrics(start_time)
        return AlphaBetaTraceResult(
            chosen_move=best_move,
            chosen_value=best_value,
            root_player=root_player,
            depth_limit=self.max_depth,
            board_size=root_state.board.size,
            evaluation_name=self.evaluation_name,
            use_move_ordering=self.use_move_ordering,
            initial_state=root_state,
            root_node_id=root_id,
            nodes=dict(self._nodes),
            metrics=metrics,
        )

    def _max_value(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        root_player: Player,
        parent_id: int,
        move_from_parent: Move,
    ) -> tuple[float, int]:
        node_id = self._create_node(
            parent_id=parent_id,
            depth=depth,
            node_type="MAX",
            current_player=state.current_player,
            move_from_parent=move_from_parent,
            alpha=alpha,
            beta=beta,
        )
        node = self._nodes[node_id]

        self._visit_node(depth)

        if rules.is_terminal(state):
            value = self._terminal_value(state, root_player)
            node.is_terminal = True
            node.leaf_value = value
            node.propagated_value = value
            node.alpha_out = alpha
            node.beta_out = beta
            return value, node_id

        if depth >= self.max_depth:
            value = self.evaluation_function(state, root_player)
            node.is_depth_cutoff = True
            node.leaf_value = value
            node.propagated_value = value
            node.alpha_out = alpha
            node.beta_out = beta
            return value, node_id

        legal_moves = rules.get_legal_moves(state)
        if not legal_moves:
            pass_move = Move.pass_turn()
            node.considered_moves = [pass_move]
            passed_state = rules.apply_move(state, pass_move)

            child_value, child_id = self._min_value(
                state=passed_state,
                depth=depth + 1,
                alpha=alpha,
                beta=beta,
                root_player=root_player,
                parent_id=node_id,
                move_from_parent=pass_move,
            )
            node.children_ids.append(child_id)

            value = child_value
            alpha = max(alpha, value)

            node.propagated_value = value
            node.alpha_out = alpha
            node.beta_out = beta
            return value, node_id

        ordered_moves = self._order_moves(
            state=state,
            legal_moves=legal_moves,
            root_player=root_player,
            maximizing=True,
        )
        node.considered_moves = list(ordered_moves)

        value = float("-inf")
        for index, move in enumerate(ordered_moves):
            child_state = rules.apply_move(state, move)
            child_value, child_id = self._min_value(
                state=child_state,
                depth=depth + 1,
                alpha=alpha,
                beta=beta,
                root_player=root_player,
                parent_id=node_id,
                move_from_parent=move,
            )
            node.children_ids.append(child_id)

            value = max(value, child_value)
            alpha = max(alpha, value)

            # Alpha-Beta prune at MAX node.
            if alpha >= beta:
                remaining_moves = ordered_moves[index + 1 :]
                self._register_pruned_branches(
                    node=node,
                    pruned_moves=remaining_moves,
                    next_node_type="MIN",
                    next_player=opponent(state.current_player),
                    alpha=alpha,
                    beta=beta,
                    reason=f"alpha ({_fmt_value(alpha)}) >= beta ({_fmt_value(beta)})",
                )
                break

        node.propagated_value = value
        node.alpha_out = alpha
        node.beta_out = beta

        return value, node_id

    def _min_value(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        root_player: Player,
        parent_id: int,
        move_from_parent: Move,
    ) -> tuple[float, int]:
        node_id = self._create_node(
            parent_id=parent_id,
            depth=depth,
            node_type="MIN",
            current_player=state.current_player,
            move_from_parent=move_from_parent,
            alpha=alpha,
            beta=beta,
        )
        node = self._nodes[node_id]

        self._visit_node(depth)

        if rules.is_terminal(state):
            value = self._terminal_value(state, root_player)
            node.is_terminal = True
            node.leaf_value = value
            node.propagated_value = value
            node.alpha_out = alpha
            node.beta_out = beta
            return value, node_id

        if depth >= self.max_depth:
            value = self.evaluation_function(state, root_player)
            node.is_depth_cutoff = True
            node.leaf_value = value
            node.propagated_value = value
            node.alpha_out = alpha
            node.beta_out = beta
            return value, node_id

        legal_moves = rules.get_legal_moves(state)
        if not legal_moves:
            pass_move = Move.pass_turn()
            node.considered_moves = [pass_move]
            passed_state = rules.apply_move(state, pass_move)

            child_value, child_id = self._max_value(
                state=passed_state,
                depth=depth + 1,
                alpha=alpha,
                beta=beta,
                root_player=root_player,
                parent_id=node_id,
                move_from_parent=pass_move,
            )
            node.children_ids.append(child_id)

            value = child_value
            beta = min(beta, value)

            node.propagated_value = value
            node.alpha_out = alpha
            node.beta_out = beta
            return value, node_id

        ordered_moves = self._order_moves(
            state=state,
            legal_moves=legal_moves,
            root_player=root_player,
            maximizing=False,
        )
        node.considered_moves = list(ordered_moves)

        value = float("inf")
        for index, move in enumerate(ordered_moves):
            child_state = rules.apply_move(state, move)
            child_value, child_id = self._max_value(
                state=child_state,
                depth=depth + 1,
                alpha=alpha,
                beta=beta,
                root_player=root_player,
                parent_id=node_id,
                move_from_parent=move,
            )
            node.children_ids.append(child_id)

            value = min(value, child_value)
            beta = min(beta, value)

            # Alpha-Beta prune at MIN node.
            if alpha >= beta:
                remaining_moves = ordered_moves[index + 1 :]
                self._register_pruned_branches(
                    node=node,
                    pruned_moves=remaining_moves,
                    next_node_type="MAX",
                    next_player=opponent(state.current_player),
                    alpha=alpha,
                    beta=beta,
                    reason=f"alpha ({_fmt_value(alpha)}) >= beta ({_fmt_value(beta)})",
                )
                break

        node.propagated_value = value
        node.alpha_out = alpha
        node.beta_out = beta

        return value, node_id

    def _register_pruned_branches(
        self,
        node: AlphaBetaTraceNode,
        pruned_moves: list[Move],
        next_node_type: str,
        next_player: Player,
        alpha: float,
        beta: float,
        reason: str,
    ) -> None:
        if not pruned_moves:
            return

        self._pruning_events += 1
        self._pruned_branches += len(pruned_moves)
        node.pruned_moves.extend(pruned_moves)
        node.prune_reason = reason

        if self.verbose:
            print(
                f"[Trace] prune at N{node.node_id} ({node.node_type}) "
                f"reason={reason} pruned={len(pruned_moves)}"
            )

        for move in pruned_moves:
            pruned_id = self._create_node(
                parent_id=node.node_id,
                depth=node.depth + 1,
                node_type=next_node_type,
                current_player=next_player,
                move_from_parent=move,
                alpha=alpha,
                beta=beta,
            )
            pruned_node = self._nodes[pruned_id]
            pruned_node.is_pruned = True
            pruned_node.prune_reason = reason
            pruned_node.alpha_out = alpha
            pruned_node.beta_out = beta
            node.children_ids.append(pruned_id)

    def _create_node(
        self,
        parent_id: int | None,
        depth: int,
        node_type: str,
        current_player: Player,
        move_from_parent: Move | None,
        alpha: float,
        beta: float,
    ) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1

        self._nodes[node_id] = AlphaBetaTraceNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            node_type=node_type,
            current_player=current_player,
            move_from_parent=move_from_parent,
            alpha_in=alpha,
            beta_in=beta,
        )

        return node_id

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
            key=lambda item: (-item[1], _move_sort_key(item[0]))
            if maximizing
            else (item[1], _move_sort_key(item[0]))
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

    def _reset_internal_state(self) -> None:
        self._nodes = {}
        self._next_node_id = 1
        self._nodes_expanded = 0
        self._pruning_events = 0
        self._pruned_branches = 0
        self._max_depth_reached = 0

    def _build_metrics(self, start_time: float) -> AlphaBetaTraceMetrics:
        return AlphaBetaTraceMetrics(
            nodes_expanded=self._nodes_expanded,
            pruning_events=self._pruning_events,
            pruned_branches=self._pruned_branches,
            max_depth_reached=self._max_depth_reached,
            decision_time_seconds=perf_counter() - start_time,
        )


def state_from_grid(
    grid: list[list[int]],
    current_player: Player,
    consecutive_passes: int = 0,
    game_config: GameConfig | None = None,
) -> GameState:
    """Create GameState from explicit integer grid (1, -1, 0)."""
    size = len(grid)
    if size == 0 or any(len(row) != size for row in grid):
        raise ValueError("Grid must be a non-empty square matrix.")

    cfg = game_config or GameConfig(board_size=size, allowed_board_sizes=None)
    if cfg.board_size != size:
        raise ValueError("game_config.board_size must match grid size.")

    board = Board(size=size, grid=[row[:] for row in grid])
    return GameState(
        board=board,
        current_player=current_player,
        consecutive_passes=consecutive_passes,
        config=cfg,
        history=[],
    )


def load_state_from_text_file(
    file_path: str,
    current_player: Player,
    consecutive_passes: int = 0,
    game_config: GameConfig | None = None,
) -> GameState:
    """Load state from a text file using tokens B/W/. or 1/-1/0."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {file_path}")

    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            cleaned = line.replace("|", " ").replace(",", " ")
            tokens = cleaned.split()
            if len(tokens) == 1 and len(tokens[0]) > 1:
                tokens = list(tokens[0])

            row = [_parse_cell_token(token) for token in tokens]
            rows.append(row)

    return state_from_grid(
        grid=rows,
        current_player=current_player,
        consecutive_passes=consecutive_passes,
        game_config=game_config,
    )


def render_trace_tree(trace: AlphaBetaTraceResult) -> str:
    """Render recorded tree in ASCII for terminal/report usage."""
    lines: list[str] = []

    def walk(node_id: int, prefix: str, is_last: bool) -> None:
        node = trace.nodes[node_id]
        branch = "+- " if is_last else "|- "
        lines.append(prefix + branch + _node_summary(node))

        if node.pruned_moves:
            pruned_str = ", ".join(format_move(move) for move in node.pruned_moves)
            marker_prefix = prefix + ("   " if is_last else "|  ")
            lines.append(
                marker_prefix
                + f"!!! PRUNED at N{node.node_id}: {pruned_str} ({node.prune_reason})"
            )

        child_prefix = prefix + ("   " if is_last else "|  ")
        for index, child_id in enumerate(node.children_ids):
            walk(child_id, child_prefix, index == len(node.children_ids) - 1)

    walk(trace.root_node_id, prefix="", is_last=True)
    return "\n".join(lines)


def build_alpha_beta_markdown(
    trace: AlphaBetaTraceResult,
    tree_text: str | None = None,
    dot_file_path: str | None = None,
) -> str:
    """Build markdown report with board, leaves, propagated values and pruning."""
    root = trace.nodes[trace.root_node_id]
    tree_content = tree_text or render_trace_tree(trace)

    root_moves = ", ".join(format_move(move) for move in root.considered_moves) or "(none)"

    leaf_nodes = [
        node
        for node in trace.nodes.values()
        if not node.is_pruned and (node.is_terminal or node.is_depth_cutoff)
    ]
    leaf_nodes.sort(key=lambda node: node.node_id)

    propagated_nodes = [
        node
        for node in trace.nodes.values()
        if not node.is_pruned and node.propagated_value is not None
    ]
    propagated_nodes.sort(key=lambda node: (node.depth, node.node_id))

    pruning_nodes = [node for node in trace.nodes.values() if node.pruned_moves]
    pruning_nodes.sort(key=lambda node: node.node_id)

    markdown_lines: list[str] = [
        "# Exemplo Minimax com Poda Alfa-Beta",
        "",
        "## Resumo",
        f"- Profundidade limite: {trace.depth_limit}",
        f"- Tamanho do tabuleiro: {trace.board_size}x{trace.board_size}",
        f"- Heuristica de avaliacao: {trace.evaluation_name}",
        f"- Ordenacao de jogadas: {trace.use_move_ordering}",
        f"- Jogada escolhida no no raiz: {format_move(trace.chosen_move)}",
        f"- Valor minimax da jogada escolhida: {_fmt_value(trace.chosen_value)}",
        f"- Nos expandidos: {trace.metrics.nodes_expanded}",
        f"- Eventos de poda: {trace.metrics.pruning_events}",
        f"- Ramos podados: {trace.metrics.pruned_branches}",
        "",
        "## Tabuleiro inicial usado",
        "```text",
        format_board(trace.initial_state.board),
        "```",
        "",
        "## Jogadas consideradas no no raiz",
        f"- {root_moves}",
        "",
        "## Jogadas consideradas por no expandido",
        "| No | Tipo | Profundidade | Jogadas consideradas |",
        "| --- | --- | --- | --- |",
    ]

    considered_nodes = [
        node
        for node in trace.nodes.values()
        if not node.is_pruned and node.considered_moves
    ]
    considered_nodes.sort(key=lambda node: (node.depth, node.node_id))

    for node in considered_nodes:
        move_list = ", ".join(format_move(move) for move in node.considered_moves)
        markdown_lines.append(
            f"| N{node.node_id} | {node.node_type} | {node.depth} | {move_list} |"
        )

    markdown_lines.extend(
        [
            "",
        "## Folhas e valores heuristicas/terminais",
        "| No | Tipo | Profundidade | Jogada pai | Valor folha |",
        "| --- | --- | --- | --- | --- |",
        ]
    )

    for node in leaf_nodes:
        leaf_kind = "terminal" if node.is_terminal else "corte_profundidade"
        markdown_lines.append(
            f"| N{node.node_id} | {leaf_kind} | {node.depth} | "
            f"{_format_optional_move(node.move_from_parent)} | {_fmt_optional_value(node.leaf_value)} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Valores propagados",
            "| No | Tipo | Profundidade | Jogada pai | Valor propagado | alfa_in -> alfa_out | beta_in -> beta_out |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for node in propagated_nodes:
        markdown_lines.append(
            f"| N{node.node_id} | {node.node_type} | {node.depth} | "
            f"{_format_optional_move(node.move_from_parent)} | {_fmt_optional_value(node.propagated_value)} | "
            f"{_fmt_value(node.alpha_in)} -> {_fmt_optional_value(node.alpha_out)} | "
            f"{_fmt_value(node.beta_in)} -> {_fmt_optional_value(node.beta_out)} |"
        )

    markdown_lines.extend(["", "## Podas Alfa-Beta"])

    if not pruning_nodes:
        markdown_lines.append("- Nenhuma poda ocorreu neste rastreamento.")
    else:
        for node in pruning_nodes:
            pruned_moves = ", ".join(format_move(move) for move in node.pruned_moves)
            markdown_lines.append(
                f"- N{node.node_id} ({node.node_type}, profundidade {node.depth}): "
                f"poda de [{pruned_moves}] com condicao {node.prune_reason}."
            )

    markdown_lines.extend(
        [
            "",
            "## Arvore textual da busca",
            "```text",
            tree_content,
            "```",
        ]
    )

    if dot_file_path:
        display_dot_path = dot_file_path.replace("\\", "/")
        markdown_lines.extend(
            [
                "",
                "## Arquivo DOT opcional",
                f"- Arquivo gerado: {display_dot_path}",
            ]
        )

    return "\n".join(markdown_lines)


def write_alpha_beta_markdown(
    trace: AlphaBetaTraceResult,
    output_path: str = "docs/alpha_beta_example.md",
    dot_file_path: str | None = None,
) -> str:
    tree_text = render_trace_tree(trace)
    markdown_content = build_alpha_beta_markdown(
        trace=trace,
        tree_text=tree_text,
        dot_file_path=dot_file_path,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_content, encoding="utf-8")
    return str(path)


def trace_to_graphviz_dot(trace: AlphaBetaTraceResult) -> str:
    """Return Graphviz DOT representation of the traced search tree."""
    lines = [
        "digraph AlphaBetaTrace {",
        "  rankdir=TB;",
        "  node [shape=box, fontname=Courier, fontsize=10];",
    ]

    for node_id in sorted(trace.nodes):
        node = trace.nodes[node_id]

        style = ""
        if node.is_pruned:
            style = ", style=\"filled,dashed\", fillcolor=\"#ffd6d6\", color=\"#d33\""
        elif node.is_terminal or node.is_depth_cutoff:
            style = ", style=\"filled\", fillcolor=\"#f0f0f0\""

        label = (
            f"N{node.node_id}\\n"
            f"{node.node_type} d={node.depth}\\n"
            f"move={_format_optional_move(node.move_from_parent)}\\n"
            f"a={_fmt_value(node.alpha_in)}->{_fmt_optional_value(node.alpha_out)}\\n"
            f"b={_fmt_value(node.beta_in)}->{_fmt_optional_value(node.beta_out)}\\n"
            f"v={_fmt_optional_value(node.propagated_value)}"
        )

        if node.is_pruned and node.prune_reason:
            label += f"\\nPRUNED: {node.prune_reason}"

        lines.append(f"  n{node.node_id} [label=\"{label}\"{style}];")

    for node in trace.nodes.values():
        if node.parent_id is None:
            continue

        edge_label = _format_optional_move(node.move_from_parent)
        lines.append(f"  n{node.parent_id} -> n{node.node_id} [label=\"{edge_label}\"]; ")

    lines.append("}")
    return "\n".join(lines)


def write_graphviz_dot(
    trace: AlphaBetaTraceResult,
    output_path: str = "docs/alpha_beta_example.dot",
) -> str:
    dot_text = trace_to_graphviz_dot(trace)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dot_text, encoding="utf-8")
    return str(path)


def generate_alpha_beta_example(
    initial_state: GameState | None = None,
    depth: int = 2,
    game_config: GameConfig | None = None,
    evaluation_name: str = "balanced",
    use_move_ordering: bool = True,
    markdown_output_path: str = "docs/alpha_beta_example.md",
    dot_output_path: str | None = None,
    print_tree: bool = True,
) -> AlphaBetaTraceResult:
    """Convenience helper that traces one search and writes report artifacts."""
    if depth <= 0:
        raise ValueError("depth must be positive.")

    state = initial_state.clone() if initial_state is not None else GameState.initial(
        game_config or GameConfig()
    )

    evaluation_fn = get_evaluation_function(evaluation_name)
    tracer = AlphaBetaTracer(
        max_depth=depth,
        evaluation_function=evaluation_fn,
        evaluation_name=evaluation_name,
        use_move_ordering=use_move_ordering,
    )
    trace = tracer.trace(state)

    tree_text = render_trace_tree(trace)
    if print_tree:
        print(tree_text)

    generated_dot_path: str | None = None
    if dot_output_path is not None:
        generated_dot_path = write_graphviz_dot(trace, output_path=dot_output_path)

    write_alpha_beta_markdown(
        trace=trace,
        output_path=markdown_output_path,
        dot_file_path=generated_dot_path,
    )

    return trace


def _parse_cell_token(token: str) -> int:
    normalized = token.strip().upper()
    mapping = {
        "B": BLACK,
        "W": WHITE,
        ".": EMPTY,
        "0": EMPTY,
        "1": BLACK,
        "-1": WHITE,
    }

    if normalized not in mapping:
        raise ValueError(f"Unsupported board token: {token}")

    return mapping[normalized]


def _node_summary(node: AlphaBetaTraceNode) -> str:
    move_label = _format_optional_move(node.move_from_parent)
    value_label = _fmt_optional_value(node.propagated_value)
    alpha_label = f"{_fmt_value(node.alpha_in)}->{_fmt_optional_value(node.alpha_out)}"
    beta_label = f"{_fmt_value(node.beta_in)}->{_fmt_optional_value(node.beta_out)}"

    status: str
    if node.is_pruned:
        status = "[PRUNED BRANCH]"
    elif node.is_terminal:
        status = f"[TERMINAL leaf={_fmt_optional_value(node.leaf_value)}]"
    elif node.is_depth_cutoff:
        status = f"[DEPTH-CUTOFF leaf={_fmt_optional_value(node.leaf_value)}]"
    else:
        status = "[INTERNAL]"

    return (
        f"N{node.node_id} {node.node_type} d={node.depth} move={move_label} "
        f"a={alpha_label} b={beta_label} v={value_label} {status}"
    )


def _fmt_value(value: float) -> str:
    if value == float("inf"):
        return "+inf"
    if value == float("-inf"):
        return "-inf"
    return f"{value:.4f}"


def _fmt_optional_value(value: float | None) -> str:
    if value is None:
        return "-"
    return _fmt_value(value)


def _format_optional_move(move: Move | None) -> str:
    if move is None:
        return "ROOT"
    return format_move(move)


def _move_sort_key(move: Move) -> tuple[int, int]:
    if move.is_pass:
        return (10**9, 10**9)

    return (int(move.row), int(move.col))
