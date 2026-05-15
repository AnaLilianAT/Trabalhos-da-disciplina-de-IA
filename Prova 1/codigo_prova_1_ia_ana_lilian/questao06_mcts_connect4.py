"""
Questão 06 — Monte Carlo Tree Search (MCTS) para Connect-4 simplificado 4x4.

Inclui rollout aleatório e rollout guloso/semi-guloso.
A recompensa é sempre do ponto de vista do jogador vermelho V:
vitória=1, empate=0.5, derrota=0.
"""
import math
import random

ROWS, COLS = 4, 4
EMPTY = "."
RED, YELLOW = "V", "A"

# Leitura usada na resolução: linha 3: A V . . ; linha 4: V A . .
BOARD0 = [list("...."), list("...."), list("AV.."), list("VA..")] 


def opponent(player):
    return YELLOW if player == RED else RED


def legal_moves(board):
    return [c for c in range(COLS) if board[0][c] == EMPTY]


def center_order(moves):
    """Prioriza as colunas centrais. Em 4x4: c2, c3, c1, c4."""
    center = (COLS - 1) / 2
    return sorted(moves, key=lambda c: (abs(c - center), c))


def board_play(board, col, player):
    b = [row[:] for row in board]
    for r in range(ROWS - 1, -1, -1):
        if b[r][col] == EMPTY:
            b[r][col] = player
            return b
    raise ValueError("coluna cheia")


def board_winner(board):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == EMPTY:
                continue
            p = board[r][c]
            for dr, dc in directions:
                ok = True
                for k in range(4):
                    rr, cc = r + dr * k, c + dc * k
                    if not (0 <= rr < ROWS and 0 <= cc < COLS) or board[rr][cc] != p:
                        ok = False
                        break
                if ok:
                    return p
    if not legal_moves(board):
        return "D"
    return None


def reward_for_red(winner):
    if winner == RED:
        return 1.0
    if winner == YELLOW:
        return 0.0
    if winner == "D":
        return 0.5
    return None


def immediate_winning_moves(board, player):
    """Retorna as colunas em que player vence imediatamente."""
    winners = []
    for col in legal_moves(board):
        next_board = board_play(board, col, player)
        if board_winner(next_board) == player:
            winners.append(col)
    return center_order(winners)


def greedy_rollout_move(board, player):
    """
    Rollout guloso/semi-guloso:
    1. se puder vencer imediatamente, vence;
    2. se o adversário puder vencer imediatamente, bloqueia;
    3. caso contrário, joga na coluna mais central disponível.
    """
    moves = legal_moves(board)
    if not moves:
        return None

    win_now = immediate_winning_moves(board, player)
    if win_now:
        return win_now[0]

    opp_win = immediate_winning_moves(board, opponent(player))
    if opp_win:
        return opp_win[0]

    return center_order(moves)[0]


def random_rollout(board, player, rng):
    """Rollout simples: escolhe movimentos aleatórios até o fim do jogo."""
    cur = player
    b = [row[:] for row in board]
    while True:
        winner = board_winner(b)
        reward = reward_for_red(winner)
        if reward is not None:
            return reward

        col = rng.choice(legal_moves(b))
        b = board_play(b, col, cur)
        cur = opponent(cur)


def greedy_rollout(board, player, rng=None):
    """Rollout guloso/semi-guloso usado para comparar com o rollout aleatório."""
    cur = player
    b = [row[:] for row in board]
    while True:
        winner = board_winner(b)
        reward = reward_for_red(winner)
        if reward is not None:
            return reward

        col = greedy_rollout_move(b, cur)
        b = board_play(b, col, cur)
        cur = opponent(cur)


def rollout(board, player, rng, policy="random"):
    normalized = policy.lower()
    if normalized in {"random", "aleatorio", "aleatório"}:
        return random_rollout(board, player, rng)
    if normalized in {"greedy", "guloso", "semi_guloso", "semi-guloso"}:
        return greedy_rollout(board, player, rng)
    raise ValueError(f"Política de rollout desconhecida: {policy}")


def uct(w, n, parent_n, C=1.4):
    if n == 0:
        return math.inf
    return (w / n) + C * math.sqrt(math.log(parent_n) / n)


class MCTSNode:
    def __init__(self, board, player, parent=None, move=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move
        self.children = {}
        self.untried = legal_moves(board)[:]
        self.N = 0
        self.W = 0.0

    def best_uct_child(self, C):
        return max(
            self.children.values(),
            key=lambda ch: (uct(ch.W, ch.N, self.N, C), -ch.move),
        )


def mcts(iterations=10, C=1.4, seed=7, rollout_policy="random"):
    rng = random.Random(seed)
    root = MCTSNode(BOARD0, RED)
    rows = []

    for it in range(1, iterations + 1):
        node = root
        path = []

        # 1) Seleção
        while not node.untried and node.children and board_winner(node.board) is None:
            node = node.best_uct_child(C)
            path.append(node.move + 1)

        # 2) Expansão
        expanded = None
        if board_winner(node.board) is None and node.untried:
            col = node.untried.pop(0)
            next_board = board_play(node.board, col, node.player)
            next_player = opponent(node.player)
            child = MCTSNode(next_board, next_player, node, col)
            node.children[col] = child
            node = child
            expanded = col + 1
            path.append(col + 1)

        # 3) Simulação/Rollout
        result = rollout(node.board, node.player, rng, policy=rollout_policy)

        # 4) Retropropagação
        cur = node
        while cur is not None:
            cur.N += 1
            cur.W += result
            cur = cur.parent

        stats = {f"c{c + 1}": (ch.N, ch.W) for c, ch in sorted(root.children.items())}
        rows.append((it, path, expanded, result, stats))

    return rows, root


def mcts_summary(root):
    summary = {}
    for col, child in sorted(root.children.items()):
        mean = child.W / child.N if child.N else 0.0
        summary[f"c{col + 1}"] = {"N": child.N, "W": child.W, "media": mean}
    return summary


def uct_table_from_root(root, C=1.4):
    parent_n = root.N
    rows = []
    for col, child in sorted(root.children.items()):
        rows.append((f"c{col + 1}", child.N, child.W, child.W / child.N, uct(child.W, child.N, parent_n, C)))
    return rows


def compare_mcts_rollouts(iterations=10, C=1.4, seed=7):
    random_rows, random_root = mcts(iterations, C, seed, rollout_policy="random")
    greedy_rows, greedy_root = mcts(iterations, C, seed, rollout_policy="greedy")
    return {
        "random": {"rows": random_rows, "summary": mcts_summary(random_root), "uct": uct_table_from_root(random_root, C)},
        "greedy": {"rows": greedy_rows, "summary": mcts_summary(greedy_root), "uct": uct_table_from_root(greedy_root, C)},
    }


def compare_c_values(iterations=10, seed=7, rollout_policy="random"):
    results = {}
    for C in [0.1, 1.4, 3.0]:
        rows, root = mcts(iterations=iterations, C=C, seed=seed, rollout_policy=rollout_policy)
        results[C] = {"rows": rows, "summary": mcts_summary(root), "uct": uct_table_from_root(root, C)}
    return results


if __name__ == "__main__":
    rows_random, root_random = mcts(iterations=10, C=1.4, seed=7, rollout_policy="random")
    rows_greedy, root_greedy = mcts(iterations=10, C=1.4, seed=7, rollout_policy="greedy")

    print("MCTS com rollout aleatório")
    for row in rows_random:
        print(row)
    print("Resumo:", mcts_summary(root_random))
    print("UCT:", uct_table_from_root(root_random, C=1.4))

    print("\nMCTS com rollout guloso")
    for row in rows_greedy:
        print(row)
    print("Resumo:", mcts_summary(root_greedy))
    print("UCT:", uct_table_from_root(root_greedy, C=1.4))

    print("\nComparação de C com rollout aleatório")
    for C, result in compare_c_values(rollout_policy="random").items():
        print("C=", C, result["summary"])
