"""
Questão 01 — Busca em largura (BFS), busca em profundidade (DFS)
e busca em profundidade iterativa (IDS).

"""
from collections import deque

G1 = {
    "A": ["B", "C", "D"],
    "B": ["E", "F"],
    "C": ["G", "H"],
    "D": ["I"],
    "E": ["J"],
    "F": ["K", "L"],
    "G": ["M"],
    "H": ["N", "O"],
    "I": ["P"],
    "J": [],
    "K": ["Q"],
    "L": [],
    "M": ["R"],
    "N": [],
    "O": ["S"],
    "P": [],
    "Q": [],
    "R": [],
    "S": [],
}

G1_MOD = dict(G1)
G1_MOD["C"] = ["H", "G"]
G1_MOD["H"] = ["O", "N"]


def bfs(graph, start="A", goal="S"):
    frontier = deque([(start, [start])])
    generated = {start}
    trace = []
    selected = []

    while frontier:
        node, path = frontier.popleft()
        selected.append(node)

        if node == goal:
            trace.append((node, [n for n, _ in frontier]))
            return path, trace, generated, selected

        for child in graph[node]:
            if child not in generated:
                generated.add(child)
                frontier.append((child, path + [child]))

        trace.append((node, [n for n, _ in frontier]))

    return None, trace, generated, selected


def dfs(graph, start="A", goal="S"):
    stack = [(start, [start])]
    generated = {start}
    trace = []
    selected = []

    while stack:
        node, path = stack.pop()
        selected.append(node)

        if node == goal:
            trace.append((node, [n for n, _ in reversed(stack)]))
            return path, trace, generated, selected

        for child in reversed(graph[node]):
            if child not in generated:
                generated.add(child)
                stack.append((child, path + [child]))

        trace.append((node, [n for n, _ in reversed(stack)]))

    return None, trace, generated, selected


def depth_limited(graph, limit, start="A", goal="S"):
    order = []

    def rec(node, path, depth):
        order.append(node)
        if node == goal:
            return path
        if depth == limit:
            return None
        for child in graph[node]:
            ans = rec(child, path + [child], depth + 1)
            if ans is not None:
                return ans
        return None

    return rec(start, [start], 0), order


def ids(graph, start="A", goal="S", max_depth=20):
    all_orders = []
    for limit in range(max_depth + 1):
        path, order = depth_limited(graph, limit, start, goal)
        all_orders.append((limit, order))
        if path is not None:
            return path, all_orders
    return None, all_orders


def print_trace(title, result):
    path, trace, generated, selected = result
    print(f"\n{title}")
    print("Caminho:", path)
    print("Nós gerados:", len(generated), sorted(generated))
    print("Nós selecionados/expandidos:", len(selected), selected)
    print("Traço:")
    for step, (node, frontier) in enumerate(trace, start=1):
        print(f"  {step:02d}. selecionado={node}, fronteira={frontier}")


if __name__ == "__main__":
    print_trace("BFS — grafo original", bfs(G1))
    print_trace("DFS — grafo original", dfs(G1))

    ids_path, ids_orders = ids(G1)
    print("\nIDS — grafo original")
    print("Caminho:", ids_path)
    for limit, order in ids_orders:
        print(f"  limite={limit}: {order}")

    print_trace("BFS — grafo modificado", bfs(G1_MOD))
    print_trace("DFS — grafo modificado", dfs(G1_MOD))

    ids_path_mod, ids_orders_mod = ids(G1_MOD)
    print("\nIDS — grafo modificado")
    print("Caminho:", ids_path_mod)
    for limit, order in ids_orders_mod:
        print(f"  limite={limit}: {order}")
