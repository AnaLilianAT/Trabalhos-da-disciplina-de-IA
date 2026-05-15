"""
Questão 02 — Busca gulosa pela melhor escolha e A*.

Inclui também as modificações na heurística:
h(B): 8 -> 2, h(C): 7 -> 12, h(R): 2 -> 0.
"""
import heapq
import itertools
import math

EDGES_Q2 = {
    "A": [("B", 2), ("C", 4), ("D", 3)],
    "B": [("E", 3), ("F", 5)],
    "C": [("G", 4), ("H", 6)],
    "D": [("I", 2)],
    "E": [("J", 4)],
    "F": [("K", 3), ("L", 5)],
    "G": [("M", 6)],
    "H": [("N", 3), ("O", 4)],
    "I": [("P", 5)],
    "J": [("Q", 4)],
    "K": [("R", 3)],
    "L": [],
    "M": [("S", 2)],
    "N": [],
    "O": [("T", 5)],
    "P": [],
    "Q": [],
    "R": [("T", 4)],
    "S": [("T", 3)],
    "T": [],
}

H_Q2 = {
    "A": 10, "B": 8, "C": 7, "D": 9, "E": 6, "F": 5,
    "G": 6, "H": 4, "I": 7, "J": 5, "K": 3, "L": 6,
    "M": 3, "N": 4, "O": 1, "P": 8, "Q": 4, "R": 2,
    "S": 1, "T": 0,
}

H_Q2_MOD = dict(H_Q2)
H_Q2_MOD.update({"B": 2, "C": 12, "R": 0})


def greedy_best_first(h, start="A", goal="T"):
    counter = itertools.count()
    pq = [(h[start], next(counter), start, [start], 0)]
    closed = set()
    generated = {start}
    trace = []

    while pq:
        _, _, node, path, g = heapq.heappop(pq)
        if node in closed:
            continue
        closed.add(node)

        if node == goal:
            trace.append((node, g, h[node], [(x[2], x[0], x[4]) for x in sorted(pq)]))
            return path, g, generated, closed, trace

        for child, cost in EDGES_Q2[node]:
            if child not in closed:
                generated.add(child)
                heapq.heappush(pq, (h[child], next(counter), child, path + [child], g + cost))

        trace.append((node, g, h[node], [(x[2], x[0], x[4]) for x in sorted(pq)]))

    return None, math.inf, generated, closed, trace


def a_star(h, start="A", goal="T"):
    # Desempate: menor f, depois menor h, depois ordem de geração.
    counter = itertools.count()
    pq = [(h[start], h[start], next(counter), start, [start], 0)]
    best_g = {start: 0}
    closed = set()
    generated = {start}
    trace = []

    while pq:
        f, _, _, node, path, g = heapq.heappop(pq)
        if node in closed:
            continue
        closed.add(node)

        if node == goal:
            trace.append((node, g, h[node], f, [(x[3], x[0], x[1], x[5]) for x in sorted(pq)]))
            return path, g, generated, closed, trace

        for child, cost in EDGES_Q2[node]:
            ng = g + cost
            if child not in best_g or ng < best_g[child]:
                best_g[child] = ng
                generated.add(child)
                heapq.heappush(pq, (ng + h[child], h[child], next(counter), child, path + [child], ng))

        trace.append((node, g, h[node], f, [(x[3], x[0], x[1], x[5]) for x in sorted(pq)]))

    return None, math.inf, generated, closed, trace


def print_greedy_result(title, result):
    path, cost, generated, closed, trace = result
    print(f"\n{title}")
    print("Caminho:", path)
    print("Custo do caminho:", cost)
    print("Nós gerados:", len(generated), sorted(generated))
    print("Nós selecionados/expandidos:", len(closed), list(closed))
    for step, (node, g, h, frontier) in enumerate(trace, start=1):
        print(f"  {step:02d}. nó={node}, g={g}, h={h}, fronteira={frontier}")


def print_astar_result(title, result):
    path, cost, generated, closed, trace = result
    print(f"\n{title}")
    print("Caminho:", path)
    print("Custo do caminho:", cost)
    print("Nós gerados:", len(generated), sorted(generated))
    print("Nós selecionados/expandidos:", len(closed), list(closed))
    for step, (node, g, h, f, frontier) in enumerate(trace, start=1):
        print(f"  {step:02d}. nó={node}, g={g}, h={h}, f={f}, fronteira={frontier}")


if __name__ == "__main__":
    print_greedy_result("Busca Gulosa — heurística original", greedy_best_first(H_Q2))
    print_astar_result("A* — heurística original", a_star(H_Q2))

    print_greedy_result("Busca Gulosa — heurística modificada", greedy_best_first(H_Q2_MOD))
    print_astar_result("A* — heurística modificada", a_star(H_Q2_MOD))
