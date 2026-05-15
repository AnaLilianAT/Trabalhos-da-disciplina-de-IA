"""
Questão 03 — Problema das N-rainhas com busca local.

Inclui Hill-Climbing, Random Restart Hill-Climbing e Simulated Annealing.
"""
from itertools import combinations
import math
import random


def queens_h(state):
    """Número de pares de rainhas em conflito."""
    conflicts = 0
    for i, j in combinations(range(len(state)), 2):
        same_row = state[i] == state[j]
        same_diag = abs(state[i] - state[j]) == abs(i - j)
        if same_row or same_diag:
            conflicts += 1
    return conflicts


def queen_neighbors(state):
    n = len(state)
    for col in range(n):
        for row in range(1, n + 1):
            if row != state[col]:
                ns = list(state)
                ns[col] = row
                yield tuple(ns), (col + 1, row), queens_h(tuple(ns))


def hill_climbing_queens(start):
    state = tuple(start)
    trace = []

    while True:
        neighbors = sorted(queen_neighbors(state), key=lambda x: (x[2], x[1][0], x[1][1]))
        best_state, move, best_h = neighbors[0]
        current_h = queens_h(state)
        trace.append((state, current_h, neighbors[:5], move, best_state, best_h))

        if best_h < current_h:
            state = best_state
        else:
            return state, current_h, trace


def random_restart_hill_climbing_queens(seed=42, runs=20):
    rng = random.Random(seed)
    results = []

    for i in range(1, runs + 1):
        init = tuple(rng.randint(1, 8) for _ in range(8))
        final, hf, trace = hill_climbing_queens(init)
        results.append((i, init, len(trace) - 1, hf, final, hf == 0))

    return results


def simulated_annealing_queens(seed=99, T0=15.0, alpha=0.97, Tmin=0.001, max_steps=5000):
    rng = random.Random(seed)
    state = tuple(rng.randint(1, 8) for _ in range(8))
    T = T0
    accepted_worse = []

    for step in range(max_steps):
        hcur = queens_h(state)
        if hcur == 0:
            return state, step, hcur, accepted_worse

        candidate = rng.choice([n for n, _, _ in queen_neighbors(state)])
        delta = queens_h(candidate) - hcur
        prob = 1.0 if delta <= 0 else math.exp(-delta / T)
        u = rng.random()

        if delta <= 0 or u < prob:
            if delta > 0 and len(accepted_worse) < 5:
                accepted_worse.append((step, state, hcur, candidate, queens_h(candidate), delta, T, prob, u))
            state = candidate

        T = max(T * alpha, Tmin)

    return state, max_steps, queens_h(state), accepted_worse


if __name__ == "__main__":
    start = [1] * 8
    final, hf, trace = hill_climbing_queens(start)
    print("Hill-Climbing")
    print("Estado inicial:", start, "h=", queens_h(tuple(start)))
    print("Estado final:", final, "h=", hf)
    print("Iterações:", len(trace))
    for i, row in enumerate(trace, start=1):
        state, hval, best_neighbors, move, best_state, best_h = row
        print(f"  {i:02d}. estado={state}, h={hval}, melhor_mov={move}, melhor={best_state}, h_melhor={best_h}")

    print("\nRandom Restart Hill-Climbing")
    rr = random_restart_hill_climbing_queens(seed=42, runs=20)
    for row in rr:
        print(row)
    print("Soluções encontradas:", sum(1 for row in rr if row[-1]))

    print("\nSimulated Annealing")
    print(simulated_annealing_queens(seed=99)[:3])
