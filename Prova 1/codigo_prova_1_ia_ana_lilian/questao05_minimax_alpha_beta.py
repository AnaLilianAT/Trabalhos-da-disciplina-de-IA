"""
Questão 05 — Minimax, poda Alpha-Beta, Alpha-Beta reordenado,
Minimax limitado e experimento adicional.
"""
import math
from copy import deepcopy

TREE_Q5 = {
    "R": ["A", "B", "C"],
    "A": ["A1", "A2"],
    "B": ["B1", "B2"],
    "C": ["C1", "C2"],
    "A1": ["L1", "L2"],
    "A2": ["L3", "L4"],
    "B1": ["L5", "L6"],
    "B2": ["L7", "L8"],
    "C1": ["L9", "L10"],
    "C2": ["L11", "L12"],
}

VALUES_Q5 = {
    "L1": 3, "L2": 5, "L3": 6, "L4": 9,
    "L5": 1, "L6": 2, "L7": 0, "L8": -1,
    "L9": 7, "L10": 4, "L11": 5, "L12": 6,
}

TYPES_Q5 = {
    "R": "MAX",
    "A": "MIN", "B": "MIN", "C": "MIN",
    "A1": "MAX", "A2": "MAX", "B1": "MAX", "B2": "MAX", "C1": "MAX", "C2": "MAX",
}

HEUR_DEPTH2 = {"A1": 4, "A2": 7, "B1": 2, "B2": 5, "C1": 6, "C2": 1}


def minimax(node="R", tree=TREE_Q5, values=VALUES_Q5, types=TYPES_Q5, trace=None):
    if trace is None:
        trace = []
    if node in values:
        trace.append((node, "FOLHA", values[node]))
        return values[node], trace

    child_values = []
    for child in tree[node]:
        val, trace = minimax(child, tree, values, types, trace)
        child_values.append(val)

    result = max(child_values) if types[node] == "MAX" else min(child_values)
    trace.append((node, types[node], result))
    return result, trace


def alpha_beta(node="R", alpha=-math.inf, beta=math.inf, tree=TREE_Q5, values=VALUES_Q5, types=TYPES_Q5, explored=None, pruned=None):
    if explored is None:
        explored = []
    if pruned is None:
        pruned = []

    explored.append(node)

    if node in values:
        return values[node], explored, pruned

    if types[node] == "MAX":
        value = -math.inf
        for i, child in enumerate(tree[node]):
            child_value, explored, pruned = alpha_beta(child, alpha, beta, tree, values, types, explored, pruned)
            value = max(value, child_value)
            alpha = max(alpha, value)
            if alpha >= beta:
                pruned.extend(tree[node][i + 1:])
                break
        return value, explored, pruned

    value = math.inf
    for i, child in enumerate(tree[node]):
        child_value, explored, pruned = alpha_beta(child, alpha, beta, tree, values, types, explored, pruned)
        value = min(value, child_value)
        beta = min(beta, value)
        if beta <= alpha:
            pruned.extend(tree[node][i + 1:])
            break
    return value, explored, pruned


def reordered_tree_for_more_pruning():
    """Reordenação usada para maximizar podas: R: C, A, B; C: C2, C1; A: A1, A2; B: B2, B1."""
    tree = deepcopy(TREE_Q5)
    tree["R"] = ["C", "A", "B"]
    tree["C"] = ["C2", "C1"]
    tree["A"] = ["A1", "A2"]
    tree["B"] = ["B2", "B1"]
    tree["C2"] = ["L12", "L11"]
    tree["C1"] = ["L9", "L10"]
    tree["A1"] = ["L2", "L1"]
    tree["A2"] = ["L4", "L3"]
    tree["B2"] = ["L7", "L8"]
    tree["B1"] = ["L6", "L5"]
    return tree


def minimax_depth_limited(node="R", depth=0, limit=2, tree=TREE_Q5, values=VALUES_Q5, types=TYPES_Q5):
    if node in values:
        return values[node]
    if depth == limit:
        return HEUR_DEPTH2[node]

    vals = [minimax_depth_limited(ch, depth + 1, limit, tree, values, types) for ch in tree[node]]
    return max(vals) if types[node] == "MAX" else min(vals)


def experiment_modified_leaves():
    """Experimento adicional: altera exatamente três folhas."""
    modified = dict(VALUES_Q5)
    modified.update({"L1": 8, "L2": 8, "L8": 4})
    value, trace = minimax(values=modified)
    ab_value, explored, pruned = alpha_beta(values=modified)
    return modified, value, trace, ab_value, explored, pruned


if __name__ == "__main__":
    value, trace = minimax()
    print("Minimax completo:", value)
    print("Ordem de cálculo:", trace)

    ab_value, explored, pruned = alpha_beta()
    print("\nAlpha-Beta:", ab_value)
    print("Nós explorados:", explored)
    print("Nós podados:", pruned)

    tree_reordered = reordered_tree_for_more_pruning()
    ab_r_value, ab_r_explored, ab_r_pruned = alpha_beta(tree=tree_reordered)
    print("\nAlpha-Beta reordenado:", ab_r_value)
    print("Nós explorados:", ab_r_explored)
    print("Nós podados:", ab_r_pruned)

    print("\nMinimax limitado na profundidade 2:", minimax_depth_limited(limit=2))

    modified, value_mod, trace_mod, ab_value_mod, explored_mod, pruned_mod = experiment_modified_leaves()
    print("\nExperimento adicional")
    print("Valores modificados:", modified)
    print("Minimax modificado:", value_mod)
    print("Alpha-Beta modificado:", ab_value_mod)
    print("Explorados:", explored_mod)
    print("Podados:", pruned_mod)
