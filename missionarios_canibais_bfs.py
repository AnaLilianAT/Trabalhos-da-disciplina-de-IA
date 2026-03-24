"""

Autores: Ana Lilian Alfonso Toledo, Tobias Viero de Oliveira

"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class State:
    """
    Representa um estado do problema.

    ml = missionários na margem esquerda
    cl = canibais na margem esquerda
    mr = missionários na margem direita
    cr = canibais na margem direita
    boat_side = lado do barco: "L" (esquerda) ou "R" (direita)
    """

    ml: int
    cl: int
    mr: int
    cr: int
    boat_side: str


def is_valid_state(state: State) -> bool:
    """Verifica se um estado respeita todas as restrições do problema."""
    # Canibais não podem superar missionários em nenhuma margem (quando há missionários)
    if state.ml > 0 and state.cl > state.ml:
        return False
    if state.mr > 0 and state.cr > state.mr:
        return False
    return True


def generate_successors(state: State, boat_capacity: int) -> List[State]:
    """
    Gera todos os sucessores válidos de um estado.

    Cada travessia transporta (m, c) passageiros, com:
    - 1 <= m + c <= boat_capacity
    - m >= 0, c >= 0
    - m == 0 ou m >= c (para evitar estados inválidos)
    """
    successors: List[State] = []

    passenger_options: List[Tuple[int, int]] = []
    for m in range(boat_capacity + 1):
        for c in range(boat_capacity + 1):
            if 1 <= m + c <= boat_capacity and (m==0 or m >= c):
                passenger_options.append((m, c))

    if state.boat_side == "L":
        # Barco sai da esquerda para a direita
        for m, c in passenger_options:
            if m <= state.ml and c <= state.cl:
                next_state = State(
                    ml=state.ml - m,
                    cl=state.cl - c,
                    mr=state.mr + m,
                    cr=state.cr + c,
                    boat_side="R",
                )
                if is_valid_state(next_state):
                    successors.append(next_state)
    else:
        # Barco sai da direita para a esquerda
        for m, c in passenger_options:
            if m <= state.mr and c <= state.cr:
                next_state = State(
                    ml=state.ml + m,
                    cl=state.cl + c,
                    mr=state.mr - m,
                    cr=state.cr - c,
                    boat_side="L",
                )
                if is_valid_state(next_state):
                    successors.append(next_state)

    return successors


def reconstruct_path(
    goal_state: State,
    parent: Dict[State, Optional[State]],
) -> List[State]:
    """Reconstrói o caminho do estado inicial até o estado objetivo."""
    states_path: List[State] = []

    current = goal_state
    while current is not None:
        states_path.append(current)
        current = parent[current]

    states_path.reverse()
    return states_path


def print_search_tree(expansion_log: List[Tuple[int, State, List[State]]]) -> None:
    """
    Imprime a árvore de busca no formato:
    - nível
    - estado pai
    - estados filhos gerados
    """
    print("\n" + "=" * 72)
    print("GERAÇÃO DA ÁRVORE DE BUSCA (BFS)")
    print("=" * 72)

    if not expansion_log:
        print("Nenhum nó foi expandido.")
        return

    for level, parent_state, children in expansion_log:
        print(f"\nNível {level} | Pai: {parent_state}")
        if children:
            print("  Filhos gerados:")
            for child in children:
                print(f"    -> {child}")
        else:
            print("  Filhos gerados: nenhum (nó folha ou apenas repetidos)")


def bfs(total_m: int, total_c: int, boat_capacity: int) -> Tuple[
    bool,
    Optional[State],
    Dict[State, Optional[State]],
    int,
    int,
    int,
    List[Tuple[int, State, List[State]]],
]:
    """
    Executa BFS e retorna:
    - se encontrou solução
    - estado objetivo (ou None)
    - mapa de pais
    - número de estados visitados (únicos)
    - número de estados gerados (incluindo duplicatas descartadas)
    - número de nós expandidos
    - log de expansão da árvore de busca
    """
    initial = State(total_m, total_c, 0, 0, "L")
    goal = State(0, 0, total_m, total_c, "R")

    if not is_valid_state(initial):
        return False, None, {}, 0, 0, 0, []

    queue = deque([(initial, 0)])
    visited: Set[State] = {initial}

    parent: Dict[State, Optional[State]] = {initial: None}

    expansion_log: List[Tuple[int, State, List[State]]] = []
    total_gerados: int = 0
    nodes_expanded: int = 0

    while queue:
        current, current_depth = queue.popleft()

        if current == goal:
            return True, current, parent, len(visited), total_gerados, nodes_expanded, expansion_log

        generated_children: List[State] = []

        for next_state in generate_successors(current, boat_capacity):
            total_gerados += 1
            if next_state not in visited:
                visited.add(next_state)
                parent[next_state] = current
                queue.append((next_state, current_depth + 1))
                generated_children.append(next_state)

        nodes_expanded += 1
        expansion_log.append((current_depth, current, generated_children))

    return False, None, parent, len(visited), total_gerados, nodes_expanded, expansion_log


def run_example(total_m: int, total_c: int, boat_capacity: int) -> None:
    initial = State(total_m, total_c, 0, 0, "L")
    goal = State(0, 0, total_m, total_c, "R")

    print("=" * 72)
    print("PROBLEMA DOS MISSIONÁRIOS E CANIBAIS - BUSCA EM LARGURA (BFS)")
    print("=" * 72)
    print(f"Parâmetros: M={total_m}, C={total_c}, B={boat_capacity}")
    print(f"Estado inicial: {initial}")
    print(f"Estado objetivo: {goal}")

    found, goal_state, parent, visited_count, generated_count, nodes_expanded, expansion_log = bfs(
        total_m, total_c, boat_capacity
    )

    print_search_tree(expansion_log)

    print("\n" + "=" * 72)
    print("RESULTADO")
    print("=" * 72)

    if not found or goal_state is None:
        print("Não existe solução para os parâmetros informados.")
        print(f"Estados visitados (únicos): {visited_count}")
        print(f"Estados testados (gerados): {generated_count}")
        print(f"Nós expandidos: {nodes_expanded}")
        return

    states_path = reconstruct_path(goal_state, parent)

    print("\nSequência ótima de estados:")
    for i, st in enumerate(states_path):
        print(f"Passo {i}: {st}")

    total_crossings = len(states_path) - 1

    print("\nResumo final:")
    print(f"- Quantidade total de travessias: {total_crossings}")
    print(f"- Número de estados visitados (únicos): {visited_count}")
    print(f"- Número de estados testados (gerados, incl. duplicatas): {generated_count}")
    print(f"- Número de nós expandidos: {nodes_expanded}")


if __name__ == "__main__":
    
    MISSIONARIES = int(input("Digite o número de missionários (M): "))
    CANNIBALS = int(input("Digite o número de canibais (C): "))
    BOAT_CAPACITY = int(input("Digite a capacidade da embarcação (B): "))

    run_example(MISSIONARIES, CANNIBALS, BOAT_CAPACITY)
