"""
Questão 04 — Problema de Satisfação de Restrições (CSP) para escala de plantões.

Foi implementado:
1. Backtracking simples.
2. Backtracking com MRV + Degree Heuristic.
3. Forward Checking
4. Backjumping simples para registrar conflitos.

Restrições:
1. Ti != T(i+1), para i = 1,...,5.
2. T3 != A.
3. T1 = B ou T2 = B.
4. Não pode ocorrer simultaneamente T2 = C e T5 = C.
5. O médico D pode aparecer no máximo duas vezes.
"""

VARS_Q4 = ["T1", "T2", "T3", "T4", "T5", "T6"]
DOCTORS = ["A", "B", "C", "D"]

# Grau aproximado das variáveis no grafo de restrições.
# T2 é a mais conectada: T1, T3, T5 e restrição T1/T2.
DEGREE = {"T1": 2, "T2": 4, "T3": 2, "T4": 2, "T5": 3, "T6": 1}


def initial_domains():
    """Domínios iniciais já aplicando a restrição unária T3 != A."""
    domains = {var: set(DOCTORS) for var in VARS_Q4}
    domains["T3"].discard("A")
    return domains


def csp_consistent(assignment):
    """Verifica se uma atribuição parcial respeita as restrições já avaliáveis."""

    # Restrição 1: o mesmo médico não pode trabalhar em turnos consecutivos.
    for i in range(1, 6):
        a, b = f"T{i}", f"T{i + 1}"
        if a in assignment and b in assignment and assignment[a] == assignment[b]:
            return False

    # Restrição 2: A não trabalha em T3.
    if assignment.get("T3") == "A":
        return False

    # Restrição 3: B deve estar em T1 ou T2.
    if "T1" in assignment and "T2" in assignment:
        if assignment["T1"] != "B" and assignment["T2"] != "B":
            return False

    # Restrição 4: C não pode aparecer simultaneamente em T2 e T5.
    if assignment.get("T2") == "C" and assignment.get("T5") == "C":
        return False

    # Restrição 5: D aparece no máximo duas vezes.
    if sum(1 for v in assignment.values() if v == "D") > 2:
        return False

    return True


def legal_values_for(var, assignment, domains=None):
    """
    Retorna os valores ainda possíveis para uma variável, considerando apenas
    a consistência direta com a atribuição parcial atual.
    """
    if domains is None:
        domains = initial_domains()

    values = []
    for val in sorted(domains[var], key=DOCTORS.index):
        test = dict(assignment)
        test[var] = val
        if csp_consistent(test):
            values.append(val)
    return values


def select_mrv_degree_variable(assignment, domains=None):
    """
    Seleciona variável usando MRV e, em caso de empate, Degree Heuristic.
    """
    if domains is None:
        domains = initial_domains()

    unassigned = [v for v in VARS_Q4 if v not in assignment]

    def key(var):
        remaining = legal_values_for(var, assignment, domains)
        return (len(remaining), -DEGREE[var], VARS_Q4.index(var))

    return min(unassigned, key=key)


def conflict_set(var, assignment):
    """Retorna variáveis anteriores envolvidas no conflito com var."""
    conflicts = set()
    idx = int(var[1])
    val = assignment.get(var)

    if var == "T3" and val == "A":
        conflicts.add("T3")

    for nb_idx in (idx - 1, idx + 1):
        nb = f"T{nb_idx}"
        if nb in assignment and assignment[nb] == val:
            conflicts.add(nb)

    if var in {"T1", "T2"} and "T1" in assignment and "T2" in assignment:
        if assignment["T1"] != "B" and assignment["T2"] != "B":
            conflicts.update({"T1", "T2"} - {var})

    if var in {"T2", "T5"} and assignment.get("T2") == "C" and assignment.get("T5") == "C":
        conflicts.update({"T2", "T5"} - {var})

    if val == "D" and sum(1 for v in assignment.values() if v == "D") > 2:
        conflicts.update(k for k, v in assignment.items() if k != var and v == "D")

    return conflicts


# 1. Backtracking simples, com ordem fixa de variáveis.
def backtracking_csp(order=VARS_Q4, values=DOCTORS):
    trace = []
    backtracks = 0

    def bt(assignment, idx):
        nonlocal backtracks

        if idx == len(order):
            return dict(assignment)

        var = order[idx]
        for val in values:
            assignment[var] = val
            ok = csp_consistent(assignment)
            trace.append((var, val, dict(assignment), ok))

            if ok:
                ans = bt(assignment, idx + 1)
                if ans is not None:
                    return ans

            del assignment[var]

        backtracks += 1
        return None

    return bt({}, 0), trace, backtracks


# 2. Backtracking com MRV + Degree.
def mrv_degree_csp(values=DOCTORS):
    domains = initial_domains()
    trace = []
    backtracks = 0

    def bt(assignment):
        nonlocal backtracks

        if len(assignment) == len(VARS_Q4):
            return dict(assignment)

        var = select_mrv_degree_variable(assignment, domains)
        possible_values = legal_values_for(var, assignment, domains)
        trace.append(("seleciona", var, "MRV+Degree", dict(assignment), possible_values))

        for val in possible_values:
            assignment[var] = val
            ok = csp_consistent(assignment)
            trace.append((var, val, dict(assignment), ok))

            if ok:
                ans = bt(assignment)
                if ans is not None:
                    return ans

            del assignment[var]

        backtracks += 1
        return None

    return bt({}), trace, backtracks


# 3. Forward Checking.
def apply_forward_checking(var, val, assignment, domains):
    """
    Aplica reduções de domínio causadas por atribuir var=val.
    Retorna (novos_domínios, reduções, domínio_vazio).
    """
    new_domains = {x: set(d) for x, d in domains.items()}
    new_domains[var] = {val}
    reductions = []

    def remove_value(target, value, reason):
        if target in new_domains and target not in assignment and value in new_domains[target]:
            new_domains[target].remove(value)
            reductions.append((target, value, reason))

    def restrict_to(target, allowed, reason):
        if target in new_domains and target not in assignment:
            before = set(new_domains[target])
            new_domains[target] &= set(allowed)
            removed = before - new_domains[target]
            for value in sorted(removed, key=DOCTORS.index):
                reductions.append((target, value, reason))

    idx = int(var[1])

    # Restrição 1: turnos consecutivos não podem ter o mesmo médico.
    for nb in [f"T{idx - 1}", f"T{idx + 1}"]:
        if nb in new_domains:
            remove_value(nb, val, f"{var}={val} consecutivo")

    # Restrição 3: T1 = B ou T2 = B.
    if var == "T1" and val != "B":
        restrict_to("T2", {"B"}, "T1 não é B; força T2=B")
    if var == "T2" and val != "B":
        restrict_to("T1", {"B"}, "T2 não é B; força T1=B")

    # Restrição 4: não pode T2=C e T5=C simultaneamente.
    if var == "T2" and val == "C":
        remove_value("T5", "C", "T2=C impede T5=C")
    if var == "T5" and val == "C":
        remove_value("T2", "C", "T5=C impede T2=C")

    # Restrição 5: D aparece no máximo duas vezes.
    d_count = sum(1 for v in assignment.values() if v == "D") + (1 if val == "D" else 0)
    if d_count == 2:
        for u in VARS_Q4:
            remove_value(u, "D", "D já apareceu duas vezes")

    empty_domain = any(
        len(new_domains[u]) == 0
        for u in VARS_Q4
        if u not in assignment and u != var
    )

    return new_domains, reductions, empty_domain


def forward_checking_csp(order=VARS_Q4):
    domains = initial_domains()
    trace = []
    backtracks = 0

    def fc(assignment, domains, idx):
        nonlocal backtracks

        if idx == len(order):
            return dict(assignment)

        var = order[idx]
        for val in sorted(domains[var], key=DOCTORS.index):
            new_assignment = dict(assignment)
            new_assignment[var] = val

            if not csp_consistent(new_assignment):
                trace.append((var, val, dict(new_assignment), "inconsistente direto", []))
                continue

            new_domains, reductions, empty_domain = apply_forward_checking(
                var, val, assignment, domains
            )

            if empty_domain:
                trace.append((var, val, dict(new_assignment), "domínio vazio", reductions))
                continue

            trace.append((var, val, dict(new_assignment), "válido", reductions))
            ans = fc(new_assignment, new_domains, idx + 1)
            if ans is not None:
                return ans

        backtracks += 1
        return None

    return fc({}, domains, 0), trace, backtracks


# 4. Backjumping simples.
def backjumping_csp(order=VARS_Q4, values=DOCTORS):
    trace = []

    def bj(assignment, idx):
        if idx == len(order):
            return dict(assignment), set()

        var = order[idx]
        accumulated_conflicts = set()

        for val in values:
            assignment[var] = val
            ok = csp_consistent(assignment)
            conflicts = set() if ok else conflict_set(var, assignment)
            trace.append((var, val, dict(assignment), ok, sorted(conflicts)))

            if ok:
                result, child_conflicts = bj(assignment, idx + 1)
                if result is not None:
                    return result, set()

                if var not in child_conflicts:
                    del assignment[var]
                    return None, child_conflicts

                accumulated_conflicts.update(child_conflicts - {var})
            else:
                accumulated_conflicts.update(conflicts)

            del assignment[var]

        accumulated_conflicts.add(var)
        return None, accumulated_conflicts

    solution, conflicts = bj({}, 0)
    return solution, trace, conflicts


if __name__ == "__main__":
    sol, trace, backtracks = backtracking_csp()
    print("Backtracking simples")
    print("Solução:", sol)
    print("Retrocessos:", backtracks)
    for step, row in enumerate(trace, start=1):
        print(step, row)

    sol_mrv, trace_mrv, backtracks_mrv = mrv_degree_csp()
    print("\nMRV + Degree")
    print("Solução:", sol_mrv)
    print("Retrocessos:", backtracks_mrv)
    for step, row in enumerate(trace_mrv, start=1):
        print(step, row)

    sol_fc, trace_fc, backtracks_fc = forward_checking_csp()
    print("\nForward Checking")
    print("Solução:", sol_fc)
    print("Retrocessos:", backtracks_fc)
    for step, row in enumerate(trace_fc, start=1):
        print(step, row)

    sol_bj, trace_bj, conflicts_bj = backjumping_csp()
    print("\nBackjumping")
    print("Solução:", sol_bj)
    print("Conflitos finais:", conflicts_bj)
    for step, row in enumerate(trace_bj, start=1):
        print(step, row)
