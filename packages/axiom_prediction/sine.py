# Clause-level basic SInE, shared by live search and saved-data evaluation.

from collections import Counter, defaultdict

from connections.syntax.formula import Variable

def settings():
    return {"sine": True, "sine_settings": {
        "tolerance": 1, "depth": None, "generality_threshold": 0,
        "selection_unit": "clause",
    }}

def select(axioms, conjecture_symbols):
    frequencies = Counter(symbol for symbols in axioms for symbol in symbols)
    triggers = defaultdict(list)
    for index, symbols in enumerate(axioms):
        if symbols:
            least = min(map(frequencies.__getitem__, symbols))
            for symbol in symbols:
                if frequencies[symbol] == least:
                    triggers[symbol].append(index)

    active = set(conjecture_symbols)
    pending = list(active)
    scores = [0.0] * len(axioms)
    while pending:
        for index in triggers.get(pending.pop(), ()):
            if scores[index]:
                continue

            scores[index] = 1.0
            new = axioms[index] - active
            active.update(new)
            pending.extend(new)

    return scores

def matrix_symbols(matrix):
    clauses = []
    for clause in matrix.clauses:
        symbols = set()
        pending = []
        for literal in clause.literals:
            symbols.add((literal.atom.symbol, 0))
            pending.extend(literal.atom.args)

        while pending:
            term = pending.pop()
            if not isinstance(term, Variable):
                symbols.add((term.symbol, 1 if term.args else 2))
                pending.extend(term.args)

        clauses.append(symbols)

    return clauses

def graph_symbols(graph):
    symbols = dict(graph.edges["sym"])
    children = defaultdict(list)
    for parent, child, *_ in graph.edges["arg_term"]:
        children[parent].append(child)

    atoms = dict(graph.edges["atom"])
    clauses = [set() for _ in graph.nodes["clause"]]
    visited = [set() for _ in clauses]
    for clause, literal in graph.edges["contains"]:
        pending = [atoms[literal]]
        while pending:
            term = pending.pop()
            if term in visited[clause]:
                continue

            visited[clause].add(term)
            clauses[clause].add(symbols[term])
            pending.extend(children[term])

    return clauses

def clause_scores(symbols, axiom_clause_ids, conjecture_clause_ids):
    conjecture = set()
    for index in conjecture_clause_ids:
        conjecture.update(symbols[index])

    return select([symbols[index] for index in axiom_clause_ids], conjecture)

def score_matrix(matrix, axiom_clause_ids, conjecture_clause_ids):
    return clause_scores(matrix_symbols(matrix), axiom_clause_ids, conjecture_clause_ids)

def predict_examples(examples):
    for example in examples:
        graph = example.graph
        yield clause_scores(
            graph_symbols(graph.graph), graph.axiom_clause_ids, graph.conjecture_clause_ids,
        )
