# Decodes the matrix and the core that `satcop --collect` prints; changed or incomplete output is rejected.

import json
import re

from connections.recursion import deep_recursion
from connections.syntax.formula import Atom, Function, Variable
from connections.syntax.matrix import Clause, Literal, Matrix

proved_statuses = ("Theorem", "Unsatisfiable")

def parse_literals(value):
    if value == "Block([])":
        return []

    items = value[len("Block(["): -len("])")].split(", ")
    matches = [re.fullmatch(r"(¬?)#(\d+)", item) for item in items]
    if not value.startswith("Block([") or any(match is None for match in matches):
        raise ValueError("unsupported SATCoP literal format")

    return [(not match[1], int(match[2])) for match in matches]

# Symbol rows hold the arity; clause rows hold (variables, first literal, literal stop, is goal).
def parse_records(text):
    tables = {"symbols": [], "clauses": []}
    core = set()
    for line in text.splitlines():
        if not line.startswith("% SATCoP "):
            continue

        kind, value = line[len("% SATCoP "):].split(" ", 1)
        symbol = re.fullmatch(r"#(\d+) (\d+)", value)
        clause = re.fullmatch(r"#(\d+) #(\d+) #(\d+)-#(\d+) (true|false)", value)
        if kind == "terms" and value.startswith("Block(["):
            tables["terms"] = json.loads(value[len("Block("):-1])
        elif kind == "literals":
            tables["literals"] = parse_literals(value)
        elif kind == "symbol" and symbol is not None and int(symbol[1]) == len(tables["symbols"]):
            tables["symbols"].append(int(symbol[2]))
        elif kind == "clause" and clause is not None and int(clause[1]) == len(tables["clauses"]):
            tables["clauses"].append((int(clause[2]), int(clause[3]), int(clause[4]), clause[5] == "true"))
        elif kind == "core":
            core.add(int(value))
        else:
            raise ValueError(f"unsupported SATCoP dump record: {line}")

    if not {"terms", "literals"} <= tables.keys() or not tables["symbols"] or not tables["clauses"]:
        raise ValueError("incomplete SATCoP matrix dump")

    return tables, sorted(core)

# A term starts with a positive symbol ID, followed by its arguments' term IDs, or with minus a variable ID.
def decode_term(tables, index, variables):
    value = tables["terms"][index]
    if value <= 0:
        return variables[-value]

    args = [
        decode_term(tables, tables["terms"][index + position + 1], variables)
        for position in range(tables["symbols"][value])
    ]

    # Native IDs keep generated and input symbols apart, whatever their names.
    return Function(f"{value}:s{value}", tuple(args))

def decode_clause(tables, index):
    variable_count, start, stop, is_goal = tables["clauses"][index]
    variables = [Variable(f"X{i}", vid = index + 1) for i in range(variable_count)]
    literals = []
    for positive, atom in tables["literals"][start:stop]:
        term = decode_term(tables, atom, variables)
        literals.append(Literal(Atom(term.symbol, term.args), polarity = positive))

    role = "conjecture" if is_goal else "axiom"
    return Clause(tuple(literals), role = role, free_variables = tuple(variables))

@deep_recursion()
def decode_matrix(tables, have_conjecture):
    clauses = [decode_clause(tables, index) for index in range(len(tables["clauses"]))]
    return Matrix(tuple(clauses), source_has_conjecture = have_conjecture)

# The matrix that `satcop --collect --clausify` prints without searching.
def parse_matrix(text):
    return decode_matrix(parse_records(text)[0], None)

# The matrix and core clause IDs of a finished `satcop --collect` proof.
def parse_proof(text):
    status = re.search(rf"^% SZS status ({'|'.join(proved_statuses)})\b", text, re.MULTILINE)
    end = re.search(r"^% SZS output end ListOfCNF(?: |$)", text, re.MULTILINE)
    if status is None or end is None:
        raise ValueError("SATCoP dump is missing a complete successful proof")

    tables, core = parse_records(text)
    if not core or core[-1] >= len(tables["clauses"]):
        raise ValueError("empty or out-of-matrix SATCoP core")

    return decode_matrix(tables, status[1] == "Theorem"), core

def axiom_clause_ids(matrix):
    return [index for index, clause in enumerate(matrix.clauses) if clause.role == "axiom"]
