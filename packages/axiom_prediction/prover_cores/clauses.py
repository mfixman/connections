# Prints Connections clauses in the syntax of an external prover.
import re

from connections.syntax.formula import Variable

class Syntax:
    def __init__(self, negation, true, false):
        self.negation = negation
        self.constants = {"$true": true, "$false": false}

tptp = Syntax("~ ", "$true", "$false")
prover9 = Syntax("-", "$T", "$F")

def symbol_text(symbol):
    if re.fullmatch("[a-z][A-Za-z0-9_]*", symbol):
        return symbol

    raise ValueError(f"unsupported symbol {symbol!r}")

def term_text(term, variables, syntax):
    if isinstance(term, Variable):
        return variables.setdefault(term, f"V{len(variables)}")

    if term.symbol in syntax.constants:
        return syntax.constants[term.symbol]

    args = [term_text(arg, variables, syntax) for arg in term.args]
    if term.symbol in ("=", "equal___"):
        return f"({args[0]} = {args[1]})"

    name = symbol_text(term.symbol)
    return f"{name}({','.join(args)})" if args else name

def clause_text(clause, syntax):
    variables = {}
    literals = [
        ("" if literal.polarity else syntax.negation) + term_text(literal.atom, variables, syntax)
        for literal in clause.literals
    ]

    return " | ".join(literals) or syntax.constants["$false"]
