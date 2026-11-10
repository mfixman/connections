# Prints Connections clauses in the syntax of an external prover.
import re

from connections.syntax.formula import Variable

class Syntax:
    def __init__(self, negation, equal, unequal, true, false):
        self.negation = negation
        self.equal = equal
        self.unequal = unequal
        self.constants = {"$true": true, "$false": false}

tptp = Syntax("~ ", "{} = {}", "{} != {}", "$true", "$false")
prover9 = Syntax("-", "({} = {})", "-({} = {})", "$T", "$F")

# Skolem and definition symbols are numbers, which no prover accepts as a functor.
def symbol_text(symbol):
    if re.fullmatch("[a-z][A-Za-z0-9_]*", symbol):
        return symbol

    return "enc__" + symbol.encode().hex()

def term_text(term, variables):
    if isinstance(term, Variable):
        return variables.setdefault(term, f"V{len(variables)}")

    name, arguments = symbol_text(term.symbol), term.args
    # Connections numbers Skolem and definition symbols in their first argument; Prover9 needs one arity per symbol.
    if name in ("f_skolem", "p_defini"):
        name, arguments = f"{name}_{arguments[0].symbol}", arguments[1:]

    args = [term_text(arg, variables) for arg in arguments]
    return f"{name}({','.join(args)})" if args else name

def literal_text(atom, positive, variables, syntax):
    if atom.symbol in ("=", "equal___"):
        pattern = syntax.equal if positive else syntax.unequal
        return pattern.format(*(term_text(arg, variables) for arg in atom.args))

    text = syntax.constants.get(atom.symbol) or term_text(atom, variables)
    return text if positive else syntax.negation + text

# Connections keeps FOF problems as a DNF of the theorem; flipping every literal gives the refutation clauses.
def clause_text(clause, syntax, flip):
    variables = {}
    literals = [
        literal_text(literal.atom, literal.polarity != flip, variables, syntax)
        for literal in clause.literals
    ]

    return " | ".join(literals) or syntax.constants["$false"]
