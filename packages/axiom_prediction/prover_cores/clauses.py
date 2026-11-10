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

    args = [term_text(arg, variables) for arg in term.args]
    name = symbol_text(term.symbol)
    return f"{name}({','.join(args)})" if args else name

def literal_text(literal, variables, syntax):
    atom = literal.atom
    if atom.symbol in ("=", "equal___"):
        pattern = syntax.equal if literal.polarity else syntax.unequal
        return pattern.format(*(term_text(arg, variables) for arg in atom.args))

    text = syntax.constants.get(atom.symbol) or term_text(atom, variables)
    return text if literal.polarity else syntax.negation + text

def clause_text(clause, syntax):
    variables = {}
    literals = [literal_text(literal, variables, syntax) for literal in clause.literals]
    return " | ".join(literals) or syntax.constants["$false"]
