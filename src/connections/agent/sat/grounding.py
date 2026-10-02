"""SATCoP's common grounding and per-instance tautology constraints."""

from collections import Counter

from connections.constraints.term import TableauVariable
from connections.environment.rules import Extension, Start
from connections.syntax.formula import Atom, Function, Variable

class Grounding:
    def __init__(self, matrix):
        constants = Counter()
        symbols = set()
        for clause in matrix.clauses:
            terms = [arg for literal in clause for arg in literal.atom.args]
            while terms:
                term = terms.pop()
                if isinstance(term, Function):
                    symbols.add(term.symbol)
                    terms.extend(term.args)
                    if not term.args and clause.role == "conjecture":
                        constants[term.symbol] += 1

        fallback = "__ground__"
        while fallback in symbols:
            fallback += "_"

        self.constant = constants.most_common(1)[0][0] if constants else fallback

    def atom_key(self, state, literal, instance_id, pending_bindings = ()):
        args = [
            self.term_key(state.constraints.terms.substitute_term(
                arg, instance_id = instance_id, pending_bindings = pending_bindings,
            ))
            for arg in literal.atom.args
        ]
        symbol = literal.atom.symbol
        return f"{symbol}({','.join(args)})" if args else symbol

    def term_key(self, term):
        if isinstance(term, (Variable, TableauVariable)):
            return self.constant
        if not isinstance(term, Function) or not term.args:
            return str(term)

        return f"{term.symbol}({','.join(self.term_key(arg) for arg in term.args)})"

def reflexivity(clause):
    if len(clause) != 1:
        return False

    literal = clause.literal(0)
    args = literal.atom.args
    return (
        literal.atom.symbol in {"=", "equal___"} and len(args) == 2
        and isinstance(args[0], Variable) and args[0] == args[1]
    )

def equality_polarity(matrix):
    # FOF matrices are dualized; CNF matrices keep their input signs.
    for clause in matrix.clauses:
        if clause.role == "axiom" and reflexivity(clause):
            return clause.literal(0).polarity

    return None

def tautological(state, pending_bindings = ()):
    equality_sign = equality_polarity(state.matrix)
    for app in state.tableau.rule_applications.values():
        rule = app.rule
        if not isinstance(rule, (Start, Extension)) or reflexivity(rule.clause):
            continue

        atoms = [
            Atom(lit.atom.symbol, tuple(
                state.constraints.terms.substitute_term(
                    arg, instance_id = rule.instance_id, pending_bindings = pending_bindings,
                ) for arg in lit.atom.args
            )) for lit in rule.clause
        ]
        for index, literal in enumerate(rule.clause):
            atom = atoms[index]
            if literal.polarity == equality_sign and atom.symbol in {"=", "equal___"}:
                left, right = atom.args
                if left == right:
                    return True

            if any(
                literal.polarity != other.polarity and atom == atoms[other_index]
                for other_index, other in enumerate(rule.clause)
                if other_index > index
            ):
                return True

    return False
