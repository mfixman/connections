from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from connections.model_finding.core import InputError, ModelClause, ModelLiteral, ModelProblem
from connections.parsing.tptp.parser import TPTPParseError, parse_tptp_file
from connections.parsing.tptp.transformer import DefinedNumberSymbol, DistinctObjectSymbol, StmtCNF, StmtFOF, StmtQMF
from connections.syntax.formula import (
    And,
    Atom,
    Box,
    Diamond,
    Eq,
    Exists,
    Forall,
    Formula,
    Function,
    Iff,
    Impl,
    Not,
    Or,
    Prefixed,
    Term,
    Variable,
)


_AXIOM_LIKE_ROLES = frozenset(
    {
        "axiom",
        "hypothesis",
        "definition",
        "assumption",
        "lemma",
        "theorem",
        "corollary",
        "plain",
        "negated_conjecture",
    }
)


def preprocess_model_problem(
    path: str | Path,
    *,
    source_roots: Iterable[str | Path] = (),
) -> ModelProblem:
    try:
        document = parse_tptp_file(path, source_roots=source_roots)
    except TPTPParseError as exc:
        raise InputError(str(exc)) from exc
    if not document.statements:
        raise InputError("model finding requires at least one FOF or CNF statement")

    user_functions: dict[str, int] = {}
    user_predicates: dict[str, int] = {}
    asserted: list[Formula] = []
    conjectures: list[Formula] = []
    raw_cnf: list[StmtCNF] = []
    fof_assertions: list[tuple[str, Formula]] = []

    for statement in document.statements:
        if isinstance(statement, StmtQMF):
            raise InputError(
                f"modal QMF statement {statement.name!r} is not supported in model mode"
            )
        if statement.role == "type":
            raise InputError(f"typed role in statement {statement.name!r} is not supported")
        if statement.role != "conjecture" and statement.role not in _AXIOM_LIKE_ROLES:
            raise InputError(
                f"formula role {statement.role!r} in statement "
                f"{statement.name!r} is not an axiom-like model input"
            )
        _validate_and_collect_signature(
            statement.formula,
            functions=user_functions,
            predicates=user_predicates,
        )
        closed = _universally_close(statement.formula)
        if statement.role == "conjecture":
            if isinstance(statement, StmtCNF):
                raise InputError("CNF conjecture roles are not supported; use negated_conjecture")
            conjectures.append(closed)
        else:
            asserted.append(closed)
            if isinstance(statement, StmtCNF):
                raw_cnf.append(statement)
            elif isinstance(statement, StmtFOF):
                fof_assertions.append((statement.name, statement.formula))
            else:  # pragma: no cover - guarded above and by parser types
                raise InputError(f"unsupported statement kind: {type(statement).__name__}")

    occupied = set(user_functions) | set(user_predicates)
    clausifier = _Clausifier(occupied)
    clauses: list[ModelClause] = []
    for statement in raw_cnf:
        standardized = clausifier.standardize(statement.formula)
        clauses.append(_raw_cnf_clause(standardized, source=statement.name))
    for source, formula in fof_assertions:
        clauses.extend(clausifier.clausify(formula, source=source))

    conjecture_formula = _conjoin(conjectures)
    if conjecture_formula is not None:
        clauses.extend(
            clausifier.clausify(
                Not(conjecture_formula),
                source="model_negated_conjectures",
            )
        )

    functions = dict(user_functions)
    predicates = dict(user_predicates)
    for clause in clauses:
        _collect_clause_signature(clause, functions, predicates)

    return ModelProblem(
        path=str(Path(path).resolve()),
        statements=list(document.statements),
        clauses=clauses,
        asserted_formulas=asserted,
        conjecture_formula=conjecture_formula,
        result_status="CounterSatisfiable" if conjecture_formula else "Satisfiable",
        function_signatures=functions,
        predicate_signatures=predicates,
        user_function_symbols=frozenset(user_functions),
        user_predicate_symbols=frozenset(user_predicates),
        hidden_function_symbols=frozenset(clausifier.hidden_functions),
        hidden_predicate_symbols=frozenset(clausifier.hidden_predicates),
    )


class _Clausifier:
    def __init__(self, occupied_symbols, variable_counter = 0, skolem_counter = 0, definition_counter = 0):
        self.occupied_symbols = set(occupied_symbols)
        self.variable_counter = variable_counter
        self.skolem_counter = skolem_counter
        self.definition_counter = definition_counter
        self.hidden_functions: set[str] = set()
        self.hidden_predicates: set[str] = set()

    def fresh_variable(self) -> Variable:
        self.variable_counter += 1
        return Variable(f"M{self.variable_counter}", vid=self.variable_counter)

    def fresh_symbol(self, stem: str) -> str:
        while True:
            if stem == "model_skolem":
                self.skolem_counter += 1
                value = self.skolem_counter
            else:
                self.definition_counter += 1
                value = self.definition_counter

            symbol = f"{stem}_{value}"
            if symbol not in self.occupied_symbols:
                self.occupied_symbols.add(symbol)
                return symbol

    def standardize(self, formula: Formula) -> Formula:
        free: dict[str, Variable] = {}

        def term(current: Term, bound: Mapping[str, Variable]) -> Term:
            if isinstance(current, Variable):
                replacement = bound.get(current.symbol)
                if replacement is None:
                    replacement = free.get(current.symbol)
                    if replacement is None:
                        replacement = self.fresh_variable()
                        free[current.symbol] = replacement
                return replacement
            if isinstance(current, Function):
                return Function(current.symbol, tuple(term(arg, bound) for arg in current.args))
            raise TypeError(f"unsupported term node: {type(current)!r}")

        def visit(current: Formula, bound: Mapping[str, Variable]) -> Formula:
            if isinstance(current, Atom):
                return Atom(current.symbol, tuple(term(arg, bound) for arg in current.args))
            if isinstance(current, Eq):
                return Eq(term(current.left, bound), term(current.right, bound))
            if isinstance(current, Not):
                return Not(visit(current.formula, bound))
            if isinstance(current, (And, Or, Impl, Iff)):
                return type(current)(visit(current.left, bound), visit(current.right, bound))
            if isinstance(current, (Forall, Exists)):
                variable = self.fresh_variable()
                next_bound = dict(bound)
                next_bound[current.variable.symbol] = variable
                return type(current)(variable, visit(current.body, next_bound))
            raise InputError(f"unsupported formula node in model mode: {type(current).__name__}")

        return visit(formula, {})

    def clausify(self, formula: Formula, *, source: str) -> list[ModelClause]:
        standardized = self.standardize(formula)
        closed = _universally_close(standardized)
        nnf = _to_nnf(closed)
        matrix = self._skolemize(nnf, [])
        definitions: list[list[ModelLiteral]] = []
        root = self._define(matrix, definitions)
        definitions.append([root])
        clauses: list[ModelClause] = []
        for literals in definitions:
            simplified = _simplify_clause(literals)
            if simplified is None:
                continue
            clauses.append(
                ModelClause(
                    literals=simplified,
                    variables=_variables_in_literals(simplified),
                    source=source,
                )
            )
        return clauses

    def _skolemize(
        self,
        formula: Formula,
        universals: list[Variable],
    ) -> Formula:
        if isinstance(formula, Forall):
            return self._skolemize(formula.body, [*universals, formula.variable])
        if isinstance(formula, Exists):
            symbol = self.fresh_symbol("model_skolem")
            self.hidden_functions.add(symbol)
            replacement = Function(symbol, tuple(universals))
            body = _substitute(formula.body, formula.variable, replacement)
            return self._skolemize(body, universals)
        if isinstance(formula, (And, Or)):
            return type(formula)(
                self._skolemize(formula.left, universals),
                self._skolemize(formula.right, universals),
            )
        if _as_literal(formula) is not None:
            return formula
        raise TypeError(f"NNF contains unsupported node: {type(formula)!r}")

    def _define(
        self,
        formula: Formula,
        clauses: list[list[ModelLiteral]],
    ) -> ModelLiteral:
        literal = _as_literal(formula)
        if literal is not None:
            return literal
        if not isinstance(formula, (And, Or)):
            raise TypeError(f"definition expected NNF formula, got {type(formula)!r}")
        left = self._define(formula.left, clauses)
        right = self._define(formula.right, clauses)
        variables = _free_variables(formula)
        symbol = self.fresh_symbol("model_definition")
        self.hidden_predicates.add(symbol)
        definition = ModelLiteral(Atom(symbol, tuple(variables)), True)
        if isinstance(formula, And):
            clauses.extend(
                (
                    [definition.complement(), left],
                    [definition.complement(), right],
                    [definition, left.complement(), right.complement()],
                )
            )
        else:
            clauses.extend(
                (
                    [definition, left.complement()],
                    [definition, right.complement()],
                    [definition.complement(), left, right],
                )
            )
        return definition


def _raw_cnf_clause(formula: Formula, *, source: str) -> ModelClause:
    parts = _flatten_or(formula)
    literals: list[ModelLiteral] = []
    for part in parts:
        constant = _boolean_constant(part)
        if constant is True:
            literals = []
            marker = Variable("Tautology", vid=-1)
            literals.append(ModelLiteral(Eq(marker, marker)))
            break
        if constant is False:
            continue
        literal = _as_literal(part)
        if literal is None:
            raise InputError(f"CNF statement {source!r} does not contain only literals")
        literals.append(literal)
    simplified = _simplify_clause(literals)
    if simplified is None:
        # A tautological source clause imposes no constraint. Represent it with
        # a built-in reflexive equality so direct CNF remains one source clause.
        marker = Variable("Tautology", vid=-1)
        simplified = [ModelLiteral(Eq(marker, marker))]
    return ModelClause(
        literals=simplified,
        variables=_variables_in_literals(simplified),
        source=source,
    )


def _boolean_constant(formula: Formula) -> bool | None:
    if (
        isinstance(formula, Impl)
        and isinstance(formula.left, Atom)
        and formula.left == formula.right
        and formula.left.symbol == "true___"
    ):
        return True
    if (
        isinstance(formula, And)
        and isinstance(formula.left, Atom)
        and formula.left.symbol == "false___"
        and isinstance(formula.right, Not)
        and formula.right.formula == formula.left
    ):
        return False
    return None


def _to_nnf(formula: Formula, negated: bool = False) -> Formula:
    if isinstance(formula, (Atom, Eq)):
        return Not(formula) if negated else formula
    if isinstance(formula, Not):
        return _to_nnf(formula.formula, not negated)
    if isinstance(formula, And):
        cls = Or if negated else And
        return cls(_to_nnf(formula.left, negated), _to_nnf(formula.right, negated))
    if isinstance(formula, Or):
        cls = And if negated else Or
        return cls(_to_nnf(formula.left, negated), _to_nnf(formula.right, negated))
    if isinstance(formula, Impl):
        rewritten = Or(Not(formula.left), formula.right)
        return _to_nnf(rewritten, negated)
    if isinstance(formula, Iff):
        if negated:
            rewritten = And(
                Or(formula.left, formula.right),
                Or(Not(formula.left), Not(formula.right)),
            )
        else:
            rewritten = And(
                Or(Not(formula.left), formula.right),
                Or(Not(formula.right), formula.left),
            )
        return _to_nnf(rewritten)
    if isinstance(formula, Forall):
        cls = Exists if negated else Forall
        return cls(formula.variable, _to_nnf(formula.body, negated))
    if isinstance(formula, Exists):
        cls = Forall if negated else Exists
        return cls(formula.variable, _to_nnf(formula.body, negated))
    raise InputError(f"unsupported formula node in model mode: {type(formula).__name__}")


def _substitute(formula: Formula, variable: Variable, replacement: Term) -> Formula:
    def term(current: Term) -> Term:
        if isinstance(current, Variable):
            return replacement if current == variable else current
        if isinstance(current, Function):
            return Function(current.symbol, tuple(term(arg) for arg in current.args))
        raise TypeError(f"unsupported term node: {type(current)!r}")

    if isinstance(formula, Atom):
        return Atom(formula.symbol, tuple(term(arg) for arg in formula.args))
    if isinstance(formula, Eq):
        return Eq(term(formula.left), term(formula.right))
    if isinstance(formula, Not):
        return Not(_substitute(formula.formula, variable, replacement))
    if isinstance(formula, (And, Or)):
        return type(formula)(
            _substitute(formula.left, variable, replacement),
            _substitute(formula.right, variable, replacement),
        )
    if isinstance(formula, (Forall, Exists)):
        if formula.variable == variable:
            return formula
        return type(formula)(
            formula.variable,
            _substitute(formula.body, variable, replacement),
        )
    raise TypeError(f"unsupported formula node during substitution: {type(formula)!r}")


def _as_literal(formula: Formula) -> ModelLiteral | None:
    if isinstance(formula, (Atom, Eq)):
        return ModelLiteral(formula, True)
    if isinstance(formula, Not) and isinstance(formula.formula, (Atom, Eq)):
        return ModelLiteral(formula.formula, False)
    return None


def _flatten_or(formula: Formula) -> list[Formula]:
    if isinstance(formula, Or):
        return [*_flatten_or(formula.left), *_flatten_or(formula.right)]
    return [formula]


def _simplify_clause(
    literals: list[ModelLiteral],
) -> list[ModelLiteral] | None:
    result: list[ModelLiteral] = []
    seen: set[ModelLiteral] = set()
    for literal in literals:
        if literal.complement() in seen:
            return None
        if literal not in seen:
            result.append(literal)
            seen.add(literal)
    return result


def _universally_close(formula: Formula) -> Formula:
    result = formula
    for variable in reversed(_free_variables(formula)):
        result = Forall(variable, result)
    return result


def _free_variables(formula: Formula) -> list[Variable]:
    found: list[Variable] = []
    seen: set[Variable] = set()

    def term(current: Term, bound: frozenset[str]):
        if isinstance(current, Variable):
            if current.symbol not in bound and current not in seen:
                seen.add(current)
                found.append(current)
        elif isinstance(current, Function):
            for arg in current.args:
                term(arg, bound)

    def visit(current: Formula, bound: frozenset[str]):
        if isinstance(current, Atom):
            for arg in current.args:
                term(arg, bound)
        elif isinstance(current, Eq):
            term(current.left, bound)
            term(current.right, bound)
        elif isinstance(current, Not):
            visit(current.formula, bound)
        elif isinstance(current, (And, Or, Impl, Iff)):
            visit(current.left, bound)
            visit(current.right, bound)
        elif isinstance(current, (Forall, Exists)):
            visit(current.body, bound | {current.variable.symbol})
        elif isinstance(current, (Box, Diamond, Prefixed)):
            raise InputError(f"modal or prefixed formula {type(current).__name__} is unsupported")

    visit(formula, frozenset())
    return found


def _variables_in_literals(literals: list[ModelLiteral]) -> list[Variable]:
    found: list[Variable] = []
    seen: set[Variable] = set()

    def term(current: Term):
        if isinstance(current, Variable):
            if current not in seen:
                seen.add(current)
                found.append(current)
        else:
            for arg in current.args:
                term(arg)

    for literal in literals:
        atom = literal.atom
        terms = atom.args if isinstance(atom, Atom) else (atom.left, atom.right)
        for item in terms:
            term(item)
    return found


def _validate_and_collect_signature(
    formula: Formula,
    *,
    functions: dict[str, int],
    predicates: dict[str, int],
):
    def symbol(value: str, *, kind: str, arity: int):
        if isinstance(value, DistinctObjectSymbol):
            raise InputError("distinct-object terms are not supported in model mode")
        if isinstance(value, DefinedNumberSymbol):
            raise InputError("numeric and arithmetic terms are not supported in model mode")
        if value.startswith("$"):
            raise InputError(f"defined/system symbol {value!r} is not supported")
        own = functions if kind == "function" else predicates
        other = predicates if kind == "function" else functions
        previous = own.get(value)
        if previous is not None and previous != arity:
            raise InputError(f"symbol {value!r} is used with inconsistent arities")
        if value in other:
            raise InputError(f"symbol {value!r} is used as both function and predicate")
        own[value] = arity

    def term(current: Term):
        if isinstance(current, Variable):
            return
        symbol(current.symbol, kind="function", arity=len(current.args))
        for arg in current.args:
            term(arg)

    def visit(current: Formula):
        if _boolean_constant(current) is not None:
            return
        if isinstance(current, Atom):
            symbol(current.symbol, kind="predicate", arity=len(current.args))
            for arg in current.args:
                term(arg)
        elif isinstance(current, Eq):
            term(current.left)
            term(current.right)
        elif isinstance(current, Not):
            visit(current.formula)
        elif isinstance(current, (And, Or, Impl, Iff)):
            visit(current.left)
            visit(current.right)
        elif isinstance(current, (Forall, Exists)):
            visit(current.body)
        else:
            raise InputError(f"unsupported formula node in model mode: {type(current).__name__}")

    visit(formula)


def _collect_clause_signature(
    clause: ModelClause,
    functions: dict[str, int],
    predicates: dict[str, int],
):
    def term(current: Term):
        if isinstance(current, Function):
            functions.setdefault(current.symbol, len(current.args))
            for arg in current.args:
                term(arg)

    for literal in clause.literals:
        atom = literal.atom
        if isinstance(atom, Atom):
            predicates.setdefault(atom.symbol, len(atom.args))
            for arg in atom.args:
                term(arg)
        else:
            term(atom.left)
            term(atom.right)


def _conjoin(formulas: list[Formula]) -> Formula | None:
    if not formulas:
        return None
    result = formulas[-1]
    for formula in reversed(formulas[:-1]):
        result = And(formula, result)
    return result


__all__ = ["preprocess_model_problem"]
