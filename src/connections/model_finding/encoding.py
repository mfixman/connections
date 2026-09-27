from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from itertools import product
from typing import Any

import pydical  # type: ignore[unresolved-import]

from connections.model_finding.core import FiniteModel, ModelClause, ModelProblem, ModelSearchBudget, ModelSearchTimeout
from connections.syntax.formula import Atom, Eq, Term, Variable


GroundInstance = tuple[int, tuple[int, ...]]
TruthLiteral = int | bool
ValueVector = tuple[TruthLiteral, ...]


class FixedDomainEncoding:
    """Relational finite-domain encoding backed by one CaDiCaL solver."""

    def __init__(
        self,
        problem: ModelProblem,
        domain_size: int,
        *,
        static_constant_symmetry: bool = False,
        budget: ModelSearchBudget | None = None,
    ):
        if domain_size < 1:
            raise ValueError("domain_size must be positive")
        self.problem = problem
        self.domain_size = domain_size
        self.domain = tuple(range(domain_size))
        self.budget = budget
        self.solver: Any = pydical.Solver()
        if hasattr(self.solver, "set"):
            self.solver.set("quiet", 1)
        if budget is not None and budget.timeout_seconds is not None:
            self.solver.connect_terminator(lambda: budget.timed_out)
        self.variable_count = 0
        self.clause_count = 0
        self.function_variables: dict[tuple[str, tuple[int, ...], int], int] = {}
        self.predicate_variables: dict[tuple[str, tuple[int, ...]], int] = {}
        self._term_values: dict[object, ValueVector] = {}
        self._atom_variables: dict[object, TruthLiteral] = {}
        self._encoded_instances: set[GroundInstance] = set()
        self._create_interpretation_tables()
        if static_constant_symmetry:
            self._add_static_constant_symmetry()
        self.solver.reserve(self.variable_count)

    @property
    def encoded_instances(self) -> frozenset[GroundInstance]:
        return frozenset(self._encoded_instances)

    def new_variable(self) -> int:
        self.variable_count += 1
        return self.variable_count

    def add_clause(self, literals: Iterable[int]):
        clause = list(literals)
        self.solver.add_clause(clause)
        self.clause_count += 1

    def encode_instance(self, instance: GroundInstance) -> bool:
        if instance in self._encoded_instances:
            return False
        clause_index, values = instance
        clause = self.problem.clauses[clause_index]
        if len(values) != len(clause.variables):
            raise ValueError("ground instance has the wrong assignment arity")
        assignment = dict(zip(clause.variables, values, strict=True))
        propositional: list[int] = []
        tautology = False
        for literal in clause.literals:
            truth = self._atom_truth(literal.atom, assignment)
            if not literal.positive:
                truth = _negate_truth(truth)
            if truth is True:
                tautology = True
                break
            if truth is not False:
                propositional.append(truth)
        if not tautology:
            self.add_clause(propositional)
        self._encoded_instances.add(instance)
        return True

    def all_instances(self) -> Iterator[GroundInstance]:
        for clause_index, clause in enumerate(self.problem.clauses):
            for values in product(self.domain, repeat=len(clause.variables)):
                if self.budget is not None and self.budget.timed_out:
                    raise ModelSearchTimeout
                yield clause_index, values

    def solve(self) -> int:
        status = self.solver.solve()
        if self.budget is not None and self.budget.timed_out:
            raise ModelSearchTimeout
        return status

    def extract_model(self) -> FiniteModel:
        functions: dict[str, dict[tuple[int, ...], int]] = {}
        for symbol, arity in sorted(self.problem.function_signatures.items()):
            table: dict[tuple[int, ...], int] = {}
            for arguments in product(self.domain, repeat=arity):
                chosen = [
                    output
                    for output in self.domain
                    if self._solver_value(self.function_variables[(symbol, arguments, output)])
                ]
                if len(chosen) != 1:
                    raise RuntimeError(f"SAT assignment did not define total function {symbol!r}")
                table[arguments] = chosen[0]
            functions[symbol] = table
        predicates: dict[str, dict[tuple[int, ...], bool]] = {}
        for symbol, arity in sorted(self.problem.predicate_signatures.items()):
            predicates[symbol] = {
                arguments: self._solver_value(self.predicate_variables[(symbol, arguments)])
                for arguments in product(self.domain, repeat=arity)
            }
        return FiniteModel(self.domain_size, functions, predicates)

    def first_falsified_instance(
        self,
        model: FiniteModel,
        *,
        exclude_encoded: bool = True,
    ) -> GroundInstance | None:
        for instance in self.all_instances():
            if exclude_encoded and instance in self._encoded_instances:
                continue
            clause_index, values = instance
            clause = self.problem.clauses[clause_index]
            assignment = dict(zip(clause.variables, values, strict=True))
            if not evaluate_clause(clause, model, assignment):
                return instance
        return None

    def _create_interpretation_tables(self):
        for symbol, arity in sorted(self.problem.function_signatures.items()):
            for arguments in product(self.domain, repeat=arity):
                choices: list[int] = []
                for output in self.domain:
                    variable = self.new_variable()
                    self.function_variables[(symbol, arguments, output)] = variable
                    choices.append(variable)
                self._exactly_one(choices)
        for symbol, arity in sorted(self.problem.predicate_signatures.items()):
            for arguments in product(self.domain, repeat=arity):
                self.predicate_variables[(symbol, arguments)] = self.new_variable()

    def _add_static_constant_symmetry(self):
        constants = sorted(
            symbol for symbol, arity in self.problem.function_signatures.items() if arity == 0
        )
        for index, symbol in enumerate(constants):
            maximum = min(index, self.domain_size - 1)
            for output in range(maximum + 1, self.domain_size):
                self.add_clause((-self.function_variables[(symbol, (), output)],))

    def _exactly_one(self, variables: list[int]):
        self.add_clause(variables)
        for index, left in enumerate(variables):
            for right in variables[index + 1 :]:
                self.add_clause((-left, -right))

    def _term_value_vector(
        self,
        term: Term,
        assignment: Mapping[Variable, int],
    ) -> ValueVector:
        if isinstance(term, Variable):
            value = assignment[term]
            return tuple(index == value for index in self.domain)
        key = _ground_term_key(term, assignment)
        cached = self._term_values.get(key)
        if cached is not None:
            return cached
        if not term.args:
            result = tuple(
                self.function_variables[(term.symbol, (), output)] for output in self.domain
            )
            self._term_values[key] = result
            return result

        argument_vectors = tuple(self._term_value_vector(arg, assignment) for arg in term.args)
        result = tuple(self.new_variable() for _ in self.domain)
        self._exactly_one(list(result))
        for arguments in product(self.domain, repeat=len(term.args)):
            condition = _condition(argument_vectors, arguments)
            if condition is None:
                continue
            for output in self.domain:
                function_var = self.function_variables[(term.symbol, arguments, output)]
                self.add_clause(
                    (
                        *(-literal for literal in condition),
                        -function_var,
                        result[output],
                    )
                )
        self._term_values[key] = result
        return result

    def _atom_truth(
        self,
        atom: Atom | Eq,
        assignment: Mapping[Variable, int],
    ) -> TruthLiteral:
        key = _ground_atom_key(atom, assignment)
        cached = self._atom_variables.get(key)
        if cached is not None:
            return cached
        if isinstance(atom, Atom):
            if not atom.args:
                result: TruthLiteral = self.predicate_variables[(atom.symbol, ())]
                self._atom_variables[key] = result
                return result
            vectors = tuple(self._term_value_vector(term, assignment) for term in atom.args)
            result = self.new_variable()
            for arguments in product(self.domain, repeat=len(atom.args)):
                condition = _condition(vectors, arguments)
                if condition is None:
                    continue
                predicate = self.predicate_variables[(atom.symbol, arguments)]
                prefix = tuple(-literal for literal in condition)
                self.add_clause((*prefix, -result, predicate))
                self.add_clause((*prefix, result, -predicate))
            self._atom_variables[key] = result
            return result

        vectors = (
            self._term_value_vector(atom.left, assignment),
            self._term_value_vector(atom.right, assignment),
        )
        result = self.new_variable()
        for values in product(self.domain, repeat=2):
            condition = _condition(vectors, values)
            if condition is None:
                continue
            prefix = tuple(-literal for literal in condition)
            self.add_clause((*prefix, result if values[0] == values[1] else -result))
        self._atom_variables[key] = result
        return result

    def _solver_value(self, variable: int) -> bool:
        value = self.solver.val(variable)
        return value > 0


def evaluate_clause(
    clause: ModelClause,
    model: FiniteModel,
    assignment: Mapping[Variable, int],
) -> bool:
    for literal in clause.literals:
        truth = evaluate_atom(literal.atom, model, assignment)
        if truth == literal.positive:
            return True
    return False


def evaluate_atom(
    atom: Atom | Eq,
    model: FiniteModel,
    assignment: Mapping[Variable, int],
) -> bool:
    if isinstance(atom, Eq):
        return evaluate_term(atom.left, model, assignment) == evaluate_term(
            atom.right, model, assignment
        )
    arguments = tuple(evaluate_term(arg, model, assignment) for arg in atom.args)
    return model.predicates[atom.symbol][arguments]


def evaluate_term(
    term: Term,
    model: FiniteModel,
    assignment: Mapping[Variable, int],
) -> int:
    if isinstance(term, Variable):
        return assignment[term]
    arguments = tuple(evaluate_term(arg, model, assignment) for arg in term.args)
    return model.functions[term.symbol][arguments]


def _condition(vectors: tuple[ValueVector, ...], values: tuple[int, ...]) -> tuple[int, ...] | None:
    literals: list[int] = []
    for vector, value in zip(vectors, values, strict=True):
        truth = vector[value]
        if truth is False:
            return None
        if truth is not True:
            literals.append(truth)
    return tuple(literals)


def _ground_term_key(term: Term, assignment: Mapping[Variable, int]) -> object:
    if isinstance(term, Variable):
        return ("value", assignment[term])
    return (term.symbol, tuple(_ground_term_key(arg, assignment) for arg in term.args))


def _ground_atom_key(atom: Atom | Eq, assignment: Mapping[Variable, int]) -> object:
    if isinstance(atom, Eq):
        return (
            "=",
            _ground_term_key(atom.left, assignment),
            _ground_term_key(atom.right, assignment),
        )
    return (
        "predicate",
        atom.symbol,
        tuple(_ground_term_key(arg, assignment) for arg in atom.args),
    )


def _negate_truth(truth: TruthLiteral) -> TruthLiteral:
    if isinstance(truth, bool):
        return not truth
    return -truth


__all__ = [
    "FixedDomainEncoding",
    "GroundInstance",
    "evaluate_atom",
    "evaluate_clause",
    "evaluate_term",
]
