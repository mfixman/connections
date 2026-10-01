from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Mapping

from connections.parsing.tptp.transformer import StmtFormula
from connections.syntax.formula import Atom, Eq, Formula, Term, Variable


class InputError(ValueError):
    """The input is outside the finite model finder's supported fragment."""


class ModelValidationError(RuntimeError):
    """A SAT candidate did not satisfy the original parsed problem."""


@dataclass(frozen=True, slots=True)
class ModelLiteral:
    atom: Atom | Eq
    positive: bool = True

    def complement(self) -> "ModelLiteral":
        return ModelLiteral(self.atom, not self.positive)


@dataclass(frozen=True, slots=True)
class ModelClause:
    literals: list[ModelLiteral]
    variables: list[Variable]
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ModelProblem:
    path: str
    statements: list[StmtFormula]
    clauses: list[ModelClause]
    asserted_formulas: list[Formula]
    conjecture_formula: Formula | None
    result_status: str
    function_signatures: Mapping[str, int]
    predicate_signatures: Mapping[str, int]
    user_function_symbols: frozenset[str] = frozenset()
    user_predicate_symbols: frozenset[str] = frozenset()
    hidden_function_symbols: frozenset[str] = frozenset()
    hidden_predicate_symbols: frozenset[str] = frozenset()

    @property
    def has_conjecture(self) -> bool:
        return self.conjecture_formula is not None

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass(frozen=True, slots=True)
class FiniteModel:
    domain_size: int
    functions: Mapping[str, Mapping[tuple[int, ...], int]]
    predicates: Mapping[str, Mapping[tuple[int, ...], bool]]

    def __post_init__(self):
        if self.domain_size < 1:
            raise ValueError("a finite model domain must be non-empty")

    @property
    def domain(self) -> list[int]:
        return list(range(self.domain_size))

    def restricted_to(self, problem: ModelProblem) -> "FiniteModel":
        return FiniteModel(
            domain_size=self.domain_size,
            functions={
                symbol: dict(table)
                for symbol, table in self.functions.items()
                if symbol in problem.user_function_symbols
            },
            predicates={
                symbol: dict(table)
                for symbol, table in self.predicates.items()
                if symbol in problem.user_predicate_symbols
            },
        )

    def to_json(self) -> dict[str, object]:
        return {
            "domain": list(self.domain),
            "domain_size": self.domain_size,
            "functions": {
                symbol: {
                    "arity": _table_arity(table),
                    "table": [
                        {"arguments": list(arguments), "value": value}
                        for arguments, value in sorted(table.items())
                    ],
                }
                for symbol, table in sorted(self.functions.items())
            },
            "predicates": {
                symbol: {
                    "arity": _table_arity(table),
                    "table": [
                        {"arguments": list(arguments), "value": value}
                        for arguments, value in sorted(table.items())
                    ],
                }
                for symbol, table in sorted(self.predicates.items())
            },
        }


@dataclass(slots=True)
class ModelSearchBudget:
    timeout_seconds: float | None = None
    max_steps: int | None = None
    max_domain: int | None = None
    started_at: float = field(default_factory=time.monotonic)
    steps: int = 0

    def __post_init__(self):
        if self.timeout_seconds is not None and self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.max_domain is not None and self.max_domain < 1:
            raise ValueError("max_domain must be positive")

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def timed_out(self) -> bool:
        return self.timeout_seconds is not None and self.elapsed_seconds >= self.timeout_seconds

    @property
    def steps_exhausted(self) -> bool:
        return self.max_steps is not None and self.steps >= self.max_steps

    def consume_step(self):
        if self.timed_out:
            raise ModelSearchTimeout
        if self.steps_exhausted:
            raise ModelSearchStepLimit
        self.steps += 1


class ModelSearchTimeout(Exception):
    pass


class ModelSearchStepLimit(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ModelSearchEvent:
    domain_size: int
    sat_calls: int
    variables: int
    clauses: int
    refinements: int
    elapsed_seconds: float
    satisfiable: bool | None
    validation_passed: bool | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "domain_size": self.domain_size,
            "sat_calls": self.sat_calls,
            "variables": self.variables,
            "clauses": self.clauses,
            "refinements": self.refinements,
            "elapsed_seconds": self.elapsed_seconds,
            "satisfiable": self.satisfiable,
            "validation_passed": self.validation_passed,
        }


@dataclass(frozen=True, slots=True)
class ModelSearchResult:
    policy: str
    status: str
    outcome: str
    solved: bool
    steps: int
    elapsed_seconds: float
    model: FiniteModel | None = None
    tptp_model: str | None = None
    events: list[ModelSearchEvent] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None

    def to_json(self, *, problem: str, path: str) -> dict[str, object]:
        return {
            "problem": problem,
            "path": path,
            "policy": self.policy,
            "status": self.status,
            "outcome": self.outcome,
            "solved": self.solved,
            "steps": self.steps,
            "elapsed_seconds": self.elapsed_seconds,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "model": None if self.model is None else self.model.to_json(),
            "tptp_model": self.tptp_model,
            "debug": {
                "provisional": True,
                "bounds": [event.to_json() for event in self.events],
            },
        }


def _table_arity(table: Mapping[tuple[int, ...], Any]) -> int:
    try:
        return len(next(iter(table)))
    except StopIteration:
        return 0


GroundAssignment = Mapping[Variable, int]
GroundFunctionTable = Mapping[tuple[int, ...], int]
GroundPredicateTable = Mapping[tuple[int, ...], bool]
ModelTerm = Term


__all__ = [
    "FiniteModel",
    "GroundAssignment",
    "GroundFunctionTable",
    "GroundPredicateTable",
    "InputError",
    "ModelClause",
    "ModelLiteral",
    "ModelProblem",
    "ModelSearchBudget",
    "ModelSearchEvent",
    "ModelSearchResult",
    "ModelSearchStepLimit",
    "ModelSearchTimeout",
    "ModelTerm",
    "ModelValidationError",
]
