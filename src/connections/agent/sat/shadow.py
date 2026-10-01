from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pydical  # type: ignore[unresolved-import]
from connections.constraints.term import TableauVariable, TermBinding
from connections.environment.state import State
from connections.syntax.formula import Function, Term, Variable
from connections.syntax.matrix import Literal


@dataclass(slots=True)
class _ShadowSAT:
    atom_ids: dict[str, int] = field(default_factory=dict)
    selector_ids: dict[int, int] = field(default_factory=dict)
    clauses: set[tuple[int, tuple[int, ...]]] = field(default_factory=set)
    clause_contents: set[tuple[int, ...]] = field(default_factory=set)
    observed_groundings: set[tuple[int, tuple[int, ...]]] = field(default_factory=set)
    model: dict[int, bool] = field(default_factory=dict)
    solver: Any = field(default_factory=pydical.Solver)
    dirty: bool = False
    satisfiable: bool = True
    new_tableau_clause: bool = False
    debug_unsat_core: bool = False
    unsat_core: list[tuple[int, ...]] = field(default_factory=list)
    sat_core_clause_texts: list[str] = field(default_factory=list)
    sat_core_clause_ids: list[int] = field(default_factory=list)
    core_available: bool = False
    next_variable_id: int = 1

    def __post_init__(self):
        if hasattr(self.solver, "set"):
            for name, value in (
                ("quiet", 1),
                ("stabilizeonly", 1),
                ("walk", 0),
                ("lucky", 0),
                ("luckyearly", 0),
                ("luckylate", 0),
                ("luckyassumptions", 0),
            ):
                self.solver.set(name, value)

    def atom_id(self, key: str) -> int:
        identifier = self.atom_ids.get(key)
        if identifier is None:
            identifier = self._fresh_variable()
            self.atom_ids[key] = identifier
        return identifier

    def selector_id(self, clause_idx: int) -> int:
        identifier = self.selector_ids.get(clause_idx)
        if identifier is None:
            identifier = self._fresh_variable()
            self.selector_ids[clause_idx] = identifier
        return identifier

    def _fresh_variable(self) -> int:
        identifier = self.next_variable_id
        self.next_variable_id += 1
        return identifier

    def add_clause(self, clause: tuple[int, ...], *, clause_idx: int, from_tableau: bool):
        sourced = (clause_idx, clause)
        if sourced in self.clauses:
            return
        self.clauses.add(sourced)
        selector = self.selector_id(clause_idx)
        self.solver.add_clause([-selector, *clause])
        self.dirty = True
        if from_tableau and clause not in self.clause_contents:
            self.new_tableau_clause = True
        self.clause_contents.add(clause)

    def solve(self) -> bool:
        if not self.dirty:
            return self.satisfiable
        for selector in self.selector_ids.values():
            self.solver.assume(selector)
        status = self.solver.solve()
        self.dirty = False
        if status == pydical.UNSATISFIABLE:
            self.model.clear()
            self.satisfiable = False
            self.core_available = True
            if self.debug_unsat_core:
                failed = {
                    clause_idx
                    for clause_idx, selector in self.selector_ids.items()
                    if self.solver.failed(selector)
                }
                self.sat_core_clause_ids = sorted(failed)
                self.unsat_core = [
                    clause
                    for clause_idx, clause in sorted(self.clauses)
                    if clause_idx in failed
                ]
            return False
        if status != pydical.SATISFIABLE:
            self.model.clear()
            self.satisfiable = True
            return True
        values = (self.solver.val(variable) for variable in range(1, self.solver.vars + 1))
        self.model = {abs(value): value > 0 for value in values if value != 0}
        self.satisfiable = True
        return True

    def literal_value(self, literal: int | None) -> bool | None:
        if literal is None:
            return None
        value = self.model.get(abs(literal))
        if value is None:
            return None
        return value if literal > 0 else not value

    def consume_new_tableau_clause(self) -> bool:
        found = self.new_tableau_clause
        self.new_tableau_clause = False
        return found


def _sat_value_score(value: bool | None) -> int:
    if value is True:
        return 0
    if value is None:
        return 1
    return 3


def _atom_key(
    state: State,
    literal: Literal,
    *,
    instance_id: int | None,
    pending_bindings: tuple[TermBinding, ...] = (),
    ground_only: bool = False,
) -> str | None:
    args: list[str] = []
    for argument in literal.atom.args:
        key = _term_key(
            state,
            argument,
            instance_id=instance_id,
            pending_bindings=pending_bindings,
            ground_only=ground_only,
        )
        if key is None:
            return None
        args.append(key)
    return f"{literal.atom.symbol}({','.join(args)})" if args else literal.atom.symbol


def _term_key(
    state: State,
    term: Term | TableauVariable,
    *,
    instance_id: int | None,
    pending_bindings: tuple[TermBinding, ...],
    ground_only: bool = False,
) -> str | None:
    if isinstance(term, TableauVariable):
        return None if ground_only else "__ground__"
    try:
        resolved = state.constraints.terms.substitute_term(
            term,
            instance_id=instance_id,
            pending_bindings=pending_bindings,
        )
    except AttributeError:
        resolved = term
    if isinstance(resolved, (Variable, TableauVariable)):
        return None if ground_only else "__ground__"
    if not isinstance(resolved, Function) or not resolved.args:
        return str(resolved)
    args: list[str] = []
    for argument in resolved.args:
        key = _term_key(
            state,
            argument,
            instance_id=None,
            pending_bindings=pending_bindings,
            ground_only=ground_only,
        )
        if key is None:
            return None
        args.append(key)
    return f"{resolved.symbol}({','.join(args)})"
