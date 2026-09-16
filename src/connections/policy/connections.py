from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

import pydical  # type: ignore[unresolved-import]

from connections.constraints.term import TableauVariable, TermBinding
from connections.policy.base import BacktrackGranularity
from connections.policy.dfs import DFSPolicyDecision
from connections.policy.id import FirstActionIDPolicy, IDPolicy
from connections.prover.actions import Action, ApplyAction, UndoAction
from connections.prover.rules import (
    Extension,
    Factorization,
    FactorizationMode,
    ModelLemma,
    Reduction,
    Start,
)
from connections.prover.state import State
from connections.prover.status import ProverOutcome
from connections.syntax.formula import Function, Term, Variable
from connections.syntax.matrix import Clause, Literal


class LeanCoPCon(FirstActionIDPolicy):
    """Connections implementation of leanCoP's cut/comp(7) policy."""

    def __init__(
        self,
        *,
        cut: bool = True,
        scut: bool = False,
        comp: int | None = 7,
        backtrack: BacktrackGranularity = "step",
        factorization: FactorizationMode = "equal",
        initial_depth: int = 1,
    ) -> None:
        super().__init__(
            cut=cut,
            scut=scut,
            comp=comp,
            backtrack=backtrack,
            factorization=factorization,
            initial_depth=initial_depth,
        )


class LeanCoPCon_2(FirstActionIDPolicy):
    """Search policy used by every phase of the leanCoP 2.1 schedule."""

    def __init__(
        self,
        *,
        cut: bool = False,
        scut: bool = False,
        comp: int | None = None,
        backtrack: BacktrackGranularity = "step",
        factorization: FactorizationMode = "equal",
        initial_depth: int = 1,
    ) -> None:
        super().__init__(
            cut=cut,
            scut=scut,
            comp=comp,
            backtrack=backtrack,
            factorization=factorization,
            initial_depth=initial_depth,
        )


@dataclass(slots=True)
class _ShadowSAT:
    atom_ids: dict[str, int] = field(default_factory=dict)
    selector_ids: dict[int, int] = field(default_factory=dict)
    clauses: set[tuple[int, tuple[int, ...]]] = field(default_factory=set)
    clause_contents: set[tuple[int, ...]] = field(default_factory=set)
    observed_groundings: set[tuple[int, tuple[int, ...]]] = field(
        default_factory=set
    )
    model: dict[int, bool] = field(default_factory=dict)
    solver: Any = field(default_factory=pydical.Solver)
    dirty: bool = False
    satisfiable: bool = True
    new_tableau_clause: bool = False
    debug_unsat_core: bool = False
    unsat_core: tuple[tuple[int, ...], ...] = ()
    sat_core_clause_texts: tuple[str, ...] = ()
    sat_core_clause_ids: tuple[int, ...] = ()
    core_available: bool = False
    next_variable_id: int = 1

    def __post_init__(self) -> None:
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

    def add_clause(self, clause: tuple[int, ...], *, clause_idx: int, from_tableau: bool) -> None:
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
                self.sat_core_clause_ids = tuple(sorted(failed))
                self.unsat_core = tuple(
                    clause
                    for clause_idx, clause in sorted(self.clauses)
                    if clause_idx in failed
                )
            return False
        if status != pydical.SATISFIABLE:
            self.model.clear()
            self.satisfiable = True
            return True
        values = (
            self.solver.val(variable)
            for variable in range(1, self.solver.vars + 1)
        )
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


class SATCoPCon(IDPolicy):
    """SAT-shadow literal-selection policy for the Connections prover.

    Ground instances exposed by the current tableau are accumulated in an
    incremental CaDiCaL shadow. Its model ranks reduction and extension heads;
    shadow UNSAT is a sound terminal proof outcome.

    The search control follows the ClassicalSAT calculus of the original
    SATCoP/SATResetCoP implementation (MyPyCop):

    - Actions with equal scores, and the order of sibling goals, are chosen by
      a ``seed``-ed random draw, like its shuffled start clauses, extension
      clauses and clause literals. Matrix order instead restarts every
      SATResetCoP iteration from the same start clause, so the shadow stops
      growing and the depth limit climbs without new ground instances.
      ``seed=None`` keeps the deterministic matrix order.
    - Model lemmas and action scores read a guidance model that is refreshed
      only when a new search iteration starts; the first iteration has none.
      Re-solving after every new instance satisfied that instance at once,
      closed its other literals as lemmas, and stalled the shadow.
    - Only literals that are ground under the current substitution have a
      model value. Extension and reduction heads are valued before their own
      unifier is applied.
    - Beyond the depth limit no extension is allowed, including extensions
      with ground clauses.

    The shadow is still solved after every new instance, so UNSAT is found as
    soon as it holds rather than at the next iteration boundary.
    """

    def __init__(
        self,
        *,
        cut: bool = True,
        scut: bool = False,
        comp: int | None = None,
        backtrack: BacktrackGranularity = "step",
        factorization: FactorizationMode = "equal",
        initial_depth: int = 1,
        debug_sat_core: bool = False,
        seed: int | None = 0,
    ) -> None:
        super().__init__(
            cut=cut,
            scut=scut,
            comp=comp,
            backtrack=backtrack,
            factorization=factorization,
            initial_depth=initial_depth,
            ground_extensions_beyond_depth=False,
        )
        self._shadow = _ShadowSAT(debug_unsat_core=debug_sat_core)
        self._seeded = False
        self._random = None if seed is None else random.Random(seed)
        self._guidance_model: dict[int, bool] = {}
        self._iteration_started = False

    def diagnostics(self) -> dict[str, object]:
        if not self._shadow.debug_unsat_core or not self._shadow.core_available:
            return {}
        atoms = {identifier: key for key, identifier in self._shadow.atom_ids.items()}
        return {
            "sat_core_clause_ids": list(self._shadow.sat_core_clause_ids),
            "sat_core_clause_texts": list(self._shadow.sat_core_clause_texts),
            "sat_core": [
                [
                    atoms[abs(literal)] if literal > 0 else f"~{atoms[abs(literal)]}"
                    for literal in clause
                ]
                for clause in self._shadow.unsat_core
            ]
        }

    def __call__(self, state: State) -> DFSPolicyDecision:
        if state.problem.logic != "classical":
            raise ValueError("SATCoPCon supports classical problems only")
        self._observe_shadow(state)
        if not self._shadow.solve():
            if self._shadow.debug_unsat_core:
                self._shadow.sat_core_clause_texts = tuple(
                    str(state.problem.matrix.clauses[clause_idx])
                    for clause_idx in self._shadow.sat_core_clause_ids
                )
            return ProverOutcome.PROVED
        if state.tableau.root.closed:
            root_application_id = state.tableau.root.applied_rule_application_id
            if root_application_id is None:
                return ProverOutcome.ID_FIXED_POINT
            return UndoAction(root_application_id)
        return super().__call__(state)

    def accepts_tableau_proof(self, state: State) -> bool:
        _ = state
        return False

    def _actions_for_goal(
        self,
        state: State,
        goal_id: int,
    ) -> tuple[Action, ...]:
        if self._goal_solved_by_model(state, goal_id):
            return (ApplyAction(goal_id, ModelLemma()),)
        return super()._actions_for_goal(state, goal_id)

    def _next_action(self, state: State, actions: tuple[Action, ...]) -> Action:
        return min(
            enumerate(actions),
            key=lambda indexed: (
                self._action_score(state, indexed[1]),
                self._tie_break_key(state, indexed[1], indexed[0]),
            ),
        )[1]

    def _tie_break_key(self, state: State, action: Action, index: int) -> float:
        _ = state, action
        return index if self._random is None else self._random.random()

    def _choose_goal_id(self, state: State, goal_ids: tuple[int, ...]) -> int:
        _ = state
        return goal_ids[0] if self._random is None else self._random.choice(goal_ids)

    def _start_next_depth(self) -> None:
        if self._iteration_started:
            self._guidance_model = dict(self._shadow.model)
        self._iteration_started = True
        super()._start_next_depth()

    def _action_score(self, state: State, action: Action) -> int:
        if isinstance(action, UndoAction):
            return 5
        rule = action.rule
        if isinstance(rule, ModelLemma):
            return 0
        if isinstance(rule, Factorization):
            return 0
        if isinstance(rule, Start):
            return 2
        if isinstance(rule, Reduction):
            context = state.literal_context_at(rule.source_goal_id)
            return _sat_value_score(self._context_value(state, context))
        if isinstance(rule, Extension):
            value = self._literal_value(
                state,
                rule.clause.literal(rule.lit_idx),
                instance_id=rule.instance_id,
            )
            return _sat_value_score(value)
        return 4

    def _goal_solved_by_model(self, state: State, goal_id: int) -> bool:
        goal = state.tableau.goals[goal_id]
        path_goal_ids = {
            path_goal_id
            for ids in goal.path_goal_ids_by_signed_symbol.values()
            for path_goal_id in ids
        }
        for candidate_id in (*sorted(path_goal_ids), goal_id):
            context = state.literal_context_at(candidate_id)
            if self._context_value(state, context) is False:
                return True
        return False

    def _shadow_seed_clause_ids(self, state: State) -> tuple[int, ...]:
        """Matrix clauses grounded into the shadow before any tableau step."""
        return state.problem.start_clause_ids

    def _observe_shadow(self, state: State) -> None:
        if not self._seeded:
            for clause_idx in self._shadow_seed_clause_ids(state):
                self._shadow.add_clause(
                    self._ground_clause(
                        state,
                        state.problem.matrix.clauses[clause_idx],
                    ),
                    clause_idx=clause_idx,
                    from_tableau=False,
                )
            self._seeded = True

        for app_id in sorted(state.tableau.rule_applications):
            application = state.tableau.rule_applications[app_id]
            rule = application.rule
            if not isinstance(rule, (Start, Extension)):
                continue
            if rule.clause_idx is None:
                raise RuntimeError("SAT shadow clause is missing source-clause provenance")
            clause = self._ground_clause(
                state,
                rule.clause,
                instance_id=rule.instance_id,
            )
            observation = (app_id, clause)
            if observation in self._shadow.observed_groundings:
                continue
            self._shadow.observed_groundings.add(observation)
            self._shadow.add_clause(
                clause,
                clause_idx=rule.clause_idx,
                from_tableau=True,
            )

    def _ground_clause(
        self,
        state: State,
        clause: Clause,
        *,
        instance_id: int | None = None,
    ) -> tuple[int, ...]:
        grounded: set[int] = set()
        for literal in clause:
            sat_literal = self._sat_literal(
                state,
                literal,
                instance_id=instance_id,
            )
            if sat_literal is None:
                raise RuntimeError("ground-clause construction did not create an atom")
            grounded.add(sat_literal)
        return tuple(
            sorted(
                grounded,
                key=lambda literal: (abs(literal), literal < 0),
            )
        )

    def _context_value(
        self,
        state: State,
        context: tuple[Literal, int | None] | None,
        *,
        pending_bindings: tuple[TermBinding, ...] = (),
    ) -> bool | None:
        if context is None:
            return None
        literal, instance_id = context
        return self._literal_value(
            state,
            literal,
            instance_id=instance_id,
            pending_bindings=pending_bindings,
        )

    def _literal_value(
        self,
        state: State,
        literal: Literal,
        *,
        instance_id: int | None,
        pending_bindings: tuple[TermBinding, ...] = (),
    ) -> bool | None:
        key = _atom_key(
            state,
            literal,
            instance_id=instance_id,
            pending_bindings=pending_bindings,
            ground_only=True,
        )
        if key is None:
            return None
        atom_id = self._shadow.atom_ids.get(key)
        if atom_id is None:
            return None
        value = self._guidance_model.get(atom_id)
        if value is None:
            return None
        return value if literal.polarity else not value

    def _sat_literal(
        self,
        state: State,
        literal: Literal,
        *,
        instance_id: int | None,
        pending_bindings: tuple[TermBinding, ...] = (),
        create: bool = True,
    ) -> int | None:
        key = _atom_key(
            state,
            literal,
            instance_id=instance_id,
            pending_bindings=pending_bindings,
        )
        if key is None:
            raise RuntimeError("ground-clause construction did not create an atom key")
        atom_id = self._shadow.atom_ids.get(key)
        if atom_id is None:
            if not create:
                return None
            atom_id = self._shadow.atom_id(key)
        return atom_id if literal.polarity else -atom_id


class SATResetCoP(SATCoPCon):
    """SATCoPCon policy that resets the tableau instead of backtracking.

    At every dead end the tableau is discarded. The next iteration keeps the
    depth limit when the finished one added a ground instance the shadow had
    not seen, and deepens otherwise. A closed tableau is not a proof: the
    search continues with the remaining start clauses, and exhausting them
    also resets rather than ending the search.
    """

    def __call__(self, state: State) -> DFSPolicyDecision:
        closed_tableau = state.tableau.root.closed
        output = super().__call__(state)
        if isinstance(output, ProverOutcome) and output is not ProverOutcome.PROVED:
            # Exhausting an iteration only means this tableau search found no
            # dead end; like the original calculus, reset and keep searching
            # until the shadow is UNSAT.
            self._reset_iteration()
            return super().__call__(state)
        if not isinstance(output, UndoAction):
            return output
        if closed_tableau:
            return output
        root_application_id = state.tableau.root.applied_rule_application_id
        if root_application_id is None:
            return output

        self._reset_iteration()
        return UndoAction(root_application_id)

    def _reset_iteration(self) -> None:
        same_depth = self._shadow.consume_new_tableau_clause()
        target_depth = self.depth_limit if same_depth else self.depth_limit + 1
        self._reset_search()
        self._pending_path_limit_plan = None
        self._path_limit_hits_before_action.clear()
        self._terminal_path_limit_hits.clear()
        self._path_limit_hit = False
        self.depth_limit = max(0, target_depth - 1)


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


__all__ = [
    "LeanCoPCon",
    "LeanCoPCon_2",
    "SATCoPCon",
    "SATResetCoP",
]
