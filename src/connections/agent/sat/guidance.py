from __future__ import annotations
from collections.abc import Sequence
import random
from connections.agent.dfs import OnlineDFSAgent
from connections.constraints.term import TermBinding
from connections.environment.actions import Action, UndoAction
from connections.environment.rules import Extension, Factorization, ModelLemma, Reduction, Start
from connections.environment.state import State
from connections.syntax.matrix import Clause, Literal
from .shadow import _ShadowSAT, _atom_key, _sat_value_score


class SATGuidance(OnlineDFSAgent):
    _shadow: _ShadowSAT
    _random: random.Random | None
    _guidance_model: dict[int, bool]

    def _shadow_seed_clause_ids(self, state: State) -> tuple[int, ...]:
        raise NotImplementedError

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
            ],
        }

    def _next_action(self, state: State, actions: Sequence[Action]) -> Action:
        return min(
            enumerate(actions),
            key=lambda indexed: (
                self._action_score(state, indexed[1]),
                self._tie_break_key(state, indexed[1], indexed[0]),
            ),
        )[1]

    def _tie_break_key(self, state: State, action: Action, index: int):
        _ = state, action
        return index if self._random is None else self._random.random()

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

    def _observe_shadow(self, state: State):
        if not self._seeded:
            for clause_idx in self._shadow_seed_clause_ids(state):
                self._shadow.add_clause(
                    self._ground_clause(
                        state,
                        state.matrix.clauses[clause_idx],
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
