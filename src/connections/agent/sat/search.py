"""Canonical SATCoP search expressed as actions on the shared tableau."""

import random
from types import GeneratorType

from connections.agent.base import AgentOptions, AgentStatus
from connections.environment.actions import ApplyAction, UndoAction
from connections.environment.dynamics import Dynamics, start_clause_ids
from connections.environment.rules import Extension, ModelLemma, Reduction

from .canonical_shadow import CanonicalShadow
from .grounding import Grounding, tautological
from .guidance import SATGuidance

class SATCoPCon(SATGuidance):
    """Reduction-first search with cuts and live SAT-selected extension goals.

    Only an UNSAT ground shadow proves the problem. The policy owns its
    search stack; SATResetCoP retains the historical implementation.
    """

    def __init__(self, options = None, *, debug_sat_core = False, seed = 0):
        options = options or AgentOptions(cut = True, start = "conjecture")
        if options.initial_depth < 1:
            raise ValueError("initial_depth must be at least 1")

        super().__init__(self._next_action, options)
        self._debug_sat_core = debug_sat_core
        self._seed = seed
        self._on_new_episode()

    def _on_new_episode(self):
        super()._on_new_episode()
        self.depth_limit = self.options.initial_depth
        self._shadow = CanonicalShadow(debug_unsat_core = self._debug_sat_core, seed = self._seed)
        self._random = None if self._seed is None else random.Random(self._seed)
        self._seeded = False
        self._guidance_model = self._shadow.model
        self.grounding = None
        self.stack = []

    def __call__(self, state):
        if state.matrix.logic != "classical":
            raise ValueError("SATCoPCon supports classical problems only")

        if state is not self._episode:
            self._episode = state
            self._on_new_episode()
            self.grounding = Grounding(state.matrix)
            self._observe_shadow(state)
            self.stack = [self.search(state)]

        action = self.advance() if self._shadow.satisfiable else None
        if not self._shadow.satisfiable:
            self._shadow.sat_core_clause_texts = [
                str(state.matrix.clauses[i]) for i in self._shadow.sat_core_clause_ids
            ]
            self.status = AgentStatus.CLOSED
            return None

        self.status = AgentStatus.GAVE_UP if action is None else AgentStatus.SEARCHING
        return action

    def advance(self):
        result = None
        while self.stack:
            try:
                item = self.stack[-1].send(result)
            except StopIteration as done:
                self.stack.pop()
                result = done.value
                continue

            result = None
            if isinstance(item, GeneratorType):
                self.stack.append(item)
            else:
                return item

        return None

    def _shadow_seed_clause_ids(self, state):
        mode = self.options.start
        if mode == "conjecture" and len(state.matrix.conjecture_clauses) == len(state.matrix):
            mode = "positive"

        return start_clause_ids(state.matrix, mode)

    def _observe_shadow(self, state):
        if self.grounding is None:
            self.grounding = Grounding(state.matrix)

        super()._observe_shadow(state)
        self._guidance_model = self._shadow.model

    def _ground_clause(self, state, clause, *, instance_id = None):
        return tuple(self._sat_literal(state, lit, instance_id = instance_id) for lit in clause)

    def _sat_literal(self, state, literal, *, instance_id, pending_bindings = (), create = True):
        if self.grounding is None:
            self.grounding = Grounding(state.matrix)

        key = self.grounding.atom_key(state, literal, instance_id, pending_bindings)
        atom = self._shadow.atom_ids.get(key)
        if atom is None and create:
            atom = self._shadow.atom_id(key)

        return None if atom is None else (atom if literal.polarity else -atom)

    def _literal_value(self, state, literal, *, instance_id, pending_bindings = ()):
        literal = self._sat_literal(
            state, literal, instance_id = instance_id,
            pending_bindings = pending_bindings, create = False,
        )
        return self._shadow.literal_value(literal)

    def _action_score(self, state, action):
        return 1 if isinstance(action.rule, Extension) else 0

    def _tie_break_key(self, state, action, index):
        return index

    def _actions_for_goal(self, state, goal_id):
        if goal_id == state.tableau.root_goal_id:
            allowed = set(self._shadow_seed_clause_ids(state))
            return tuple(
                ApplyAction(goal_id, rule) for rule in Dynamics.start_rules_for(state, "all")
                if rule.clause_idx in allowed
            )

        if not self.regular(state, goal_id):
            return ()

        for rule in reversed(Dynamics.reduction_rules_for(state, goal_id)):
            if not tautological(state, rule.constraint_delta.term_bindings):
                return (ApplyAction(goal_id, rule),)

        if state.tableau.goals[goal_id].depth >= self.depth_limit:
            return ()

        return self.extensions(state, goal_id)

    def regular(self, state, goal_id):
        literal, instance = state.literal_context_at(goal_id)
        for path_id in state.path_goal_ids_for(goal_id, literal.signed_symbol):
            other, other_instance = state.literal_context_at(path_id)
            if state.constraints.terms.equal_literals(
                literal, left_instance = instance, right = other, right_instance = other_instance,
            ):
                return False

        return True

    def extensions(self, state, goal_id):
        literal, _ = state.literal_context_at(goal_id)
        positions = list(state.matrix.connection_graph.get(literal.complement_symbol, ()))
        if self._random is not None:
            self._random.shuffle(positions)

        extensions = []
        for clause_id, literal_id in positions:
            action = Dynamics.extension_action_for_position(
                state, goal_id, clause_id, literal_id, instance_id = state.fresh_instance_id(),
            )
            if action is not None:
                extensions.append(action)

        return tuple(extensions)

    def search(self, state):
        root = state.tableau.root
        if root.applied_rule_application_id is not None:
            yield UndoAction(root.applied_rule_application_id)

        while self._shadow.satisfiable:
            starts = list(self._actions_for_goal(state, root.goal_id))
            if not starts:
                return

            while starts:
                action = self._next_action(state, starts)
                starts.remove(action)
                yield action
                yield self.promises(state, root.applied_rule_application_id, select_by_model = False)
                if not self._shadow.satisfiable:
                    return

                yield UndoAction(root.applied_rule_application_id)

            self._start_next_depth()

    def _start_next_depth(self):
        if not self._shadow.consume_new_tableau_clause():
            self.depth_limit += 1

    def prove(self, state, goal_id):
        alternatives = list(self._actions_for_goal(state, goal_id))
        while alternatives:
            action = self._next_action(state, alternatives)
            alternatives.remove(action)
            yield action
            app_id = state.tableau.goals[goal_id].applied_rule_application_id
            if tautological(state):
                yield UndoAction(app_id)
                continue

            self._observe_shadow(state)
            if not self._shadow.satisfiable or isinstance(action.rule, Reduction):
                return True

            if (yield self.promises(state, app_id, select_by_model = True)):
                return True

            yield UndoAction(app_id)

        return False

    def promises(self, state, app_id, *, select_by_model):
        pending = list(state.tableau.rule_applications[app_id].child_goal_ids)
        if self._random is not None:
            self._random.shuffle(pending)

        while pending and self._shadow.satisfiable:
            index = next((
                i for i, goal_id in enumerate(pending)
                if not select_by_model or self._context_value(state, state.literal_context_at(goal_id)) is True
            ), None)
            if index is None:
                for goal_id in pending:
                    yield ApplyAction(goal_id, ModelLemma())
                return True

            goal_id = pending[index]
            if select_by_model:
                pending[index] = pending[-1]
                pending.pop()
            else:
                pending.pop(index)

            if not (yield self.prove(state, goal_id)):
                return False

        return True
