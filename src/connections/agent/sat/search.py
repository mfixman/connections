from __future__ import annotations

from dataclasses import replace
import random

from connections.agent.base import AgentOptions, AgentStatus
from connections.environment.actions import ApplyAction, UndoAction
from connections.environment.dynamics import Dynamics, start_clause_ids
from connections.environment.rules import ModelLemma

from .guidance import SATGuidance
from .shadow import _ShadowSAT


class SATCoPCon(SATGuidance):
    """Iterative DFS guided by a SAT model; only shadow UNSAT proves the problem.

    Model lemmas close search branches, not proofs. Ground instances retain
    their source clause IDs so an UNSAT core can label the original matrix.
    """

    def __init__(self, options=None, *, debug_sat_core=False, seed=0):
        options = options or AgentOptions(cut=True, factorization="equal", start="all")
        if options.initial_depth < 1:
            raise ValueError("initial_depth must be at least 1")
        super().__init__(self._next_action, options)
        self._constructed_options = options
        self._debug_sat_core = debug_sat_core
        self._seed = seed
        self._on_new_episode()

    def _on_new_episode(self):
        super()._on_new_episode()
        self.options = self._constructed_options
        self.depth_limit = self.options.initial_depth
        self._shadow = _ShadowSAT(debug_unsat_core=self._debug_sat_core)
        self._seeded = False
        self._random = None if self._seed is None else random.Random(self._seed)
        self._guidance_model = {}
        self._child_order = {}
        self._path_limit_hit = False

    def __call__(self, state):
        if state.matrix.logic != "classical":
            raise ValueError("SATCoPCon supports classical problems only")
        if state is not self._episode:
            self._episode = state
            self._on_new_episode()
        self._observe_shadow(state)
        if not self._shadow.solve():
            self._shadow.sat_core_clause_texts = [
                str(state.matrix.clauses[i])
                for i in self._shadow.sat_core_clause_ids
            ]
            self.status = AgentStatus.CLOSED
            return None
        if state.tableau.root.closed:
            app_id = state.tableau.root.applied_rule_application_id
            self._forget_subtree(state, app_id)
            return UndoAction(app_id)
        return super().__call__(state)

    def _shadow_seed_clause_ids(self, state):
        return start_clause_ids(state.matrix, self.options.start)

    def _current_goal(self, state):
        goal = state.tableau.root
        while goal.applied_rule_application_id is not None:
            app_id = goal.applied_rule_application_id
            order = self._child_order.get(app_id)
            if order is None:
                order = list(state.tableau.rule_applications[app_id].child_goal_ids)
                if self._random is not None:
                    self._random.shuffle(order)
                self._child_order[app_id] = order
            goal = next(state.tableau.goals[i] for i in order if not state.tableau.goals[i].closed)
        return goal.goal_id

    def _available_actions(self, state):
        while True:
            if self.options.cut:
                self._commit_closed(state)
            goal_id = self._current_goal(state)
            alternatives = self._alternatives.get(goal_id)
            if alternatives is None:
                alternatives = list(self._actions_for_goal(state, goal_id))
                if goal_id == state.tableau.root_goal_id and not alternatives:
                    return ()
                self._apply_scut(state, goal_id, alternatives)
                self._alternatives[goal_id] = alternatives
            if alternatives:
                return tuple(alternatives)
            self._forget(goal_id)
            actions = self._backtrack(state)
            if actions:
                return actions
            if not self._next_iteration():
                return ()
            self._clear_iteration()

    def _actions_for_goal(self, state, goal_id):
        if self._goal_solved_by_model(state, goal_id):
            return (ApplyAction(goal_id, ModelLemma()),)
        goal = state.tableau.goals[goal_id]
        actions = Dynamics.apply_actions(
            state,
            goal,
            factorization=self.options.factorization,
            start=self.options.start,
            depth_limit=self.depth_limit,
        )
        self._path_limit_hit |= bool(actions.path_limit_hits)
        if goal_id != state.tableau.root_goal_id and goal.depth + 1 >= self.depth_limit:
            self._path_limit_hit |= bool(actions.extension)
            return actions.start + actions.factorization + actions.reduction
        return actions.ordered()

    def _clear_iteration(self):
        self._alternatives.clear()
        self._committed.clear()
        self._scut_goal_id = None
        self._child_order.clear()
        self._path_limit_hit = False

    def _start_next_depth(self):
        self.depth_limit += 1
        self._guidance_model = dict(self._shadow.model)

    def _next_iteration(self):
        if self.options.comp is not None:
            if self.depth_limit >= self.options.comp:
                self.options = replace(self.options, comp=None, cut=False, scut=False)
                self.depth_limit = 0
        elif not self._path_limit_hit:
            return False
        self._start_next_depth()
        return True

    def _exhaustion_status(self):
        return AgentStatus.GAVE_UP

    def _forget_subtree(self, state, app_id):
        self._child_order.pop(app_id, None)
        super()._forget_subtree(state, app_id)

    def _resumable_ancestor_application(self, state):
        goal = state.tableau.goals[self._current_goal(state)]
        while goal.parent_rule_application_id is not None:
            app = state.tableau.rule_applications[goal.parent_rule_application_id]
            goal = state.tableau.goals[app.parent_goal_id]
            if self._alternatives.get(goal.goal_id):
                return goal.applied_rule_application_id
        return None
