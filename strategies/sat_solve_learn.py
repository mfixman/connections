"""Initial learned-policy scaffold for SAT-guided ties."""

from __future__ import annotations

import json
import os

from connections.calculi.classicalsat import ConnectionAction, Extension, Start
from connections.utils.primitives import Literal

from .sat_solve import SATSolveLiteral
from .sat_solve_ties import _SATSolveLiteralTieMixin


class SATSolveLearnInitial(_SATSolveLiteralTieMixin, SATSolveLiteral):
    """Initial learned-policy scaffold on top of ``SATSolveLiteral``.

    Policy:
    - keep ``SATSolveLiteral``'s primary SAT-model ordering exactly;
    - if the best bucket is unique, do nothing special;
    - if the best bucket is SAT-unknown and tied, rank the tied actions
      using whole-action SAT features and record a training row.

    This is intended as the first data-collection and feature-stability
    step before plugging in an external learned ranker.
    """

    _learn_schema = 'sat_solve_learn_initial/v1'
    _learn_short_clause_bound = 2
    _learn_row_sample_limit = 8

    def __init__(self, env):
        super().__init__(env)
        self._init_learning_guidance()

    def _init_learning_guidance(self) -> None:
        self._learn_count_clauses = set()
        self._learn_atom_counts = {}
        self._learn_signed_counts = {}
        self._learn_short_atom_counts = {}
        self._learn_decisions = 0
        self._learn_candidates = 0
        self._learn_agreed_default = 0
        self._learn_changed_default = 0
        self._learn_rows_dumped = 0
        self._learn_rows = []
        self._learn_dump_path = os.getenv('MYPYCOP_LEARN_INITIAL_ROWS')

    def training_rows(self) -> list[dict]:
        return list(self._learn_rows)

    def _sync_learning_counts(self) -> None:
        current_clauses = set(getattr(self.env.state.shadow, 'ground_clauses', ()))
        if not self._learn_count_clauses.issubset(current_clauses):
            self._learn_count_clauses = set()
            self._learn_atom_counts = {}
            self._learn_signed_counts = {}
            self._learn_short_atom_counts = {}

        for clause in current_clauses - self._learn_count_clauses:
            short_clause = len(clause) <= self._learn_short_clause_bound
            for sat_lit in clause:
                atom_id = abs(sat_lit)
                self._learn_atom_counts[atom_id] = self._learn_atom_counts.get(atom_id, 0) + 1
                self._learn_signed_counts[sat_lit] = self._learn_signed_counts.get(sat_lit, 0) + 1
                if short_clause:
                    self._learn_short_atom_counts[atom_id] = self._learn_short_atom_counts.get(atom_id, 0) + 1
            self._learn_count_clauses.add(clause)

    def _sat_literal_for_literal(self, literal: Literal) -> int | None:
        state = self.env.state
        if literal is None or not hasattr(state, '_atom_key'):
            return None

        substitution = getattr(state, 'substitution', None)
        grounded_literal = substitution(literal) if callable(substitution) else literal
        atom_id = state.shadow.atom_map.get(state._atom_key(grounded_literal))
        if atom_id is None:
            return None
        return -atom_id if grounded_literal.neg else atom_id

    def _child_literals(self, action: ConnectionAction) -> list[Literal]:
        match action:
            case Extension(clause_copy = clause_copy, lit_idx = lit_idx):
                return [
                    literal
                    for idx, literal in enumerate(clause_copy)
                    if idx != lit_idx
                ]
            case Start(clause_copy = clause_copy):
                return list(clause_copy or [])
        return []

    def _path_length(self) -> int:
        goal = getattr(self.env.state, 'goal', None)
        if goal is None or not hasattr(goal, 'path'):
            return 0
        try:
            return len(goal.path())
        except Exception:
            return 0

    def _evsids(self, sat_lit: int | None) -> float:
        if sat_lit is None:
            return 0.0
        solver = getattr(self.env.state.shadow, 'solver', None)
        if solver is None or not hasattr(solver, 'score'):
            return 0.0
        try:
            return float(solver.score(abs(sat_lit)))
        except Exception:
            return 0.0

    def _lookahead_literal(self) -> int | None:
        solver = getattr(self.env.state.shadow, 'solver', None)
        if solver is None or not hasattr(solver, 'lookahead'):
            return None
        try:
            lookahead_lit = solver.lookahead()
        except Exception:
            return None
        return lookahead_lit if isinstance(lookahead_lit, int) and lookahead_lit != 0 else None

    def _action_feature_row(self, action: ConnectionAction, lookahead_lit: int | None) -> dict:
        self._sync_learning_counts()

        fallback_kind, _ = self._connection_tie_key(action)
        selected_sat_lit = self._selected_sat_literal(action)
        child_literals = self._child_literals(action)
        child_sat_literals = [
            sat_lit
            for literal in child_literals
            if (sat_lit := self._sat_literal_for_literal(literal)) is not None
        ]

        child_values = [self.env.state.sat_literal_value(literal) for literal in child_literals]
        child_frequencies = [self._learn_atom_counts.get(abs(sat_lit), 0) for sat_lit in child_sat_literals]
        child_short_frequencies = [self._learn_short_atom_counts.get(abs(sat_lit), 0) for sat_lit in child_sat_literals]

        lookahead_candidates = [
            sat_lit
            for sat_lit in [selected_sat_lit, *child_sat_literals]
            if sat_lit is not None
        ]

        return {
            'action_id': action.id,
            'action_kind': type(action).__name__.lower(),
            'action_kind_index': fallback_kind,
            'depth': getattr(getattr(self.env.state, 'goal', None), 'depth', 0),
            'path_length': self._path_length(),
            'clause_size': len(getattr(action, 'clause_copy', []) or []),
            'selected_mapped': int(selected_sat_lit is not None),
            'selected_frequency': self._learn_atom_counts.get(abs(selected_sat_lit), 0) if selected_sat_lit is not None else 0,
            'selected_opposite_frequency': self._learn_signed_counts.get(-selected_sat_lit, 0) if selected_sat_lit is not None else 0,
            'selected_short_frequency': self._learn_short_atom_counts.get(abs(selected_sat_lit), 0) if selected_sat_lit is not None else 0,
            'selected_evsids': self._evsids(selected_sat_lit),
            'lookahead_exact': int(
                lookahead_lit is not None
                and any(sat_lit == lookahead_lit for sat_lit in lookahead_candidates)
            ),
            'lookahead_variable': int(
                lookahead_lit is not None
                and any(abs(sat_lit) == abs(lookahead_lit) for sat_lit in lookahead_candidates)
            ),
            'child_count': len(child_literals),
            'child_mapped': len(child_sat_literals),
            'child_model_false': sum(1 for value in child_values if value is False),
            'child_model_true': sum(1 for value in child_values if value is True),
            'child_model_unknown': sum(1 for value in child_values if value is None),
            'child_total_frequency': sum(child_frequencies),
            'child_min_frequency': min(child_frequencies) if child_frequencies else 0,
            'child_total_short_frequency': sum(child_short_frequencies),
            'child_min_short_frequency': min(child_short_frequencies) if child_short_frequencies else 0,
        }

    def _learn_initial_key(self, features: dict) -> tuple:
        fallback_kind = features['action_kind_index']
        fallback_size = features['clause_size'] - 1 if features['action_kind'] == 'extension' else 0
        reduction_bonus = int(features['action_kind'] == 'reduction')
        return (
            reduction_bonus,
            features['lookahead_exact'],
            features['lookahead_variable'],
            features['child_model_false'],
            features['child_mapped'],
            features['child_total_short_frequency'],
            features['child_total_frequency'],
            features['selected_short_frequency'],
            features['selected_opposite_frequency'],
            features['selected_frequency'],
            int(features['selected_evsids'] > 0),
            features['selected_evsids'],
            -features['child_model_true'],
            -features['child_count'],
            -fallback_kind,
            -fallback_size,
        )

    def _record_learning_row(
        self,
        tied: list[ConnectionAction],
        features_by_id: dict[str, dict],
        chosen: ConnectionAction,
        default_choice: ConnectionAction,
        lookahead_lit: int | None,
    ) -> None:
        goal = getattr(self.env.state, 'goal', None)
        row = {
            'schema': self._learn_schema,
            'best_score': 1,
            'depth': getattr(goal, 'depth', 0),
            'path_length': self._path_length(),
            'choice_size': len(tied),
            'shadow_clause_count': len(getattr(self.env.state.shadow, 'ground_clauses', ())),
            'shadow_new_clause_count': len(getattr(self.env.state.shadow, 'new_ground_clauses', ())),
            'lookahead_lit': lookahead_lit,
            'default_action_id': default_choice.id,
            'chosen_action_id': chosen.id,
            'candidates': [],
        }

        for action in tied:
            features = dict(features_by_id[action.id])
            features['label'] = int(action is chosen)
            features['default'] = int(action is default_choice)
            row['candidates'].append(features)

        if len(self._learn_rows) < self._learn_row_sample_limit:
            self._learn_rows.append(row)

        if self._learn_dump_path:
            with open(self._learn_dump_path, 'a', encoding = 'utf-8') as handle:
                handle.write(json.dumps(row, sort_keys = True) + '\n')
            self._learn_rows_dumped += 1

    def _choose_learn_initial_tie(self, tied: list[ConnectionAction]) -> ConnectionAction | None:
        if not tied:
            return None

        lookahead_lit = self._lookahead_literal()
        features_by_id = {
            action.id: self._action_feature_row(action, lookahead_lit)
            for action in tied
        }
        scored = [
            (action, self._learn_initial_key(features_by_id[action.id]))
            for action in tied
        ]
        default_choice = tied[0]
        chosen = max(scored, key = lambda item: item[1])[0]

        self._learn_decisions += 1
        self._learn_candidates += len(tied)
        if chosen is default_choice:
            self._learn_agreed_default += 1
        else:
            self._learn_changed_default += 1

        self._record_learning_row(tied, features_by_id, chosen, default_choice, lookahead_lit)
        return chosen

    def final_info(self) -> str | None:
        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        if self._learn_decisions:
            parts.append(
                'learn='
                f'schema:{self._learn_schema},'
                f'decisions:{self._learn_decisions},'
                f'candidates:{self._learn_candidates},'
                f'agreed_default:{self._learn_agreed_default},'
                f'changed_default:{self._learn_changed_default},'
                f'row_samples:{len(self._learn_rows)},'
                f'rows_dumped:{self._learn_rows_dumped}'
            )
        return '; '.join(parts) if parts else None

    def _choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        tied = self._best_score_tied_actions(actions)
        if not tied:
            return None
        if self.score(tied[0]) != 1 or len(tied) == 1:
            return tied[0]
        return self._choose_learn_initial_tie(tied)
