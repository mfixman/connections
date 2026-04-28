"""ML-lite learning strategies layered on top of ``SATSolveLiteral``."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import uuid

from connections.calculi.classicalsat import (
    Backtrack,
    ConnectionAction,
    Extension,
    Reduction,
    Reset,
    Start,
)
from connections.utils.primitives import Literal

from .sat_solve import SATSolveLiteral
from .sat_solve_ties import _SATSolveLiteralTieMixin


class _BanditTableMixin:
    def _zero_bandit_table(self, contexts: tuple[str, ...], arms: tuple[str, ...], value):
        return {
            context: {arm: value() for arm in arms}
            for context in contexts
        }


class _LearnStoreMixin(_BanditTableMixin):
    """Read and write small persistent bandit summaries."""

    _learn_store_schema = 'sat_solve_learn_store/v1'
    _learn_store_path = Path(__file__).resolve().parents[1] / '.satsolvelearn_store'

    def load_bandit_store(self, section: str, contexts: tuple[str, ...], arms: tuple[str, ...]) -> tuple[dict, dict]:
        pulls = self._zero_bandit_table(contexts, arms, int)
        rewards = self._zero_bandit_table(contexts, arms, float)

        path = Path(self._learn_store_path)
        if not path.exists() or not path.is_dir():
            return pulls, rewards

        for store_file in sorted(path.glob(f'{section}-*.json')):
            try:
                data = json.loads(store_file.read_text(encoding = 'utf-8'))
            except json.JSONDecodeError:
                continue
            if data.get('schema') != self._learn_store_schema:
                continue

            raw_section = data.get(section, {})
            for context in contexts:
                raw_context = raw_section.get(context, {})
                for arm in arms:
                    raw_arm = raw_context.get(arm, {})
                    pulls[context][arm] += int(raw_arm.get('pulls', 0))
                    rewards[context][arm] += float(raw_arm.get('reward', 0.0))
        return pulls, rewards

    def persist_bandit_store(
        self,
        section: str,
        contexts: tuple[str, ...],
        arms: tuple[str, ...],
        pulls: dict,
        rewards: dict,
    ) -> None:
        path = Path(self._learn_store_path)
        path.parent.mkdir(parents = True, exist_ok = True)
        path.mkdir(parents = True, exist_ok = True)

        data = {'schema': self._learn_store_schema, section: {}}
        raw_section = data[section]
        for context in contexts:
            raw_context = raw_section.setdefault(context, {})
            for arm in arms:
                raw_context[arm] = {
                    'pulls': int(pulls[context][arm]),
                    'reward': float(rewards[context][arm]),
                }

        unique = f'{section}-{os.getpid()}-{time.time_ns()}-{uuid.uuid4().hex}'
        tmp_path = path / f'.{unique}.tmp'
        final_path = path / f'{unique}.json'
        tmp_path.write_text(json.dumps(data, sort_keys = True) + '\n', encoding = 'utf-8')
        tmp_path.replace(final_path)

    def bandit_store_file_count(self, section: str) -> int:
        path = Path(self._learn_store_path)
        if not path.exists() or not path.is_dir():
            return 0
        return sum(1 for _ in path.glob(f'{section}-*.json'))


class _ForwardBanditMixin(_BanditTableMixin):
    """Unknown-tie bandit for forward SATSolveLiteral actions."""

    _learn_schema = 'sat_solve_learn_initial_bandit/v1'
    _learn_short_clause_bound = 2
    _learn_row_sample_limit = 8

    _forward_arms = ('default', 'short', 'lookahead', 'whole')
    _forward_contexts = ('with_reduction', 'extensions_only')
    _forward_prior_count = 2.0
    _forward_ucb_scale = 0.40
    _forward_prior_reward_mass = {
        'with_reduction': {
            'default': 1.0,
            'short': 1.2,
            'lookahead': 0.9,
            'whole': 1.1,
        },
        'extensions_only': {
            'default': 1.0,
            'short': 0.9,
            'lookahead': 1.1,
            'whole': 1.2,
        },
    }
    _forward_arm_alias = {
        'default': 'de',
        'short': 'sh',
        'lookahead': 'lo',
        'whole': 'wh',
    }
    _forward_context_alias = {
        'with_reduction': 'mix',
        'extensions_only': 'ext',
    }

    def _init_forward_bandit(self) -> None:
        self._learn_count_clauses = set()
        self._learn_atom_counts = {}
        self._learn_signed_counts = {}
        self._learn_short_atom_counts = {}
        self._learn_rows_dumped = 0
        self._learn_rows = []
        self._learn_dump_path = os.getenv('MYPYCOP_LEARN_INITIAL_ROWS')

        self._forward_base_pulls = self._zero_bandit_table(self._forward_contexts, self._forward_arms, int)
        self._forward_base_rewards = self._zero_bandit_table(self._forward_contexts, self._forward_arms, float)
        self._forward_pulls = self._zero_bandit_table(self._forward_contexts, self._forward_arms, int)
        self._forward_rewards = self._zero_bandit_table(self._forward_contexts, self._forward_arms, float)
        self._forward_online_updates = True
        self._forward_explore = True
        self._forward_record_rows = True

        self._forward_decisions = 0
        self._forward_updates = 0
        self._forward_open = []
        self._forward_agreed_default = 0
        self._forward_changed_default = 0

    def training_rows(self) -> list[dict]:
        return list(self._learn_rows)

    def _set_forward_store(self, pulls: dict, rewards: dict) -> None:
        self._forward_base_pulls = pulls
        self._forward_base_rewards = rewards

    def _forward_total_pulls(self, context: str, arm: str) -> float:
        return self._forward_prior_count + self._forward_base_pulls[context][arm] + self._forward_pulls[context][arm]

    def _forward_total_reward(self, context: str, arm: str) -> float:
        return (
            self._forward_prior_reward_mass[context][arm]
            + self._forward_base_rewards[context][arm]
            + self._forward_rewards[context][arm]
        )

    def _forward_loaded_pulls(self) -> int:
        return sum(
            pulls
            for context in self._forward_contexts
            for pulls in self._forward_base_pulls[context].values()
        )

    def _forward_local_pulls(self) -> int:
        return sum(
            pulls
            for context in self._forward_contexts
            for pulls in self._forward_pulls[context].values()
        )

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

    def _whole_action_key(self, features: dict) -> tuple:
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

    def _lookahead_tie_key(self, action: ConnectionAction, lookahead_lit: int | None) -> tuple[int, int, float, int, int]:
        sat_lit = self._selected_sat_literal(action)
        evsids = self._evsids(sat_lit)
        exact_lookahead = int(sat_lit is not None and sat_lit == lookahead_lit)
        variable_lookahead = int(
            sat_lit is not None
            and lookahead_lit is not None
            and abs(sat_lit) == abs(lookahead_lit)
        )
        fallback_kind, fallback_size = self._connection_tie_key(action)
        return (exact_lookahead, variable_lookahead, evsids, -fallback_kind, -fallback_size)

    def _forward_context(self, tied: list[ConnectionAction]) -> str:
        if any(isinstance(action, Reduction) for action in tied):
            return 'with_reduction'
        return 'extensions_only'

    def _forward_context_from_features(
        self,
        tied: list[ConnectionAction],
        features_by_id: dict[str, dict],
        lookahead_lit: int | None,
    ) -> str:
        del features_by_id, lookahead_lit
        return self._forward_context(tied)

    def _forward_bandit_score(self, context: str, arm: str) -> float:
        count = self._forward_total_pulls(context, arm)
        reward = self._forward_total_reward(context, arm)
        mean = reward / count
        if not self._forward_explore:
            return mean

        total = sum(self._forward_total_pulls(context, other) for other in self._forward_arms)
        bonus = self._forward_ucb_scale * math.sqrt(math.log(max(total, 2.0)) / count)
        return mean + bonus

    def _forward_arm_action_choices(
        self,
        tied: list[ConnectionAction],
        features_by_id: dict[str, dict],
        lookahead_lit: int | None,
    ) -> dict[str, ConnectionAction]:
        return {
            'default': tied[0],
            'short': min(tied, key = self._connection_tie_key),
            'lookahead': max(tied, key = lambda action: self._lookahead_tie_key(action, lookahead_lit)),
            'whole': max(tied, key = lambda action: self._whole_action_key(features_by_id[action.id])),
        }

    def _record_forward_row(
        self,
        context: str,
        tied: list[ConnectionAction],
        features_by_id: dict[str, dict],
        arm_choices: dict[str, ConnectionAction],
        arm_scores: dict[str, float],
        chosen_arm: str,
        chosen: ConnectionAction,
        default_choice: ConnectionAction,
        lookahead_lit: int | None,
    ) -> dict | None:
        if not self._forward_record_rows and not self._learn_dump_path:
            return None

        goal = getattr(self.env.state, 'goal', None)
        row = {
            'schema': self._learn_schema,
            'context': context,
            'depth': getattr(goal, 'depth', 0),
            'path_length': self._path_length(),
            'choice_size': len(tied),
            'shadow_clause_count': len(getattr(self.env.state.shadow, 'ground_clauses', ())),
            'lookahead_lit': lookahead_lit,
            'default_action_id': default_choice.id,
            'chosen_arm': chosen_arm,
            'chosen_action_id': chosen.id,
            'arm_scores': {arm: round(score, 6) for arm, score in arm_scores.items()},
            'arm_choices': {arm: action.id for arm, action in arm_choices.items()},
            'candidates': [],
        }

        for action in tied:
            features = dict(features_by_id[action.id])
            features['label'] = int(action is chosen)
            features['default'] = int(action is default_choice)
            row['candidates'].append(features)

        if self._forward_record_rows and len(self._learn_rows) < self._learn_row_sample_limit:
            self._learn_rows.append(row)
        return row

    def _dump_learning_row(self, row: dict) -> None:
        if not self._learn_dump_path:
            return
        with open(self._learn_dump_path, 'a', encoding = 'utf-8') as handle:
            handle.write(json.dumps(row, sort_keys = True) + '\n')
        self._learn_rows_dumped += 1

    def _finalize_forward_record(self, record: dict, reward: float, outcome: str, ground_gain: int) -> None:
        context = record['context']
        arm = record['arm']

        if self._forward_online_updates:
            self._forward_pulls[context][arm] += 1
            self._forward_rewards[context][arm] += reward
            self._forward_updates += 1

        row = record.get('row')
        if row is not None:
            row['reward'] = reward
            row['outcome'] = outcome
            row['ground_gain'] = ground_gain
            self._dump_learning_row(row)

    def _settle_forward_feedback(self, force: bool = False) -> None:
        if not self._forward_open:
            return

        state = self.env.state
        current_proof = list(getattr(state, 'proof_sequence', ()))
        current_ground_clause_count = len(getattr(state.shadow, 'ground_clauses', ()))
        theorem = bool(
            getattr(state, 'is_terminal', False)
            and isinstance(getattr(state, 'info', None), str)
            and state.info.startswith('Theorem')
        )

        remaining = []
        for record in self._forward_open:
            ground_gain = max(0, current_ground_clause_count - record['ground_clause_count'])
            proof_index = record['proof_index']
            action = record['action']
            action_alive = proof_index < len(current_proof) and current_proof[proof_index] is action

            if theorem:
                self._finalize_forward_record(record, 1.0, 'theorem', ground_gain)
            elif ground_gain > 0:
                self._finalize_forward_record(record, 1.0, 'ground', ground_gain)
            elif not action_alive or force:
                outcome = 'backtrack' if not action_alive else 'forced'
                self._finalize_forward_record(record, 0.0, outcome, ground_gain)
            else:
                remaining.append(record)

        self._forward_open = remaining

    def _choose_forward_tie(self, tied: list[ConnectionAction]) -> ConnectionAction | None:
        if not tied:
            return None

        lookahead_lit = self._lookahead_literal()
        features_by_id = {
            action.id: self._action_feature_row(action, lookahead_lit)
            for action in tied
        }
        context = self._forward_context_from_features(tied, features_by_id, lookahead_lit)
        arm_choices = self._forward_arm_action_choices(tied, features_by_id, lookahead_lit)
        arm_scores = {
            arm: self._forward_bandit_score(context, arm)
            for arm in self._forward_arms
        }
        chosen_arm = max(self._forward_arms, key = lambda arm: arm_scores[arm])
        chosen = arm_choices[chosen_arm]
        default_choice = tied[0]
        row = self._record_forward_row(
            context,
            tied,
            features_by_id,
            arm_choices,
            arm_scores,
            chosen_arm,
            chosen,
            default_choice,
            lookahead_lit,
        )

        self._forward_decisions += 1
        if chosen is default_choice:
            self._forward_agreed_default += 1
        else:
            self._forward_changed_default += 1

        if self._forward_online_updates or row is not None:
            self._forward_open.append(
                {
                    'context': context,
                    'arm': chosen_arm,
                    'action': chosen,
                    'proof_index': len(getattr(self.env.state, 'proof_sequence', ())),
                    'ground_clause_count': len(getattr(self.env.state.shadow, 'ground_clauses', ())),
                    'row': row,
                }
            )
        return chosen

    def choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        tied = self._best_score_tied_actions(actions)
        if not tied:
            return None
        if self.score(tied[0]) != 1 or len(tied) == 1:
            return tied[0]
        return self._choose_forward_tie(tied)

    def _forward_context_summary(self, context: str) -> str:
        total = sum(self._forward_pulls[context].values())
        arm_parts = []
        for arm in self._forward_arms:
            pulls = self._forward_pulls[context][arm]
            mean = self._forward_rewards[context][arm] / pulls if pulls else 0.0
            arm_parts.append(f'{self._forward_arm_alias[arm]}:{pulls}/{mean:.2f}')
        return f'{self._forward_context_alias[context]}:{total}[{",".join(arm_parts)}]'

    def _forward_nonzero_contexts(self) -> list[str]:
        return [
            context
            for context in self._forward_contexts
            if sum(self._forward_pulls[context].values()) > 0
        ]

    def _forward_final_info_piece(self) -> str | None:
        if not self._forward_decisions:
            return None
        used_contexts = self._forward_nonzero_contexts()
        context_summary = ','.join(
            self._forward_context_summary(context)
            for context in used_contexts
        )
        return (
            'learn='
            f'schema:{self._learn_schema},'
            f'decisions:{self._forward_decisions},'
            f'updates:{self._forward_updates},'
            f'agreed_default:{self._forward_agreed_default},'
            f'changed_default:{self._forward_changed_default},'
            f'used_contexts:{len(used_contexts)},'
            f'row_samples:{len(self._learn_rows)},'
            f'rows_dumped:{self._learn_rows_dumped},'
            f'{context_summary}'
        )


class _ControlBanditMixin(_BanditTableMixin):
    """Bandit over backtrack / small backjump / reset control actions."""

    _control_schema = 'sat_solve_learn_control_bandit/v1'
    _control_arms = ('bt1', 'bt2', 'reset')
    _control_contexts = ('shadow_growing', 'shadow_stagnant')
    _control_prior_count = 1.0
    _control_ucb_scale = 0.40
    _control_prior_reward_mass = {
        'shadow_growing': {'bt1': 1.0, 'bt2': 1.0, 'reset': 1.0},
        'shadow_stagnant': {'bt1': 1.0, 'bt2': 1.0, 'reset': 1.0},
    }
    _control_arm_alias = {'bt1': 'b1', 'bt2': 'b2', 'reset': 'rs'}
    _control_context_alias = {'shadow_growing': 'grow', 'shadow_stagnant': 'stall'}

    def _init_control_bandit(self) -> None:
        self._control_pulls = self._zero_bandit_table(self._control_contexts, self._control_arms, int)
        self._control_rewards = self._zero_bandit_table(self._control_contexts, self._control_arms, float)
        self._control_decisions = 0
        self._control_updates = 0
        self._control_agreed_default = 0
        self._control_changed_default = 0
        self._control_open = None

    def _control_context(self) -> str:
        if getattr(self.env.state.shadow, 'new_ground_clauses', ()):
            return 'shadow_growing'
        return 'shadow_stagnant'

    def _control_score(self, context: str, arm: str) -> float:
        count = self._control_prior_count + self._control_pulls[context][arm]
        reward = self._control_prior_reward_mass[context][arm] + self._control_rewards[context][arm]
        mean = reward / count
        total = len(self._control_arms) * self._control_prior_count + sum(self._control_pulls[context].values())
        bonus = self._control_ucb_scale * math.sqrt(math.log(max(total, 2.0)) / count)
        return mean + bonus

    def _control_arm_choices(self, actions: list[ConnectionAction]) -> dict[str, ConnectionAction]:
        choices = {}
        backtrack_action = next((action for action in actions if isinstance(action, Backtrack)), None)
        if backtrack_action is not None:
            choices['bt1'] = backtrack_action
            goal = getattr(self.env.state, 'goal', None)
            if (
                goal is not None
                and getattr(goal, 'parent', None) is not None
                and getattr(goal.parent, 'parent', None) is not None
            ):
                choices['bt2'] = Backtrack(id = 'bt2', steps = 2)

        reset_action = next((action for action in actions if isinstance(action, Reset)), None)
        if reset_action is not None:
            choices['reset'] = reset_action
        return choices

    def _finalize_control_record(self, reward: float, outcome: str, ground_gain: int) -> None:
        if self._control_open is None:
            return

        context = self._control_open['context']
        arm = self._control_open['arm']
        self._control_pulls[context][arm] += 1
        self._control_rewards[context][arm] += reward
        self._control_updates += 1
        self._control_open = None

    def _settle_control_feedback(self, actions: list[ConnectionAction], force: bool = False) -> None:
        if self._control_open is None:
            return

        state = self.env.state
        ground_gain = max(0, len(getattr(state.shadow, 'ground_clauses', ())) - self._control_open['ground_clause_count'])
        theorem = bool(
            getattr(state, 'is_terminal', False)
            and isinstance(getattr(state, 'info', None), str)
            and state.info.startswith('Theorem')
        )
        forward_available = any(isinstance(action, (Start, Reduction, Extension)) for action in actions)

        if theorem:
            self._finalize_control_record(1.0, 'theorem', ground_gain)
        elif ground_gain > 0:
            self._finalize_control_record(1.0, 'ground', ground_gain)
        elif force or not forward_available:
            self._finalize_control_record(0.0, 'forced' if force else 'stalled', ground_gain)

    def _choose_control_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        choices = self._control_arm_choices(actions)
        if not choices:
            return None
        if len(choices) == 1:
            return next(iter(choices.values()))

        context = self._control_context()
        arm_scores = {
            arm: self._control_score(context, arm)
            for arm in choices
        }
        chosen_arm = max(choices, key = lambda arm: arm_scores[arm])
        chosen = choices[chosen_arm]
        default_choice = choices['bt1'] if 'bt1' in choices else next(iter(choices.values()))

        self._control_decisions += 1
        if chosen is default_choice:
            self._control_agreed_default += 1
        else:
            self._control_changed_default += 1

        self._control_open = {
            'context': context,
            'arm': chosen_arm,
            'ground_clause_count': len(getattr(self.env.state.shadow, 'ground_clauses', ())),
        }
        return chosen

    def _control_context_summary(self, context: str) -> str:
        total = sum(self._control_pulls[context].values())
        arm_parts = []
        for arm in self._control_arms:
            pulls = self._control_pulls[context][arm]
            mean = self._control_rewards[context][arm] / pulls if pulls else 0.0
            arm_parts.append(f'{self._control_arm_alias[arm]}:{pulls}/{mean:.2f}')
        return f'{self._control_context_alias[context]}:{total}[{",".join(arm_parts)}]'

    def _control_nonzero_contexts(self) -> list[str]:
        return [
            context
            for context in self._control_contexts
            if sum(self._control_pulls[context].values()) > 0
        ]

    def _control_final_info_piece(self) -> str | None:
        if not self._control_decisions:
            return None
        used_contexts = self._control_nonzero_contexts()
        context_summary = ','.join(
            self._control_context_summary(context)
            for context in used_contexts
        )
        return (
            'control='
            f'schema:{self._control_schema},'
            f'decisions:{self._control_decisions},'
            f'updates:{self._control_updates},'
            f'agreed_default:{self._control_agreed_default},'
            f'changed_default:{self._control_changed_default},'
            f'used_contexts:{len(used_contexts)},'
            f'{context_summary}'
        )


class SATSolveLearnInitial(_ForwardBanditMixin, _SATSolveLiteralTieMixin, SATSolveLiteral):
    """Online UCB over four cheap tie-breakers inside SAT-unknown ties."""

    def __init__(self, env):
        super().__init__(env)
        self._init_forward_bandit()

    def step(self):
        self._settle_forward_feedback()
        return super().step()

    def final_info(self) -> str | None:
        self._settle_forward_feedback(force = True)

        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        forward_info = self._forward_final_info_piece()
        if forward_info is not None:
            parts.append(forward_info)
        return '; '.join(parts) if parts else None


class SATSolveLearnTrain(_LearnStoreMixin, SATSolveLearnInitial):
    """Train ``SATSolveLearnInitial`` and persist aggregate arm statistics."""

    def __init__(self, env):
        super().__init__(env)
        pulls, rewards = self.load_bandit_store('forward', self._forward_contexts, self._forward_arms)
        self._set_forward_store(pulls, rewards)
        self._forward_record_rows = False
        self._learn_rows = []
        self._learn_dump_path = None
        self._forward_store_saved = False

    def final_info(self) -> str | None:
        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)

        if not self._forward_store_saved:
            self.persist_bandit_store(
                'forward',
                self._forward_contexts,
                self._forward_arms,
                self._forward_pulls,
                self._forward_rewards,
            )
            self._forward_store_saved = True

        parts.append(
            'learn_store='
            f'mode:train,'
            f'path:{Path(self._learn_store_path).name},'
            f'files:{self.bandit_store_file_count("forward")},'
            f'loaded_pulls:{self._forward_loaded_pulls()},'
            f'saved_pulls:{self._forward_local_pulls()}'
        )
        return '; '.join(parts) if parts else None


class SATSolveLearnEval(_LearnStoreMixin, SATSolveLearnInitial):
    """Evaluate with frozen arm means loaded from the persistent store."""

    def __init__(self, env):
        super().__init__(env)
        pulls, rewards = self.load_bandit_store('forward', self._forward_contexts, self._forward_arms)
        self._set_forward_store(pulls, rewards)
        self._forward_online_updates = False
        self._forward_explore = False
        self._forward_record_rows = False
        self._learn_rows = []
        self._learn_dump_path = None

    def final_info(self) -> str | None:
        parts = []
        base_info = SATSolveLiteral.final_info(self)
        if base_info is not None:
            parts.append(base_info)
        if self._forward_decisions:
            parts.append(
                'learn_eval='
                f'schema:{self._learn_schema},'
                f'decisions:{self._forward_decisions},'
                f'files:{self.bandit_store_file_count("forward")},'
                f'loaded_pulls:{self._forward_loaded_pulls()},'
                f'path:{Path(self._learn_store_path).name}'
            )
        return '; '.join(parts) if parts else None


class SATSolveLearnControl(_ControlBanditMixin, SATSolveLiteral):
    """Online UCB over ``Backtrack(1)``, ``Backtrack(2)``, and ``Reset``."""

    def __init__(self, env):
        super().__init__(env)
        self._init_control_bandit()

    def step(self):
        state = self.env.state
        actions = self.env.legal_actions()
        self._settle_control_feedback(actions)
        self._record_nonzero_atom_scores()
        action = self.choose_action(actions)
        self._record_literal_choice_diagnostics(actions, action)

        if isinstance(action, (Reduction, Extension)):
            action.cut = True

        if isinstance(action, (Start, Reduction, Extension)):
            return action

        if (
            state.goal.parent is None
            and any(isinstance(candidate, Reset) for candidate in actions)
            and state.shadow.new_ground_clauses
        ):
            return Reset(id = 'rs', depth = state.max_depth)

        if action := self._choose_control_action(actions):
            return action

        raise LookupError('expected Backtrack or Reset action')

    def final_info(self) -> str | None:
        self._settle_control_feedback(self.env.legal_actions(), force = True)

        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        control_info = self._control_final_info_piece()
        if control_info is not None:
            parts.append(control_info)
        return '; '.join(parts) if parts else None


class SATSolveLearnCombined(_ControlBanditMixin, SATSolveLearnInitial):
    """Combine forward unknown-tie learning with control-action learning."""

    def __init__(self, env):
        super().__init__(env)
        self._init_control_bandit()

    def step(self):
        state = self.env.state
        actions = self.env.legal_actions()
        self._settle_forward_feedback()
        self._settle_control_feedback(actions)
        self._record_nonzero_atom_scores()
        action = self.choose_action(actions)
        self._record_literal_choice_diagnostics(actions, action)

        if isinstance(action, (Reduction, Extension)):
            action.cut = True

        if isinstance(action, (Start, Reduction, Extension)):
            return action

        if (
            state.goal.parent is None
            and any(isinstance(candidate, Reset) for candidate in actions)
            and state.shadow.new_ground_clauses
        ):
            return Reset(id = 'rs', depth = state.max_depth)

        if action := self._choose_control_action(actions):
            return action

        raise LookupError('expected Backtrack or Reset action')

    def final_info(self) -> str | None:
        self._settle_forward_feedback(force = True)
        self._settle_control_feedback(self.env.legal_actions(), force = True)

        parts = []
        base_info = SATSolveLiteral.final_info(self)
        if base_info is not None:
            parts.append(base_info)
        forward_info = self._forward_final_info_piece()
        if forward_info is not None:
            parts.append(forward_info)
        control_info = self._control_final_info_piece()
        if control_info is not None:
            parts.append(control_info)
        return '; '.join(parts) if parts else None


class SATSolveLearnSATContext(SATSolveLearnCombined):
    """SATSolveLearnCombined with richer SAT-aware bandit contexts.

    This keeps the same forward and control arms as ``SATSolveLearnCombined``,
    but replaces the coarse contexts with SAT-shadow summaries:

    - Forward unknown-tie contexts add:
      - whether any tied action matches CaDiCaL's lookahead variable;
      - whether any tied action has non-zero EVSIDS score.
    - Control dead-end contexts add:
      - whether a SAT model is currently available;
      - whether the solver has exposed any non-zero atom score so far.

    The goal is to let the bandit separate regimes where SAT-side guidance is
    likely to be informative from those where the shadow carries little signal.
    """

    _learn_schema = 'sat_solve_learn_sat_context_forward_bandit/v1'
    _control_schema = 'sat_solve_learn_sat_context_control_bandit/v1'

    _forward_contexts = tuple(
        f'{base}|{lookahead}|{score}'
        for base in ('with_reduction', 'extensions_only')
        for lookahead in ('lookahead_hit', 'no_lookahead_hit')
        for score in ('evsids', 'flat')
    )
    _forward_context_alias = {
        f'{base}|{lookahead}|{score}': (
            f'{"mix" if base == "with_reduction" else "ext"}'
            f'/{ "lh" if lookahead == "lookahead_hit" else "nlh"}'
            f'/{ "ev" if score == "evsids" else "fl"}'
        )
        for base in ('with_reduction', 'extensions_only')
        for lookahead in ('lookahead_hit', 'no_lookahead_hit')
        for score in ('evsids', 'flat')
    }
    _forward_prior_reward_mass = {
        f'{base}|{lookahead}|{score}': dict(
            _ForwardBanditMixin._forward_prior_reward_mass[base],
        )
        for base in ('with_reduction', 'extensions_only')
        for lookahead in ('lookahead_hit', 'no_lookahead_hit')
        for score in ('evsids', 'flat')
    }

    _control_contexts = tuple(
        f'{growth}|{model}|{score}'
        for growth in ('shadow_growing', 'shadow_stagnant')
        for model in ('model', 'nomodel')
        for score in ('scored', 'flat')
    )
    _control_context_alias = {
        f'{growth}|{model}|{score}': (
            f'{"grow" if growth == "shadow_growing" else "stall"}'
            f'/{ "mdl" if model == "model" else "nmdl"}'
            f'/{ "sc" if score == "scored" else "fl"}'
        )
        for growth in ('shadow_growing', 'shadow_stagnant')
        for model in ('model', 'nomodel')
        for score in ('scored', 'flat')
    }
    _control_prior_reward_mass = {
        f'{growth}|{model}|{score}': dict(
            _ControlBanditMixin._control_prior_reward_mass[growth],
        )
        for growth in ('shadow_growing', 'shadow_stagnant')
        for model in ('model', 'nomodel')
        for score in ('scored', 'flat')
    }

    def _forward_context_from_features(
        self,
        tied: list[ConnectionAction],
        features_by_id: dict[str, dict],
        lookahead_lit: int | None,
    ) -> str:
        base = super()._forward_context(tied)
        del lookahead_lit

        lookahead_bucket = 'no_lookahead_hit'
        if any(
            features['lookahead_exact'] or features['lookahead_variable']
            for features in features_by_id.values()
        ):
            lookahead_bucket = 'lookahead_hit'

        score_bucket = 'flat'
        if any(features['selected_evsids'] > 0 for features in features_by_id.values()):
            score_bucket = 'evsids'

        return f'{base}|{lookahead_bucket}|{score_bucket}'

    def _control_context(self) -> str:
        base = super()._control_context()
        model_bucket = 'model' if getattr(self.env.state.shadow, 'model', {}) else 'nomodel'
        score_bucket = 'scored' if getattr(self, '_sat_nonzero_atom_score', None) not in (None, 0) else 'flat'
        return f'{base}|{model_bucket}|{score_bucket}'
