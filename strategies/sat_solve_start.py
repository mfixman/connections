"""Root/start-clause SAT-guided variants."""

from __future__ import annotations

from typing import NamedTuple

from connections.calculi.classicalsat import ConnectionAction, Reset, Start

from .sat_solve import SATSolveLiteral


class _StartClauseSignals(NamedTuple):
    mapped: int
    unmapped: int
    min_frequency: int
    total_frequency: int
    length: int
    model_false: int
    model_true: int
    model_unknown: int


class _SATSolveLiteralStartMixin:
    """Choose root start clauses using cheap statistics from the shadow CNF."""

    _start_mode = 'rare'

    def _init_start_guidance(self) -> None:
        self._start_frequency_clauses = set()
        self._start_atom_counts = {}
        self._start_signed_counts = {}
        self._start_decisions = 0
        self._start_candidates = 0
        self._start_mapped = 0
        self._start_unmapped = 0
        self._start_changed_default = 0
        self._start_agreed_default = 0
        self._start_unique_best = 0
        self._start_ties = 0
        self._start_total_min_frequency = 0
        self._start_model_false = 0
        self._start_model_true = 0
        self._start_model_unknown = 0

    def _sync_start_frequency_counts(self) -> None:
        current_clauses = set(self.env.state.shadow.ground_clauses)
        if not self._start_frequency_clauses.issubset(current_clauses):
            self._start_frequency_clauses = set()
            self._start_atom_counts = {}
            self._start_signed_counts = {}

        for clause in current_clauses - self._start_frequency_clauses:
            for sat_lit in clause:
                atom_id = abs(sat_lit)
                self._start_atom_counts[atom_id] = self._start_atom_counts.get(atom_id, 0) + 1
                self._start_signed_counts[sat_lit] = self._start_signed_counts.get(sat_lit, 0) + 1
            self._start_frequency_clauses.add(clause)

    def _start_sat_literals(self, action: Start) -> list[int]:
        if action.clause_copy is None:
            return []

        state = self.env.state
        if not hasattr(state, '_atom_key'):
            return []

        substitution = getattr(state, 'substitution', None)
        sat_literals = []
        seen = set()
        for literal in action.clause_copy:
            grounded_literal = substitution(literal) if callable(substitution) else literal
            atom_id = state.shadow.atom_map.get(state._atom_key(grounded_literal))
            if atom_id is None:
                continue
            sat_lit = -atom_id if grounded_literal.neg else atom_id
            if sat_lit not in seen:
                sat_literals.append(sat_lit)
                seen.add(sat_lit)
        return sat_literals

    def _start_clause_signals(self, action: Start) -> _StartClauseSignals:
        sat_literals = self._start_sat_literals(action)
        frequencies = [self._start_atom_counts.get(abs(sat_lit), 0) for sat_lit in sat_literals]
        model = getattr(self.env.state.shadow, 'model', {})

        model_false = 0
        model_true = 0
        model_unknown = 0
        for sat_lit in sat_literals:
            value = model.get(abs(sat_lit))
            if value is None:
                model_unknown += 1
                continue
            lit_value = value if sat_lit > 0 else (not value)
            if lit_value:
                model_true += 1
            else:
                model_false += 1

        length = len(action.clause_copy or [])
        return _StartClauseSignals(
            mapped = len(sat_literals),
            unmapped = max(0, length - len(sat_literals)),
            min_frequency = min(frequencies) if frequencies else 0,
            total_frequency = sum(frequencies),
            length = length,
            model_false = model_false,
            model_true = model_true,
            model_unknown = model_unknown,
        )

    def _start_clause_score(self, signals: _StartClauseSignals) -> tuple[int, ...]:
        mapped = int(signals.mapped > 0)
        rare = (-signals.min_frequency, -signals.total_frequency, -signals.length)
        if self._start_mode == 'model':
            return (
                mapped,
                signals.model_false,
                -signals.model_true,
                -signals.model_unknown,
                *rare,
            )
        return (mapped, *rare)

    def _choose_start_clause(self, starts: list[Start]) -> Start | None:
        if not starts:
            return None

        self._sync_start_frequency_counts()
        default_choice = starts[0]
        scored = [
            (action, self._start_clause_signals(action))
            for action in starts
        ]

        self._start_decisions += 1
        self._start_candidates += len(starts)
        self._start_mapped += sum(1 for _, signals in scored if signals.mapped > 0)
        self._start_unmapped += sum(1 for _, signals in scored if signals.mapped == 0)
        self._start_total_min_frequency += sum(signals.min_frequency for _, signals in scored)
        self._start_model_false += sum(signals.model_false for _, signals in scored)
        self._start_model_true += sum(signals.model_true for _, signals in scored)
        self._start_model_unknown += sum(signals.model_unknown for _, signals in scored)

        best_score = max(self._start_clause_score(signals) for _, signals in scored)
        best = [
            action
            for action, signals in scored
            if self._start_clause_score(signals) == best_score
        ]
        if len(best) == 1:
            self._start_unique_best += 1
        else:
            self._start_ties += 1

        chosen = best[0]
        if chosen is default_choice:
            self._start_agreed_default += 1
        else:
            self._start_changed_default += 1
        return chosen

    def choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        starts = [action for action in actions if isinstance(action, Start)]
        if len(starts) > 1 and min(self.score(action) for action in actions) == self.score(starts[0]):
            return self._choose_start_clause(starts)
        return super().choose_action(actions)

    def final_info(self) -> str | None:
        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        if self._start_decisions:
            avg_min_frequency = self._start_total_min_frequency / self._start_candidates
            parts.append(
                'start='
                f'mode:{self._start_mode},'
                f'decisions:{self._start_decisions},'
                f'candidates:{self._start_candidates},'
                f'mapped:{self._start_mapped},'
                f'unmapped:{self._start_unmapped},'
                f'unique:{self._start_unique_best},'
                f'ties:{self._start_ties},'
                f'avg_min_freq:{avg_min_frequency:.2f},'
                f'model_false:{self._start_model_false},'
                f'model_true:{self._start_model_true},'
                f'model_unknown:{self._start_model_unknown},'
                f'agreed_default:{self._start_agreed_default},'
                f'changed_default:{self._start_changed_default}'
            )
        return '; '.join(parts) if parts else None


class SATSolveLiteralRareStart(_SATSolveLiteralStartMixin, SATSolveLiteral):
    """SATSolveLiteral with rare-SAT-atom ordering for root start clauses.

    The strategy leaves Rawson-Reger literal selection unchanged after a
    start clause has been selected.  At the root, where SATSolveLiteral
    otherwise follows the randomised start order, it prefers start
    clauses containing rarer atoms in the current shadow CNF and falls
    back to shorter clauses.
    """

    _start_mode = 'rare'

    def __init__(self, env):
        super().__init__(env)
        self._init_start_guidance()


class SATSolveLiteralModelStart(_SATSolveLiteralStartMixin, SATSolveLiteral):
    """SATSolveLiteral with SAT-model-aware root start ordering.

    After the first SAT iteration this prefers start clauses with more
    literals currently false in the SAT model, because model-false goals
    are immediately closed by SATCoP's model lemmata.  Before a model is
    available, or inside equal model buckets, it uses the same rare-atom
    fallback as ``SATSolveLiteralRareStart``.
    """

    _start_mode = 'model'

    def __init__(self, env):
        super().__init__(env)
        self._init_start_guidance()


class SATSolveLiteralFirstStart(_SATSolveLiteralStartMixin, SATSolveLiteral):
    """SATSolveLiteral with one SAT-guided start choice per fresh root list.

    This is the conservative version of the Start-guidance experiment.
    When a reset creates a fresh root list containing all start clauses,
    the strategy scores only the first few randomized starts and uses
    SAT metadata once to choose the first start clause to try.  After
    backtracking returns to root, the remaining starts are left in the
    ordinary SATSolveLiteral/randomised order.  The score is model-aware
    when a SAT model exists, and otherwise falls back to the rare-atom
    signal from the initial shadow CNF.
    """

    _start_mode = 'model'
    _start_candidate_limit = 32

    def __init__(self, env):
        super().__init__(env)
        self._init_start_guidance()
        self._fresh_start_count = None

    def _root_start_count(self) -> int:
        state = self.env.state
        if hasattr(state, '_start_clause_indices'):
            return max(1, len(state._start_clause_indices()))
        if self._fresh_start_count is None:
            self._fresh_start_count = 0
        return self._fresh_start_count

    def _is_fresh_root_start_list(self, starts: list[Start]) -> bool:
        state = self.env.state
        goal = getattr(state, 'goal', None)
        if goal is None or getattr(goal, 'parent', None) is not None:
            return False

        expected = self._root_start_count()
        if expected == 0:
            self._fresh_start_count = len(starts)
            expected = self._fresh_start_count
        return len(starts) == expected

    def choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        if not actions:
            return None

        starts = [action for action in actions if isinstance(action, Start)]
        best_score = min(self.score(action) for action in actions)
        if (
            len(starts) > 1
            and best_score == self.score(starts[0])
            and self._is_fresh_root_start_list(starts)
        ):
            return self._choose_start_clause(starts[:self._start_candidate_limit])

        return SATSolveLiteral.choose_action(self, actions)


class _SATSolveLiteralFridayMixin(_SATSolveLiteralStartMixin):
    """Root-control experiments guided by SAT start-clause metadata."""

    _start_mode = 'model'
    _friday_enable_start_guidance = True
    _friday_commit_on_confident_start = False
    _friday_enable_root_stagnation = False
    _friday_use_first_start = False
    _friday_root_visit_limit = 8
    _start_candidate_limit = 32

    def _init_friday_controls(self) -> None:
        self._init_start_guidance()
        self._fresh_start_count = None
        self._friday_committed_depth = None
        self._friday_root_depth = None
        self._friday_root_visits = 0
        self._friday_root_clause_count = None
        self._friday_confident_starts = 0
        self._friday_commit_resets = 0
        self._friday_stagnation_resets = 0

    def _root_start_count(self) -> int:
        state = self.env.state
        if hasattr(state, '_start_clause_indices'):
            return max(1, len(state._start_clause_indices()))
        if self._fresh_start_count is None:
            self._fresh_start_count = 0
        return self._fresh_start_count

    def _is_fresh_root_start_list(self, starts: list[Start]) -> bool:
        state = self.env.state
        goal = getattr(state, 'goal', None)
        if goal is None or getattr(goal, 'parent', None) is not None:
            return False

        expected = self._root_start_count()
        if expected == 0:
            self._fresh_start_count = len(starts)
            expected = self._fresh_start_count
        return len(starts) == expected

    def _choose_start_clause_with_confidence(self, starts: list[Start]) -> tuple[Start | None, bool]:
        if not starts:
            return None, False

        self._sync_start_frequency_counts()
        default_choice = starts[0]
        scored = [
            (action, self._start_clause_signals(action))
            for action in starts
        ]

        self._start_decisions += 1
        self._start_candidates += len(starts)
        self._start_mapped += sum(1 for _, signals in scored if signals.mapped > 0)
        self._start_unmapped += sum(1 for _, signals in scored if signals.mapped == 0)
        self._start_total_min_frequency += sum(signals.min_frequency for _, signals in scored)
        self._start_model_false += sum(signals.model_false for _, signals in scored)
        self._start_model_true += sum(signals.model_true for _, signals in scored)
        self._start_model_unknown += sum(signals.model_unknown for _, signals in scored)

        best_score = max(self._start_clause_score(signals) for _, signals in scored)
        best = [
            (action, signals)
            for action, signals in scored
            if self._start_clause_score(signals) == best_score
        ]
        if len(best) == 1:
            self._start_unique_best += 1
        else:
            self._start_ties += 1

        chosen, chosen_signals = best[0]
        if chosen is default_choice:
            self._start_agreed_default += 1
        else:
            self._start_changed_default += 1

        if len(best) != 1:
            return chosen, False

        other_signals = [
            signals
            for action, signals in scored
            if action is not chosen
        ]
        if not other_signals or chosen_signals.mapped == 0:
            return chosen, False

        strong_model = (
            chosen_signals.model_false > 0
            and chosen_signals.model_false > max(signals.model_false for signals in other_signals)
        )
        strong_rarity = chosen_signals.min_frequency < min(
            signals.min_frequency for signals in other_signals
        )
        return chosen, (strong_model or strong_rarity)

    def _maybe_friday_root_reset(self, starts: list[Start]) -> Reset | None:
        state = self.env.state
        expected = self._root_start_count()
        clause_count = len(getattr(state.shadow, 'ground_clauses', ()))

        if self._friday_root_depth != state.max_depth:
            self._friday_committed_depth = None
            self._friday_root_depth = state.max_depth
            self._friday_root_visits = 0
            self._friday_root_clause_count = clause_count

        if (
            self._friday_commit_on_confident_start
            and self._friday_committed_depth == state.max_depth
            and 0 < len(starts) < expected
        ):
            self._friday_committed_depth = None
            self._friday_commit_resets += 1
            self._friday_root_visits = 0
            self._friday_root_clause_count = clause_count
            return Reset(id = 'rs', depth = state.max_depth + 1, reason = 'friday committed start failed')

        if not self._friday_enable_root_stagnation:
            return None

        if len(starts) >= expected:
            self._friday_root_visits = 0
            self._friday_root_clause_count = clause_count
            return None

        if getattr(state.shadow, 'new_ground_clauses', set()) or clause_count > (self._friday_root_clause_count or 0):
            self._friday_root_visits = 0
            self._friday_root_clause_count = clause_count
            return None

        self._friday_root_visits += 1
        if self._friday_root_visits < self._friday_root_visit_limit:
            return None

        self._friday_stagnation_resets += 1
        self._friday_root_visits = 0
        self._friday_root_clause_count = clause_count
        return Reset(id = 'rs', depth = state.max_depth + 1, reason = 'friday root stagnation')

    def choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        if not actions:
            return None

        state = self.env.state
        starts = [action for action in actions if isinstance(action, Start)]
        goal = getattr(state, 'goal', None)
        at_root = goal is not None and getattr(goal, 'parent', None) is None

        if at_root and starts:
            if reset := self._maybe_friday_root_reset(starts):
                return reset

            best_score = min(self.score(action) for action in actions)
            if (
                self._friday_enable_start_guidance
                and len(starts) > 1
                and best_score == self.score(starts[0])
            ):
                if self._friday_use_first_start and not self._is_fresh_root_start_list(starts):
                    return SATSolveLiteral.choose_action(self, actions)

                candidates = starts[:self._start_candidate_limit] if self._friday_use_first_start else starts
                chosen, confident = self._choose_start_clause_with_confidence(candidates)
                if chosen is not None:
                    if confident and self._friday_commit_on_confident_start:
                        self._friday_confident_starts += 1
                        self._friday_committed_depth = state.max_depth
                    return chosen

        return SATSolveLiteral.choose_action(self, actions)

    def final_info(self) -> str | None:
        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        if self._friday_confident_starts or self._friday_commit_resets or self._friday_stagnation_resets:
            parts.append(
                'friday='
                f'confident_starts:{self._friday_confident_starts},'
                f'commit_resets:{self._friday_commit_resets},'
                f'stagnation_resets:{self._friday_stagnation_resets}'
            )
        return '; '.join(parts) if parts else None


class SATSolveLiteralFridayStartCut(_SATSolveLiteralFridayMixin, SATSolveLiteral):
    """SAT-guided root-start choice with selective root commitment."""

    _friday_commit_on_confident_start = True

    def __init__(self, env):
        super().__init__(env)
        self._init_friday_controls()


class SATSolveLiteralFridayRootReset(_SATSolveLiteralFridayMixin, SATSolveLiteral):
    """SATSolveLiteral with early reset on repeated stale root revisits."""

    _friday_enable_start_guidance = False
    _friday_enable_root_stagnation = True

    def __init__(self, env):
        super().__init__(env)
        self._init_friday_controls()


class SATSolveLiteralFridayHybrid(_SATSolveLiteralFridayMixin, SATSolveLiteral):
    """Conservative Friday hybrid: first-start guidance, commitment, stagnation reset."""

    _friday_commit_on_confident_start = True
    _friday_enable_root_stagnation = True
    _friday_use_first_start = True

    def __init__(self, env):
        super().__init__(env)
        self._init_friday_controls()
