"""Faithful, explicit-state leanCoP 2.x strategy configurations."""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass
from pathlib import Path

from connections.calculi.classicalcut import Extension, Reduction, Reset
from connections.calculi.leancop2 import Finish, Lemma
from connections.utils.classical_problem_parsing import (
    problem_metadata,
    translate_tptp_problem,
)
from connections.utils.primitives import Matrix

from .base import Strategy


@dataclass(frozen=True)
class _LeanCoPPhase:
    # TIME_PC value from leanCoP 2.1's leancop.sh.
    share: int
    settings: tuple[str, ...]
    complete: bool = False

    @property
    def cut(self) -> bool:
        return 'cut' in self.settings

    @property
    def scut(self) -> bool:
        return 'scut' in self.settings

    @property
    def conjecture_start(self) -> bool:
        return 'conj' in self.settings

    @property
    def comp_limit(self) -> int | None:
        for setting in self.settings:
            match = re.fullmatch(r'comp\((\d+)\)', setting)
            if match:
                return int(match.group(1))
        return None


class LeanCoP_2(Strategy):
    """A close, but not binary-identical, reimplementation of leanCoP 2.1.

    Experimental provenance
    -----------------------
    The reference implementation for the search procedure and schedule is the
    official leanCoP 2.1 distribution in ``~/packages/leancop21``, especially
    ``leancop21_swi.pl`` and ``leancop.sh``.  In a paper this strategy is best
    described as ``leanCoP_2.1^Con``: a Connections-based reimplementation of
    the Prolog search core and fixed schedule.  It should not be described as
    the original leanCoP executable.

    What is intended to match leanCoP 2.1
    --------------------------------------
    * regularity over the current remaining subgoal and active path;
    * scoped lemmata, tried before reduction and extension;
    * reduction-before-extension search order;
    * inclusive restricted backtracking for ``cut`` and the corresponding
      restricted first start-clause choice for ``scut``;
    * positive/conjecture-restricted starts for ``conj``;
    * leanCoP's virtual ``-#`` start/path literal and reduction against ``#``;
    * iterative path-depth deepening, including the ground-clause exemption;
    * ``comp(J)``, which abandons restricted search after depth ``J`` and
      restarts unrestricted search at depth one; and
    * the 30 schedule entries below, including their ``def``, ``nodef``,
      ``conj``, ``cut``, ``scut``, ``comp`` and ``reo`` constants.

    The TPTP frontend uses the official translator bundled with this project.
    It is the later 2.2-style frontend: it accepts direct TPTP input, accepts
    CNF statements as well as FOF statements, and adds the corresponding
    equality axioms.  The differences from 2.1's clausifier are naming-level
    changes for Skolem functions and definition predicates, plus CNF input
    support; the logical transformations and schedule settings are retained.

    Observable differences from the Prolog program
    -----------------------------------------------
    * Search is represented by explicit Python states and actions rather than
      Prolog clauses, SLD choice points and Prolog's built-in cut.
    * Legal actions are generated eagerly and use this project's indexing and
      union-find unifier.  The Prolog program generates alternatives lazily.
      Consequently variable names, memory use, timing, proof traces, and rare
      path-limit edge cases need not be identical even when the same proof is
      found.
    * An eagerly observed path-limit hit can cause one harmless extra deepening
      round even if a later alternative would make that round unnecessary.
    * The original shell divides every ``TIME_PC`` by 111 even though all phase
      values sum to 194, allowing the final complete ``def`` run to exceed the
      advertised time limit.  The same denominator is used here, but the
      caller's outer per-problem time or step limit remains a hard cap and can
      stop that final extra allowance.
    * Phase translation is prepared outside the runner's timed ``step`` call,
      although that elapsed preparation time is deducted from the phase's own
      wall-clock allowance.
    * Raw ``.p`` TPTP files receive per-phase clausification and ``reo``.  An
      already translated Connections matrix cannot be reconstructed as FOF,
      so ``def``, ``nodef`` and ``conj`` cannot re-clausify it; only the core
      search options and a local exact perfect shuffle for ``reo`` are used.
      Raw TPTP input should therefore be used for comparative experiments.
    * Proof objects and reporting use this project's action format, not
      leanCoP's Prolog proof-term/output format.

    Useful methodological references in ``papers/`` are Otten, "Restricting
    Backtracking in Connection Calculi" (2010), for the cut variants, and
    Otten, "20 years of leanCoP -- an overview of the provers" (2023), for
    version history and implementation context.
    """

    # Exact settings, order, and TIME_PC weights from leanCoP 2.1's
    # leancop.sh.  ``complete`` corresponds to COMP=y in that script.
    phases = (
        _LeanCoPPhase(10, ('cut', 'comp(7)'), complete=True),
        _LeanCoPPhase(15, ('conj', 'def', 'cut')),
        _LeanCoPPhase(15, ('nodef', 'scut', 'cut')),
        _LeanCoPPhase(10, ('scut',)),
        _LeanCoPPhase(5, ('def', 'cut')),
        _LeanCoPPhase(4, ('conj', 'nodef', 'cut')),
        _LeanCoPPhase(2, ('def', 'scut', 'cut')),
        _LeanCoPPhase(2, ('scut', 'cut')),
        _LeanCoPPhase(1, ('conj', 'def')),
        _LeanCoPPhase(4, ('reo(40)', 'conj', 'nodef', 'scut', 'cut')),
        _LeanCoPPhase(4, ('reo(42)', 'def', 'scut', 'cut')),
        _LeanCoPPhase(2, ('reo(12)', 'def', 'scut', 'cut')),
        _LeanCoPPhase(2, ('reo(72)', 'def', 'scut', 'cut')),
        _LeanCoPPhase(2, ('reo(39)', 'nodef', 'cut')),
        _LeanCoPPhase(2, ('reo(38)', 'conj', 'def', 'cut')),
        _LeanCoPPhase(2, ('reo(15)', 'conj', 'def', 'cut')),
        _LeanCoPPhase(1, ('reo(73)', 'conj', 'def', 'cut')),
        _LeanCoPPhase(1, ('reo(57)', 'conj', 'def', 'cut')),
        _LeanCoPPhase(1, ('reo(13)', 'conj', 'nodef', 'scut', 'cut')),
        _LeanCoPPhase(1, ('reo(59)', 'conj', 'nodef', 'scut', 'cut')),
        _LeanCoPPhase(1, ('reo(75)', 'conj', 'nodef', 'scut', 'cut')),
        _LeanCoPPhase(1, ('reo(36)', 'conj', 'nodef', 'scut')),
        _LeanCoPPhase(1, ('reo(16)', 'conj', 'nodef', 'scut')),
        _LeanCoPPhase(1, ('reo(71)', 'def', 'scut')),
        _LeanCoPPhase(1, ('reo(58)', 'def', 'scut', 'cut')),
        _LeanCoPPhase(1, ('reo(76)', 'def', 'scut', 'cut')),
        _LeanCoPPhase(1, ('reo(74)', 'nodef', 'cut')),
        _LeanCoPPhase(1, ('reo(14)', 'nodef', 'cut')),
        _LeanCoPPhase(1, ('reo(37)', 'nodef', 'scut')),
        _LeanCoPPhase(99, ('def',), complete=True),
    )
    _schedule_weight = sum(phase.share for phase in phases)
    _schedule_denominator = 111

    def __init__(self, env):
        super().__init__(env)
        self._phase_index = 0
        self._phase_started: float | None = None
        self._phase_inferences = 0
        self._phase_complete_search = False
        self._phase_matrices = {}
        self._original_matrix = copy.deepcopy(env.matrix)
        self._raw_tptp = not self._is_connections_matrix(env.path)
        self._install_phase(0)

    @staticmethod
    def _is_connections_matrix(path: str) -> bool:
        text = Path(path).read_text(encoding='utf-8')
        normalized = re.sub(r'^\s*%.*$', '', text, flags=re.MULTILINE)
        normalized = re.sub(r'\s+', '', normalized)
        return re.match(r'^cnf[i]?\([^,]+,[^,]+,\[', normalized) is not None

    def _matrix_for_phase(self, phase: _LeanCoPPhase):
        if phase.settings not in self._phase_matrices:
            if self._raw_tptp:
                matrix = translate_tptp_problem(
                    self.env.path,
                    problem_metadata(self.env.path),
                    phase.settings,
                )
            else:
                # Already-translated connection matrices cannot be returned to
                # FOF and re-clausified.  They still exercise every core search
                # feature; raw TPTP inputs get the exact phase translations.
                matrix = copy.deepcopy(self._original_matrix)
                reorder_count = self._reorder_count(phase.settings)
                if reorder_count:
                    matrix = self._reorder_matrix(matrix, reorder_count)
            self._phase_matrices[phase.settings] = matrix
        return self._phase_matrices[phase.settings]

    @staticmethod
    def _reorder_count(settings: tuple[str, ...]) -> int:
        for setting in settings:
            match = re.fullmatch(r'reo\((\d+)\)', setting)
            if match:
                return int(match.group(1))
        return 0

    @staticmethod
    def _reorder_matrix(matrix: Matrix, count: int) -> Matrix:
        """Apply def_mm.pl's three-way perfect shuffle ``count`` times."""
        clauses = list(matrix.clauses)
        for _ in range(count):
            third = len(clauses) // 3
            first = clauses[:third]
            middle = clauses[third:len(clauses) - third]
            last = clauses[len(clauses) - third:]
            shuffled = []
            for last_clause, first_clause, middle_clause in zip(last, first, middle):
                shuffled.extend((last_clause, first_clause, middle_clause))
            shuffled.extend(middle[len(first):])
            clauses = shuffled
        return Matrix(
            clauses,
            metadata=matrix.metadata,
            start_clauses=matrix.start_clauses,
        )

    def _install_phase(self, index: int) -> None:
        self._phase_index = index
        phase = self.phases[index]
        self._phase_complete_search = phase.complete and phase.comp_limit is None
        self._phase_started = time.perf_counter()
        settings = self.env.settings
        settings.leancop_cut = phase.cut
        settings.leancop_scut = phase.scut
        settings.leancop_comp_limit = phase.comp_limit
        settings.leancop_conjecture_start = phase.conjecture_start
        settings.connection_constraints = False

        self.env.matrix = self._matrix_for_phase(phase)
        self.env.matrix.reset()
        self.env._init_state()
        self._phase_inferences = 0

    def _phase_budget_exhausted(self) -> bool:
        phase = self.phases[self._phase_index]
        time_limit = self.env.settings.time_limit_seconds
        if time_limit is not None:
            assert self._phase_started is not None
            elapsed = time.perf_counter() - self._phase_started
            return elapsed >= time_limit * phase.share / self._schedule_denominator

        step_limit = self.env.settings.step_limit
        if step_limit is None:
            return False
        phase_budget = max(1, step_limit * phase.share // self._schedule_denominator)
        return self._phase_inferences >= phase_budget

    def _advance_phase(self) -> bool:
        next_index = self._phase_index + 1
        if next_index >= len(self.phases):
            return False
        self._install_phase(next_index)
        return True

    def _complete_comp_phase(self, reset: Reset) -> Reset:
        state = self.env.state
        phase = self.phases[self._phase_index]
        if phase.comp_limit is None or reset.depth <= phase.comp_limit:
            return reset

        # comp(J): restart at depth one with cut and scut switched off.
        state.restricted_backtracking = False
        state.restricted_start = False
        state.comp_limit = None
        state.settings.leancop_cut = False
        state.settings.leancop_scut = False
        state.settings.leancop_comp_limit = None
        self._phase_complete_search = True
        return Reset(id='rs', depth=1, reason='comp')

    def step(self):
        while True:
            if self._phase_budget_exhausted() and self._advance_phase():
                continue

            actions = self.env.legal_actions()
            if not actions:
                if self._phase_complete_search:
                    return Finish(id='done')
                if self._advance_phase():
                    continue
                return Finish(id='done')

            action = actions[0]
            if isinstance(action, (Lemma, Reduction, Extension)):
                if self.env.state.restricted_backtracking:
                    action.cut = True
            if isinstance(action, Reset):
                action = self._complete_comp_phase(action)

            if isinstance(action, (Lemma, Reduction, Extension)) or action.__class__.__name__ == 'Start':
                self._phase_inferences += 1
            return action

    def final_info(self) -> str:
        phase = self.phases[self._phase_index]
        options = ','.join(phase.settings) or 'complete'
        return f'LeanCoP_2_phase={self._phase_index + 1}:{options}'
