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
    share: int
    settings: tuple[str, ...]

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
    """A close, but not binary-identical, reimplementation of leanCoP 2.0.

    Experimental provenance
    -----------------------
    The reference implementation for the search procedure and the schedule is
    the official leanCoP 2.0 distribution, in particular ``leancop_main.sh``
    and the 2007 ``leancop.sh``.  In a paper this strategy is best described as
    ``leanCoP_2^Con``: a Connections-based reimplementation of the leanCoP 2.0
    search core and its fixed schedule, using the later leanCoP clausifier.  It
    should not be described as the original leanCoP executable.

    What is intended to match leanCoP 2.0
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
    * the ten schedule entries below, including their ``def``, ``nodef``,
      ``conj``, ``cut``, ``scut``, ``comp`` and ``reo`` constants.

    The TPTP frontend uses the official translator bundled with this project.
    It is closer to the later leanCoP 2.1/2.2 frontend than to the literal 2.0
    frontend: it accepts direct TPTP input and adds the corresponding equality
    axioms.  Thus the overall strategy is a deliberate 2.0-search/2.2-frontend
    hybrid, not a complete reproduction of either later leanCoP release.  In
    particular it does not reproduce their optimized schedules or their proof
    presentation.

    Observable differences from the Prolog program
    -----------------------------------------------
    * Search is represented by explicit Python states and actions rather than
      Prolog clauses, SLD choice points and Prolog's built-in cut.
    * Legal actions are generated eagerly and use this project's indexing and
      union-find unifier.  The Prolog program generates alternatives lazily.
      Consequently variable names, memory use, timing, proof traces, and rare
      path-limit/reordering edge cases need not be identical even when the same
      proof is found.
    * An eagerly observed path-limit hit can cause one harmless extra deepening
      round even if a later alternative would make that round unnecessary.
    * The original shell gives its final unrestricted run a full time limit
      after allocating 92 percent to the preceding runs, so its worst-case
      total is 192 percent of the advertised limit.  Here the final phase gets
      the remaining 8 percent: the schedule is normalized to one caller-owned
      budget.  With no wall-clock limit, the same shares are applied to the
      proof-step limit.
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

    phases = (
        _LeanCoPPhase(2, ('cut', 'comp(7)')),
        _LeanCoPPhase(60, ('conj', 'cut')),
        _LeanCoPPhase(16, ('def', 'scut', 'cut')),
        _LeanCoPPhase(4, ('nodef', 'scut', 'cut')),
        _LeanCoPPhase(2, ('conj', 'cut', 'reo(23)')),
        _LeanCoPPhase(2, ('def', 'scut', 'cut', 'reo(29)')),
        _LeanCoPPhase(2, ('conj', 'cut', 'reo(37)')),
        _LeanCoPPhase(2, ('nodef', 'scut', 'cut', 'reo(41)')),
        _LeanCoPPhase(2, ('conj', 'cut', 'reo(47)')),
        _LeanCoPPhase(8, ()),
    )

    def __init__(self, env):
        super().__init__(env)
        self._phase_index = 0
        self._phase_started: float | None = None
        self._phase_inferences = 0
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
            return elapsed >= time_limit * phase.share / 100.0

        step_limit = self.env.settings.step_limit
        if step_limit is None:
            return False
        phase_budget = max(1, step_limit * phase.share // 100)
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
        return Reset(id='rs', depth=1, reason='comp')

    def step(self):
        while True:
            if self._phase_budget_exhausted() and self._advance_phase():
                continue

            actions = self.env.legal_actions()
            if not actions:
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
