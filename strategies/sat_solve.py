"""Strategy for the SAT-aware classical connection calculus."""

from __future__ import annotations

from connections.calculi.classicalsat import (
    Backtrack,
    ConnectionAction,
    Extension,
    Reduction,
    Reset,
    Start,
    pydical,
)
from connections.utils.primitives import Literal
from .base import Strategy

class _SATScoreTracker:
    """Track the first solver atom score that becomes non-zero."""

    def _init_sat_score_tracker(self) -> None:
        self._sat_nonzero_atom_score = None

    def _sync_shadow_solver(self) -> None:
        state = self.env.state
        state._flush_new_ground_clauses()
        state._solve_ground_clauses()

    def _record_nonzero_atom_scores(self) -> None:
        if self._sat_nonzero_atom_score is not None:
            return

        state = self.env.state
        solver = state.shadow.solver
        if solver is None:
            return

        for atom_id in state.shadow.atom_map.values():
            try:
                atom_score = solver.score(atom_id)
            except Exception:
                continue
            if atom_score not in (0, None):
                self._sat_nonzero_atom_score = atom_score
                return

    def final_info(self) -> str | None:
        if self._sat_nonzero_atom_score is None:
            self._sync_shadow_solver()
            self._record_nonzero_atom_scores()
        if self._sat_nonzero_atom_score is not None:
            return f'score={self._sat_nonzero_atom_score}'
        return None

class SATSolve(_SATScoreTracker, Strategy):
    """Paper-faithful SATCoP control for ``ClassicalSAT``.

    This mirrors Rawson and Reger (2021), Algorithm 2:
    randomised forward selection from the calculus, restricted
    backtracking on successful reduction/extension steps, depth control
    based on whether new grounded clauses were found, and model-based
    lemmata handled by the state machine.
    """

    def __init__(self, env):
        super().__init__(env)
        self._init_sat_score_tracker()
        if not hasattr(self.env.state, 'sat_literal_value'):
            raise TypeError('SATSolve requires the ClassicalSAT calculus')
        self.env.state.enable_paper_mode(True)

    def step(self):
        actions = self.env.legal_actions()

        self._sync_shadow_solver()
        self._record_nonzero_atom_scores()

        for action in actions:
            if isinstance(action, (Reduction, Extension)):
                action.cut = True

            if isinstance(action, (Start, Reduction, Extension)):
                return action

        if (
            self.env.state.goal.parent is None
            and any(isinstance(action, Reset) for action in actions)
            and self.env.state.shadow.new_ground_clauses
        ):
            return Reset(id = 'rs', depth = self.env.state.max_depth, reason = 'ground shadow still growing')

        if action := next((a for a in actions if isinstance(a, Backtrack)), None):
            return action

        if action := next((a for a in actions if isinstance(a, Reset)), None):
            return action

        raise LookupError('Expected at least one action to be Start, Reduction, Extension, Backtrack, or Reset.')

class SATSolveGuided(SATSolve):
    """SATCoP control with SAT-shadow scoring for extension choice.

    This keeps ``SATSolve``'s control policy, including paper mode,
    restricted backtracking, root resets, and backtracking order.  The
    only difference is at extension choice points: when ``SATSolve``
    would pick an extension and more than one extension is available,
    choose the extension with the best score from the current SAT
    shadow.

    Sources and scope:
        This is not the Rawson/Reger literal-selection rule; that is
        already represented by ``SATSolveLiteral``.  This strategy is an
        experiment in using SAT-solver-internal properties to guide the
        first-order connection prover.

        The SAT-side ingredients are standard CDCL signals rather than
        new prover machinery:
        - EVSIDS-style variable activity, introduced by the Chaff SAT
          solver (Moskewicz et al., 2001), exposed here by CaDiCaL as
          ``score(var)``.
        - Forced assignments / failed assumptions, a standard
          assumption-based incremental SAT interface; PySAT documents
          the same assumption-call style for its solver wrappers.
        - Variable involvement in propagation and conflict analysis as
          a useful importance signal for SAT search, e.g. Liu et al.
          (2024), "NeuroSelect: Learning to Select Clauses in SAT
          Solvers", §3.1-3.2.

        I did not find an existing paper in ``papers/`` that applies
        these SAT-internal signals to guide extension choice in a
        connection prover.  The originality here is intentionally small:
        map candidate extension subgoals to existing SAT atoms and ask
        CaDiCaL which atoms currently look important.  The score is a
        lexicographic tuple, not a weighted sum: first prefer assumption
        probes that refute the current shadow CNF, then forced literals,
        then larger EVSIDS activity, then fewer remaining subgoals.
    """

    def step(self):
        actions = self.env.legal_actions()

        self._sync_shadow_solver()
        self._record_nonzero_atom_scores()

        for action in actions:
            if isinstance(action, (Reduction, Extension)):
                action.cut = True

        for action in actions:
            if isinstance(action, Start):
                return action
            if isinstance(action, Reduction):
                return action
            if isinstance(action, Extension):
                extensions = [a for a in actions if isinstance(a, Extension)]
                if len(extensions) <= 1:
                    return action
                return max(extensions, key = self._extension_score)

        if (
            self.env.state.goal.parent is None
            and any(isinstance(action, Reset) for action in actions)
            and self.env.state.shadow.new_ground_clauses
        ):
            return Reset(id = 'rs', depth = self.env.state.max_depth, reason = 'ground shadow still growing')

        if action := next((a for a in actions if isinstance(a, Backtrack)), None):
            return action

        if action := next((a for a in actions if isinstance(a, Reset)), None):
            return action

        raise LookupError('Expected at least one action to be Start, Reduction, Extension, Backtrack, or Reset.')

    def _extension_score(self, action: Extension) -> tuple[int, int, float, int]:
        state = self.env.state
        subgoals = self._remaining_subgoals(action)

        state.substitution.update(action.sub_updates)
        try:
            sat_literals = self._sat_literals(subgoals)
        finally:
            state.substitution.backtrack()

        solver = state.shadow.solver
        probe_conflicts = 0
        forced_true = 0
        evsids = 0.0

        for sat_lit in sat_literals:
            if solver.fixed(sat_lit) > 0:
                forced_true += 1

            evsids += solver.score(abs(sat_lit))

            solver.assume(-sat_lit)
            try:
                if solver.solve() == pydical.UNSATISFIABLE:
                    probe_conflicts += 1
            finally:
                solver.reset_assumptions()

        return (probe_conflicts, forced_true, evsids, -len(subgoals))

    def _remaining_subgoals(self, action: Extension) -> list[Literal]:
        return [
            literal
            for idx, literal in enumerate(action.clause_copy)
            if idx != action.lit_idx
        ]

    def _sat_literals(self, literals: list[Literal]) -> list[int]:
        state = self.env.state
        result = []

        for literal in literals:
            grounded_literal = state.substitution(literal)
            atom_id = state.shadow.atom_map.get(state._atom_key(grounded_literal))
            if atom_id is None:
                continue
            result.append(-atom_id if grounded_literal.neg else atom_id)

        return result

class SATSolveLiteral(_SATScoreTracker, Strategy):
    """Strict SATCoP with §9.1 literal-selection refinement.

    Implements Rawson and Reger (2021), Algorithm 2 with the
    "literal selection" extension from Sect. 9.1: when several forward
    actions are available, prefer those whose head literal is assigned
    ``True`` in the current SAT model (i.e. the first goal literal the
    model would close), falling back to the randomised order from the
    calculus when the model has nothing to say.

    Like ``SATSolve``, this strategy:
    - Enables ``paper_mode`` on the calculus, which gives:
      * initial grounding of start clauses (§4),
      * model-based lemmata that skip goals already assigned false
        in the SAT model (§6),
      * refusal to accept a tableau-level proof — acceptance comes
        only when the SAT solver reports UNSAT (§4.1).
    - Applies restricted backtracking (``cut=True``) on successful
      reduction/extension steps, matching Algorithm 2.
    - Controls iterative deepening via the calculus: stays at the
      same depth while new ground clauses are being discovered, only
      incrementing when they stop (§5).

    Differences from ``SATSolve``:
    - Forward actions are scored by SAT truth value of their head
      literal, so reductions/extensions the SAT model already believes
      are closest to closing fire first (§9.1 "literal selection").
    - When no SAT information is available for the candidate literals
      (e.g. first iteration, or all literals still non-ground), the
      behaviour degenerates to ``SATSolve``'s "first shuffled action".

    Known differences from the paper that are shared with SATSolve:
    - The SAT backend is CaDiCaL rather than the custom CDCL
      from §4.3.
    - Start clauses are taken from the positive+negative pool rather
      than strictly "clauses derived from the conjecture".
    """

    def __init__(self, env):
        super().__init__(env)
        self._init_sat_score_tracker()
        if not hasattr(self.env.state, 'sat_literal_value'):
            raise TypeError('SATSolveLiteral requires the ClassicalSAT calculus')
        self.env.state.enable_paper_mode(True)

    def score(self, action: ConnectionAction):
        match action:
            case Start():
                return 2

            case Reduction(path_lit = path_lit):
                value = self.env.state.sat_literal_value(path_lit)
                if value is True:
                    return 0
                elif value is None:
                    return 1
                else:
                    return 3

            case Extension(clause_copy = clause_copy, lit_idx = lit_idx):
                value = None
                if 0 <= lit_idx < len(clause_copy):
                    value = self.env.state.sat_literal_value(clause_copy[lit_idx])
                if value is True:
                    return 0
                elif value is None:
                    return 1
                else:
                    return 3

        return 4

    def step(self):
        state = self.env.state
        actions = self.env.legal_actions()
        self._record_nonzero_atom_scores()
        action = min(actions, key = self.score, default = None)

        if isinstance(action, (Reduction, Extension)):
            action.cut = True

        if isinstance(action, (Start, Reduction, Extension)):
            return action

        if (
            state.goal.parent is None
            and any(isinstance(action, Reset) for action in actions)
            and state.shadow.new_ground_clauses
        ):
            return Reset(id = 'rs', depth = state.max_depth)

        for action in actions:
            if isinstance(action, Backtrack):
                return action

        for action in actions:
            if isinstance(action, Reset):
                return action

        raise LookupError('expected Backtrack or Reset action')
