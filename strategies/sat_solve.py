"""Core SAT-aware connection-tableau strategies."""

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
        self._init_literal_choice_diagnostics()
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

    def _init_literal_choice_diagnostics(self) -> None:
        self._literal_choice_points = 0
        self._literal_forward_actions = 0
        self._literal_unique_best = 0
        self._literal_ties = 0
        self._literal_tied_actions = 0
        self._literal_best_true = 0
        self._literal_best_unknown = 0
        self._literal_best_start = 0
        self._literal_best_false = 0
        self._literal_best_other = 0
        self._literal_tied_mappable = 0
        self._literal_tied_mapped = 0
        self._literal_chosen_start = 0
        self._literal_chosen_reduction = 0
        self._literal_chosen_extension = 0

    def _selected_literal(self, action: ConnectionAction) -> Literal | None:
        match action:
            case Reduction(path_lit = path_lit):
                return path_lit
            case Extension(clause_copy = clause_copy, lit_idx = lit_idx):
                if 0 <= lit_idx < len(clause_copy):
                    return clause_copy[lit_idx]
        return None

    def _selected_sat_literal(self, action: ConnectionAction) -> int | None:
        literal = self._selected_literal(action)
        if literal is None:
            return None

        state = self.env.state
        if not hasattr(state, '_atom_key'):
            return None

        substitution = getattr(state, 'substitution', None)
        grounded_literal = substitution(literal) if callable(substitution) else literal
        atom_id = state.shadow.atom_map.get(state._atom_key(grounded_literal))
        if atom_id is None:
            return None
        return -atom_id if grounded_literal.neg else atom_id

    def _record_literal_choice_diagnostics(
        self,
        actions: list[ConnectionAction],
        chosen: ConnectionAction | None,
    ) -> None:
        forward = [
            action
            for action in actions
            if isinstance(action, (Start, Reduction, Extension))
        ]
        if not forward:
            return

        scored = [(action, self.score(action)) for action in forward]
        best_score = min(score for _, score in scored)
        tied = [action for action, score in scored if score == best_score]

        self._literal_choice_points += 1
        self._literal_forward_actions += len(forward)
        if len(tied) == 1:
            self._literal_unique_best += 1
        else:
            self._literal_ties += 1
            self._literal_tied_actions += len(tied)

        match best_score:
            case 0:
                self._literal_best_true += 1
            case 1:
                self._literal_best_unknown += 1
            case 2:
                self._literal_best_start += 1
            case 3:
                self._literal_best_false += 1
            case _:
                self._literal_best_other += 1

        tied_with_selected_literals = [
            action
            for action in tied
            if self._selected_literal(action) is not None
        ]
        self._literal_tied_mappable += len(tied_with_selected_literals)
        self._literal_tied_mapped += sum(
            1
            for action in tied_with_selected_literals
            if self._selected_sat_literal(action) is not None
        )

        if isinstance(chosen, Start):
            self._literal_chosen_start += 1
        elif isinstance(chosen, Reduction):
            self._literal_chosen_reduction += 1
        elif isinstance(chosen, Extension):
            self._literal_chosen_extension += 1

    def choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        return min(actions, key = self.score, default = None)

    def step(self):
        state = self.env.state
        actions = self.env.legal_actions()
        self._record_nonzero_atom_scores()
        action = self.choose_action(actions)
        self._record_literal_choice_diagnostics(actions, action)

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

    def final_info(self) -> str | None:
        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        if self._literal_choice_points:
            avg_tie = (
                self._literal_tied_actions / self._literal_ties
                if self._literal_ties
                else 0.0
            )
            parts.append(
                'literal='
                f'choices:{self._literal_choice_points},'
                f'forward:{self._literal_forward_actions},'
                f'unique:{self._literal_unique_best},'
                f'ties:{self._literal_ties},'
                f'avg_tie:{avg_tie:.2f},'
                f'best_true:{self._literal_best_true},'
                f'best_unknown:{self._literal_best_unknown},'
                f'best_start:{self._literal_best_start},'
                f'best_false:{self._literal_best_false},'
                f'best_other:{self._literal_best_other},'
                f'tied_mappable:{self._literal_tied_mappable},'
                f'tied_mapped:{self._literal_tied_mapped},'
                f'chosen_start:{self._literal_chosen_start},'
                f'chosen_reduction:{self._literal_chosen_reduction},'
                f'chosen_extension:{self._literal_chosen_extension}'
            )
        return '; '.join(parts) if parts else None
