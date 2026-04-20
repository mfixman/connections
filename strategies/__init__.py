import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeVar

from connections.env import ConnectionEnv
from connections.calculi.classicalcut import ConnectionAction, Start, Reduction, Extension, Backtrack, Reset
from .base import Strategy

TAction = TypeVar('TAction', bound = ConnectionAction)

def backtrack_or_reset(actions: list[ConnectionAction]) -> ConnectionAction:
    backtracks = (a for a in actions if isinstance(a, Backtrack))
    resets = (a for a in actions if isinstance(a, Reset))
    return next(backtracks, None) or next(resets)

def _action_score(action: ConnectionAction) -> float:
    """Heuristic score for a forward action.  Higher is better.

    Source:
        Inspired by the fewest-solutions and shortest-clause principles
        from Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2.

    Design choices (not from the paper):
        The paper states qualitative ordering principles — prefer
        subgoals with fewer solutions, and use clause length as a proxy
        — but never proposes a numeric scoring function.  The specific
        values (Reduction = 2.0, Extension/Start = 1 / clause_length)
        are our adaptation.  Reductions score highest because they close
        a subgoal immediately (0 new subgoals); extensions score
        inversely with clause length as a rough proxy for the
        shortest-clause principle.
    """
    if isinstance(action, Reduction):
        return 2.0
    elif isinstance(action, (Extension, Start)):
        return 1.0 / max(len(action.clause_copy), 1)
    return 0.0

def _forward_actions(actions: list[ConnectionAction]) -> list[ConnectionAction]:
    return [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

class _ConfiguredCutStrategy(Strategy):
    """Base class for strategies that choose a forward action and
    optionally annotate it with inclusive cuts.

    This supports two orthogonal design choices already present in the
    literature and codebase:

    - action ordering (default, heuristic, subgoal-reordered)
    - inclusive cuts on reduction and/or extension steps

    If ``max_cut_height`` is set, cuts are disabled once iterative
    deepening exceeds that height and the prover restarts from depth 1,
    mirroring leanCoP's ``comp(J)`` behaviour from Otten (2010), §5.4.
    """

    cut_reduction: bool = False
    cut_extension: bool = False
    max_cut_height: Optional[int] = None
    should_comp: bool

    def __init__(self, env):
        super().__init__(env)
        self.should_comp = self.max_cut_height is not None

    def choose_forward(self, forward: list[ConnectionAction]) -> ConnectionAction:
        return forward[0]

    def _cuts_enabled(self) -> bool:
        return self.max_cut_height is None or self.should_comp

    def _decorate_forward(self, action: ConnectionAction) -> ConnectionAction:
        if self._cuts_enabled():
            if isinstance(action, Reduction) and self.cut_reduction:
                action.cut = True
            elif isinstance(action, Extension) and self.cut_extension:
                action.cut = True
        return action

    def _handle_control(self, actions: list[ConnectionAction]) -> ConnectionAction:
        action = backtrack_or_reset(actions)
        if (
            self.max_cut_height is not None
            and self.should_comp
            and isinstance(action, Reset)
            and action.depth > self.max_cut_height
        ):
            self.should_comp = False
            return Reset(id = 'rs', depth = 1)
        return action

    def step(self):
        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        if not forward:
            return self._handle_control(actions)

        chosen = self.choose_forward(forward)
        return self._decorate_forward(chosen)

class FirstAction(Strategy):
    def step(self):
        return self.env.legal_actions()[0]

class LeanCopCut(Strategy):
    """Mirror leanCoP's ``cut`` option for forward proof steps.

    This strategy follows the baseline proof-search order of
    ``FirstAction``, but if the selected action is an ``Extension`` or a
    ``Reduction``, it sets ``cut=True`` before returning it.

    Source:
        Otten (2010), "Restricting backtracking in connection calculi",
        §4.3, Definition 8.  Restricted backtracking cuts off
        alternative rule applications once a literal has been solved.
        The Prolog implementation inserts a ``!`` (cut) after each
        successful proof step (line 14 of Figure 11).  The ``scut``
        variant additionally restricts start-clause backtracking
        (§5.4, line c).

    Consistency with leanCoP:
        The Prolog ``cut`` option in leanCoP applies restricted
        backtracking after successful reduction and extension steps.
        It does not force a cut on start steps; that is the separate
        ``scut`` option.  This strategy therefore leaves ``Start``
        actions unchanged.

    Design choices (not from the paper):
        leanCoP's ``cut`` is unconditional: every reduction/extension
        step gets a cut.  This strategy reproduces that behaviour
        exactly for the ``cut`` option; ``scut`` is not implemented
        here.  The ``comp(J)`` logic (``should_comp`` and
        ``max_cut_height``) follows Otten (2010), §5.4 verbatim:
        run with cut until iterative deepening exceeds depth J,
        then restart from depth 1 without cut.
    """

    max_cut_height: int = 7
    should_comp: bool

    def __init__(self, env):
        super().__init__(env)
        self.should_comp = True

    def step(self):
        action = self.env.legal_actions()[0]

        if not self.should_comp:
            return action

        match action:
            case Reduction() | Extension():
                action.cut = True

            case Reset(depth = depth):
                if depth > self.max_cut_height:
                    self.should_comp = False
                    return Reset(id = 'rs', depth = 1)

        return action

class ReductionCut(_ConfiguredCutStrategy):
    """Baseline depth-first order with cut on reductions only.

    Source:
        Färber (2024), "A Curiously Effective Backtracking Strategy
        for Connection Tableaux", §4 and §6.  The paper decomposes
        restricted backtracking into cut on reduction steps (R) and cut
        on extension steps (EI/EX), and reports that adding R is often
        beneficial.

    Design choice:
        This keeps the default depth-first action order and applies an
        inclusive cut only to ``Reduction`` actions.  Unlike
        ``LeanCopCut``, it does not force cuts on extensions.
    """

    cut_reduction = True

class HeuristicSelection(Strategy):
    """Select the highest-scored forward action.

    Source:
        Inspired by Letz and Stenz (2001), "Model Elimination and
        Connection Tableau Procedures", especially the fewest-solutions
        and shortest-clause principles (§2.6.1–2.6.2).

    Design choice:
        The paper gives qualitative ordering principles (prefer
        reductions, then shorter clauses) applied to subgoal reordering
        within a clause.  We repurpose them as a numeric scoring
        function for action selection at choice points.  See
        ``_action_score`` for details.
    """

    def step(self):
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        if not forward:
            return backtrack_or_reset(actions)

        if len(forward) == 1:
            return forward[0]

        scores = [_action_score(a) for a in forward]
        best_idx = max(range(len(forward)), key = lambda i: scores[i])
        return forward[best_idx]

class HeuristicCut(_ConfiguredCutStrategy):
    """Heuristic forward selection with leanCoP-style inclusive cuts.

    Sources:
        Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2: fewest-solutions and
        shortest-clause principles.

        Otten (2010), "Restricting backtracking in connection
        calculi", §4.3 and §5.4: inclusive cut on successful
        reduction/extension steps, with ``comp(J)`` for completeness.

    Design choice:
        This fills a gap in the current strategy set: your existing
        strategies test ordering without cut (``HeuristicSelection``)
        and cut without reordered choice (``LeanCopCut``), but not the
        combination.  The combination itself appears to be original to
        this project.
    """

    cut_reduction = True
    cut_extension = True
    max_cut_height = 7

    def choose_forward(self, forward: list[ConnectionAction]) -> ConnectionAction:
        if len(forward) == 1:
            return forward[0]
        scores = [_action_score(a) for a in forward]
        best_idx = max(range(len(forward)), key = lambda i: scores[i])
        return forward[best_idx]

class SubgoalReordering(Strategy):
    """Strict categorical ordering: reductions first, then extensions
    sorted by clause length (ascending), then starts by clause length.

    Source:
        Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2.  The fewest-solutions
        principle states that subgoals producing fewer solutions should
        be preferred.  Reductions close the current subgoal immediately
        (0 new subgoals), so they are tried first.  Among extensions,
        the shortest-clause principle prefers clauses with fewer
        literals, since the number of solutions from a clause is the
        product of the solutions of its subgoals (footnote 3, p. 2031).

    Design choice:
        The paper describes subgoal alternation — interleaving choice
        points across multiple open subgoals — which our step-at-a-time
        framework cannot directly express.  We implement the core
        ranking: reductions > extensions sorted by number of new
        subgoals (len(clause_copy) - 1).  Ties preserve the default
        depth-first order, matching the paper's baseline (§2.6, "depth-
        first left-most/right-most selection").
    """

    def _action_key(self, action: ConnectionAction) -> tuple[int, int]:
        """Sort key: (category, new_subgoal_count).

        Category 0 = Reduction (closes branch, 0 new subgoals).
        Category 1 = Extension (sorted by clause length ascending).
        Category 2 = Start (sorted by clause length ascending).
        """
        if isinstance(action, Reduction):
            return (0, 0)
        elif isinstance(action, Extension):
            # clause_copy includes the head literal; new subgoals = len - 1
            return (1, len(action.clause_copy) - 1)
        elif isinstance(action, Start):
            return (2, len(action.clause_copy))
        return (3, 0)

    def step(self):
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        if not forward:
            return backtrack_or_reset(actions)

        if len(forward) == 1:
            return forward[0]

        # Stable sort preserves depth-first order within same category+length.
        forward.sort(key=self._action_key)
        return forward[0]

class OrderedCut(_ConfiguredCutStrategy):
    """Subgoal-reordered forward selection with leanCoP-style cuts.

    Sources:
        Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2.

        Otten (2010), "Restricting backtracking in connection
        calculi", §4.3 and §5.4.

    Design choice:
        This is the same order-plus-cut combination as ``HeuristicCut``
        but uses the stricter categorical ordering from
        ``SubgoalReordering``: reductions first, then shorter
        extensions, then shorter starts.
    """

    cut_reduction = True
    cut_extension = True
    max_cut_height = 7

    def _action_key(self, action: ConnectionAction) -> tuple[int, int]:
        if isinstance(action, Reduction):
            return (0, 0)
        elif isinstance(action, Extension):
            return (1, len(action.clause_copy) - 1)
        elif isinstance(action, Start):
            return (2, len(action.clause_copy))
        return (3, 0)

    def choose_forward(self, forward: list[ConnectionAction]) -> ConnectionAction:
        if len(forward) == 1:
            return forward[0]
        return sorted(forward, key = self._action_key)[0]

class HeuristicReductionCut(_ConfiguredCutStrategy):
    """Heuristic forward selection with cut on reductions only.

    Sources:
        Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2.

        Färber (2024), "A Curiously Effective Backtracking Strategy
        for Connection Tableaux", §4 and §6: reduction cut (R) as a
        separate design dimension.

    Design choice:
        This tests whether the main gain comes from choosing stronger
        forward actions while keeping extension alternatives available.
        It is cheaper and less committal than ``HeuristicCut``.
    """

    cut_reduction = True

    def choose_forward(self, forward: list[ConnectionAction]) -> ConnectionAction:
        if len(forward) == 1:
            return forward[0]
        scores = [_action_score(a) for a in forward]
        best_idx = max(range(len(forward)), key = lambda i: scores[i])
        return forward[best_idx]

class EntropyCut(Strategy):
    """Selective restricted backtracking based on action-distribution
    entropy.

    Analogous to leanCoP's ``cut`` option (Otten 2010, Definition 8),
    which cuts off non-essential backtracking after every successful
    proof step.  Rather than cutting unconditionally, this strategy
    cuts only when the heuristic score distribution has low entropy
    (i.e. one action type clearly dominates).

    Source:
        Otten (2010), "Restricting backtracking in connection calculi",
        §4.3, Definition 8: restricted backtracking cuts off
        alternative rule applications once a literal is solved.  The
        Prolog implementation inserts a ``!`` after each successful
        proof step (line 14 of Figure 11).

        Zombori et al. (2021), "The Role of Entropy in Guiding a
        Connection Prover": uses entropy of a learned neural policy to
        measure search confidence.  When entropy is low the prover is
        confident and can commit to a branch.

    Design choices (not from the papers):
        leanCoP's ``cut`` is unconditional — every forward step is
        cut.  Zombori et al. use a *learned* neural policy whose
        entropy is a natural confidence measure.  We combine the two
        ideas but replace the neural policy with hand-written heuristic
        scores (see ``_action_score``).  The specific mechanism —
        softmax over heuristic scores, then cut when Shannon entropy
        < 0.69 — is original to this project.

        The threshold 0.69 ≈ ln(2) is the minimum entropy for two
        equally-scored actions; a threshold just below this means we
        cut only when one action type clearly dominates (e.g. one
        reduction vs. one extension).

        This strategy does *not* reorder forward actions; it always
        takes the first available action (baseline depth-first order)
        and only decides whether to annotate it with ``cut=True``.
    """

    entropy_threshold: float = 0.69

    def _softmax_entropy(self, scores: list[float]) -> float:
        """Shannon entropy of the softmax distribution over scores."""
        max_s = max(scores)
        exp_scores = [math.exp(s - max_s) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]
        return -sum(p * math.log(p) for p in probs if p > 0)

    def _apply_entropy_cut(self, chosen: ConnectionAction, forward: list[ConnectionAction]) -> ConnectionAction:
        """Set ``cut=True`` on *chosen* if the action distribution has
        low entropy (high confidence)."""
        scores = [_action_score(a) for a in forward]
        entropy = self._softmax_entropy(scores)
        chosen.cut = (entropy < self.entropy_threshold)
        return chosen

    def step(self):
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        if not forward:
            return backtrack_or_reset(actions)

        if len(forward) == 1:
            return forward[0]

        return self._apply_entropy_cut(forward[0], forward)

class AdaptiveReset(Strategy):
    """Adaptive restart policy: backtrack, reset(+1), or reset(+2).

    Analogous to leanCoP's ``comp(J)`` option (Otten 2010, §5.4),
    which restarts proof search with a complete (cut-free) strategy
    once iterative deepening exceeds a fixed depth limit J.  Rather
    than a fixed threshold, this strategy uses dynamic signals to
    decide when and how far to reset.

    Source:
        Otten (2010), "Restricting backtracking in connection calculi",
        §5.4: ``comp(Limit)`` restores completeness by restarting from
        depth 1 without cut once PathLim exceeds Limit.

        Holden (2021), "Machine Learning for Automated Theorem
        Proving", Chapter 3: surveys SAT restart strategies that adapt
        restart frequency based on solver state (Luby restarts, glucose-
        style dynamic restarts).

        Li et al. (2024), "A Reinforcement Learning based Reset Policy
        for CDCL SAT Solvers": proposes Thompson Sampling to choose
        between restart policies in SAT.  Does not transfer to
        connection calculi (see project memory for negative result).

    Design choices (not from the papers):
        leanCoP's ``comp(J)`` uses a fixed depth limit; we replace it
        with two dynamic triggers:

        1. Consecutive failure count ≥ 5: if the prover hits 5
           dead-ends in a row, the current search direction is
           unlikely to lead to a proof.  The threshold of 5 is
           hand-tuned.
        2. Constraint saturation: if the number of learned constraints
           exceeds 2 × num_lits, further search at the current depth
           is unpromising.  This test is original to this project.

        The depth increment (+1 or +2) is decided by per-depth failure
        rate: if more than 80% of steps at the current depth were
        dead-ends, skip a level (+2); otherwise increment normally
        (+1).  This heuristic has no paper basis.

        Like ``EntropyCut``, this strategy does *not* reorder forward
        actions; it only affects the dead-end handling (backtrack vs
        reset).
    """

    max_consecutive_failures: int = 5

    def __init__(self, env):
        super().__init__(env)
        self.consecutive_failures: int = 0
        self.steps_at_depth: dict[int, int] = defaultdict(int)
        self.failures_at_depth: dict[int, int] = defaultdict(int)

    def _track_step(self, has_forward: bool) -> None:
        """Update per-depth statistics and failure counter."""
        state = self.env.state
        self.steps_at_depth[state.max_depth] += 1
        if has_forward:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.failures_at_depth[state.max_depth] += 1

    def _should_reset(self) -> tuple[bool, str]:
        """Decide whether to reset instead of backtracking."""
        if self.consecutive_failures >= self.max_consecutive_failures:
            return True, f'{self.consecutive_failures} consecutive failures'
        state = self.env.state
        if len(state.constraints) > state.matrix.num_lits * 2:
            return True, f'{len(state.constraints)} constraints (> {state.matrix.num_lits * 2})'
        return False, ''

    def _choose_reset_depth(self) -> int:
        """Choose the depth for the next iteration after a reset."""
        state = self.env.state
        current = state.max_depth
        steps = self.steps_at_depth[current]
        failures = self.failures_at_depth[current]
        if steps > 0 and failures / steps > 0.8:
            return current + 2
        return current + 1

    def _handle_dead_end(self, actions: list[ConnectionAction]) -> ConnectionAction:
        """Return a Reset (with chosen depth) or fall back to Backtrack."""
        do_reset, reason = self._should_reset()
        if do_reset:
            state = self.env.state
            depth = self._choose_reset_depth()
            delta = depth - state.max_depth
            self.consecutive_failures = 0
            return Reset(id = 'rs', depth = depth, reason = f'{reason}, +{delta} depth')
        return backtrack_or_reset(actions)

    def step(self):
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        self._track_step(bool(forward))

        if not forward:
            return self._handle_dead_end(actions)

        return forward[0]

class SmartCut(AdaptiveReset, EntropyCut):
    """Entropy-based cut combined with adaptive reset.

    Analogous to leanCoP's ``[cut, comp(J)]`` strategy combination
    (Otten 2010, §5.5): restricted backtracking is active during the
    search, with a restart mechanism to recover from dead-end regions.

    Combines:
        ``EntropyCut`` — selective restricted backtracking (≈ ``cut``).
        ``AdaptiveReset`` — adaptive restart policy (≈ ``comp(J)``).

    This class does *not* reorder forward actions.  It follows the
    baseline depth-first order and only adds cut annotations and
    reset decisions.
    """

    def step(self):
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        self._track_step(bool(forward))

        if not forward:
            return self._handle_dead_end(actions)

        if len(forward) == 1:
            return forward[0]

        return self._apply_entropy_cut(forward[0], forward)

class PathComplementCut(_ConfiguredCutStrategy):
    """Order extensions by how many new subgoals have complements on the
    current path (i.e. can immediately reduce), then apply leanCoP-style
    inclusive cuts.

    Source:
        Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2: the fewest-solutions
        principle recommends preferring subgoals that are easiest to
        close.  Reductions close subgoals immediately and produce no
        new work; among extensions, those that generate subgoals
        reducible against the current path are closest to closing.

        Otten (2010), "Restricting backtracking in connection calculi",
        §4.3 and §5.4: inclusive cut on successful proof steps with
        ``comp(J)`` for completeness recovery.

    Design choices (not from the papers):
        Letz and Stenz rank by total solution count, which is
        expensive to compute exactly.  We approximate it: an extension
        whose new subgoals already have path complements is likely to
        produce fewer remaining subgoals (the reducible ones can be
        closed at zero cost).  The specific scoring function —
        ``-complement_count`` as primary key, clause length as
        tiebreaker — is original to this project.

        When no path is available (root/start), falls back to clause
        length ordering identical to ``OrderedCut``.
    """

    cut_reduction = True
    cut_extension = True
    max_cut_height = 7

    def _complement_count(self, action: Extension) -> int:
        """Count new subgoals that have a complement on the path."""
        path_lits = self.env.state.goal.path()
        # A subgoal `s` can reduce against path literal `p` when
        # p.neg != s.neg and p.symbol == s.symbol.  Build the set of
        # (neg, symbol) pairs that would match.
        reducible = {(not lit.neg, lit.symbol) for lit in path_lits}
        return sum(
            1 for i, lit in enumerate(action.clause_copy)
            if i != action.lit_idx and (lit.neg, lit.symbol) in reducible
        )

    def _action_key(self, action: ConnectionAction) -> tuple:
        if isinstance(action, Reduction):
            return (0, 0, 0)
        elif isinstance(action, Extension):
            return (1, -self._complement_count(action), len(action.clause_copy))
        elif isinstance(action, Start):
            return (2, 0, len(action.clause_copy))
        return (3, 0, 0)

    def choose_forward(self, forward: list[ConnectionAction]) -> ConnectionAction:
        if len(forward) == 1:
            return forward[0]
        return min(forward, key=self._action_key)

class GoalDistanceCut(_ConfiguredCutStrategy):
    """Order extensions by estimated proof distance: prefer subgoals
    that are either path-reducible or have many matrix complements.

    Sources:
        Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2: fewest-solutions and
        shortest-clause principles.

        Otten (2010), "Restricting backtracking in connection
        calculi", §4.3 and §5.4: inclusive cut with ``comp(J)``.

    Design choices (not from the papers):
        The fewest-solutions principle says to prefer subgoals with
        fewer solutions — but computing this exactly requires
        exhaustive search.  We estimate proof distance per subgoal:
        a path-reducible subgoal costs 0 (one reduction step), while
        a subgoal needing an extension costs ``1 / complement_count``
        (more matrix complements = more chances to find a match).
        A subgoal with zero complements gets a high penalty.  The
        total over all new subgoals gives the extension's score.
        This is strictly more informative than clause length alone.
    """

    cut_reduction = True
    cut_extension = True
    max_cut_height = 7

    def _extension_distance(self, action: Extension) -> float:
        """Estimated proof distance for new subgoals.  Lower = better."""
        path_lits = self.env.state.goal.path()
        reducible = {(not lit.neg, lit.symbol) for lit in path_lits}
        matrix = self.env.state.matrix

        distance = 0.0
        for i, lit in enumerate(action.clause_copy):
            if i == action.lit_idx:
                continue
            if (lit.neg, lit.symbol) in reducible:
                pass  # free: can reduce
            else:
                n_complements = len(matrix.complements(lit))
                if n_complements == 0:
                    distance += 100.0
                else:
                    distance += 1.0 / n_complements
        return distance

    def choose_forward(self, forward: list[ConnectionAction]) -> ConnectionAction:
        if len(forward) == 1:
            return forward[0]

        def key(action: ConnectionAction) -> tuple:
            if isinstance(action, Reduction):
                return (0, 0.0)
            elif isinstance(action, Extension):
                return (1, self._extension_distance(action))
            elif isinstance(action, Start):
                return (2, float(len(action.clause_copy)))
            return (3, 0.0)

        return min(forward, key=key)

class FewAlternativesCut(Strategy):
    """Cut only when few forward alternatives exist, preserving
    exploration when the choice point is ambiguous.

    Source:
        Otten (2010), "Restricting backtracking in connection calculi",
        §4.3, Definition 8: restricted backtracking.

        Färber (2024), "A Curiously Effective Backtracking Strategy
        for Connection Tableaux", §4: decomposing restricted
        backtracking reveals that the benefit depends on how many
        alternatives are pruned.

    Design choices (not from the papers):
        leanCoP's ``cut`` is unconditional — every forward step is
        cut regardless of how many alternatives exist.  This strategy
        hypothesises that cutting is most valuable when few
        alternatives exist (the prover is unlikely to need them) and
        risky when many alternatives exist (the first choice may be
        wrong).  The threshold of 2 is hand-tuned: cut when ≤ 2
        forward actions, skip cut when ≥ 3.

        Uses ``comp(J)`` with J = 7 identically to leanCoP.
    """

    max_alternatives_for_cut: int = 2
    max_cut_height: int = 7

    def __init__(self, env):
        super().__init__(env)
        self.should_comp: bool = True

    def step(self):
        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        if not forward:
            action = backtrack_or_reset(actions)
            if (
                self.should_comp
                and isinstance(action, Reset)
                and action.depth > self.max_cut_height
            ):
                self.should_comp = False
                return Reset(id='rs', depth=1)
            return action

        chosen = forward[0]
        if (
            self.should_comp
            and len(forward) <= self.max_alternatives_for_cut
            and isinstance(chosen, (Reduction, Extension))
        ):
            chosen.cut = True
        return chosen

class DepthPhasedCut(Strategy):
    """Depth-dependent cut policy: cut reductions at all depths, cut
    extensions only at shallow depths.

    Source:
        Otten (2010), "Restricting backtracking in connection calculi",
        §4.3 and §5.4: inclusive cut and ``comp(J)``.

        Färber (2024), "A Curiously Effective Backtracking Strategy
        for Connection Tableaux", §4 and §6: decomposing restricted
        backtracking into reduction cut (R) and extension cut (EI/EX)
        as separate dimensions.

    Design choices (not from the papers):
        Färber shows that reduction cuts are almost always beneficial,
        while extension cuts are more problem-dependent.  We
        hypothesise that extension cuts are most valuable at shallow
        depths (where the search tree is wide and pruning saves the
        most work) and risky at deeper depths (where committing to a
        wrong extension is harder to recover from).

        The threshold is half the current iterative-deepening depth
        limit: extensions at depth ≤ max_depth/2 are cut, deeper ones
        are not.  This lets the strategy adapt as iterative deepening
        increases the depth limit.

        Uses ``comp(10)`` instead of ``comp(7)`` to give the phased
        policy more room before falling back to complete search.
    """

    max_cut_height: int = 10

    def __init__(self, env):
        super().__init__(env)
        self.should_comp: bool = True

    def step(self):
        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        if not forward:
            action = backtrack_or_reset(actions)
            if (
                self.should_comp
                and isinstance(action, Reset)
                and action.depth > self.max_cut_height
            ):
                self.should_comp = False
                return Reset(id='rs', depth=1)
            return action

        chosen = forward[0]
        if self.should_comp and isinstance(chosen, (Reduction, Extension)):
            if isinstance(chosen, Reduction):
                chosen.cut = True
            elif isinstance(chosen, Extension):
                goal = self.env.state.goal
                max_depth = self.env.state.max_depth
                if goal is not None and goal.depth <= max_depth // 2:
                    chosen.cut = True
        return chosen

class ProgressiveComp(Strategy):
    """Multi-phase completeness recovery: try aggressive cuts at
    progressively higher depth limits before falling back to complete
    (cut-free) search.

    Source:
        Otten (2010), "Restricting backtracking in connection calculi",
        §5.4: ``comp(J)`` runs with cut until depth J, then restarts
        from depth 1 without cut.  This is a single-phase policy.

    Design choices (not from the paper):
        leanCoP uses a single ``comp(J)`` transition.  This strategy
        generalises to multiple phases: ``comp(3) → comp(5) → comp(7)
        → comp(10) → complete``.  Each phase restarts the search with
        cuts active up to a higher depth limit.  The rationale is that
        a proof at depth 4 might require different constraint
        learning than a proof at depth 8; restarting with a fresh
        constraint set at each phase gives the prover independent
        chances to find it.

        The specific schedule [3, 5, 7, 10] is hand-tuned.  It
        favours early termination for easy problems (comp(3) is very
        fast) while still reaching depth 10 before falling back to
        the expensive complete search.
    """

    comp_schedule: tuple[int, ...] = (3, 5, 7, 10)

    def __init__(self, env):
        super().__init__(env)
        self.phase: int = 0

    def _cuts_active(self) -> bool:
        return self.phase < len(self.comp_schedule)

    def _current_threshold(self) -> Optional[int]:
        if self.phase < len(self.comp_schedule):
            return self.comp_schedule[self.phase]
        return None

    def step(self):
        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        if not forward:
            action = backtrack_or_reset(actions)
            threshold = self._current_threshold()
            if (
                isinstance(action, Reset)
                and threshold is not None
                and action.depth > threshold
            ):
                self.phase += 1
                return Reset(id='rs', depth=1)
            return action

        chosen = forward[0]
        if self._cuts_active() and isinstance(chosen, (Reduction, Extension)):
            chosen.cut = True
        return chosen

class GoalDistanceFewCut(Strategy):
    """Goal-distance ordering with selective cuts: only cut when few
    alternatives exist.

    Sources:
        Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2: fewest-solutions principle
        (estimated via goal distance).

        Otten (2010), "Restricting backtracking in connection calculi",
        §4.3: restricted backtracking.

        Färber (2024), "A Curiously Effective Backtracking Strategy
        for Connection Tableaux", §4: decomposing cut dimensions.

    Design choices (not from the papers):
        Combines two independently effective ideas:
        (1) ``GoalDistanceCut``'s ordering by estimated proof distance
            (path-complement + matrix-complement scoring).
        (2) ``FewAlternativesCut``'s selective cut policy (cut only
            when ≤ 2 alternatives).
        The combination is original to this project; neither paper
        proposes pairing ordering heuristics with selective cuts.
    """

    max_alternatives_for_cut: int = 2
    max_cut_height: int = 7

    def __init__(self, env):
        super().__init__(env)
        self.should_comp: bool = True

    def _extension_distance(self, action: Extension) -> float:
        path_lits = self.env.state.goal.path()
        reducible = {(not lit.neg, lit.symbol) for lit in path_lits}
        matrix = self.env.state.matrix
        distance = 0.0
        for i, lit in enumerate(action.clause_copy):
            if i == action.lit_idx:
                continue
            if (lit.neg, lit.symbol) in reducible:
                pass
            else:
                n_complements = len(matrix.complements(lit))
                distance += 100.0 if n_complements == 0 else 1.0 / n_complements
        return distance

    def _choose_forward(self, forward: list[ConnectionAction]) -> ConnectionAction:
        if len(forward) == 1:
            return forward[0]

        def key(action: ConnectionAction) -> tuple:
            if isinstance(action, Reduction):
                return (0, 0.0)
            elif isinstance(action, Extension):
                return (1, self._extension_distance(action))
            elif isinstance(action, Start):
                return (2, float(len(action.clause_copy)))
            return (3, 0.0)

        return min(forward, key=key)

    def step(self):
        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        if not forward:
            action = backtrack_or_reset(actions)
            if (
                self.should_comp
                and isinstance(action, Reset)
                and action.depth > self.max_cut_height
            ):
                self.should_comp = False
                return Reset(id='rs', depth=1)
            return action

        chosen = self._choose_forward(forward)
        if (
            self.should_comp
            and len(forward) <= self.max_alternatives_for_cut
            and isinstance(chosen, (Reduction, Extension))
        ):
            chosen.cut = True
        return chosen

class PathComplementFewCut(Strategy):
    """Path-complement ordering with selective cuts.

    Sources:
        Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2: fewest-solutions principle
        (reductions > path-reducible extensions > others).

        Otten (2010), "Restricting backtracking in connection calculi",
        §4.3: restricted backtracking.

    Design choices (not from the papers):
        Combines ``PathComplementCut``'s ordering (prefer extensions
        whose subgoals can reduce against the path) with
        ``FewAlternativesCut``'s selective cut policy.
    """

    max_alternatives_for_cut: int = 2
    max_cut_height: int = 7

    def __init__(self, env):
        super().__init__(env)
        self.should_comp: bool = True

    def _complement_count(self, action: Extension) -> int:
        path_lits = self.env.state.goal.path()
        reducible = {(not lit.neg, lit.symbol) for lit in path_lits}
        return sum(
            1 for i, lit in enumerate(action.clause_copy)
            if i != action.lit_idx and (lit.neg, lit.symbol) in reducible
        )

    def _choose_forward(self, forward: list[ConnectionAction]) -> ConnectionAction:
        if len(forward) == 1:
            return forward[0]

        def key(action: ConnectionAction) -> tuple:
            if isinstance(action, Reduction):
                return (0, 0, 0)
            elif isinstance(action, Extension):
                return (1, -self._complement_count(action), len(action.clause_copy))
            elif isinstance(action, Start):
                return (2, 0, len(action.clause_copy))
            return (3, 0, 0)

        return min(forward, key=key)

    def step(self):
        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        if not forward:
            action = backtrack_or_reset(actions)
            if (
                self.should_comp
                and isinstance(action, Reset)
                and action.depth > self.max_cut_height
            ):
                self.should_comp = False
                return Reset(id='rs', depth=1)
            return action

        chosen = self._choose_forward(forward)
        if (
            self.should_comp
            and len(forward) <= self.max_alternatives_for_cut
            and isinstance(chosen, (Reduction, Extension))
        ):
            chosen.cut = True
        return chosen

class ReductionCutOrdered(_ConfiguredCutStrategy):
    """Subgoal-reordered forward selection with cut on reductions only.

    Sources:
        Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2: subgoal reordering.

        Färber (2024), "A Curiously Effective Backtracking Strategy
        for Connection Tableaux", §4 and §6: reduction-only cut.

    Design choices (not from the papers):
        Combines the categorical ordering from ``SubgoalReordering``
        with the reduction-only cut from ``ReductionCut``.  The
        ordering improves action selection while reduction cuts add
        safe pruning without the aggressiveness of full cuts.
    """

    cut_reduction = True
    cut_extension = False

    def _action_key(self, action: ConnectionAction) -> tuple[int, int]:
        if isinstance(action, Reduction):
            return (0, 0)
        elif isinstance(action, Extension):
            return (1, len(action.clause_copy) - 1)
        elif isinstance(action, Start):
            return (2, len(action.clause_copy))
        return (3, 0)

    def choose_forward(self, forward: list[ConnectionAction]) -> ConnectionAction:
        if len(forward) == 1:
            return forward[0]
        return sorted(forward, key=self._action_key)[0]

@dataclass(frozen=True)
class ScheduledStrategySlot:
    strategy_name: str
    step_budget: int

class InitialSchedule(Strategy):
    """Run a fixed multi-strategy schedule loaded from JSON.

    The schedule file specifies a sequence of strategies and absolute
    step budgets.  Each slot runs its underlying strategy from a fresh
    prover state.  When the current slot exhausts its step budget, the
    environment is reset and the next strategy takes over.

    This mirrors the execution model used by
    ``paramils_schedule_wrapper.py`` for static schedules, except that
    budgets here are expressed directly in proof steps.
    """

    schedule_path = Path(__file__).resolve().parent.parent / 'schedules' / 'initial_schedule.json'

    def __init__(self, env):
        super().__init__(env)
        self._slots = self._load_slots()
        self._slot_index = 0
        self._slot_steps = 0
        self._active_strategy = self._instantiate_strategy(self._slots[0].strategy_name)

    @classmethod
    def _load_slots(cls) -> tuple[ScheduledStrategySlot, ...]:
        with open(cls.schedule_path, encoding = 'utf-8') as handle:
            raw = json.load(handle)

        raw_slots = raw.get('slots')
        if not isinstance(raw_slots, list) or not raw_slots:
            raise ValueError(f'{cls.schedule_path} must define a non-empty "slots" list')

        slots: list[ScheduledStrategySlot] = []
        for index, raw_slot in enumerate(raw_slots, start = 1):
            if not isinstance(raw_slot, dict):
                raise ValueError(f'slot {index} in {cls.schedule_path} must be an object')

            strategy_name = raw_slot.get('strategy')
            step_budget = raw_slot.get('step_budget')
            if not isinstance(strategy_name, str) or not strategy_name:
                raise ValueError(f'slot {index} in {cls.schedule_path} is missing a valid strategy name')
            if not isinstance(step_budget, int) or step_budget <= 0:
                raise ValueError(f'slot {index} in {cls.schedule_path} must have a positive integer step_budget')

            slots.append(ScheduledStrategySlot(strategy_name = strategy_name, step_budget = step_budget))

        return tuple(slots)

    def _instantiate_strategy(self, strategy_name: str) -> Strategy:
        strategies = Strategy.strategies()
        if strategy_name == type(self).__name__:
            raise ValueError(f'{type(self).__name__} cannot schedule itself')
        if strategy_name not in strategies:
            choices = ', '.join(sorted(strategies))
            raise ValueError(f'unknown scheduled strategy {strategy_name!r}; available: {choices}')
        return strategies[strategy_name](self.env)

    def _advance_slot(self) -> bool:
        if self._slot_index + 1 >= len(self._slots):
            return False
        self._slot_index += 1
        self._slot_steps = 0
        self.env.reset()
        self._active_strategy = self._instantiate_strategy(self._slots[self._slot_index].strategy_name)
        return True

    def step(self):
        current = self._slots[self._slot_index]
        if self._slot_steps >= current.step_budget:
            self._advance_slot()

        action = self._active_strategy.step()
        self._slot_steps += 1
        return action

from .sat_solve import SATSolve, SATSolveGuided, SATSolveLiteral  # noqa: E402,F401
