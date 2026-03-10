import math
from collections import defaultdict

from connections.env import ConnectionEnv
from connections.calculi.classicalcut import (
    ConnectionAction, Start, Reduction, Extension, Backtrack, Reset,
)


class Strategy:
    env: ConnectionEnv

    def __init__(self, env):
        self.env = env

    def step(self):
        raise NotImplemented('Abstract class')

    @classmethod
    def strategies(cls):
        return {
            'FirstAction': FirstAction,
            'SmartCut': SmartCut,
        }


class FirstAction(Strategy):
    def step(self):
        return self.env.legal_actions()[0]


class SmartCut(Strategy):
    """Strategy combining entropy-based cut with constraint-aware control.

    Forward action selection and cut:
        Actions are scored by a heuristic, converted to a probability
        distribution via softmax, and the Shannon entropy of that
        distribution decides whether to cut. Low entropy means the
        policy is confident, so cut is safe.
        (Zombori et al. 2021, \"The Role of Entropy in Guiding a
        Connection Prover\")

    Backtrack vs reset:
        When no forward actions are available, the strategy decides
        between backtracking and resetting based on consecutive failure
        count and constraint saturation.
        (Li et al. 2024, \"A Reinforcement Learning based Reset Policy
        for CDCL SAT Solvers\")

    Action scoring heuristic:
        Reductions are preferred over extensions (they close branches).
        Among extensions, shorter clauses are preferred.
        (Letz and Stenz 2001, \"Model Elimination and Connection Tableau
        Procedures\" — fewest-solutions and shortest-clause principles)
    """

    def __init__(self, env):
        super().__init__(env)

        # Entropy threshold below which we set cut=True.
        # Lower values = more conservative (cut less often).
        # With softmax over our heuristic scores, entropy of ln(2)
        # ≈ 0.693 is the minimum for 2 actions. A threshold just
        # below this means we only cut when there is a single dominant
        # action type (e.g., one reduction vs one extension).
        self.entropy_threshold = 0.69

        # How many consecutive dead-ends before considering a reset.
        self.max_consecutive_failures = 5

        # Tracks consecutive steps where no forward action was available.
        self.consecutive_failures = 0

        # Per-depth productivity tracking for adaptive reset depth
        # (inspired by Li et al. 2024 restart window learning rate).
        self.steps_at_depth: dict[int, int] = defaultdict(int)
        self.failures_at_depth: dict[int, int] = defaultdict(int)

    def score(self, action: ConnectionAction) -> float:
        """Heuristic score for a forward action. Higher is better.

        Reductions close branches immediately and are strongly
        preferred (score 2.0). Extensions are scored inversely to
        clause length — shorter clauses introduce fewer new subgoals.
        Start actions are scored like extensions.

        Follows the fewest-solutions and shortest-clause principles
        from Letz and Stenz (2001).
        """
        if isinstance(action, Reduction):
            return 2.0
        elif isinstance(action, Extension):
            return 1.0 / max(len(action.clause_copy), 1)
        elif isinstance(action, Start):
            return 1.0 / max(len(action.clause_copy), 1)
        return 0.0

    def softmax_entropy(self, scores: list[float]) -> tuple[list[float], float]:
        """Compute softmax probabilities and Shannon entropy.

        Returns (probabilities, entropy). Used to decide cut: low
        entropy means the policy is confident about which action to
        take (Zombori et al. 2021).
        """
        max_s = max(scores)
        exp_scores = [math.exp(s - max_s) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        return probs, entropy

    def should_reset(self) -> tuple[bool, str]:
        """Decide whether to reset instead of backtracking.

        Returns (should_reset, reason_string).

        Resets when the search appears unproductive: too many
        consecutive failures, or the constraint store is saturated
        (many constraints relative to the problem size suggest the
        current depth level is mostly exhausted).

        The consecutive-failure threshold is analogous to the restart
        condition in CDCL solvers (Holden 2021). The constraint
        saturation check is inspired by the productivity signal
        (restart window global learning rate) from Li et al. (2024).
        """
        if self.consecutive_failures >= self.max_consecutive_failures:
            return True, f'{self.consecutive_failures} consecutive failures'
        state = self.env.state
        if len(state.constraints) > state.matrix.num_lits * 2:
            return True, f'{len(state.constraints)} constraints (> {state.matrix.num_lits * 2})'
        return False, ''

    def choose_reset_depth(self) -> int:
        """Choose the depth for the next iteration after a reset.

        Standard iterative deepening increments by 1. If the current
        depth has been unproductive (high failure-to-step ratio), we
        skip ahead by 2 — a simplified version of the partial reset
        from Li et al. (2024), where the amount of \"forgetting\" is
        adapted to search productivity.
        """
        state = self.env.state
        current = state.max_depth
        steps = self.steps_at_depth[current]
        failures = self.failures_at_depth[current]
        if steps > 0 and failures / steps > 0.8:
            return current + 2
        return current + 1

    def step(self):
        state = self.env.state
        actions = self.env.legal_actions()

        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        # Track per-depth statistics.
        self.steps_at_depth[state.max_depth] += 1

        if not forward:
            # Dead end: decide backtrack vs reset.
            self.consecutive_failures += 1
            self.failures_at_depth[state.max_depth] += 1

            do_reset, reason = self.should_reset()
            if do_reset:
                depth = self.choose_reset_depth()
                delta = depth - state.max_depth
                self.consecutive_failures = 0
                return Reset(id = 'rs', depth = depth,
                             reason = f'{reason}, +{delta} depth')
            return Backtrack(id = 'bt')

        self.consecutive_failures = 0

        if len(forward) == 1:
            # Only one forward action: take it, no cut decision needed.
            return forward[0]

        # Score all forward actions and compute entropy.
        scores = [self.score(a) for a in forward]
        probs, entropy = self.softmax_entropy(scores)

        # Select the highest-scored action.
        best_idx = max(range(len(forward)), key = lambda i: scores[i])
        chosen = forward[best_idx]

        # Low entropy => high confidence => cut is safe.
        chosen.cut = (entropy < self.entropy_threshold)

        return chosen
