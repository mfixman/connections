# MyPyCoP Development Notes

---

## 1. SmartCut: Design, Implementation, and Results

### Overview

SmartCut is a hand-crafted strategy for the connection calculus that combines four ideas from
the literature into a single proof-search policy. It operates on top of the `ClassicalCut`
calculus, which already performs constraint learning transparently before SmartCut sees any
actions. The strategy makes two decisions at each step: (a) which forward action to take, and
(b) whether to annotate that action with a cut.  When no forward actions are available, it
makes a third decision: whether to backtrack or reset to a higher depth.

### Component 1: Constraint Learning (Rawson et al. 2026)

Before SmartCut sees any actions, `ClassicalCut` filters them against a set of learned
constraints. When the prover reaches a dead end (no reductions or extensions apply),
`analyze_failure()` records the current set of connection facts on the proof path as a
constraint. A *connection fact* is a pair `(goal_pos, target_pos)` identifying which matrix
positions were connected. Any future action that would recreate a superset of a known-failed
connection pattern is pruned by `violates_constraint()`.

This is a simplified version of the conflict analysis in Rawson et al. (2026): rather than
computing a minimal reason for failure, the full path is recorded. The constraint fires only
when the exact same set of connections is rebuilt, which is sound but less precise than a
minimal reason. Up to 500 constraints are stored; older ones are evicted when the limit is
reached.

On backtrack, `goal.actions` is re-filtered against constraints learned since the node was last
visited — so the constraint store improves the search even within a single depth level.

### Component 2: Action Scoring and Selection (Letz & Stenz 2001)

Among the forward actions available, SmartCut selects the one with the highest heuristic
score:

- **Reductions** score 2.0 — they close an open branch immediately and are strongly preferred.
- **Extensions** score 1 / clause\_length — shorter clauses introduce fewer new subgoals.
- **Start actions** are scored like extensions.

This follows the *fewest-solutions* and *shortest-clause* principles from Letz & Stenz (2001,
Chapter 28), which favour actions that constrain the search most tightly. Reductions dominate
because they require no new literals to be proved.

### Component 3: Entropy-Based Cut Decision (Zombori et al. 2021)

After selecting the best forward action, SmartCut decides whether to set `cut=True` on it.
The cut operator prunes backtracking into the current branch once it succeeds — a sound but
incomplete optimisation that trades completeness-per-depth for speed.

The decision is based on the Shannon entropy of the softmax distribution over the action
scores. Low entropy means the policy is confident (one action dominates); in that case, the
unchosen alternatives are unlikely to be useful, and cutting is safe. High entropy means the
alternatives are plausible, so we do not cut.

Formally, softmax probabilities are computed as $p_i = \exp(s_i) / \sum_j \exp(s_j)$, and the
entropy $H = -\sum_i p_i \log p_i$ is compared against a threshold of 0.69. This threshold
sits just below $\ln 2 \approx 0.693$, the minimum possible entropy for two actions with
equal scores. In practice, the cut fires when a reduction is available alongside extensions
(score 2.0 vs ≤ 1.0), making the entropy collapse.

This adapts the entropy-regularisation analysis of Zombori et al. (2021), who show that a
neural guidance policy should avoid overconfidence: too little entropy causes the prover to
commit to wrong branches. Here, the entropy threshold plays the opposite role — it is used to
*identify* high-confidence situations where commitment (cutting) is safe.

When there is only one forward action, the cut logic is bypassed entirely. With a single
action, entropy is zero by definition, which would always trigger a cut — an overly aggressive
policy. The action is returned as-is.

### Component 4: Backtrack vs Reset (Holden 2021; Li et al. 2024)

When no forward actions remain (a dead end), SmartCut must choose between:

- **Backtrack**: undo the last action and try an alternative at the parent node.
- **Reset**: abandon the entire current search and restart at a higher depth bound.

Two conditions trigger a reset (either suffices):

1. **Consecutive failure count** ≥ 5: if the prover has hit dead ends on five consecutive steps
   without any forward progress, the current depth level is likely exhausted. This is directly
   analogous to the restart condition in CDCL SAT solvers (Holden 2021).

2. **Constraint saturation**: if `len(constraints) > 2 × num_literals`, the constraint store
   is dense relative to the problem size, suggesting the current depth level has been largely
   explored.

Once reset is triggered, the new depth is chosen adaptively. If the failure-to-step ratio at
the current depth exceeds 80%, the depth is incremented by 2 (skipping a mostly-unproductive
level); otherwise it increments by 1. This is a simplified version of the *partial reset*
from Li et al. (2024), where the amount of "forgetting" is adapted to search productivity via
a restart window global learning rate.

### Results

Comparison against `FirstAction` (which always takes the first legal action, performing
exhaustive depth-first search) on the test suite with a limit of 50,000 steps and 60-second
timeout:

```
Problem          FirstAction   SmartCut     Delta
-------------------------------------------------
GEO081+1.p       timeout       2373         SmartCut only
GEO185+1.p       15038         timeout      FirstAction only
LAT381+3.p       17093         1981         −88%
NUM531+1.p       426           timeout      FirstAction only
SET027+3.p       463           221          −52%
SET875+1.p       559           476          −14%
SET877+1.p       2355          timeout      FirstAction only
SYN001+1.p       34            34           tie
(7 problems both timeout or not applicable)
-------------------------------------------------
Solved: FirstAction=7/15   SmartCut=5/15
Wins:   FirstAction=3      SmartCut=4      Ties=1
```

SmartCut wins on problems where iterative deepening with aggressive cut pruning finds the
proof much faster than exhaustive search. It loses on problems where the proof requires
exploring many branches at the correct depth — the cut operator occasionally prunes branches
that are needed, and the heuristic scoring occasionally mis-orders actions.

---

## 2. Attempted Experiment: Thompson Sampling Bandit for Reset Decisions

### Motivation

The backtrack/reset logic in SmartCut (Component 4) uses hand-crafted thresholds. Li et al.
(2024) propose modelling the restart decision in CDCL SAT solvers as a 2-armed bandit problem
and learning a policy via Thompson Sampling, showing consistent improvements over fixed
restart schedules. We attempted to adapt this approach to connection calculi, replacing the
threshold logic with a learned bandit policy.

### Setup: Li et al. 2024

In the SAT setting, the two arms are *restart* and *no-restart*, and the decision is made at
each restart boundary. The reward signal is the *restart window global learning rate*
(rw\_glr): the number of clauses learned divided by the number of decisions taken in the
current restart window. Success is declared when the current window's rw\_glr meets or exceeds
its exponential moving average (EMA), meaning the arm is keeping pace with historical
productivity.

Posteriors are maintained as Beta distributions $\text{Beta}(\alpha_i, \beta_i)$ for each
arm $i$. At each decision, a sample is drawn from each posterior and the arm with the higher
sample is chosen (Thompson Sampling). After observing the outcome, the winning arm's $\alpha$
or $\beta$ is incremented, and all parameters are decayed by a factor of 0.8 before each
update to handle the non-stationarity of the solver's search distribution. Thompson Sampling
was found to consistently outperform UCB across SAT Competition 2022, 2023, and Satcoin
benchmarks.

### Our Adaptation

- **Arms**: `Backtrack` (arm 0) vs `Reset(depth+1)` (arm 1).
- **Decision trigger**: every dead end (no forward actions available).
- **Reward signal (first attempt)**: constraints learned / total steps since last bandit
  decision, directly mirroring rw\_glr.
- **Reward signal (second attempt)**: forward steps (Start, Reduction, Extension) taken
  between successive dead ends — a more direct measure of search productivity.
- **Prior**: initially uniform Beta(1, 1) for both arms; later adjusted to Beta(5, 1) / Beta(1, 1)
  to bias toward backtracking.

### Results

```
Problem          FirstAction   SmartCut (bandit)   vs. baseline SmartCut
------------------------------------------------------------------------
LAT381+3.p       17093         timeout             regression
SET875+1.p       559           timeout             regression (was 476)
SET877+1.p       2355          timeout             regression
GEO081+1.p       timeout       25                  slight improvement
SET027+3.p       463           26                  improvement
SYN001+1.p       34            12                  improvement
------------------------------------------------------------------------
Solved: FirstAction=7/15   SmartCut (bandit)=3/15
Wins:   FirstAction=5      SmartCut=3      Ties=0
```

The bandit improved performance on the two problems where SmartCut already won decisively,
but caused regressions on all problems that required careful backtracking without early resets.

### Why It Failed

**1. Sparse reward signal.**
The fundamental incompatibility is in the reward signal. In SAT, every conflict produces a
new learned clause, so the rw\_glr signal is dense and continuously informative — the solver
generates hundreds of clause-learning events per restart cycle. In connection calculi,
constraints are learned only at dead ends, which are rare relative to forward steps. For most
of the search, constraint\_delta = 0, so rw\_glr = 0 throughout. With the EMA also converging
to zero, the success condition `rw_glr ≥ ema_rw_glr` reduces to `0 ≥ 0 = True` — every arm
choice is classified as a success, and the bandit receives no useful learning signal. Both
arms accumulate equal pseudo-successes, and the Beta posteriors cannot differentiate them.

Switching the signal to forward steps between dead ends partially addressed the degeneracy
(the signal is always positive and nonzero), but both arms still produced similar window
lengths for most problems, leaving the bandit unable to distinguish them.

**2. Short episode length.**
Li et al. (2024) demonstrate their approach on SAT Competition instances where the solver
runs for thousands of seconds and millions of decisions, generating many hundreds of restart
cycles per run. This gives the bandit sufficient data to converge. Our test problems are
solved — or time out — in hundreds to low thousands of steps, spanning at most a few dozen
bandit decisions. With fewer data points than the initial Beta prior dominates, the
bandit has not converged before the episode ends.

**3. Domain knowledge outperforms uninitialised learning.**
The threshold conditions (`consecutive_failures ≥ 5` and `len(constraints) > 2 × num_lits`)
directly encode the same productivity signal the bandit would need to discover: consecutive
failures measure depth-level exhaustion, and constraint saturation measures how thoroughly
the constraint store has been populated. For problems of this scale, these domain-specific
conditions are immediately effective from the first step, whereas the bandit requires many
cycles of exploration before its policy is meaningful.

**4. The structure of useful resets differs between SAT and connection calculi.**
In SAT, restarts are frequent and fine-grained — the solver restarts after a small number of
conflicts, retaining learned clauses. In connection calculi, a reset discards the entire
current proof attempt and increases the depth bound. The cost of a premature reset is high:
unexplored branches at the current depth are abandoned, and the constraint store is cleared.
For problems like SET875+1.p, the proof is accessible without any reset — `FirstAction` finds
it in 559 steps by pure backtracking — and any stochastic reset decision hurts by interrupting
a potentially productive branch.

### Conclusion

The Thompson Sampling bandit of Li et al. (2024) is designed for a regime characterised by
high-volume decision-making, dense reward signals, and long solver runs. Connection calculi
proof search matches none of these properties at the problem scales in our test suite. The
adaptation is not straightforward without either (a) a richer per-step signal analogous to
LBD in SAT, or (b) a problem set with significantly harder instances where the solver runs
long enough for the bandit to learn.

The threshold-based approach remains in use. The failed experiment is worth documenting as
a negative result: it clarifies the boundary conditions under which RL-based restart policies
are applicable, and motivates future work on finding a meaningful dense signal for connection
calculi (e.g., an analogue of LBD based on constraint specificity or proof-path depth).

---

## References

- Holden, S. B. (2021). *Machine Learning for Automated Theorem Proving: Learning to Solve SAT and QSAT*. University of Cambridge.
- Letz, R. and Stenz, G. (2001). Model Elimination and Connection Tableau Procedures. In *Handbook of Automated Reasoning*, Chapter 28.
- Li, C., Liu, C., Chung, J., Lu, Z., Jha, P., and Ganesh, V. (2024). A Reinforcement Learning based Reset Policy for CDCL SAT Solvers. *CVIT 2024*.
- Rawson, M., Eisenhofer, C., and Kovács, L. (2026). Constraint Learning for Non-confluent Proof Search. *TABLEAUX 2025*, LNAI.
- Zombori, Z., Urban, J., and Olsák, M. (2021). The Role of Entropy in Guiding a Connection Prover. *TABLEAUX 2021*, LNAI.
