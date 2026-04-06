from connections.utils.unification import Substitution
from connections.utils.primitives import Matrix, Literal
from connections.utils.tableau import Tableau

from dataclasses import dataclass, field
from typing import Optional, cast

@dataclass(frozen = True)
class ConnectionFact:
    """A connection between two matrix positions.

    Records which goal literal was connected to which target literal,
    identified by their positions in the original matrix. Used for
    constraint learning to prune previously-failed connection patterns.

    Based on the "connection atoms" from Rawson et al. (2026),
    "Constraint Learning for Non-confluent Proof Search".
    """
    goal_pos: tuple[int, int]
    target_pos: tuple[int, int]

@dataclass
class ConnectionAction:
    id: str
    cut: bool = False

@dataclass
class Start(ConnectionAction):
    clause_copy: list[Literal] = field(default_factory = lambda: [])
    sub_updates: list[Literal] = field(default_factory = lambda: [])

    def __repr__(self) -> str:
        return f"Start {self.id}: {str(self.clause_copy)}"

@dataclass
class Reduction(ConnectionAction):
    principle_node: Tableau = field(default = None)
    sub_updates: list[object] = field(default_factory = lambda: [])
    path_lit: Literal = field(default = None)

    def __repr__(self) -> str:
        suffix = " [cut]" if self.cut else ""
        return f"Reduction {self.id}: {str(self.principle_node.literal)} <- {str(self.path_lit)}{suffix}"

@dataclass
class Extension(ConnectionAction):
    principle_node: Tableau = field(default = None)
    sub_updates: list[object] = field(default_factory = lambda: [])
    lit_idx: int = -1
    clause_copy: list[Literal] = field(default_factory = lambda: [])

    def __repr__(self) -> str:
        suffix = " [cut]" if self.cut else ""
        return f"Excention {self.id}: {str(self.principle_node.literal)} -> {str(self.clause_copy)}{suffix}"

@dataclass
class Backtrack(ConnectionAction):
    def __repr__(self) -> str:
        return 'Backtrack'

@dataclass
class Reset(ConnectionAction):
    depth: int = 1
    reason: str = ''

    def __repr__(self) -> str:
        return f'Reset {self.depth}'

class ClassicalCut:
    matrix: Matrix
    tableau: Tableau
    goal: Tableau
    substitution: Substitution
    max_depth: int
    info: Optional[str]
    is_terminal: bool
    proof_sequence: list[ConnectionAction]

    def __init__(self, matrix: Matrix, settings) -> None:
        self.matrix = matrix
        self.settings = settings
        self.reset()

    def finalize_pending_cuts(self) -> None:
        stack = [self.tableau]
        while stack:
            node = stack.pop()
            if node.pending_cut and node.proven:
                node.pending_cut = False
                node.cut_committed = True
            stack.extend(node.children)

    def __str__(self) -> str:
        substitution = "\n".join(f"{k} → {v}" for k, v in self.substitution.to_dict().items())
        actions = None
        if self.goal is not None:
            actions = "\n".join(
                f"{i}. {str(action)}" for i, action in enumerate(self.goal.actions.values())
            )
        return (
            f"=========================\n"
            f"Tableau:\n{self.tableau}\n\n"
            f"Substitution:\n{substitution}\n\n"
            f"Available Actions:\n{actions}"
            f"\n\nMax Depth: {self.max_depth}"
            f"\n========================="
        )

    def reset(self, depth: Optional[int] = None) -> None:
        self.max_depth = depth if depth is not None else 10
        if depth is None:
            self.max_depth = self.settings.iterative_deepening_initial_depth

        self.tableau = Tableau()
        self.goal = self.tableau
        self.substitution = Substitution()
        self.goal.actions = self.legal_actions()

        self.info = None
        self.is_terminal = False
        self.proof_sequence = []

        # Constraint learning state (Rawson et al. 2026).
        # Constraints learned at one depth level are not reused at the
        # next, as the different depth bound changes the search space.
        self.constraints: list[frozenset[ConnectionFact]] = []
        self.failure_count: int = 0
        # Incrementally maintained set of connection facts on the
        # current proof path, to avoid recomputing from proof_sequence.
        self._cached_facts: set[ConnectionFact] = set()

    def starts(self) -> list[Start]:
        starts: list[Start] = []
        start_clause_candidates = self.matrix.positive_clauses

        if not self.settings.positive_start_clauses:
            start_clause_candidates += self.matrix.negative_clauses

        for clause in start_clause_candidates:
            clause_copy = self.matrix.copy(clause)
            starts.append(Start(id = "st" + str(len(starts)), clause_copy = clause_copy))
        if not starts:
            starts.append(Start(id = "st0"))

        return starts

    def backtracks(self) -> list[Backtrack]:
        if self.goal.parent is None:
            return []
        return [Backtrack(id = 'bt')]

    def resets(self) -> list[Reset]:
        return [Reset(id = 'rs', depth = self.max_depth + 1)]

    def extensions(self) -> list[Extension]:
        extensions: list[Extension] = []

        for clause_idx, lit_idx in self.matrix.complements(self.goal.literal):
            clause_copy = self.matrix.copy(clause_idx)
            unifies, updates = self.substitution.can_unify(self.goal.literal, clause_copy[lit_idx])

            if not unifies:
                continue

            extensions.append(
                Extension(
                    principle_node = self.goal,
                    sub_updates = updates,
                    lit_idx = lit_idx,
                    clause_copy = clause_copy,
                    id = "ex" + str(len(extensions)),
                )
            )

        return extensions

    def reductions(self) -> list[Reduction]:
        reductions: list[Reduction] = []
        for lit in self.goal.path():
            unifies = False
            if self.goal.literal.neg != lit.neg and self.goal.literal.symbol == lit.symbol:
                unifies, updates = self.substitution.can_unify(self.goal.literal, lit)
            if unifies:
                reductions.append(
                    Reduction(
                        principle_node = self.goal,
                        sub_updates = updates,
                        path_lit = lit,
                        id = "re" + str(len(reductions)),
                    )
                )
        return reductions

    def regularizable(self, clause: list[Literal]) -> bool:
        for path_lit in self.goal.path():
            for clause_lit in clause:
                if path_lit.neg == clause_lit.neg and path_lit.symbol == clause_lit.symbol:
                    if self.substitution.equal(path_lit, clause_lit):
                        return True
        return False

    def action_to_fact(self, action: ConnectionAction) -> Optional[ConnectionFact]:
        """Extract the connection fact from an Extension or Reduction."""
        if isinstance(action, Extension):
            goal_pos = action.principle_node.literal.matrix_pos
            target_pos = action.clause_copy[action.lit_idx].matrix_pos
            if goal_pos is not None and target_pos is not None:
                return ConnectionFact(goal_pos, target_pos)
        elif isinstance(action, Reduction):
            goal_pos = action.principle_node.literal.matrix_pos
            target_pos = action.path_lit.matrix_pos
            if goal_pos is not None and target_pos is not None:
                return ConnectionFact(goal_pos, target_pos)
        return None

    def current_facts(self) -> frozenset[ConnectionFact]:
        """Return the set of connection facts on the current proof path."""
        return frozenset(self._cached_facts)

    # Maximum number of constraints to store. Limits the cost of the
    # subset check in violates_constraint(). Older constraints are
    # evicted when the limit is reached.
    MAX_CONSTRAINTS = 500

    def analyze_failure(self) -> None:
        """Record a constraint when no forward actions are available.

        When the prover is stuck (no reductions or extensions apply),
        the current set of connections on the proof path is recorded as
        a constraint. Future actions that would recreate this pattern
        are pruned.

        This is a simplified version of the conflict analysis in Rawson
        et al. (2026): we record the full path rather than computing a
        minimal reason. This is sound but less precise — the constraint
        only fires when the exact same set of connections is rebuilt.
        """
        facts = self.current_facts()
        if facts and facts not in self.constraints:
            if len(self.constraints) >= self.MAX_CONSTRAINTS:
                self.constraints.pop(0)
            self.constraints.append(facts)
        self.failure_count += 1

    def violates_constraint(self, action: ConnectionAction) -> bool:
        """Check whether taking this action would recreate a known-bad
        connection pattern.

        An action violates a constraint if the set of connection facts
        on the current path, augmented with the new fact from this
        action, is a superset of any learned constraint.
        """
        if not self.constraints:
            return False
        new_fact = self.action_to_fact(action)
        if new_fact is None:
            return False
        augmented = self.current_facts() | {new_fact}
        return any(constraint <= augmented for constraint in self.constraints)

    def control_actions(self) -> list[ConnectionAction]:
        """Return control actions (backtrack/reset) available to the
        strategy. These are not stored on the tableau node."""
        if self.goal.parent is None:
            return []
        return cast(list[ConnectionAction], self.backtracks() + self.resets())

    def legal_forward_actions(self) -> dict[str, ConnectionAction]:
        current_clause = [node.literal for node in self.goal.parent.children[1:]]
        reg = self.regularizable(current_clause)

        if (self.goal is None) or reg:
            actions = []
        elif self.settings.iterative_deepening and (self.goal.depth >= self.max_depth):
            actions = cast(list[ConnectionAction], self.reductions())
        else:
            actions = cast(list[ConnectionAction], self.reductions() + self.extensions())

        # Filter actions against learned constraints (Rawson et al. 2026).
        actions = [a for a in actions if not self.violates_constraint(a)]

        return {action.id: action for action in actions}

    def legal_actions(self) -> dict[str, ConnectionAction]:
        if self.goal.parent is None:
            actions = cast(list[ConnectionAction], self.starts() + self.resets())
            return {action.id: action for action in actions}

        back = cast(list[ConnectionAction], self.backtracks() + self.resets())
        return {**self.legal_forward_actions(), **{a.id: a for a in back}}

    def backtrack(self) -> None:
        if self.goal.parent is None:
            return

        while True:
            old_goal = self.goal
            self.goal = self.goal.find_prev()
            pop_frame = (self.goal is old_goal.parent)

            if pop_frame and self.proof_sequence:
                popped = self.proof_sequence.pop()
                fact = self.action_to_fact(popped)
                if fact is not None:
                    self._cached_facts.discard(fact)

            if self.goal.pending_cut:
                self.goal.pending_cut = False

            if pop_frame:
                self.substitution.backtrack()
            self.goal.proven = False
            self.goal.children = []

            # Re-filter remaining actions against constraints learned
            # since this node was last visited (Rawson et al. 2026).
            self.goal.actions = {
                k: v for k, v in self.goal.actions.items()
                if not self.violates_constraint(v)
            }

            if self.goal.cut_committed or not self.goal.actions:
                continue

            break

    def update_goal(self, action: ConnectionAction) -> None:
        # Control actions (Backtrack, Reset) may not be in goal.actions
        # when the strategy creates them directly.
        if action.id in self.goal.actions:
            del self.goal.actions[action.id]
        self.goal.num_attempted += 1

        if isinstance(action, Start | Reduction | Extension):
            self.substitution.update(action.sub_updates)
            self.proof_sequence.append(action)
            fact = self.action_to_fact(action)
            if fact is not None:
                self._cached_facts.add(fact)

        match action:
            case Backtrack():
                self.backtrack()
                return

            case Reset(depth = depth):
                self.reset(depth = depth)
                return

            case Start(clause_copy = clause_copy):
                if clause_copy is None:
                    self.info = 'Non-Theorem: no positive start clauses'
                    self.is_terminal = True
                    return

                self.goal.children = [Tableau(lit, self.goal) for lit in clause_copy]

            case Extension(clause_copy = clause_copy, lit_idx = lit_idx, cut = cut):
                self.goal.children = [Tableau(lit, self.goal) for lit in clause_copy]
                self.goal.children[lit_idx].proven = True
                self.goal.children.insert(0, self.goal.children.pop(lit_idx))
                if cut:
                    self.goal.pending_cut = True

            case Reduction(cut = cut):
                self.goal.proven = True
                if cut:
                    self.goal.cut_committed = True

        self.theorem_or_next()

    def theorem_or_next(self) -> None:
        self.goal = self.goal.find_next()
        self.finalize_pending_cuts()
        if self.goal is None:
            self.info = 'Theorem'
            self.is_terminal = True
            return

        self.goal.actions = self.legal_forward_actions()
        if not self.goal.actions:
            self.analyze_failure()
