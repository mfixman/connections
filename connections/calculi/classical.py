from connections.utils.unification import Substitution
from connections.utils.primitives import Matrix, Literal
from connections.utils.tableau import Tableau

from dataclasses import dataclass, field
from typing import Optional, cast

@dataclass
class ConnectionAction:
    id: str

@dataclass
class Start(ConnectionAction):
    clause_copy: list[Literal] = field(default_factory = lambda: [])
    sub_updates: list[Literal] = field(default_factory = lambda: [])

    def __repr__(self) -> str:
        return f"{self.id}: {str(self.clause_copy)}"

@dataclass
class Reduction(ConnectionAction):
    principle_node: Tableau
    sub_updates: list[object]
    path_lit: Literal

    def __repr__(self) -> str:
        return f"{self.id}: {str(self.principle_node.literal)} <- {str(self.path_lit)}"

@dataclass
class Extension(ConnectionAction):
    principle_node: Tableau
    sub_updates: list[object]
    lit_idx: int
    clause_copy: list[Literal]

    def __repr__(self) -> str:
        return f"{self.id}: {str(self.principle_node.literal)} -> {str(self.clause_copy)}"

@dataclass
class Backtrack(ConnectionAction):
    def __repr__(self) -> str:
        return 'Backtrack'

class ConnectionState:
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
        # Tableau fields
        self.max_depth = depth if depth is not None else 10
        if depth is None:
            self.max_depth = self.settings.iterative_deepening_initial_depth

        self.tableau = Tableau()
        self.goal = self.tableau
        self.substitution = Substitution()
        self.goal.actions = self.legal_actions()

        # Proof fields
        self.info = None
        self.is_terminal = False
        self.proof_sequence = []

    def starts(self) -> list[Start]:
        starts: list[Start] = []
        start_clause_candidates = self.matrix.positive_clauses

        if not self.settings.positive_start_clauses:
            start_clause_candidates += self.matrix.negative_clauses

        for clause in start_clause_candidates:
            clause_copy = self.matrix.copy(clause)
            starts.append( Start(id = "st" + str(len(starts)), clause_copy = clause_copy))
        if not starts:
            starts.append(Start(id = "st0"))

        return starts

    def backtracks(self) -> list[Backtrack]:
        return [Backtrack(id = 'bt')]

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

    def legal_actions(self) -> dict[str, ConnectionAction]:
        if self.goal.parent is None:
            return {action.id: action for action in self.starts()}

        current_clause = [node.literal for node in self.goal.parent.children[1:]]
        reg = self.regularizable(current_clause)

        actions: list[ConnectionAction]
        if self.goal is None or reg:
            actions = cast(list[ConnectionAction], self.backtracks())
        elif self.settings.iterative_deepening and (self.goal.depth >= self.max_depth):
            actions = cast(list[ConnectionAction], self.reductions() + self.backtracks())
        else:
            actions = cast(list[ConnectionAction], self.reductions() + self.extensions() + self.backtracks())

        return {action.id: action for action in actions}

    def _refresh_remaining_actions(self, remaining_ids: set[str]) -> dict[str, ConnectionAction]:
        """Recompute actions under the restored substitution.

        Backtracking changes the substitution state, so any cached
        action objects may carry stale ``sub_updates``. Rebuild the
        current node's actions from scratch and keep only ids that have
        not yet been tried at this node.
        """
        fresh_actions = self.legal_actions()
        return {action_id: fresh_actions[action_id] for action_id in remaining_ids if action_id in fresh_actions}

    def backtrack(self) -> None:
        # Backtrack to previous choice point (goal). If no choice points left, reset.
        actions: dict[str, ConnectionAction] = {}

        limit = self.settings.backtrack_after if self.settings.restricted_backtracking else float('inf')
        while all(isinstance(x, Backtrack) for x in actions.values()) or self.goal.num_attempted > limit:
            old_goal = self.goal
            self.goal = self.goal.find_prev()
            pop_frame = (self.goal is old_goal.parent)
            remaining_ids = set(self.goal.actions)

            if pop_frame and self.proof_sequence:
                self.proof_sequence.pop()

            if pop_frame:
                self.substitution.backtrack()
            self.goal.proven = False
            self.goal.children = []
            self.goal.actions = self._refresh_remaining_actions(remaining_ids)
            actions = self.goal.actions

            # If no new actions available for previous goals increase depth
            if self.goal is self.tableau and not self.goal.actions:
                self.reset(depth = self.max_depth + 1)
                break

    def update_goal(self, action: ConnectionAction) -> None:
        del self.goal.actions[action.id]
        self.goal.num_attempted += 1

        if isinstance(action, Start | Reduction | Extension):
            self.substitution.update(action.sub_updates)
            self.proof_sequence.append(action)

        match action:
            case Backtrack():
                self.backtrack()
                return

            case Start(clause_copy = clause_copy):
                if clause_copy is None:
                    self.info = 'Non-Theorem: no positive start clauses'
                    self.is_terminal = True
                    return

                self.goal.children = [Tableau(lit, self.goal) for lit in clause_copy]

            case Extension(clause_copy = clause_copy, lit_idx = lit_idx):
                self.goal.children = [Tableau(lit, self.goal) for lit in clause_copy]
                self.goal.children[lit_idx].proven = True
                self.goal.children.insert(0, self.goal.children.pop(lit_idx))

            case Reduction():
                self.goal.proven = True

        self.theorem_or_next()

    def theorem_or_next(self) -> None:
        self.goal = self.goal.find_next()
        if self.goal is None:
            self.info = 'Theorem'
            self.is_terminal = True
            return

        self.goal.actions = self.legal_actions()
