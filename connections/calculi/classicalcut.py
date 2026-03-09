from connections.utils.unification import Substitution
from connections.utils.primitives import Matrix, Literal
from connections.utils.tableau import Tableau

from dataclasses import dataclass, field
from typing import Optional, cast

@dataclass
class ConnectionAction:
    id: str
    cut: bool = False

@dataclass
class Start(ConnectionAction):
    clause_copy: list[Literal] = field(default_factory = lambda: [])
    sub_updates: list[Literal] = field(default_factory = lambda: [])

    def __repr__(self) -> str:
        return f"{self.id}: {str(self.clause_copy)}"

@dataclass
class Reduction(ConnectionAction):
    principle_node: Tableau = field(default = None)
    sub_updates: list[object] = field(default_factory = lambda: [])
    path_lit: Literal = field(default = None)

    def __repr__(self) -> str:
        suffix = " [cut]" if self.cut else ""
        return f"{self.id}: {str(self.principle_node.literal)} <- {str(self.path_lit)}{suffix}"

@dataclass
class Extension(ConnectionAction):
    principle_node: Tableau = field(default = None)
    sub_updates: list[object] = field(default_factory = lambda: [])
    lit_idx: int = -1
    clause_copy: list[Literal] = field(default_factory = lambda: [])

    def __repr__(self) -> str:
        suffix = " [cut]" if self.cut else ""
        return f"{self.id}: {str(self.principle_node.literal)} -> {str(self.clause_copy)}{suffix}"

@dataclass
class Backtrack(ConnectionAction):
    def __repr__(self) -> str:
        return 'Backtrack'


@dataclass
class Reset(ConnectionAction):
    def __repr__(self) -> str:
        return 'Reset'


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
        return [Backtrack(id = 'bt')]

    def resets(self) -> list[Reset]:
        return [Reset(id = 'rs')]

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

    def legal_forward_actions(self) -> dict[str, ConnectionAction]:
        current_clause = [node.literal for node in self.goal.parent.children[1:]]
        reg = self.regularizable(current_clause)

        if (self.goal is None) or reg:
            actions = []
        elif self.settings.iterative_deepening and (self.goal.depth >= self.max_depth):
            actions = cast(list[ConnectionAction], self.reductions())
        else:
            actions = cast(list[ConnectionAction], self.reductions() + self.extensions())

        return {action.id: action for action in actions}

    def legal_actions(self) -> dict[str, ConnectionAction]:
        if self.goal.parent is None:
            return {action.id: action for action in self.starts()}

        back = cast(list[ConnectionAction], self.backtracks() + self.resets())
        return {**self.legal_forward_actions(), **{a.id: a for a in back}}

    def backtrack(self) -> None:
        while True:
            self.goal = self.goal.find_prev()

            if self.proof_sequence:
                self.proof_sequence.pop()

            if self.goal.pending_cut:
                self.goal.pending_cut = False

            self.substitution.backtrack()
            self.goal.proven = False
            self.goal.children = []

            actions = self.goal.actions
            only_backtracks = all(isinstance(x, Backtrack) for x in actions.values())

            if self.goal is self.tableau and not self.goal.actions:
                self.reset(depth = self.max_depth + 1)
                break

            if self.goal.cut_committed or only_backtracks:
                continue

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

            case Reset():
                self.reset(depth = self.max_depth + 1)
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
