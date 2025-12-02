from connections.utils.unification import Substitution
from connections.utils.primitives import Matrix
from typing import Optional

import re
from enum import StrEnum, auto

import pysat
from pysat.solver import Solver

class Tableau:
    def __init__(self, literal = None, parent = None):
        self.literal = literal
        self.parent = parent
        self.children = []
        self.proven = False
        self.depth = parent.depth + 1 if parent is not None else -1
        self.num_attempted = 0
        self.actions = {}

        # if self.literal is not None and re.search(r'_[0-9]{6,}', str(self.literal)):
            # import ipdb
            # ipdb.set_trace()

    def __str__(self):
        angle = "└── " if self.depth >= 0 else ""
        ret = "    " * (self.depth) + angle + str(self.literal) + "\n"
        for child in self.children:
            ret += str(child)
        return ret

    def path(self):
        path = []
        current = self.parent
        while current.literal is not None:
            path.append(current.literal)
            current = current.parent
        return path

    # DFS over tree, find next unproven node.
    def find_next(self):
        parent = self
        while parent is not None:
            for child in parent.children:
                if not child.proven:
                    return child

            parent.proven = True
            parent = parent.parent

        return None

    def find_prev(self):
        parent = self.parent
        self_idx = parent.children.index(self)
        if self_idx > 1 or (self_idx == 1 and parent.literal is None):
            prev = parent.children[self_idx - 1]
            while len(prev.children) > 1:
                prev = prev.children[-1]
        else:
            prev = parent
        return prev

# This should be several dataclasses that inherit from ConnectionAction.
class State(StrEnum):
   Start = auto()
   Extension = auto()
   Reduction = auto()
   Backtrack = auto()

class ConnectionAction:
    """
    Abstract action class defines functions required of an action in an
    action space defined by a problem searched by an agent.
    """

    def __init__(
            self,
            type: State,
            principle_node = None,
            sub_updates = [],
            id = None,
            lit_idx = None,
            path_lit = None,
            clause_copy = None,
    ):
        self.type = type
        self.principle_node = principle_node
        self.sub_updates = sub_updates
        self.path_lit = path_lit
        self.lit_idx = lit_idx
        self.clause_copy = clause_copy
        self.id = id

    def __repr__(self):
        match self.type:
            case State.Extension:
                return f"{self.id}: {str(self.principle_node.literal)} -> {str(self.clause_copy)}"
            case State.Reduction:
                return f"{self.id}: {str(self.principle_node.literal)} <- {str(self.path_lit)}"
            case State.Start:
                return f"{self.id}: {str(self.clause_copy)}"
            case State.Backtrack:
                return 'Backtrack'

class SATConnectionState:
    matrix: Matrix

    tableau: Tableau
    goal: Tableau
    substitution: Substitution
    max_depth: int
    type: State

    info: Optional[str]
    is_terminal: bool
    proof_sequence: list[ConnectionAction]

    sat_solver: Solver
    atom_map: dict[str, int]

    def __init__(self, matrix: Matrix, settings):
        self.matrix = matrix
        self.settings = settings
        self.reset()

        self.sat_solver = Solver(name = 'picosat')
        self.sat_atom_map = {}

    def __str__(self) -> str:
        substitution = "\n".join(
            f"{k} → {v}" for k, v in self.substitution.to_dict().items()
        )
        actions = None
        if self.goal is not None:
            actions = "\n".join(
                f"{i}. {str(action)}" for i, action in enumerate(self.goal.actions.values())
            )
        return (
            f" = == = == = == = == = == = == = == = == = \n"
            f"Tableau:\n{self.tableau}\n\n"
            f"Substitution:\n{substitution}\n\n"
            f"Available Actions:\n{actions}"
            f"\n\nMax Depth: {self.max_depth}"
            f"\n = == = == = == = == = == = == = == = == = "
        )

    def reset(self, depth: Optional[int] = None):
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

        # SAT Actions
        for action in self.starts():
            if action.clause_copy:
                self.solver.add_clause(self.ground(action.clause_copy)

    def legal_actions(self) -> dict[int, ConnectionAction]:
        if self.goal.parent == None:
            return {action.id: action for action in self.starts()}

        current_clause = [node.literal for node in self.goal.parent.children[1:]]
        reg = self.regularizable(current_clause)
        if (self.goal is None) or reg:
            actions = self.backtracks()
        elif self.settings.iterative_deepening and (self.goal.depth >= self.max_depth):
            actions = self.reductions() + self.backtracks()
        else:
            actions = self.reductions() + self.extensions() + self.backtracks()

        return {action.id: action for action in actions}

    def starts(self) -> list[ConnectionAction]:
        starts: list[ConnectionAction] = []
        start_clause_candidates = self.matrix.positive_clauses

        if not self.settings.positive_start_clauses:
            start_clause_candidates = self.matrix.clauses

        for clause in start_clause_candidates:
            clause_copy = self.matrix.copy(clause)
            starts.append(
                    ConnectionAction(
                        type = State.Start,
                        clause_copy = clause_copy,
                        id = "st" + str(len(starts))
                    )
                )

        if not starts:
            starts.append(ConnectionAction(type = State.Start, id = "st0"))

        return starts

    def regularizable(self, clause):
        for path_lit in self.goal.path():
            for clause_lit in clause:
                if path_lit.neg == clause_lit.neg and path_lit.symbol == clause_lit.symbol:
                    if self.substitution.equal(path_lit, clause_lit):
                        return True
        return False

    def reductions(self) -> list[ConnectionAction]:
        reductions: list[ConnectionAction] = []
        for lit in self.goal.path():
            unifies = False
            if self.goal.literal.neg != lit.neg and self.goal.literal.symbol == lit.symbol:
                unifies, updates = self.substitution.can_unify(self.goal.literal, lit)
            if unifies:
                reductions.append(
                    ConnectionAction(
                        type = State.Reduction,
                        principle_node = self.goal,
                        sub_updates = updates,
                        path_lit = lit,
                        id = "re" + str(len(reductions)),
                    )
                )
        return reductions

    def extensions(self) -> list[ConnectionAction]:
        extensions: list[ConnectionAction] = []

        for clause_idx, lit_idx in self.matrix.complements(self.goal.literal):
            clause_copy = self.matrix.copy(clause_idx)
            unifies, updates = self.substitution.can_unify(self.goal.literal,clause_copy[lit_idx])
            if unifies:
                extensions.append(
                    ConnectionAction(
                        type = State.Extension,
                        principle_node = self.goal,
                        sub_updates = updates,
                        lit_idx = lit_idx,
                        clause_copy = clause_copy,
                        id = "ex" + str(len(extensions)),
                    )
                )
        return extensions

    def backtracks(self) -> list[ConnectionAction]:
        return [ConnectionAction(type = State.Backtrack, id = 'bt')]

    def backtrack(self) -> None:
        # Backtrack to previous choice point (goal). If no choice points left, reset. 
        actions: dict[int, ConnectionAction] = {}
        while not actions or actions.keys() == [State.Backtrack] or (self.settings.restricted_backtracking and (self.goal.num_attempted > self.settings.backtrack_after)):
            self.goal = self.goal.find_prev()

            if self.proof_sequence:
                self.proof_sequence.pop()

            actions = self.goal.actions
            self.substitution.backtrack()
            self.goal.proven = False
            self.goal.children = []

            # If no new actions available for previous goals increase depth
            if self.goal is self.tableau and not self.goal.actions:
                self.reset(depth = self.max_depth + 1)
                break

    def update_goal(self, action):
        del self.goal.actions[action.id]
        self.goal.num_attempted += 1

        if action.type != State.Backtrack:
            self.substitution.update(action.sub_updates)
            self.proof_sequence.append(action)

        match action.type:
            case State.Backtrack:
                self.backtrack()
                return

            case State.Start :
                if action.clause_copy is None:
                    self.info = 'Non-Theorem: no positive start clauses'
                    self.is_terminal = True
                    return
                self.goal.children = [Tableau(lit, self.goal) for lit in action.clause_copy]

            # Make literal extended to child and mark as proven for backtracking purposes
            case State.Extension:
                self.goal.children = [Tableau(lit, self.goal) for lit in action.clause_copy]
                self.goal.children[action.lit_idx].proven = True
                self.goal.children.insert(0, self.goal.children.pop(action.lit_idx))

            case State.Reduction:
                self.goal.proven = True

        # Find next goal node, if None, a proof has been found, otherwise backtrack
        self.theorem_or_next()

    def theorem_or_next(self):
        self.goal = self.goal.find_next()
        if self.goal is None:
            self.info = 'Theorem'
            self.is_terminal = True
            return
        self.goal.actions = self.legal_actions()
