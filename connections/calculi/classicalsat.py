from connections.utils.unification import Substitution
from connections.utils.primitives import Matrix, Literal, Variable
from connections.utils.tableau import Tableau
from typing import Optional
from enum import StrEnum, auto
import re

import sys

from pydical import Solver as Cadical, UNSATISFIABLE
from pysat.solvers import Solver, Glucose4, Cadical153, Minisat22

import logging
from dataclasses import dataclass, field
from typing import cast

@dataclass
class ConnectionAction:
    id: str

@dataclass
class Start(ConnectionAction):
    clause_copy: list[Literal] = field(default_factory = lambda: [])
    sub_updates: list[Literal] = field(default_factory = lambda: [])

    def __repr__(self):
        return f"{self.id}: {str(self.clause_copy)}"

@dataclass
class Reduction(ConnectionAction):
    principle_node: Tableau
    sub_updates: int
    path_lit: int

    def __repr__(self):
        return f"{self.id}: {str(self.principle_node.literal)} <- {str(self.path_lit)}"

@dataclass
class Extension(ConnectionAction):
    principle_node: Tableau
    sub_updates: int
    lit_idx: int
    clause_copy: list[Literal]

    def __repr__(self):
        return f"{self.id}: {str(self.principle_node.literal)} -> {str(self.clause_copy)}"

@dataclass
class Backtrack(ConnectionAction):
    def __repr__(self):
        return 'Backtrack'

class SATConnectionState:
    matrix: Matrix

    tableau: Tableau
    goal: Tableau
    substitution: Substitution
    max_depth: int

    solver: Solver
    atom_map: dict[str, int]
    next_atom_id: int

    info: dict[str, str]
    is_terminal: bool
    proof_sequence: list[ConnectionAction]

    def __init__(self, matrix: Matrix, settings):
        self.matrix = matrix
        self.settings = settings

        self.cadical = Cadical()
        self.cadical.set('verbose', 0)
        self.pysat = Minisat22()

        self.solver = self.cadical

        self.atom_map = {}
        self.next_atom_id = 1
        self.substitution = Substitution()

        self.clauses = []

        for action in self.starts():
            if action.clause_copy:
                sat_clause = self.ground_clause(action.clause_copy)
                self.clauses.append((sat_clause, action.clause_copy))
                logging.info((sat_clause, action.clause_copy))
                self.solver.add_clause(sat_clause)

        if not self.solver.solve():
            self.info['status'] = 'Theorem (SAT)'
            self.is_terminal = True

            failed = [x for x in range(-self.next_atom_id + 1, self.next_atom_id) if self.solver.failed(x)]
            self.info['core'] = str(failed)

            return

        self.reset()

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

        logging.info(f"Resetting Tableau (Depth {self.max_depth})")

        self.tableau = Tableau()
        self.goal = self.tableau
        self.goal.actions = self.legal_actions()

        # Proof fields
        self.info = {}
        self.is_terminal = False
        self.proof_sequence = []

    # Converts a logical Literal to a SAT integer.
    def ground_literal(self, literal: Literal) -> int:
        # Apply substitution to get the most grounded version
        if self.substitution:
            ground_lit = self.substitution(literal)
        else:
            ground_lit = literal

        # Canonicalize: Map all variables to a single constant '*'
        # This implements the "simplest scheme" from the paper.
        atom_str = self.canonicalize_atom(ground_lit)

        if atom_str not in self.atom_map:
            self.atom_map[atom_str] = self.next_atom_id
            self.next_atom_id += 1

        sat_id = self.atom_map[atom_str]
        return -sat_id if literal.neg else sat_id

    def canonicalize_atom(self, literal: Literal) -> str:
        return self.stringify_term(literal)

    def stringify_term(self, term) -> str:
        if isinstance(term, Variable):
            # This is the weird part of the paper. Check it later.
            return "var"

        # if hasattr(term, 'args') and term.args:
            # args_str = ",".join(self.stringify_term(a) for a in term.args)
            # return f"{term.symbol}({args_str})"

        return str(term)

    def ground_clause(self, clause: list[Literal]) -> list[int]:
        return [self.ground_literal(lit) for lit in clause]

    def legal_actions(self) -> dict[str, ConnectionAction]:
        if self.goal.parent == None:
            return {action.id: action for action in self.starts()}

        current_clause = [node.literal for node in self.goal.parent.children[1:]]
        reg = self.regularizable(current_clause)

        actions: list[ConnectionAction]
        if (self.goal is None) or reg:
            actions = cast(list[ConnectionAction], self.backtracks())
        elif self.settings.iterative_deepening and (self.goal.depth >= self.max_depth):
            actions = cast(list[ConnectionAction], self.reductions() + self.backtracks())
        else:
            actions = cast(list[ConnectionAction], self.reductions() + self.extensions() + self.backtracks())

        return {action.id: action for action in actions}

    def starts(self) -> list[Start]:
        starts: list[Start] = []
        start_clause_candidates = self.matrix.positive_clauses

        if not self.settings.positive_start_clauses:
            start_clause_candidates += self.matrix.negative_clauses

        for clause in start_clause_candidates:
            clause_copy = self.matrix.copy(clause)
            starts.append(
                Start(
                    id = "st" + str(len(starts)),
                    clause_copy = clause_copy,
                )
            )
        if not starts:
            starts.append(Start(id = "st0"))

        return starts

    def regularizable(self, clause):
        for path_lit in self.goal.path():
            for clause_lit in clause:
                if path_lit.neg == clause_lit.neg and path_lit.symbol == clause_lit.symbol:
                    if self.substitution.equal(path_lit, clause_lit):
                        return True
        return False

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

    def backtracks(self) -> list[Backtrack]:
        return [Backtrack(id = 'bt')]

    def backtrack(self) -> None:
        # Backtrack to previous choice point (goal). If no choice points left, reset. 
        actions: dict[str, ConnectionAction] = {}

        limit = self.settings.backtrack_after if self.settings.restricted_backtracking else float('inf')
        while all(isinstance(x, Backtrack) for x in actions.values()) or self.goal.num_attempted > limit:
            self.goal = self.goal.find_prev()

            if self.proof_sequence:
                self.proof_sequence.pop()

            actions = self.goal.actions
            self.substitution.backtrack()
            self.goal.proven = False
            self.goal.children = []

            # If no new actions available for previous goals increase depth
            if self.goal is self.tableau and not self.goal.actions:
                logging.info(f'Increasing depth to {self.max_depth + 1}')
                self.reset(depth = self.max_depth + 1)
                break

    steps = 0
    def update_goal(self, action: ConnectionAction):
        self.steps += 1

        del self.goal.actions[action.id]
        self.goal.num_attempted += 1

        if not isinstance(action, Backtrack):
            self.substitution.update(action.sub_updates)
            self.proof_sequence.append(action)

        # print(self.tableau)
        # logging.info(action)
        match action:
            case Backtrack():
                self.backtrack()
                return

            case Start(clause_copy = clause_copy):
                if clause_copy is None:
                    self.info['status'] = 'Non-Theorem: no positive start clauses'
                    self.is_terminal = True
                    return

                self.goal.children = [Tableau(lit, self.goal) for lit in clause_copy]

            case Extension(clause_copy = clause_copy, lit_idx = lit_idx):
                # Ground the clause used for extension
                sat_clause = self.ground_clause(clause_copy)
                self.clauses.append((sat_clause, clause_copy))

                logging.info((sat_clause, clause_copy))
                self.solver.add_clause(sat_clause)

                if self.solver.solve() in (False, UNSATISFIABLE):
                    self.info['status'] = 'Theorem (SAT)'
                    self.is_terminal = True

                    failed = [x for x in range(-self.next_atom_id + 1, self.next_atom_id) if x != 0 and self.solver.failed(x)]
                    self.info['core'] = str(failed)
                    return

                self.goal.children = [Tableau(lit, self.goal) for lit in clause_copy]
                self.goal.children[lit_idx].proven = True
                self.goal.children.insert(0, self.goal.children.pop(lit_idx))

            case Reduction():
                self.goal.proven = True

        self.theorem_or_next()

    def clause_score(self, lit: Literal) -> int:
        sat_clause = self.ground_literal(lit)
        return self.solver.score(sat_clause)

    def theorem_or_next(self):
        self.goal = self.goal.find_best(self.clause_score)
        # self.goal = self.goal.find_next()
        if self.goal is None:
            # Standard success condition if no SAT pruning
            self.info['status'] = 'Theorem'
            self.is_terminal = True
            return

        self.goal.actions = self.legal_actions()
