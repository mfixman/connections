from dataclasses import dataclass, field
import random
from typing import Optional

from connections.calculi.classicalcut import (
    Backtrack,
    ClassicalCut,
    ConnectionAction,
    Extension,
    Reduction,
    Reset,
    Start,
)
from connections.utils.primitives import Expression, Literal, Variable

try:
    import pycosat
except ModuleNotFoundError:
    pycosat = None

try:
    from pysat.solvers import Minisat22 as PySATSolver
except ModuleNotFoundError:
    PySATSolver = None

def sign(x):
    if x == 0:
        return '0'
    return '+' if x > 0 else '-'

@dataclass
class ShadowSAT:
    atom_map: dict[str, int] = field(default_factory = dict)
    ground_clauses: set[tuple[int, ...]] = field(default_factory = set)
    new_ground_clauses: set[tuple[int, ...]] = field(default_factory = set)
    model: dict[int, bool] = field(default_factory = dict)
    satisfiable: Optional[bool] = None

    def atom_id(self, atom: str) -> int:
        atom_id = self.atom_map.get(atom)
        if atom_id is None:
            atom_id = len(self.atom_map) + 1
            self.atom_map[atom] = atom_id
        return atom_id

class SATConnectionState(ClassicalCut):
    shadow: ShadowSAT

    def __init__(self, matrix, settings) -> None:
        self.sat_solver_backend = getattr(settings, 'sat_solver', 'pycosat').lower()
        self._require_sat_backend(self.sat_solver_backend)
        self.shadow = ShadowSAT()
        self.random = random.Random(settings.random_seed)
        super().__init__(matrix, settings)

    def sat_model(self) -> dict[int, bool]:
        return self.shadow.model

    def sat_status(self) -> Optional[bool]:
        return self.shadow.satisfiable

    def ground_clauses(self) -> set[tuple[int, ...]]:
        return self.shadow.ground_clauses

    def new_ground_clauses(self) -> set[tuple[int, ...]]:
        return self.shadow.new_ground_clauses

    def reset(self, depth: Optional[int] = None) -> None:
        if depth is None:
            self.shadow = ShadowSAT()
            self._seed_ground_with_start_clauses()
        super().reset(depth)

    def analyze_failure(self) -> None:
        return

    def violates_constraint(self, action: ConnectionAction) -> bool:
        return False

    def starts(self) -> list[Start]:
        starts = super().starts()
        for action in starts:
            action.clause_copy = self._shuffled_clause(action.clause_copy)
        self.random.shuffle(starts)
        return starts

    def extensions(self) -> list[Extension]:
        extensions = super().extensions()
        for action in extensions:
            action.clause_copy, action.lit_idx = self._shuffled_extension_clause(action.clause_copy, action.lit_idx)
        self.random.shuffle(extensions)
        return extensions

    def backtrack(self) -> None:
        if self.goal.parent is None:
            return

        while True:
            old_goal = self.goal
            self.goal = self.goal.find_prev()
            pop_frame = (self.goal is old_goal.parent)
            remaining_ids = set(self.goal.actions)

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
            self.goal.actions = self._refresh_remaining_actions(remaining_ids)

            if self.goal.cut_committed or not self.goal.actions:
                continue

            break

        self._refresh_goal()

    def update_goal(self, action: ConnectionAction) -> None:
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
                self._finish_iteration()
                if self.is_terminal:
                    return
                self.reset(depth = depth)
                return

            case Start(clause_copy = clause_copy):
                if clause_copy is None:
                    self.info = 'Non-Theorem: no positive start clauses'
                    self.is_terminal = True
                    return

                self.goal.children = [self._new_tableau_child(lit) for lit in clause_copy]

            case Extension(clause_copy = clause_copy, lit_idx = lit_idx, cut = cut):
                self.goal.children = [self._new_tableau_child(lit) for lit in clause_copy]
                self.goal.children[lit_idx].proven = True
                self.goal.children.insert(0, self.goal.children.pop(lit_idx))
                if cut:
                    self.goal.pending_cut = True
                self._record_tableau_clauses()

            case Reduction(cut = cut):
                self.goal.proven = True
                if cut:
                    self.goal.cut_committed = True
                self._record_tableau_clauses()

        self.theorem_or_next()

    def theorem_or_next(self) -> None:
        self.goal = self.goal.find_next()
        self._refresh_goal()

    def sat_literal_value(self, literal: Literal) -> Optional[bool]:
        grounded = self.substitution(literal)
        if not self._is_ground(grounded):
            return None
        sat_lit = self._ground_literal(grounded)
        value = self.shadow.model.get(abs(sat_lit))
        if value is None:
            return None
        return value if sat_lit > 0 else (not value)

    def ground_clause(self, clause: list[Literal]) -> tuple[int, ...]:
        grounded = {self._ground_literal(self.substitution(lit)) for lit in clause}
        return tuple(sorted(grounded, key = lambda lit: (abs(lit), lit < 0)))

    def _refresh_goal(self) -> None:
        self.finalize_pending_cuts()

        if self.goal is None:
            self.info = 'Theorem'
            self.is_terminal = True
            return

        if self.goal.parent is None:
            self.goal.actions = self.legal_actions()
            return

        self.goal.actions = self.legal_forward_actions()
        if not self.goal.actions:
            self.analyze_failure()

    def _finish_iteration(self) -> None:
        if self.shadow.new_ground_clauses:
            self.shadow.ground_clauses |= self.shadow.new_ground_clauses
            self.shadow.new_ground_clauses = set()
        self._solve_ground_clauses()
        if self.shadow.satisfiable is False:
            self.info = '; '.join(['Theorem (SAT)'] + [str(x) for x in self.shadow.ground_clauses])
            self.is_terminal = True

    def _solve_ground_clauses(self) -> None:
        if not self.shadow.ground_clauses:
            self.shadow.model = {}
            self.shadow.satisfiable = True
            return

        result = self._solve_shadow_cnf([list(clause) for clause in self.shadow.ground_clauses])
        if result == 'UNSAT':
            self.shadow.model = {}
            self.shadow.satisfiable = False
        elif result == 'UNKNOWN':
            self.shadow.model = {}
            self.shadow.satisfiable = None
        else:
            self.shadow.model = {abs(lit): lit > 0 for lit in result if lit != 0}
            self.shadow.satisfiable = True

    def _require_sat_backend(self, backend: str) -> None:
        if backend == 'pycosat':
            if pycosat is None:
                raise ModuleNotFoundError(
                    'ClassicalSAT requires pycosat for --sat-solver pycosat. '
                    'Install it in ~/venv and run with that environment.'
                )
            return

        if backend == 'pysat':
            if PySATSolver is None:
                raise ModuleNotFoundError(
                    'ClassicalSAT requires python-sat for --sat-solver pysat. '
                    'Install it in ~/venv and run with that environment.'
                )
            return

        raise LookupError(f'Unknown SAT solver backend: {backend}')

    def _solve_shadow_cnf(self, clauses: list[list[int]]):
        if self.sat_solver_backend == 'pycosat':
            return pycosat.solve(clauses)

        solver = PySATSolver(bootstrap_with = clauses)
        try:
            satisfiable = solver.solve()
            if satisfiable is False:
                return 'UNSAT'
            if satisfiable is None:
                return 'UNKNOWN'
            return solver.get_model() or []
        finally:
            solver.delete()

    def _record_tableau_clauses(self) -> None:
        for clause in self._tableau_clauses():
            if clause not in self.shadow.ground_clauses:
                self.shadow.new_ground_clauses.add(clause)

    def _tableau_clauses(self) -> set[tuple[int, ...]]:
        clauses = set()
        stack = [self.tableau]
        while stack:
            node = stack.pop()
            if node.children:
                clauses.add(self.ground_clause([child.literal for child in node.children]))
            stack.extend(node.children)
        return clauses

    def _seed_ground_with_start_clauses(self) -> None:
        for clause_idx in self._start_clause_indices():
            clause = self.matrix.clauses[clause_idx]
            self.shadow.ground_clauses.add(self._ground_clause_without_substitution(clause))

    def _start_clause_indices(self) -> list[int]:
        clause_indices = list(self.matrix.positive_clauses)
        if not self.settings.positive_start_clauses:
            clause_indices += self.matrix.negative_clauses
        return clause_indices

    def _ground_clause_without_substitution(self, clause: list[Literal]) -> tuple[int, ...]:
        grounded = {self._ground_literal(lit) for lit in clause}
        return tuple(sorted(grounded, key = lambda lit: (abs(lit), lit < 0)))

    def _ground_literal(self, literal: Literal) -> int:
        atom = self._atom_key(literal)
        atom_id = self.shadow.atom_id(atom)
        return -atom_id if literal.neg else atom_id

    def _atom_key(self, literal: Literal) -> str:
        args = ','.join(self._term_key(arg) for arg in literal.args)
        return f'{literal.symbol}({args})' if args else literal.symbol

    def _term_key(self, term: Expression) -> str:
        if isinstance(term, Variable):
            return f'c_{term.symbol}_{term.copy_num}'
        if not isinstance(term, Expression) or not term.args:
            return str(term)
        args = ','.join(self._term_key(arg) for arg in term.args)
        return f'{term.symbol}({args})'

    def _is_ground(self, expression: Expression) -> bool:
        if isinstance(expression, Variable):
            return False
        return all(self._is_ground(arg) for arg in expression.args)

    def _new_tableau_child(self, literal: Literal):
        from connections.utils.tableau import Tableau

        return Tableau(literal, self.goal)

    def _shuffled_clause(self, clause: list[Literal]) -> list[Literal]:
        shuffled = list(clause)
        self.random.shuffle(shuffled)
        return shuffled

    def _shuffled_extension_clause(self, clause: list[Literal], lit_idx: int) -> tuple[list[Literal], int]:
        order = list(range(len(clause)))
        self.random.shuffle(order)
        shuffled = [clause[i] for i in order]
        return shuffled, order.index(lit_idx)
