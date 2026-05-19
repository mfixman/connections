from dataclasses import dataclass, field
from enum import auto
import random
from typing import Any, Literal as TypeLiteral, Optional, cast

from connections.Utils import UncaseEnum
from connections.utils.primitives import Expression, Literal, Matrix, Variable
from connections.utils.tableau import Tableau
from connections.utils.unification import Substitution

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
        return f"Extension {self.id}: {str(self.principle_node.literal)} -> {str(self.clause_copy)}{suffix}"

@dataclass
class Backtrack(ConnectionAction):
    steps: int = 1

    def __repr__(self) -> str:
        if self.steps == 1:
            return 'Backtrack'
        return f'Backtrack {self.steps}'

@dataclass
class Reset(ConnectionAction):
    depth: int = 1
    reason: str = ''

    def __repr__(self) -> str:
        return f'Reset {self.depth}'

class SATSolver(UncaseEnum):
    CaDiCaL = auto()

try:
    import pydical
except ModuleNotFoundError:
    pydical = None

SATClause = tuple[int, ...]

@dataclass
class ShadowSAT:
    atom_map: dict[str, int] = field(default_factory = dict)
    ground_clauses: set[SATClause] = field(default_factory = set)
    new_ground_clauses: set[SATClause] = field(default_factory = set)
    solver: Any = None
    solver_clauses: set[SATClause] = field(default_factory = set)
    expanded_clauses: dict[SATClause, list[str]] = field(default_factory = dict)
    unsat_core: list[SATClause] = field(default_factory = list)
    model: dict[int, bool] = field(default_factory = dict)
    satisfiable: bool = True

    def atom_id(self, atom: str) -> int:
        atom_id = self.atom_map.get(atom)
        if atom_id is None:
            atom_id = len(self.atom_map) + 1
            self.atom_map[atom] = atom_id
        return atom_id

class SATConnectionState:
    matrix: Matrix
    tableau: Tableau
    goal: Tableau
    substitution: Substitution
    max_depth: int
    info: Optional[str]
    is_terminal: bool
    proof_sequence: list[ConnectionAction]
    shadow: ShadowSAT
    sat_solver_backend: SATSolver

    def __init__(self, matrix: Matrix, settings) -> None:
        self.sat_solver_backend = settings.sat_solver
        self._require_sat_backend()
        self.random = random.Random(settings.random_seed)
        # Sect. 4.2 grounds every remaining tableau variable to the
        # same fresh constant.  Keeping variable identities distinct
        # makes the shadow CNF much less propositional than intended.
        self._ground_variable_constant = '__ground__'
        self.paper_mode = False
        self.matrix = matrix
        self.settings = settings
        self.init()

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

    def backtracks(self) -> list[Backtrack]:
        if self.goal.parent is None:
            return []
        return [Backtrack(id = 'bt')]

    def resets(self) -> list[Reset]:
        return [Reset(id = 'rs', depth = self.max_depth + 1)]

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

    def control_actions(self) -> list[ConnectionAction]:
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

        return {action.id: action for action in actions}

    def legal_actions(self) -> dict[str, ConnectionAction]:
        if self.goal.parent is None:
            actions = cast(list[ConnectionAction], self.starts() + self.resets())
            return {action.id: action for action in actions}

        back = cast(list[ConnectionAction], self.backtracks() + self.resets())
        return self.legal_forward_actions() | {action.id: action for action in back}

    def _refresh_remaining_actions(self, remaining_ids: list[str]) -> dict[str, ConnectionAction]:
        if self.goal.parent is None:
            fresh_actions = self.legal_actions()
        else:
            fresh_actions = self.legal_forward_actions()
        return {action_id: fresh_actions[action_id] for action_id in remaining_ids if action_id in fresh_actions}

    def enable_paper_mode(self, enabled: bool = True) -> None:
        self.paper_mode = enabled

    def initial_status(self) -> tuple[bool, Optional[str]]:
        if self.shadow.satisfiable:
            return False, None
        if not self.settings.debug:
            return True, 'Theorem (SAT)'

        clauses = self.shadow.unsat_core or self._sorted_shadow_clauses(self.shadow.ground_clauses)
        expanded = [self.shadow.expanded_clauses.get(clause, [str(lit) for lit in clause]) for clause in clauses]
        info = '; '.join(['Theorem (SAT)'] + ['(' + ', '.join(clause) + ')' for clause in expanded])
        return True, info

    def init(self) -> None:
        solver = None
        if self.sat_solver_backend == SATSolver.CaDiCaL:
            solver = self._new_cadical_solver()

        self.shadow = ShadowSAT(solver = solver)
        self._iteration_new_ground_clause = False
        self._seed_ground_with_start_clauses()
        self.reset(self.settings.iterative_deepening_initial_depth)

        if self.paper_mode:
            self._solve_ground_clauses()
            done, info = self.initial_status()
            if done:
                self.info = info
                self.is_terminal = True
            else:
                # Algorithm 2 only uses the SAT model after the first
                # completed iteration.
                self.shadow.model = {}
                self.shadow.satisfiable = True

    def reset(self, depth: int) -> None:
        self.max_depth = depth
        self.tableau = Tableau()
        self.goal = self.tableau
        self.substitution = Substitution()
        self.goal.actions = self.legal_actions()

        self.info = None
        self.is_terminal = False
        self.proof_sequence = []
        self._iteration_new_ground_clause = False

    def starts(self) -> list[Start]:
        starts: list[Start] = []
        for clause in self._start_clause_indices():
            starts.append(Start(id = "st" + str(len(starts)), clause_copy = self.matrix.copy(clause)))
        if not starts:
            starts.append(Start(id = "st0", clause_copy = None))

        for action in starts:
            if action.clause_copy is not None:
                self.random.shuffle(action.clause_copy)
        self.random.shuffle(starts)
        return starts

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

        for action in extensions:
            action.clause_copy, action.lit_idx = self._shuffled_extension_clause(action.clause_copy, action.lit_idx)
        self.random.shuffle(extensions)
        return extensions

    def _backtrack_once(self) -> None:
        if self.goal.parent is None:
            return

        while True:
            old_goal = self.goal
            self.goal = self.goal.find_prev()
            pop_frame = (self.goal is old_goal.parent)
            remaining_ids = list(self.goal.actions)

            if pop_frame and self.proof_sequence:
                self.proof_sequence.pop()

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

        self.finalize_pending_cuts()

        while True:
            if self.goal is None:
                if self.paper_mode:
                    self._restart_same_iteration()
                    return
                self.info = 'Theorem'
                self.is_terminal = True
                return

            if self.goal.parent is None:
                return

            if self.paper_mode and self._solved_by_model():
                self.goal.proven = True
                self.goal = self.goal.find_next()
                self.finalize_pending_cuts()
                continue

            return

    def backtrack(self, steps: int = 1) -> None:
        for _ in range(max(1, steps)):
            self._backtrack_once()
            if self.is_terminal or self.goal.parent is None:
                break

    def update_goal(self, action: ConnectionAction) -> None:
        if action.id in self.goal.actions:
            del self.goal.actions[action.id]
        self.goal.num_attempted += 1

        if isinstance(action, Start | Reduction | Extension):
            self.substitution.update(action.sub_updates)
            self.proof_sequence.append(action)

        match action:
            case Backtrack(steps = steps):
                self.backtrack(steps)
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

                if clause_copy == []:
                    # Degenerate start with an empty clause: there is no
                    # subtree to inspect, so the ordinary tableau proof is
                    # already complete. This occurs on translated $true/$false
                    # edge cases when an actual empty clause survives parsing.
                    self.info = 'Theorem'
                    self.is_terminal = True
                    return

                self.goal.children = [Tableau(lit, self.goal) for lit in clause_copy]

            case Extension(clause_copy = clause_copy, lit_idx = lit_idx, cut = cut):
                self.goal.children = [Tableau(lit, self.goal) for lit in clause_copy]
                self.goal.children[lit_idx].proven = True
                self.goal.children.insert(0, self.goal.children.pop(lit_idx))
                if cut:
                    self.goal.pending_cut = True
                if not getattr(action, 'skip_tableau_clause_recording', False):
                    self._record_tableau_clauses()
                    if self.paper_mode and self._sync_shadow_and_accept():
                        return

            case Reduction(cut = cut):
                self.goal.proven = True
                if cut:
                    self.goal.cut_committed = True
                self._record_tableau_clauses()
                if self.paper_mode and self._sync_shadow_and_accept():
                    return

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

    def _refresh_goal(self) -> None:
        self.finalize_pending_cuts()

        while True:
            if self.goal is None:
                if self.paper_mode:
                    self._restart_same_iteration()
                    return
                self.info = 'Theorem'
                self.is_terminal = True
                return

            if self.goal.parent is None:
                self.goal.actions = self.legal_actions()
                return

            if self.paper_mode and self._solved_by_model():
                self.goal.proven = True
                self.goal = self.goal.find_next()
                self.finalize_pending_cuts()
                continue

            self.goal.actions = self.legal_forward_actions()
            return

    def _finish_iteration(self) -> None:
        self._sync_shadow_and_accept()

    def _sync_shadow_and_accept(self) -> bool:
        self._flush_new_ground_clauses()
        self._solve_ground_clauses()
        done, info = self.initial_status()
        if done:
            self.info = info
            self.is_terminal = True
        return done

    def _solve_ground_clauses(self) -> None:
        if not self.shadow.ground_clauses:
            self.shadow.unsat_core = []
            self.shadow.model = {}
            self.shadow.satisfiable = True
            return

        result = self._solve_current_shadow_cnf_with_pydical()
        if result == 'UNSAT':
            ground_clauses = self._sorted_shadow_clauses(self.shadow.ground_clauses)
            if getattr(getattr(self, 'settings', None), 'debug', False):
                # In debug mode, print an irreducible UNSAT subset instead of
                # the full grounded shadow CNF.  This keeps theorem witnesses
                # readable without changing the underlying proof search.
                self.shadow.unsat_core = self._debug_unsat_subset(ground_clauses) or ground_clauses
            else:
                self.shadow.unsat_core = []
            self.shadow.model = {}
            self.shadow.satisfiable = False
        elif result == 'UNKNOWN':
            self.shadow.unsat_core = []
            self.shadow.model = {}
            self.shadow.satisfiable = True
        else:
            self.shadow.unsat_core = []
            self.shadow.model = {abs(lit): lit > 0 for lit in result if lit != 0}
            self.shadow.satisfiable = True

    def _require_sat_backend(self) -> None:
        if pydical is None:
            raise ModuleNotFoundError(
                'ClassicalSAT requires pydical. '
                'Install the project dependencies so the pinned pydical fork is available.'
            )

    def _solve_current_shadow_cnf_with_pydical(self) -> list[int] | TypeLiteral['UNSAT', 'UNKNOWN']:
        solver = self._cadical_solver()
        for clause in self._sorted_shadow_clauses(self.shadow.ground_clauses):
            self._add_clause_to_cadical_solver(clause)
        return self._read_pydical_result(solver)

    def _solve_clause_set_with_pydical(self, clauses: list[SATClause]) -> list[int] | TypeLiteral['UNSAT', 'UNKNOWN']:
        solver = self._new_cadical_solver()
        for clause in clauses:
            solver.add_clause(list(clause))
        return self._read_pydical_result(solver)

    def _debug_unsat_subset(self, clauses: list[SATClause]) -> list[SATClause]:
        if not clauses:
            return []

        if self._solve_clause_set_with_pydical(clauses) != 'UNSAT':
            return []

        kept = list(clauses)
        idx = 0
        while idx < len(kept):
            trial = kept[:idx] + kept[idx + 1:]
            if trial and self._solve_clause_set_with_pydical(trial) == 'UNSAT':
                kept = trial
                continue
            idx += 1
        return kept

    def _read_pydical_result(self, solver) -> list[int] | TypeLiteral['UNSAT', 'UNKNOWN']:
        status = solver.solve()
        if status == pydical.UNSATISFIABLE:
            return 'UNSAT'
        if status == pydical.UNSOLVED:
            return 'UNKNOWN'
        if status != pydical.SATISFIABLE:
            return 'UNKNOWN'

        return [solver.val(var) for var in range(1, solver.vars + 1)]

    def _flush_new_ground_clauses(self) -> None:
        if not self.shadow.new_ground_clauses:
            return

        self._iteration_new_ground_clause = True
        for clause in self._sorted_shadow_clauses(self.shadow.new_ground_clauses):
            self._add_ground_clause(clause)
        self.shadow.new_ground_clauses = set()

    def _add_ground_clause(self, clause: SATClause) -> None:
        self.shadow.ground_clauses.add(clause)
        self._add_clause_to_cadical_solver(clause)

    def _add_clause_to_cadical_solver(self, clause: SATClause) -> None:
        if clause in self.shadow.solver_clauses:
            return

        self._cadical_solver().add_clause(list(clause))
        self.shadow.solver_clauses.add(clause)

    def _cadical_solver(self):
        if self.shadow.solver is None:
            self.shadow.solver = self._new_cadical_solver()
        return self.shadow.solver

    def _new_cadical_solver(self):
        solver = pydical.Solver()
        if hasattr(solver, 'set'):
            solver.set('quiet', 1)
            solver.set('stabilizeonly', 1)
            solver.set('walk', 0)
            solver.set('lucky', 0)
            solver.set('luckyearly', 0)
            solver.set('luckylate', 0)
            solver.set('luckyassumptions', 0)
        return solver

    def _record_tableau_clauses(self) -> None:
        for clause, expanded in self._tableau_clauses().items():
            self.shadow.expanded_clauses.setdefault(clause, expanded)
            if clause not in self.shadow.ground_clauses:
                self._iteration_new_ground_clause = True
                self.shadow.new_ground_clauses.add(clause)

    def iteration_discovered_ground_clause(self) -> bool:
        return bool(
            self._iteration_new_ground_clause
            or self.shadow.new_ground_clauses
        )

    def _sorted_shadow_clauses(self, clauses: set[SATClause]) -> list[SATClause]:
        return sorted(clauses, key = lambda clause: (-len(clause), clause))

    def _tableau_clauses(self) -> dict[SATClause, list[str]]:
        clauses = {}
        stack = [self.tableau]
        while stack:
            node = stack.pop()
            if node.children:
                clause, expanded = self._ground_clause_data(
                    [child.literal for child in node.children],
                    apply_substitution = True,
                )
                clauses.setdefault(clause, expanded)
            stack.extend(node.children)
        return clauses

    def _seed_ground_with_start_clauses(self) -> None:
        for clause_idx in self._start_clause_indices():
            clause = self.matrix.clauses[clause_idx]
            grounded_clause, expanded_clause = self._ground_clause_data(clause, apply_substitution = False)
            self._add_ground_clause(grounded_clause)
            self.shadow.expanded_clauses[grounded_clause] = expanded_clause

    def _restart_same_iteration(self) -> None:
        remaining_actions = self.tableau.actions.copy()
        self.tableau = Tableau()
        self.tableau.actions = remaining_actions
        self.goal = self.tableau
        self.substitution = Substitution()
        self.proof_sequence = []

    def _solved_by_model(self) -> bool:
        literals = self.goal.path() + [self.goal.literal]
        return any(self.sat_literal_value(literal) is False for literal in literals)

    def _start_clause_indices(self) -> list[int]:
        if self.settings.rust_start_clauses and self.matrix.start_clauses is not None:
            return list(self.matrix.start_clauses)

        clause_indices = list(self.matrix.positive_clauses)
        if not self.settings.positive_start_clauses:
            clause_indices += self.matrix.negative_clauses
        return clause_indices

    def _ground_clause_data(
        self,
        clause: list[Literal],
        *,
        apply_substitution: bool,
    ) -> tuple[SATClause, list[str]]:
        grounded_pairs = {}
        for literal in clause:
            grounded_literal = self.substitution(literal) if apply_substitution else literal
            sat_literal = self._ground_literal(grounded_literal)
            grounded_pairs[sat_literal] = self._debug_literal(grounded_literal)

        ordered_pairs = tuple(sorted(grounded_pairs.items(), key = lambda item: (abs(item[0]), item[0] < 0)))
        return (
            tuple(sat_literal for sat_literal, _ in ordered_pairs),
            [debug_literal for _, debug_literal in ordered_pairs],
        )

    def _ground_literal(self, literal: Literal) -> int:
        atom = self._atom_key(literal)
        atom_id = self.shadow.atom_id(atom)
        return -atom_id if literal.neg else atom_id

    def _debug_literal(self, literal: Literal) -> str:
        args = ', '.join(self._debug_term(arg) for arg in literal.args)
        atom = f'{literal.symbol}({args})' if args else literal.symbol
        return f'-{atom}' if literal.neg else atom

    def _debug_term(self, term: Expression) -> str:
        if isinstance(term, Variable):
            return self._ground_variable_constant
        if not isinstance(term, Expression) or not term.args:
            return str(term)
        args = ', '.join(self._debug_term(arg) for arg in term.args)
        return f'{term.symbol}({args})'

    def _atom_key(self, literal: Literal) -> str:
        args = ','.join(self._term_key(arg) for arg in literal.args)
        return f'{literal.symbol}({args})' if args else literal.symbol

    def _term_key(self, term: Expression) -> str:
        if isinstance(term, Variable):
            return self._ground_variable_constant
        if not isinstance(term, Expression) or not term.args:
            return str(term)
        args = ','.join(self._term_key(arg) for arg in term.args)
        return f'{term.symbol}({args})'

    def _is_ground(self, expression: Expression) -> bool:
        if isinstance(expression, Variable):
            return False
        return all(self._is_ground(arg) for arg in expression.args)

    def _shuffled_extension_clause(self, clause: list[Literal], lit_idx: int) -> tuple[list[Literal], int]:
        order = list(range(len(clause)))
        self.random.shuffle(order)
        shuffled = [clause[i] for i in order]
        return shuffled, order.index(lit_idx)
