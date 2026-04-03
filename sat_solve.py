"""SAT-guided strategy using a PicoSAT-backed pycosat model.

This is intentionally a small controller-level sketch of the ideas from
Rawson and Reger (2021), not a full SATCoP reimplementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Optional

from connections.calculi.classicalcut import Backtrack, ConnectionAction, Extension, Reduction, Reset, Start
from connections.utils.primitives import Expression, Literal, Variable
from strategy_base import Strategy

try:
    import pycosat
except ModuleNotFoundError:
    pycosat = None

def _backtrack_or_reset(actions: list[ConnectionAction]) -> ConnectionAction:
    for action in actions:
        if isinstance(action, Backtrack):
            return action
    for action in actions:
        if isinstance(action, Reset):
            return action
    raise LookupError('expected Backtrack or Reset action')

def _forward_actions(actions: list[ConnectionAction]) -> list[ConnectionAction]:
    return [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

@dataclass
class _ShadowSAT:
    atom_map: dict[str, int] = field(default_factory = dict)
    clauses: set[tuple[int, ...]] = field(default_factory = set)
    model: dict[int, bool] = field(default_factory = dict)
    satisfiable: Optional[bool] = None
    dirty: bool = False

    def add_clause(self, clause: tuple[int, ...]) -> bool:
        if not clause or clause in self.clauses:
            return False
        self.clauses.add(clause)
        self.dirty = True
        return True

    def atom_id(self, atom: str) -> int:
        atom_id = self.atom_map.get(atom)
        if atom_id is None:
            atom_id = len(self.atom_map) + 1
            self.atom_map[atom] = atom_id
        return atom_id

    def literal_value(self, lit: int) -> Optional[bool]:
        self.solve()
        value = self.model.get(abs(lit))
        if value is None:
            return None
        return value if lit > 0 else (not value)

    def solve(self) -> None:
        if not self.dirty:
            return
        if not self.clauses:
            self.model = {}
            self.satisfiable = True
            self.dirty = False
            return

        result = pycosat.solve([list(clause) for clause in self.clauses])
        if result == 'UNSAT':
            self.model = {}
            self.satisfiable = False
        elif result == 'UNKNOWN':
            self.model = {}
            self.satisfiable = None
        else:
            self.model = {abs(lit): lit > 0 for lit in result if lit != 0}
            self.satisfiable = True
        self.dirty = False

class SATSolve(Strategy):
    """Use a grounded shadow clause set and a SAT model to steer search."""

    def __init__(self, env, seed: int = 0):
        if pycosat is None:
            raise ModuleNotFoundError(
                'SATSolve requires pycosat (PicoSAT wrapper). '
                'Install it in ~/venv and run with that environment.'
            )
        super().__init__(env)
        self.random = random.Random(seed)
        self.shadow = _ShadowSAT()
        self.new_clauses = 0
        self.last_depth = getattr(self.env.state, 'max_depth', 0)

    def step(self):
        self._sync_shadow()
        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        if not forward:
            return self._dead_end_action(actions)
        if len(forward) == 1:
            return forward[0]

        self.random.shuffle(forward)
        return max(forward, key = self._score)

    def _dead_end_action(self, actions: list[ConnectionAction]) -> ConnectionAction:
        reset = next((action for action in actions if isinstance(action, Reset)), None)
        if reset is not None and self.new_clauses > 0:
            self.new_clauses = 0
            return Reset(id = 'rs', depth = self.env.state.max_depth, reason = 'ground shadow still growing')
        return _backtrack_or_reset(actions)

    def _score(self, action: ConnectionAction) -> tuple[int, ...]:
        if isinstance(action, Reduction):
            truth = self._literal_truth(action.principle_node.literal)
            truth_rank = 2 if truth is True else 1 if truth is None else 0
            return (2, truth_rank)

        if isinstance(action, Start):
            clause = action.clause_copy
            residual = clause
            kind_rank = 0
        else:
            clause = action.clause_copy
            residual = [lit for i, lit in enumerate(clause) if i != action.lit_idx]
            kind_rank = 1

        grounded_clause = self._ground_clause(clause)
        novelty_rank = int(grounded_clause not in self.shadow.clauses)
        true_count = 0
        false_count = 0
        for lit in residual:
            truth = self._literal_truth(lit)
            if truth is True:
                true_count += 1
            elif truth is False:
                false_count += 1
        return (kind_rank, novelty_rank, true_count, -false_count, -len(residual))

    def _sync_shadow(self) -> None:
        depth = getattr(self.env.state, 'max_depth', self.last_depth)
        if depth != self.last_depth:
            self.new_clauses = 0
            self.last_depth = depth

        for clause in self._tableau_clauses():
            if self.shadow.add_clause(clause):
                self.new_clauses += 1

    def _tableau_clauses(self) -> set[tuple[int, ...]]:
        tableau = getattr(self.env.state, 'tableau', None)
        if tableau is None:
            return set()

        clauses = set()
        stack = [tableau]
        while stack:
            node = stack.pop()
            if node.children:
                clauses.add(self._ground_clause([child.literal for child in node.children]))
            stack.extend(node.children)
        return clauses

    def _literal_truth(self, literal: Literal) -> Optional[bool]:
        return self.shadow.literal_value(self._ground_literal(literal))

    def _ground_clause(self, clause: list[Literal]) -> tuple[int, ...]:
        return tuple(sorted({self._ground_literal(lit) for lit in clause}, key = lambda lit: (abs(lit), lit < 0)))

    def _ground_literal(self, literal: Literal) -> int:
        grounded = self.env.state.substitution(literal)
        atom = self._atom_key(grounded)
        atom_id = self.shadow.atom_id(atom)
        return -atom_id if grounded.neg else atom_id

    def _atom_key(self, literal: Literal) -> str:
        args = ','.join(self._term_key(arg) for arg in literal.args)
        return f'{literal.symbol}({args})' if args else literal.symbol

    def _term_key(self, term: Expression) -> str:
        if isinstance(term, Variable):
            return '*'
        if not isinstance(term, Expression) or not term.args:
            return str(term)
        args = ','.join(self._term_key(arg) for arg in term.args)
        return f'{term.symbol}({args})'
