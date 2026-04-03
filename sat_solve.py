"""SAT-guided proof-search strategy.

This is a controller-level sketch inspired by Rawson and Reger (2021),
"Eliminating Models During Model Elimination".

Important limitation:
    The paper's full SATCoP design adds new solver-side capabilities to
    the prover itself:

    - clauses from the current tableau are grounded and persisted
    - propositional UNSAT can terminate the proof search
    - model-based lemmata can skip goals directly

    A ``Strategy`` can only *choose existing actions* from the current
    environment. It cannot add new inference rules or new success
    conditions. This file therefore implements the part that *can* live
    at strategy level:

    - maintain a cumulative grounded "shadow" of clauses seen so far
    - build a lightweight propositional model approximation
    - use that approximation to rank forward actions
    - use same-depth resets while the current depth is still producing
      new ground clauses, then deepen when learning stalls

The resulting behaviour should be read as a practical bridge between the
paper's control ideas and the current project architecture, not as a
complete reimplementation of SATCoP.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import random
from typing import Optional

from connections.calculi.classicalcut import Backtrack, ConnectionAction, Extension, Reduction, Reset, Start
from connections.utils.primitives import Constant, Expression, Literal, Variable

from strategy_base import Strategy

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

def _base_action_score(action: ConnectionAction) -> float:
    if isinstance(action, Reduction):
        return 2.0
    if isinstance(action, (Extension, Start)):
        return 1.0 / max(len(action.clause_copy), 1)
    return 0.0

@dataclass
class _ClauseShadow:
    """Incremental ground-clause store with a bounded SAT approximation.

    Exact SAT solving would be better, but that requires either a solver
    dependency or environment-level support. For a strategy-level sketch
    we instead maintain:

    - a cumulative set of grounded clauses
    - a bounded DPLL check on small shadows
    - otherwise a greedy partial model used only for scoring
    """

    atom_map: dict[str, int] = field(default_factory = dict)
    next_atom_id: int = 1
    clauses: set[tuple[int, ...]] = field(default_factory = set)
    model: dict[int, bool] = field(default_factory = dict)
    satisfiable: Optional[bool] = None
    dirty: bool = False

    max_exact_atoms: int = 18
    max_dpll_nodes: int = 2_000

    def add_clause(self, clause: tuple[int, ...]) -> bool:
        if not clause or clause in self.clauses:
            return False
        self.clauses.add(clause)
        self.dirty = True
        return True

    def atom_id(self, atom_key: str) -> int:
        existing = self.atom_map.get(atom_key)
        if existing is not None:
            return existing
        atom_id = self.next_atom_id
        self.atom_map[atom_key] = atom_id
        self.next_atom_id += 1
        return atom_id

    def literal_value(self, sat_lit: int) -> Optional[bool]:
        self.ensure_model()
        assignment = self.model.get(abs(sat_lit))
        if assignment is None:
            return None
        return assignment if sat_lit > 0 else (not assignment)

    def ensure_model(self) -> None:
        if not self.dirty:
            return

        clauses = [list(clause) for clause in self.clauses]
        atoms = {abs(lit) for clause in clauses for lit in clause}

        if len(atoms) <= self.max_exact_atoms:
            status, model = self._bounded_dpll(clauses, {})
            self.satisfiable = status
            self.model = self._greedy_completion(clauses, model)
        else:
            self.satisfiable = None
            self.model = self._greedy_completion(clauses, {})

        self.dirty = False

    def _bounded_dpll(
        self,
        clauses: list[list[int]],
        assignment: dict[int, bool],
        budget: Optional[list[int]] = None,
    ) -> tuple[Optional[bool], dict[int, bool]]:
        if budget is None:
            budget = [self.max_dpll_nodes]

        simplified, propagated = self._unit_propagate(clauses, assignment)
        if simplified is None:
            return False, propagated
        if not simplified:
            return True, propagated
        if budget[0] <= 0:
            return None, propagated
        budget[0] -= 1

        counter = Counter(abs(lit) for clause in simplified for lit in clause)
        atom = max(counter, key = counter.get)

        positive_first = sum(1 for clause in simplified for lit in clause if lit == atom)
        branch_values = (True, False) if positive_first >= counter[atom] / 2 else (False, True)
        unknown_model = propagated

        for value in branch_values:
            status, model = self._bounded_dpll(simplified, propagated | {atom: value}, budget)
            if status is True:
                return True, model
            if status is None:
                unknown_model = model

        if budget[0] <= 0:
            return None, unknown_model
        return False, propagated

    def _unit_propagate(
        self,
        clauses: list[list[int]],
        assignment: dict[int, bool],
    ) -> tuple[Optional[list[list[int]]], dict[int, bool]]:
        current = [list(clause) for clause in clauses]
        model = dict(assignment)

        while True:
            changed = False
            reduced: list[list[int]] = []

            for clause in current:
                satisfied = False
                remaining: list[int] = []
                for lit in clause:
                    value = model.get(abs(lit))
                    if value is None:
                        remaining.append(lit)
                        continue
                    lit_value = value if lit > 0 else (not value)
                    if lit_value:
                        satisfied = True
                        break
                if satisfied:
                    continue
                if not remaining:
                    return None, model
                reduced.append(remaining)

            current = reduced
            units = [clause[0] for clause in current if len(clause) == 1]
            if not units:
                return current, model

            for unit in units:
                atom = abs(unit)
                value = unit > 0
                existing = model.get(atom)
                if existing is not None and existing != value:
                    return None, model
                if existing is None:
                    model[atom] = value
                    changed = True

            if not changed:
                return current, model

    def _greedy_completion(
        self,
        clauses: list[list[int]],
        assignment: dict[int, bool],
    ) -> dict[int, bool]:
        model = dict(assignment)
        for clause in clauses:
            satisfied = False
            first_unassigned: Optional[int] = None
            for lit in clause:
                value = model.get(abs(lit))
                if value is None and first_unassigned is None:
                    first_unassigned = lit
                    continue
                if value is not None and (value if lit > 0 else (not value)):
                    satisfied = True
                    break
            if not satisfied and first_unassigned is not None:
                model[abs(first_unassigned)] = first_unassigned > 0
        return model

class SATSolve(Strategy):
    """SAT-guided control policy for ``ClassicalCut``-style environments.

    Paper basis:
        Rawson and Reger (2021), "Eliminating Models During Model
        Elimination": grounding clauses seen during connection-tableau
        search, reusing that information propositionally, randomising
        local search order, and staying at the same depth while new
        ground clauses are still being learned.

    What this strategy actually implements:
        - persistent ground shadow clauses extracted from the current
          tableau under the current substitution
        - action scoring using a lightweight propositional model
        - pseudo-random tie-breaking among similarly-ranked actions
        - same-depth resets once a local search pass appears to have
          learned something new, and depth increase only once learning
          stalls

    What it cannot implement without calculus support:
        - SAT-based theorem reporting
        - literal skipping / model-based lemmata as a true inference
          shortcut
    """

    # Reset only after a short burst of local dead ends; before that we
    # still want ordinary backtracking to exhaust nearby alternatives.
    dead_end_patience: int = 6
    novelty_bonus: float = 0.35
    goal_true_bonus: float = 0.40
    goal_false_penalty: float = 0.25
    residual_true_bonus: float = 0.18
    residual_false_penalty: float = 0.08

    def __init__(self, env, seed: int = 0):
        super().__init__(env)
        self.random = random.Random(seed)
        self.shadow = _ClauseShadow()
        self.dead_ends_since_restart = 0
        self.new_clauses_since_restart = 0
        self.last_depth = getattr(self.env.state, 'max_depth', 0)

    def step(self):
        self._sync_shadow()

        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        if not forward:
            return self._handle_dead_end(actions)

        self.dead_ends_since_restart = 0
        return self._choose_forward(forward)

    def _handle_dead_end(self, actions: list[ConnectionAction]) -> ConnectionAction:
        self.dead_ends_since_restart += 1

        if self.dead_ends_since_restart < self.dead_end_patience:
            return _backtrack_or_reset(actions)

        state = self.env.state
        self.dead_ends_since_restart = 0

        # Rough analogue of SATCoP's depth-control loop:
        # if this pass still learned something new, restart at the same
        # depth and explore a different part of the search space;
        # otherwise deepen.
        if self.new_clauses_since_restart > 0:
            self.new_clauses_since_restart = 0
            return Reset(
                id = 'rs',
                depth = state.max_depth,
                reason = 'ground shadow still growing',
            )

        return Reset(
            id = 'rs',
            depth = state.max_depth + 1,
            reason = 'ground shadow stalled',
        )

    def _choose_forward(self, forward: list[ConnectionAction]) -> ConnectionAction:
        if len(forward) == 1:
            return forward[0]

        self.random.shuffle(forward)
        return max(forward, key = self._score_action)

    def _score_action(self, action: ConnectionAction) -> float:
        score = _base_action_score(action)

        if isinstance(action, Reduction):
            goal_value = self._literal_truth(action.principle_node.literal)
            if goal_value is True:
                score += self.goal_true_bonus
            elif goal_value is False:
                score -= self.goal_false_penalty
            return score

        if isinstance(action, Start):
            clause = action.clause_copy
            residual = clause
        elif isinstance(action, Extension):
            clause = action.clause_copy
            residual = [lit for i, lit in enumerate(clause) if i != action.lit_idx]
        else:
            return score

        grounded_clause = self._ground_clause(clause)
        if grounded_clause not in self.shadow.clauses:
            score += self.novelty_bonus

        for literal in residual:
            truth = self._literal_truth(literal)
            if truth is True:
                score += self.residual_true_bonus
            elif truth is False:
                score -= self.residual_false_penalty

        return score

    def _sync_shadow(self) -> None:
        state = self.env.state
        depth = getattr(state, 'max_depth', self.last_depth)
        if depth != self.last_depth:
            self.dead_ends_since_restart = 0
            self.new_clauses_since_restart = 0
            self.last_depth = depth

        for clause in self._tableau_clauses():
            if self.shadow.add_clause(clause):
                self.new_clauses_since_restart += 1

    def _tableau_clauses(self) -> set[tuple[int, ...]]:
        tableau = getattr(self.env.state, 'tableau', None)
        if tableau is None:
            return set()

        clauses: set[tuple[int, ...]] = set()
        stack = [tableau]
        while stack:
            node = stack.pop()
            if node.children:
                clause = self._ground_clause([child.literal for child in node.children])
                if clause:
                    clauses.add(clause)
            stack.extend(node.children)
        return clauses

    def _literal_truth(self, literal: Literal) -> Optional[bool]:
        sat_lit = self._ground_literal(literal)
        return self.shadow.literal_value(sat_lit)

    def _ground_clause(self, clause: list[Literal]) -> tuple[int, ...]:
        grounded = {self._ground_literal(literal) for literal in clause}
        return tuple(sorted(grounded, key = lambda lit: (abs(lit), lit < 0)))

    def _ground_literal(self, literal: Literal) -> int:
        substitution = self.env.state.substitution
        grounded = substitution(literal)
        atom_key = self._literal_atom_key(grounded)
        atom_id = self.shadow.atom_id(atom_key)
        return -atom_id if grounded.neg else atom_id

    def _literal_atom_key(self, literal: Literal) -> str:
        args = ','.join(self._term_key(arg) for arg in literal.args)
        return f'{literal.symbol}({args})' if args else literal.symbol

    def _term_key(self, term: Expression) -> str:
        if isinstance(term, Variable):
            # Simple grounding scheme from the paper: collapse remaining
            # variables to a placeholder constant.
            return Constant('*').symbol
        if not isinstance(term, Expression) or not term.args:
            return str(term)
        args = ','.join(self._term_key(arg) for arg in term.args)
        return f'{term.symbol}({args})'
