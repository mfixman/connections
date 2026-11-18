# SAT policies that prefer the clauses an axiom predictor (or SInE) scores highly.

from __future__ import annotations

import hashlib
import math

from connections.agent.sat import SATCoPCon, SATResetCoP
from connections.environment.actions import Action, ApplyAction
from connections.environment.rules import Extension, Start
from connections.environment.state import State
from connections.syntax.matrix import Matrix

def matrix_digest(matrix: Matrix) -> str:
    h = hashlib.sha256()
    for clause in matrix.clauses:
        h.update(str(clause).encode("utf-8"))
        h.update(b"\n")

    return h.hexdigest()

# Scores axioms with the predictor, or SInE without one; conjectures weigh 1.
def clause_weights(matrix, axiom_ids, conjecture_ids, predictor = None) -> dict[int, float]:
    if predictor is None:
        from .sine import score_matrix

        weights = dict(zip(axiom_ids, score_matrix(matrix, axiom_ids, conjecture_ids), strict = True))
    else:
        predictions = predictor.predict(
            matrix,
            axiom_clause_ids = axiom_ids,
            conjecture_clause_ids = conjecture_ids,
        )
        weights = {p.clause_index: p.probability for p in predictions}

    weights.update({i: 1.0 for i in conjecture_ids})
    return weights

# Orders the prover's alternatives by a Gumbel-perturbed clause weight, a weighted random permutation.
class AxiomGuided:
    def __init__(self, *, clause_weights: dict[int, float], temperature = 1.0, matrix_digest = None, **kwargs):
        min_weight = 1e-6

        super().__init__(**kwargs)
        if self._random is None:
            raise ValueError("weighted guidance needs a seed")

        self.temperature = temperature
        self.weights = {int(i): max(min_weight, float(w)) for i, w in clause_weights.items()}
        logs = list(map(math.log, self.weights.values()))
        self.neutral = math.exp(sum(logs) / len(logs)) if logs else 1.0

        self.digest = matrix_digest
        self.checked = False

    def __call__(self, state: State):
        if state is not self._episode:
            self.checked = False

        if not self.checked:
            self.check_matrix(state)

        return super().__call__(state)

    def check_matrix(self, state: State):
        if self.digest is not None and matrix_digest(state.matrix) != self.digest:
            raise RuntimeError("the prover's matrix differs from the matrix the axiom predictor scored")

        self.checked = True

    # Actions without a clause, such as reductions, get the geometric mean weight.
    def weight(self, action: Action) -> float:
        i = clause_index(action)
        return self.neutral if i is None else self.weights.get(i, self.neutral)

    def _tie_break_key(self, state: State, action: Action, index: int):
        min_uniform = 1e-300
        u = max(min_uniform, self._random.random())
        return (self.temperature * math.log(-math.log(u)) - math.log(self.weight(action)),)

def clause_index(action: Action) -> int | None:
    if isinstance(action, ApplyAction) and isinstance(action.rule, (Start, Extension)):
        return action.rule.clause_idx

    return None

class AxiomGuidedSATResetCoP(AxiomGuided, SATResetCoP):
    pass

class AxiomGuidedSATCoP(AxiomGuided, SATCoPCon):
    pass
