from __future__ import annotations

from .choices import GuidanceMode

from collections import Counter
from collections.abc import Sequence
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

class AxiomGuided:
    def __init__(
        self,
        *,
        clause_weights: dict[int, float],
        mode: str = GuidanceMode.Weighted,
        temperature: float = 1.0,

        allowed_clause_ids: list[int] | None = None,
        matrix_digest: str | None = None,
        **kwargs,
    ):
        min_weight = 1e-6

        guided_modes = (GuidanceMode.Weighted, GuidanceMode.Strict)

        super().__init__(**kwargs)
        mode = GuidanceMode(mode)
        if mode not in guided_modes:
            raise ValueError(f"unknown guided mode {mode!r}; choose from {', '.join(guided_modes)}")

        if self._random is None and mode == GuidanceMode.Weighted:
            raise ValueError("weighted mode needs a seed")

        self.mode = mode
        self.temperature = temperature
        self.weights = {
            int(i): max(min_weight, float(w))
            for i, w in clause_weights.items()
        }

        logs = list(map(math.log, self.weights.values()))
        self.neutral = math.exp(sum(logs) / len(logs)) if logs else 1.0

        self.allowed = None if allowed_clause_ids is None else frozenset(
            allowed_clause_ids
        )

        self.digest = matrix_digest
        self.checked = False

        self.starts_used: set[int] = set()
        self.starts_depth = None
        self.start_count = None
        self.clause_uses: Counter[int] = Counter()

    def __call__(self, state: State):
        if state is not self._episode:
            self.checked = False
            self.starts_used.clear()
            self.clause_uses.clear()
            self.starts_depth = None

        if not self.checked:
            self.check_matrix(state)

        return super().__call__(state)

    def check_matrix(self, state: State):
        if self.digest is not None and matrix_digest(state.matrix) != self.digest:
            raise RuntimeError(
                "the prover's matrix differs from the matrix the axiom predictor scored"
            )

        ids = super()._shadow_seed_clause_ids(state)
        self.start_count = len(
            ids if self.allowed is None else [i for i in ids if i in self.allowed]
        )

        self.checked = True

    def _shadow_seed_clause_ids(self, state: State) -> tuple[int, ...]:
        ids = super()._shadow_seed_clause_ids(state)
        return ids if self.allowed is None else tuple(
            i for i in ids if i in self.allowed
        )

    def _actions_for_goal(self, state: State, goal_id: int) -> tuple[Action, ...]:
        actions = super()._actions_for_goal(state, goal_id)
        if self.allowed is None:
            return actions

        return tuple(
            a for a in actions if (i := clause_index(a)) is None or i in self.allowed
        )

    def _start_next_depth(self):
        super()._start_next_depth()
        if self.starts_depth != self.depth_limit:
            self.clause_uses.clear()
            self.starts_depth = self.depth_limit

        if self.start_count is not None and len(self.starts_used) >= self.start_count:
            self.starts_used.clear()

    def weight(self, action: Action) -> float:
        i = clause_index(action)
        return self.neutral if i is None else self.weights.get(i, self.neutral)

    def _tie_break_key(self, state: State, action: Action, index: int):
        w = self.weight(action)
        if self.mode == GuidanceMode.Strict:
            i = clause_index(action)
            if i is None:
                repeats = 0
            elif isinstance(action, ApplyAction) and isinstance(action.rule, Start):
                repeats = int(i in self.starts_used)
            else:
                repeats = self.clause_uses[i]

            return (repeats, -w, index)

        assert self._random is not None
        min_uniform = 1e-300
        u = max(min_uniform, self._random.random())
        return (self.temperature * math.log(-math.log(u)) - math.log(w),)

    def _next_action(self, state: State, actions: Sequence[Action]) -> Action:
        action = super()._next_action(state, actions)
        i = clause_index(action)
        if i is not None:
            if isinstance(action, ApplyAction) and isinstance(action.rule, Start):
                self.starts_used.add(i)
            else:
                self.clause_uses[i] += 1

        return action

def clause_index(action: Action) -> int | None:
    if isinstance(action, ApplyAction) and isinstance(action.rule, (Start, Extension)):
        return action.rule.clause_idx

    return None

class AxiomGuidedSATResetCoP(AxiomGuided, SATResetCoP):
    pass

class AxiomGuidedSATCoP(AxiomGuided, SATCoPCon):
    pass
