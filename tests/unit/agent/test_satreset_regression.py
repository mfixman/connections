"""Trace baselines captured at 56c9514, before separating the SATCoP policies."""

import hashlib
import json
from pathlib import Path
import random

import pytest

from connections.agent.sat import SATResetCoP
from connections.environment.dynamics import Dynamics
from connections.environment.state import State
from connections.environment.tableau import Tableau
from connections.interaction.records import action_record
from connections.syntax.formula import Atom, Function, Variable
from connections.syntax.matrix import Clause, Literal, Matrix

def matrices():
    rng = random.Random(1847)
    terms = [Variable("X"), Function("a"), Function("b"), Function("f", (Variable("X"),))]
    for _ in range(24):
        clauses = []
        for index in range(5):
            literals = tuple(
                Literal(Atom(rng.choice("pqr"), (rng.choice(terms),)), polarity = rng.choice((True, False)))
                for _ in range(rng.randint(1, 3))
            )
            clauses.append(Clause(literals, role = "conjecture" if index == 4 else "axiom"))

        yield Matrix(tuple(clauses))

def trace_digest(matrix, mode, seed):
    if mode == "base":
        policy = SATResetCoP(seed = seed, debug_sat_core = True)
    else:
        from axiom_prediction.guided import AxiomGuidedSATResetCoP
        policy = AxiomGuidedSATResetCoP(
            clause_weights = {i: (i + 1) / 6 for i in range(5)},
            mode = mode, seed = seed, debug_sat_core = True,
        )

    state = State(matrix, Tableau())
    trace = []
    for _ in range(100):
        action = policy(state)
        trace.append([
            None if action is None else action_record(action),
            policy.status.value, policy.depth_limit,
            sorted(policy._shadow.clauses), sorted(policy._shadow.model.items()),
            policy.diagnostics(),
        ])
        if action is None:
            break

        Dynamics.transition(state, action)

    return hashlib.sha256(json.dumps(trace, sort_keys = True).encode()).hexdigest()

@pytest.mark.parametrize("mode", ["base", "strict", "weighted"])
@pytest.mark.parametrize("seed", [0, 42])
def test_satreset_trace_unchanged(mode, seed):
    if mode != "base":
        pytest.importorskip("axiom_prediction")

    expected = json.loads(Path(__file__).with_name("satreset_traces.json").read_text())
    actual = [trace_digest(matrix, mode, seed) for matrix in matrices()]
    assert actual == expected[f"{mode}-{seed}"]
