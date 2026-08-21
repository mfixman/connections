from __future__ import annotations

from connections.policy import (
    LeanCoPCon,
    LeanCoPCon_2,
    SATCoPCon,
    SATResetCoP,
)
from connections.prover.actions import ApplyAction, UndoAction
from connections.prover.prover import Problem
from connections.prover.rules import Extension
from connections.prover.state import State
from connections.prover.status import ProverOutcome
from connections.prover.tableau import Tableau
from connections.syntax.formula import Atom
from connections.syntax.matrix import Clause, Literal, Matrix


def _lit(symbol: str, *, positive: bool = True) -> Literal:
    return Literal(Atom(symbol), polarity=positive)


def _state(*clauses: Clause) -> State:
    return State(
        Problem(Matrix(tuple(clauses)), start_clauses="positive"),
        Tableau(),
    )


def test_leancop_policy_defaults_match_named_implementations():
    leancop = LeanCoPCon()
    leancop2 = LeanCoPCon_2()

    assert leancop.cut_enabled is True
    assert leancop.comp == 7
    assert leancop.factorization == "equal"
    assert leancop2.cut_enabled is False
    assert leancop2.comp is None
    assert leancop2.factorization == "equal"


def test_satcop_shadow_unsat_returns_proved():
    positive = Clause((_lit("p"),))
    negative = Clause((_lit("p", positive=False),), role="conjecture")
    state = _state(positive, negative)
    state.apply_rule(
        parent_goal_id=state.tableau.root_goal_id,
        rule=next(iter(_start_rules(state))),
    )
    goal_id = state.fringe[0].goal_id
    state.apply_rule(
        parent_goal_id=goal_id,
        rule=Extension(
            lit_idx=0,
            clause=negative,
            clause_idx=1,
            instance_id=state.fresh_instance_id(),
        ),
    )

    assert SATCoPCon()(state) is ProverOutcome.PROVED


def test_satcop_prefers_head_made_true_by_shadow_model():
    state = _state(Clause((_lit("p"),)))
    policy = SATCoPCon()
    policy._shadow.add_clause((policy._shadow.atom_id("p"),), from_tableau=False)
    assert policy._shadow.solve() is True
    positive = ApplyAction(
        state.tableau.root_goal_id,
        Extension(lit_idx=0, clause=Clause((_lit("p"),))),
    )
    negative = ApplyAction(
        state.tableau.root_goal_id,
        Extension(lit_idx=0, clause=Clause((_lit("p", positive=False),))),
    )

    assert policy._next_action(state, (negative, positive)) is positive


def test_satreset_replaces_nested_backtrack_with_root_reset():
    start = Clause((_lit("p"), _lit("q")))
    extension = Clause((_lit("p", positive=False), _lit("r")), role="conjecture")
    state = _state(start, extension)
    policy = SATResetCoP()

    start_action = policy(state)
    assert isinstance(start_action, ApplyAction)
    state.apply_rule(
        parent_goal_id=start_action.goal_id,
        rule=start_action.rule,
    )
    root_application_id = state.tableau.root.applied_rule_application_id
    assert root_application_id is not None

    extension_action = policy(state)
    assert isinstance(extension_action, ApplyAction)
    state.apply_rule(
        parent_goal_id=extension_action.goal_id,
        rule=extension_action.rule,
    )

    reset = policy(state)

    assert reset == UndoAction(root_application_id)


def _start_rules(state: State):
    from connections.prover.dynamics import Dynamics

    return Dynamics.start_rules_for(state, state.tableau.root_goal_id)
