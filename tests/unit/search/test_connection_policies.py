from __future__ import annotations

import pytest

from connections.policy import (
    LeanCoPCon,
    LeanCoPCon_2,
    SATCoPCon,
    SATResetCoP,
)
from connections.prover.actions import ApplyAction, UndoAction
from connections.prover.prover import Problem, Prover
from connections.prover.rules import Extension, Start
from connections.prover.state import State
from connections.prover.status import ProverOutcome
from connections.prover.tableau import Tableau
from connections.syntax.formula import Atom, Function, Variable
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
    policy._shadow.add_clause(
        (policy._shadow.atom_id("p"),), clause_idx=0, from_tableau=False
    )
    assert policy._shadow.solve() is True
    policy._guidance_model = dict(policy._shadow.model)
    positive = ApplyAction(
        state.tableau.root_goal_id,
        Extension(lit_idx=0, clause=Clause((_lit("p"),))),
    )
    negative = ApplyAction(
        state.tableau.root_goal_id,
        Extension(lit_idx=0, clause=Clause((_lit("p", positive=False),))),
    )

    assert policy._next_action(state, (negative, positive)) is positive


def test_satcop_breaks_score_ties_randomly_unless_seed_is_none():
    clauses = tuple(Clause((_lit(f"p{index}"),)) for index in range(8))
    state = _state(*clauses)
    starts = tuple(
        ApplyAction(state.tableau.root_goal_id, rule)
        for rule in _start_rules(state)
    )

    seeded = SATCoPCon(seed=0)
    chosen = {seeded._next_action(state, starts) for _ in range(20)}
    assert len(chosen) > 1
    assert [SATCoPCon(seed=3)._next_action(state, starts) for _ in range(5)] == [
        SATCoPCon(seed=3)._next_action(state, starts) for _ in range(5)
    ]
    assert SATCoPCon(seed=None)._next_action(state, starts) is starts[0]


def test_sat_core_uses_source_selectors_and_excludes_irrelevant_clause():
    shadow = SATCoPCon(debug_sat_core=True)._shadow
    p = shadow.atom_id("p")
    selector = shadow.selector_id(7)
    assert selector != p
    shadow.add_clause((p,), clause_idx=2, from_tableau=False)
    shadow.add_clause((-p,), clause_idx=3, from_tableau=False)
    shadow.add_clause((shadow.atom_id("q"),), clause_idx=4, from_tableau=False)

    assert shadow.solve() is False
    assert shadow.sat_core_clause_ids == (2, 3)


def test_sat_core_preserves_repeated_groundings_for_one_source_clause():
    shadow = SATCoPCon(debug_sat_core=True)._shadow
    shadow.add_clause((shadow.atom_id("p"),), clause_idx=2, from_tableau=False)
    shadow.add_clause((shadow.atom_id("q"),), clause_idx=2, from_tableau=True)

    assert len(shadow.clauses) == 2
    assert len(shadow.selector_ids) == 1


@pytest.mark.parametrize("policy_class", [SATCoPCon, SATResetCoP])
def test_sat_policies_report_source_clause_core(policy_class):
    positive = Clause((_lit("p"),))
    negative = Clause((_lit("p", positive=False),), role="conjecture")
    state = State(
        Problem(Matrix((positive, negative)), start_clauses="all"),
        Tableau(),
    )
    policy = policy_class(debug_sat_core=True)

    assert policy(state) is ProverOutcome.PROVED
    diagnostics = policy.diagnostics()
    assert diagnostics["sat_core_clause_ids"] == [0, 1]
    assert diagnostics["sat_core"] == [["p"], ["~p"]]


def test_sat_shadow_rejects_missing_source_clause_provenance():
    positive = Clause((_lit("p"),))
    negative = Clause((_lit("p", positive=False),), role="conjecture")
    state = _state(positive, negative)
    state.apply_rule(
        parent_goal_id=state.tableau.root_goal_id,
        rule=Start(positive, clause_idx=0, instance_id=state.fresh_instance_id()),
    )
    state.apply_rule(
        parent_goal_id=state.fringe[0].goal_id,
        rule=Extension(
            lit_idx=0,
            clause=negative,
            instance_id=state.fresh_instance_id(),
        ),
    )

    with pytest.raises(RuntimeError, match="source-clause provenance"):
        SATCoPCon()(state)


def test_satreset_replaces_nested_backtrack_with_root_reset():
    start = Clause((_lit("p"), _lit("q")))
    extension = Clause((_lit("p", positive=False), _lit("r")), role="conjecture")
    state = _state(start, extension)
    policy = SATResetCoP(initial_depth=2, seed=None)

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


def _atom_literal(symbol: str, argument, *, positive: bool = True) -> Literal:
    return Literal(Atom(symbol, (argument,)), polarity=positive)


def test_sat_guidance_model_waits_for_the_next_iteration():
    state = _state(Clause((_lit("p"),)), Clause((_lit("p", positive=False), _lit("q"))))
    policy = SATCoPCon()
    policy._shadow.add_clause((policy._shadow.atom_id("p"),), clause_idx=0, from_tableau=False)
    assert policy._shadow.solve() is True

    policy._start_next_depth()
    assert policy._literal_value(state, _lit("p"), instance_id=None) is None
    policy._start_next_depth()
    assert policy._literal_value(state, _lit("p"), instance_id=None) is True
    assert policy._literal_value(state, _lit("p", positive=False), instance_id=None) is False


def test_sat_guidance_only_values_ground_literals():
    state = _state(Clause((_lit("p"),)))
    policy = SATCoPCon()
    for key in ("p(__ground__)", "p(a)"):
        policy._shadow.add_clause((policy._shadow.atom_id(key),), clause_idx=0, from_tableau=False)
    assert policy._shadow.solve() is True
    policy._guidance_model = dict(policy._shadow.model)

    ground = _atom_literal("p", Function("a"))
    open_literal = _atom_literal("p", Variable("X"))
    assert policy._literal_value(state, ground, instance_id=1) is True
    assert policy._literal_value(state, open_literal, instance_id=1) is None


def test_satreset_resets_instead_of_stopping_when_start_clauses_are_exhausted():
    # Both start clauses close the tableau, so the search exhausts its start
    # clauses before the shadow sees p(a) and ~p(a) together.
    positive = Clause((_atom_literal("p", Variable("X")),))
    negative = Clause((_atom_literal("p", Function("a"), positive=False),))
    state = State(Problem(Matrix((positive, negative)), start_clauses="all"), Tableau())

    _steps, _inferences, outcome = Prover()._run_strategy_loop(
        state,
        policy=SATResetCoP(),
        step_limit=200,
    )

    assert outcome is ProverOutcome.PROVED


def test_sat_policies_do_not_extend_with_ground_clauses_beyond_the_depth_limit():
    assert SATCoPCon().ground_extensions_beyond_depth is False
    assert LeanCoPCon().ground_extensions_beyond_depth is True


def test_deep_axiom_conjunctions_clausify(tmp_path):
    from connections.clausification import matrix_from_file

    problem = tmp_path / "deep.p"
    axioms = "".join(f"fof(a{index}, axiom, p{index}).\n" for index in range(3000))
    problem.write_text(axioms + "fof(c, conjecture, p0).\n", encoding="utf-8")

    matrix = matrix_from_file(problem, start_clauses="all")

    assert len(matrix.clauses) == 3001


def test_sat_shadow_seed_clauses_can_be_restricted():
    class Restricted(SATCoPCon):
        def _shadow_seed_clause_ids(self, state):
            return (0,)

    state = State(
        Problem(Matrix((Clause((_lit("p"),)), Clause((_lit("p", positive=False),)))), start_clauses="all"),
        Tableau(),
    )
    policy = Restricted()
    policy._observe_shadow(state)

    assert {clause_idx for clause_idx, _ in policy._shadow.clauses} == {0}
    assert policy._shadow.solve() is True
