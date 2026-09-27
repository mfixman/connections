import pytest
from connections.agent import AgentOptions, AgentStatus
from connections.agent.sat import SATCoPCon, SATResetCoP
from connections.environment.actions import ApplyAction, UndoAction
from connections.environment.rules import Extension, Start
from connections.environment.state import State
from connections.environment.tableau import Tableau
from connections.interaction.rollout import rollout
from connections.environment.rules import ModelLemma
from connections.environment.dynamics import Dynamics
from connections.interaction.records import action_record, resolve_record
from connections.syntax.formula import Atom, Function, Variable
from connections.syntax.matrix import Clause, Literal, Matrix


def _lit(symbol: str, *, positive: bool = True) -> Literal:
    return Literal(Atom(symbol), polarity=positive)


def _state(*clauses: Clause) -> State:
    return State(
        Matrix(tuple(clauses)),
        Tableau(),
    )


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

    agent = SATCoPCon()
    assert agent(state) is None
    assert agent.status is AgentStatus.CLOSED


def test_satcop_prefers_head_made_true_by_shadow_model():
    state = _state(Clause((_lit("p"),)))
    policy = SATCoPCon()
    policy._shadow.add_clause((policy._shadow.atom_id("p"),), clause_idx=0, from_tableau=False)
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
    starts = tuple(ApplyAction(state.tableau.root_goal_id, rule) for rule in _start_rules(state))

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
        Matrix((positive, negative)),
        Tableau(),
    )
    policy = policy_class(debug_sat_core=True)

    assert policy(state) is None
    assert policy.status is AgentStatus.CLOSED
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
    policy = SATResetCoP(AgentOptions(initial_depth=2, start="all"), seed=None)

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
    from connections.environment.dynamics import Dynamics

    return Dynamics.start_rules_for(state, "all")


def _atom_literal(symbol: str, argument, *, positive: bool = True) -> Literal:
    return Literal(Atom(symbol, (argument,)), polarity=positive)


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
    state = State(Matrix((positive, negative)), Tableau())

    result = rollout(
        state,
        SATResetCoP(),
        step_limit=200,
    )

    assert result.status is AgentStatus.CLOSED


def test_deep_axiom_conjunctions_clausify(tmp_path):
    from connections.clausification import matrix_from_file

    problem = tmp_path / "deep.p"
    axioms = "".join(f"fof(a{index}, axiom, p{index}).\n" for index in range(3000))
    problem.write_text(axioms + "fof(c, conjecture, p0).\n", encoding="utf-8")

    matrix = matrix_from_file(problem)

    assert len(matrix.clauses) == 3001


def test_sat_shadow_seed_clauses_can_be_restricted():
    class Restricted(SATCoPCon):
        def _shadow_seed_clause_ids(self, state):
            return (0,)

    state = State(
        Matrix((Clause((_lit("p"),)), Clause((_lit("p", positive=False),)))),
        Tableau(),
    )
    policy = Restricted()
    policy._observe_shadow(state)

    assert {clause_idx for clause_idx, _ in policy._shadow.clauses} == {0}
    assert policy._shadow.solve() is True


@pytest.mark.parametrize("agent_class", [SATCoPCon, SATResetCoP])
def test_model_lemma_closure_is_not_a_proof(agent_class):
    state = _state(Clause((_lit("p"),)))
    agent = agent_class()
    action = agent(state)
    state.apply_rule(parent_goal_id=action.goal_id, rule=action.rule)
    state.apply_rule(parent_goal_id=state.fringe[0].goal_id, rule=ModelLemma())
    assert state.tableau.root.closed
    assert isinstance(agent(state), UndoAction)
    assert agent.status is not AgentStatus.CLOSED


@pytest.mark.parametrize("agent_class", [SATCoPCon, SATResetCoP])
def test_empty_matrix_terminates_without_a_proof(agent_class):
    result = rollout(_state(), agent_class(), step_limit=10)
    assert result.status is AgentStatus.GAVE_UP


def test_depth_gate_blocks_even_ground_extensions():
    state = _state(Clause((_lit("p"),)), Clause((_lit("p", positive=False), _lit("q"))))
    start = next(iter(_start_rules(state)))
    state.apply_rule(parent_goal_id=state.tableau.root_goal_id, rule=start)
    agent = SATCoPCon()
    assert not any(
        isinstance(action.rule, Extension)
        for action in agent._actions_for_goal(state, state.fringe[0].goal_id)
    )
    assert agent._path_limit_hit


def test_sat_actions_record_and_replay_negative_starts_and_model_lemmas():
    state = _state(Clause((_lit("p", positive=False),)))
    replay = _state(Clause((_lit("p", positive=False),)))
    action = ApplyAction(state.tableau.root_goal_id, _start_rules(state)[0])
    for action in (action, ApplyAction(1, ModelLemma()), UndoAction(0)):
        record = action_record(action)
        restored = resolve_record(replay, record)
        assert restored is not None
        Dynamics.transition(state, action)
        Dynamics.transition(replay, restored)
        assert str(replay.tableau) == str(state.tableau)
