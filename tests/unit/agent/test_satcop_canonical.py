import itertools
import random

from connections.agent import AgentOptions, AgentStatus
from connections.agent.sat import SATCoPCon
from connections.agent.sat.canonical_shadow import CanonicalShadow
from connections.agent.sat.grounding import Grounding, tautological
from connections.environment.actions import ApplyAction, UndoAction
from connections.environment.dynamics import Dynamics
from connections.environment.rules import Extension, ModelLemma, Reduction, Start
from connections.environment.state import State
from connections.environment.tableau import Tableau
from connections.interaction.rollout import rollout
from connections.syntax.formula import Atom, Function, Variable
from connections.syntax.matrix import Clause, Literal, Matrix

def lit(name, term = None, positive = True):
    return Literal(Atom(name, () if term is None else (term,)), polarity = positive)

def state_of(*clauses):
    return State(Matrix(tuple(clauses)), Tableau())

def apply(state, rule, goal_id = None):
    goal_id = state.fringe[0].goal_id if goal_id is None else goal_id
    return state.apply_rule(parent_goal_id = goal_id, rule = rule)

def test_starts_use_conjecture_then_positive_fallback_in_matrix_order():
    state = state_of(Clause((lit("p"),)), Clause((lit("q", positive = False),), role = "conjecture"))
    policy = SATCoPCon()
    assert policy._shadow_seed_clause_ids(state) == (1,)
    all_goals = state_of(*(Clause(c.literals, role = "conjecture") for c in state.matrix.clauses))
    assert policy._shadow_seed_clause_ids(all_goals) == (0,)

    positive = state_of(Clause((lit("p"),)), Clause((lit("q"),)))
    actions = policy._actions_for_goal(positive, 0)
    assert policy._next_action(positive, actions).rule.clause_idx == 0

def test_grounding_uses_frequent_goal_constant_even_for_open_literals():
    state = state_of(Clause((lit("p", Function("a")), lit("q", Function("a"))), role = "conjecture"))
    policy = SATCoPCon()
    policy._observe_shadow(state)
    assert policy.grounding.constant == "a"
    assert policy._sat_literal(state, lit("p", Variable("X")), instance_id = 1) == policy._sat_literal(
        state, lit("p", Function("a")), instance_id = None,
    )
    assert policy._literal_value(state, lit("p", Variable("X")), instance_id = 1) is not None

    collision = state_of(Clause((lit("p", Function("__ground__")),)))
    assert Grounding(collision.matrix).constant != "__ground__"

def test_depth_one_allows_one_extension_and_reductions_at_the_limit():
    positive = Clause((lit("p"),))
    negative = Clause((lit("p", positive = False), lit("q")))
    state = state_of(positive, negative)
    apply(state, Dynamics.start_rules_for(state, "all")[0])
    policy = SATCoPCon(seed = None)
    actions = policy._actions_for_goal(state, state.fringe[0].goal_id)
    extension = next(a for a in actions if isinstance(a.rule, Extension))
    Dynamics.transition(state, extension)
    child = state.fringe[0].goal_id
    assert not any(isinstance(a.rule, Extension) for a in policy._actions_for_goal(state, child))

    reduction_state = state_of(positive)
    apply(reduction_state, Start(positive, clause_idx = 0, instance_id = 1))
    parent = reduction_state.fringe[0].goal_id
    apply(reduction_state, Extension(0, Clause((lit("p", positive = False), lit("p", positive = False))), clause_idx = 0))
    reductions = policy._actions_for_goal(reduction_state, reduction_state.fringe[0].goal_id)
    assert reductions and isinstance(reductions[0].rule, Reduction)
    assert reductions[0].rule.source_goal_id == parent

def test_new_instances_repeat_depth_then_no_progress_increases_it():
    policy = SATCoPCon()
    shadow = policy._shadow
    shadow.add_clause((shadow.atom_id("p"),), clause_idx = 0, from_tableau = True)
    policy._start_next_depth()
    assert policy.depth_limit == 1
    policy._start_next_depth()
    assert policy.depth_limit == 2

def test_extension_promises_follow_live_model_and_do_not_skip_start_goals():
    state = state_of(Clause((lit("p"),)), Clause((lit("p", positive = False), lit("q"), lit("r"))))
    apply(state, Start(state.matrix.clauses[0], clause_idx = 0, instance_id = 1))
    app = apply(state, Extension(0, state.matrix.clauses[1], clause_idx = 1, instance_id = 2))
    policy = SATCoPCon(seed = None)
    q, r = app.child_goal_ids
    policy._shadow.model[policy._shadow.atom_id("q")] = False
    policy._shadow.model[policy._shadow.atom_id("r")] = True
    visited = []

    def prove(state, goal_id):
        visited.append(goal_id)
        if goal_id == r:
            policy._shadow.model[policy._shadow.atom_id("q")] = True
        yield ApplyAction(goal_id, ModelLemma())
        return True

    policy.prove = prove
    policy.stack = [policy.promises(state, app.rule_application_id, select_by_model = True)]
    while (action := policy.advance()) is not None:
        Dynamics.transition(state, action)

    assert visited == [r, q]
    policy._shadow.model[policy._shadow.atom_id("q")] = False
    policy.stack = [policy.promises(state, app.rule_application_id, select_by_model = False)]
    assert policy.advance().goal_id == q

def test_failed_later_start_goal_does_not_reopen_successful_sibling():
    state = state_of(
        Clause((lit("p"), lit("q"))),
        Clause((lit("p", positive = False), lit("r"))),
        Clause((lit("p", positive = False), lit("s"))),
    )
    policy = SATCoPCon(AgentOptions(start = "positive"), seed = None)
    result = rollout(state, policy, step_limit = 5)
    assert [type(a.rule) if isinstance(a, ApplyAction) else UndoAction for a in result.actions] == [
        Start, Extension, ModelLemma, UndoAction, Start,
    ]
    assert result.actions[3].step_id == 0
    assert policy.depth_limit == 1  # First pass learned a clause.

def test_tautology_constraints_use_substitution_not_common_grounding():
    x, a = Variable("X"), Function("a")
    clause = Clause((lit("p", x), lit("p", a, positive = False)))
    state = state_of(clause, Clause((lit("p", a, positive = False),)))
    apply(state, Start(clause, clause_idx = 0, instance_id = 1))
    assert not tautological(state)
    action = Dynamics.extension_action_for_position(state, state.fringe[0].goal_id, 1, 0, instance_id = 2)
    Dynamics.transition(state, action)
    assert tautological(state)

def test_equality_constraints_respect_dualized_sign_and_exempt_reflexivity():
    a, x = Function("a"), Variable("X")
    reflexive = Clause((Literal(Atom("equal___", (x, x)), polarity = False),))
    for sign in (True, False):
        clause = Clause((Literal(Atom("equal___", (a, a)), polarity = sign), lit("p")))
        state = state_of(reflexive, clause)
        apply(state, Start(reflexive, clause_idx = 0, instance_id = 1))
        assert not tautological(state)
        apply(state, Extension(0, clause, clause_idx = 1, instance_id = 2))
        assert tautological(state) == (not sign)

def test_regularity_checks_selected_goal_without_pruning_its_siblings():
    positive = Clause((lit("p"),))
    extension = Clause((lit("p", positive = False), lit("p"), lit("q")))
    state = state_of(positive, extension, Clause((lit("q", positive = False),)))
    apply(state, Start(positive, clause_idx = 0, instance_id = 1))
    app = apply(state, Extension(0, extension, clause_idx = 1, instance_id = 2))
    repeated, other = app.child_goal_ids
    policy = SATCoPCon(AgentOptions(initial_depth = 2), seed = None)
    assert not policy._actions_for_goal(state, repeated)
    assert policy._actions_for_goal(state, other)

def test_first_order_proof_requires_learning_not_just_tableau_closure():
    a, x = Function("a"), Variable("X")
    state = state_of(Clause((lit("p", x),)), Clause((lit("p", a, positive = False),), role = "conjecture"))
    policy = SATCoPCon(debug_sat_core = True)
    result = rollout(state, policy, step_limit = 50)
    assert result.status is AgentStatus.CLOSED
    assert result.steps == 2
    assert policy.diagnostics()["sat_core_clause_ids"] == [0, 1]

def test_equality_congruence_with_shared_fof_clausification(tmp_path):
    from connections.clausification import matrix_from_file
    problem = tmp_path / "equality.p"
    problem.write_text("fof(e,axiom,a=b).\nfof(p,axiom,p(a)).\nfof(c,conjecture,p(b)).\n")
    matrix = matrix_from_file(problem, mark_conjecture = True)
    result = rollout(State(matrix, Tableau()), SATCoPCon(), step_limit = 1000)
    assert result.status is AgentStatus.CLOSED

def test_incremental_walk_and_cdcl_agree_with_exhaustive_truth_tables():
    rng = random.Random(123)
    for _ in range(30):
        shadow = CanonicalShadow(debug_unsat_core = True)
        atoms = [shadow.atom_id(str(i)) for i in range(4)]
        clauses = []
        for index in range(15):
            clause = tuple(rng.choice(atoms) * rng.choice((-1, 1)) for _ in range(rng.randrange(4)))
            clauses.append(clause)
            shadow.add_clause(clause, clause_idx = index, from_tableau = True)
            satisfiable = any(
                all(any(values[abs(lit) - 1] == (lit > 0) for lit in c) for c in clauses)
                for values in itertools.product((False, True), repeat = 4)
            )
            assert shadow.solve() == satisfiable
            if not satisfiable:
                assert shadow.core_available and shadow.sat_core_clause_ids
                break

            assert all(any(shadow.literal_value(lit) for lit in c) for c in clauses)
