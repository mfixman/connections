import os
import pytest

import connections.calculi.classicalsat as classicalsat
from types import SimpleNamespace

from connections.env import ConnectionEnv, Settings
from connections.calculi.classicalsat import Backtrack, Extension, Reset, SATConnectionState, SATSolver, ShadowSAT, pycosat
from connections.utils.primitives import Literal, Variable
from connections.utils.tableau import Tableau
from problem_inputs import resolve_problem_path
from strategies import Strategy

SYN344_1 = """\
cnf(clause1,negated_conjecture,
    ( f(a,z(Y1,Y2))
    | f(a,a) ) ).

cnf(clause2,negated_conjecture,
    ( f(Y1,z(Y1,Y2))
    | f(a,a) ) ).

cnf(clause3,negated_conjecture,
    ( ~ f(Y2,a)
    | f(a,a) ) ).

cnf(clause4,negated_conjecture,
    ( ~ f(a,a)
    | f(Y1,Y2) ) ).

cnf(clause5,negated_conjecture,
    ( ~ f(a,a)
    | ~ f(z(Y1,Y2),z(Y1,Y2)) ) ).
"""

def test_sat_solve_is_registered():
    assert 'SATSolve' in Strategy.strategies()

@pytest.mark.skipif(pycosat is None, reason = 'pycosat is not installed')
def test_sat_solve_runs_in_classical_cut_env():
    env = ConnectionEnv(
        'tests/cnf_problems/SYN081+1.cnf',
        Settings(logic = 'ClassicalSAT'),
    )
    strategy = Strategy.strategies()['SATSolve'](env)
    env.reset()

    for _ in range(20):
        action = strategy.step()
        _, _, done, _ = env.step(action)
        if done:
            break

    assert len(env.state.shadow.ground_clauses) > 0

@pytest.mark.skipif(pycosat is None, reason = 'pycosat is not installed')
def test_sat_solve_can_backtrack_to_root_without_crashing():
    env = ConnectionEnv(
        'tests/cnf_problems/SYN081+1.cnf',
        Settings(logic = 'ClassicalSAT'),
    )
    strategy = Strategy.strategies()['SATSolve'](env)
    env.reset()

    for _ in range(100):
        action = strategy.step()
        _, _, done, _ = env.step(action)
        if done:
            break

    assert env.state.goal is None or hasattr(env.state.goal, 'actions')

@pytest.mark.skipif(pycosat is None, reason = 'pycosat is not installed')
def test_sat_solve_solves_easy_syn_problem():
    env = ConnectionEnv(
        'tests/cnf_problems/SYN081+1.cnf',
        Settings(logic = 'ClassicalSAT'),
    )
    strategy = Strategy.strategies()['SATSolve'](env)
    env.reset()

    done = False
    info = None
    for _ in range(100):
        action = strategy.step()
        _, _, done, info = env.step(action)
        if done:
            break

    assert done is True
    assert isinstance(info, str)
    assert info.startswith('Theorem (SAT)')

@pytest.mark.skipif(pycosat is None, reason = 'pycosat is not installed')
def test_sat_solve_handles_empty_start_clause_via_sat(tmp_path):
    problem = tmp_path / 'empty_start-1.p'
    problem.write_text('cnf(c1,axiom,$false).', encoding = 'utf-8')

    env = ConnectionEnv(str(problem), Settings(logic = 'ClassicalSAT'))
    Strategy.strategies()['SATSolve'](env)
    env.reset()

    assert env.state.is_terminal is True
    assert isinstance(env.state.info, str)
    assert env.state.info.startswith('Theorem (SAT)')

@pytest.mark.skipif(pycosat is None, reason = 'pycosat is not installed')
@pytest.mark.skipif(os.getenv('TPTP') is None, reason = 'TPTP is not set')
def test_sat_solve_solves_tptp_syn000_3():
    path = resolve_problem_path('SYN000-3.p')
    env = ConnectionEnv(path, Settings(logic = 'ClassicalSAT'))
    strategy = Strategy.strategies()['SATSolve'](env)
    env.reset()

    done = False
    info = None
    for _ in range(100):
        action = strategy.step()
        _, _, done, info = env.step(action)
        if done:
            break

    assert done is True
    assert isinstance(info, str)
    assert info.startswith('Theorem (SAT)')

@pytest.mark.skipif(pycosat is None, reason = 'pycosat is not installed')
@pytest.mark.skipif(os.getenv('TPTP') is None, reason = 'TPTP is not set')
@pytest.mark.parametrize('problem_name', ['SYN916+1.p', 'SYN916-1.p'])
def test_sat_solve_rejects_degenerate_empty_translation(problem_name):
    path = resolve_problem_path(problem_name)
    env = ConnectionEnv(path, Settings(logic = 'ClassicalSAT'))
    strategy = Strategy.strategies()['SATSolve'](env)
    env.reset()

    done = False
    info = None
    for _ in range(3):
        action = strategy.step()
        _, _, done, info = env.step(action)
        if done:
            break

    assert done is True
    assert info == 'Non-Theorem: no positive start clauses'

def test_sat_connection_state_can_use_pysat_backend(monkeypatch):
    calls = {}

    class FakeSolver:
        def __init__(self, *, bootstrap_with):
            calls['clauses'] = bootstrap_with

        def solve(self):
            return True

        def get_model(self):
            return [1, -2]

        def delete(self):
            calls['deleted'] = True

    monkeypatch.setattr(classicalsat, 'PySATSolver', FakeSolver)

    state = object.__new__(SATConnectionState)
    state.sat_solver_backend = SATSolver.PySAT
    state.shadow = ShadowSAT(ground_clauses = {(1, -2)})

    state._solve_ground_clauses()

    assert calls['clauses'] == [[1, -2]]
    assert calls['deleted'] is True
    assert state.shadow.model == {1: True, 2: False}
    assert state.shadow.satisfiable is True

def test_sat_connection_state_can_use_cadical_backend(monkeypatch):
    calls = {}

    class FakeSolver:
        vars = 2

        def add_clauses(self, clauses):
            calls['clauses'] = clauses

        def solve(self):
            return 10

        def val(self, var):
            return {1: -1, 2: 2}[var]

    monkeypatch.setattr(
        classicalsat,
        'pydical',
        SimpleNamespace(Solver = FakeSolver, SATISFIABLE = 10, UNSATISFIABLE = 20, UNSOLVED = 0),
    )

    state = object.__new__(SATConnectionState)
    state.sat_solver_backend = SATSolver.CaDiCaL
    state.shadow = ShadowSAT(ground_clauses = {(-1, 2)})

    state._solve_ground_clauses()

    assert calls['clauses'] == [[-1, 2]]
    assert state.shadow.model == {1: False, 2: True}
    assert state.shadow.satisfiable is True

@pytest.mark.skipif(classicalsat.pydical is None, reason = 'pydical is not installed')
def test_sat_connection_state_can_use_real_pydical_backend():
    state = object.__new__(SATConnectionState)
    state.sat_solver_backend = SATSolver.CaDiCaL

    result = state._solve_shadow_cnf([[1], [-1, 2]])

    assert result not in ('UNSAT', 'UNKNOWN')
    assert {abs(lit): lit > 0 for lit in result if lit != 0} == {1: True, 2: True}
    assert state._solve_shadow_cnf([[1], [-1]]) == 'UNSAT'

def test_missing_cadical_does_not_disable_other_sat_backends(monkeypatch):
    monkeypatch.setattr(classicalsat, 'pydical', None)
    monkeypatch.setattr(classicalsat, 'pycosat', object())

    state = object.__new__(SATConnectionState)

    state._require_sat_backend(SATSolver.PycoSAT)

def test_unsat_core_is_reduced_before_reporting():
    state = object.__new__(SATConnectionState)
    state.settings = SimpleNamespace(debug = True)
    state.shadow = ShadowSAT(
        ground_clauses = {(1,), (-1,), (2, 3)},
        expanded_clauses = {
            (1,): ['p(a)'],
            (-1,): ['-p(a)'],
            (2, 3): ['q(a)', 'r(a)'],
        },
    )

    def fake_solve(clauses):
        clause_set = {tuple(clause) for clause in clauses}
        if (1,) in clause_set and (-1,) in clause_set:
            return 'UNSAT'
        return [1]

    state._solve_shadow_cnf = fake_solve

    state._solve_ground_clauses()

    assert state.shadow.satisfiable is False
    assert set(state.shadow.unsat_core) == {(1,), (-1,)}
    done, info = state.initial_status()
    assert done is True
    assert 'q(a)' not in info
    assert '(p(a))' in info
    assert '(-p(a))' in info

def test_initial_status_reports_unsat_shadow_result():
    state = object.__new__(SATConnectionState)
    state.settings = SimpleNamespace(debug = True)
    state.shadow = ShadowSAT(
        ground_clauses = {(1,), (-1,), (2, 3)},
        expanded_clauses = {
            (1,): ['p(a)'],
            (-1,): ['-p(a)'],
            (2, 3): ['q(a)', 'r(a)'],
        },
        unsat_core = [(1,), (-1,)],
        satisfiable = False,
    )

    done, info = state.initial_status()

    assert done is True
    assert isinstance(info, str)
    assert info.startswith('Theorem (SAT)')
    assert '(p(a))' in info
    assert '(-p(a))' in info
    assert 'q(a)' not in info

def test_initial_status_is_terse_without_debug():
    state = object.__new__(SATConnectionState)
    state.settings = SimpleNamespace(debug = False)
    state.shadow = ShadowSAT(
        ground_clauses = {(1,), (-1,)},
        expanded_clauses = {
            (1,): ['p(a)'],
            (-1,): ['-p(a)'],
        },
        unsat_core = [(1,), (-1,)],
        satisfiable = False,
    )

    done, info = state.initial_status()

    assert done is True
    assert info == 'Theorem (SAT)'

def test_initial_status_reports_unsolved_shadow_result():
    state = object.__new__(SATConnectionState)
    state.shadow = ShadowSAT(
        ground_clauses = {(1, -2)},
        satisfiable = True,
    )

    done, info = state.initial_status()

    assert done is False
    assert info is None

def test_classical_cut_initial_status_defaults_to_not_done():
    env = ConnectionEnv(
        'tests/cnf_problems/SYN081+1.cnf',
        Settings(logic = 'ClassicalCut'),
    )

    done, info = env.state.initial_status()

    assert done is False
    assert info is None

def test_grounding_collapses_all_remaining_variables_to_one_fresh_constant():
    state = object.__new__(SATConnectionState)
    state.shadow = ShadowSAT()
    state._ground_variable_constant = '__ground__'

    lit1 = Literal('p', [Variable('X')])
    lit2 = Literal('p', [Variable('Y')])

    assert state._ground_literal(lit1) == state._ground_literal(lit2)

def test_sat_solve_enables_paper_mode():
    env = SimpleNamespace(
        legal_actions = lambda: [],
        state = SimpleNamespace(
            sat_literal_value = lambda literal: None,
            enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled),
        ),
    )

    Strategy.strategies()['SATSolve'](env)

    assert env.state.paper_mode is True

def test_paper_mode_closed_tableau_restarts_same_iteration():
    state = object.__new__(SATConnectionState)
    state.paper_mode = True
    state.goal = None
    state.tableau = Tableau()
    state.tableau.actions = {'rs': Reset(id = 'rs', depth = 2)}
    state.proof_sequence = [Extension(id = 'ex0')]
    state.is_terminal = False
    state.info = None

    state._refresh_goal()

    assert state.goal is state.tableau
    assert state.goal.parent is None
    assert 'rs' in state.goal.actions
    assert state.is_terminal is False
    assert state.proof_sequence == []

def test_paper_mode_skips_goal_when_model_assigns_false():
    state = object.__new__(SATConnectionState)
    state.paper_mode = True
    state.tableau = Tableau()
    state.tableau.actions = {'rs': Reset(id = 'rs', depth = 2)}
    state.goal = Tableau(Literal('p', []), state.tableau)
    state.tableau.children = [state.goal]
    state.proof_sequence = []
    state.is_terminal = False
    state.info = None
    state.sat_literal_value = lambda literal: False if literal.symbol == 'p' else None

    state._refresh_goal()

    assert state.goal is state.tableau
    assert state.goal.parent is None
    assert state.is_terminal is False

@pytest.mark.parametrize('strategy_name', ['SATSolve', 'SATSolveLiteral'])
def test_sat_strategies_use_restricted_backtracking_on_forward_steps(strategy_name):
    env = SimpleNamespace(
        legal_actions = lambda: [Extension(id = 'ex0')],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            shadow = SimpleNamespace(new_ground_clauses = set()),
            sat_literal_value = lambda literal: None,
            enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled),
        ),
    )
    strategy = Strategy.strategies()[strategy_name](env)
    action = strategy.step()

    assert action.cut is True

def test_sat_solve_uses_first_randomised_forward_action():
    first = Extension(id = 'ex0')
    second = Extension(id = 'ex1')
    env = SimpleNamespace(
        legal_actions = lambda: [first, second],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            shadow = SimpleNamespace(new_ground_clauses = set()),
            sat_literal_value = lambda literal: None,
            enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled),
        ),
    )

    strategy = Strategy.strategies()['SATSolve'](env)

    assert strategy.step() is first

@pytest.mark.parametrize('strategy_name', ['SATSolve', 'SATSolveLiteral'])
def test_sat_strategies_only_reset_at_root_when_shadow_is_still_growing(strategy_name):
    env = SimpleNamespace(
        legal_actions = lambda: [Backtrack(id = 'bt'), Reset(id = 'rs', depth = 3)],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            shadow = SimpleNamespace(new_ground_clauses = {(1,)}),
            sat_literal_value = lambda literal: None,
            enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled),
        ),
    )
    strategy = Strategy.strategies()[strategy_name](env)

    assert isinstance(strategy.step(), Backtrack)

def test_starts_do_not_duplicate_clause_indices_across_calls():
    env = ConnectionEnv(
        'tests/cnf_problems/SYN081+1.cnf',
        Settings(logic = 'ClassicalCut'),
    )

    before = list(env.matrix.positive_clauses)
    env.state.starts()
    env.state.starts()

    assert env.matrix.positive_clauses == before

@pytest.mark.parametrize('strategy_name', ['SATSolve', 'SATSolveLiteral'])
@pytest.mark.skipif(pycosat is None, reason = 'pycosat is not installed')
def test_sat_strategies_do_not_refute_satisfiable_problem(tmp_path, strategy_name):
    problem = tmp_path / 'SYN344-1.p'
    problem.write_text(SYN344_1, encoding = 'utf-8')

    env = ConnectionEnv(str(problem), Settings(logic = 'ClassicalSAT'))
    strategy = Strategy.strategies()[strategy_name](env)
    env.reset()

    done = False
    info = None
    for _ in range(5):
        action = strategy.step()
        _, _, done, info = env.step(action)
        if done:
            break

    assert not done
    assert info != 'Theorem'
