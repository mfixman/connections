import pytest

import connections.calculi.classicalsat as classicalsat
from connections.env import ConnectionEnv, Settings
from connections.calculi.classicalsat import pycosat
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

    assert len(env.state.ground_clauses()) > 0

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

def test_old_sat_logic_is_still_available():
    pytest.importorskip('pydical')
    pytest.importorskip('pysat')
    env = ConnectionEnv(
        'tests/cnf_problems/SYN081+1.cnf',
        Settings(logic = 'ClassicalSATOld'),
    )
    assert env.state.__class__.__name__ == 'SATConnectionState'

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

    state = object.__new__(classicalsat.SATConnectionState)
    state.sat_solver_backend = 'pysat'
    state.shadow = classicalsat.ShadowSAT(ground_clauses = {(1, -2)})

    state._solve_ground_clauses()

    assert calls['clauses'] == [[1, -2]]
    assert calls['deleted'] is True
    assert state.shadow.model == {1: True, 2: False}
    assert state.shadow.satisfiable is True

@pytest.mark.skipif(pycosat is None, reason = 'pycosat is not installed')
def test_sat_solve_literal_does_not_refute_satisfiable_problem(tmp_path):
    problem = tmp_path / 'SYN344-1.p'
    problem.write_text(SYN344_1, encoding = 'utf-8')

    env = ConnectionEnv(str(problem), Settings(logic = 'ClassicalSAT'))
    strategy = Strategy.strategies()['SATSolveLiteral'](env)
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
