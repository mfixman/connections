import pytest

from connections.env import ConnectionEnv, Settings
from connections.calculi.classicalsat import pycosat
from strategies import Strategy

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
