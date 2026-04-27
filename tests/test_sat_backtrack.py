import pytest

from connections.calculi.classicalsat import Extension, SATConnectionState, Start, pydical
from connections.env import ConnectionEnv, Settings
from connections.utils.primitives import Literal
from connections.utils.tableau import Tableau

@pytest.mark.skipif(pydical is None, reason = 'pydical is not installed')
def test_sat_backtrack_keeps_parent_frame_when_stepping_to_sibling():
    env = ConnectionEnv(
        'tests/cnf_problems/SYN081+1.cnf',
        Settings(logic = 'ClassicalSAT'),
    )
    state = env.reset()

    parent = Tableau(Literal('parent', []), state.tableau)
    proven = Tableau(Literal('done', []), parent)
    sibling = Tableau(Literal('sibling', []), parent)
    current = Tableau(Literal('current', []), parent)
    proven.proven = True
    sibling.actions = {}
    current.actions = {}
    parent.actions = {'ex0': Extension(id = 'ex0')}
    parent.children = [proven, sibling, current]
    state.tableau.children = [parent]
    state.goal = current

    state.substitution.trail = [[], []]
    state.proof_sequence = [
        Start(id = 'st0'),
        Extension(
            id = 'ex0',
            principle_node = parent,
            clause_copy = [proven.literal, sibling.literal, current.literal],
            lit_idx = 0,
        ),
    ]
    state._refresh_remaining_actions = lambda remaining_ids: {
        action_id: Extension(id = action_id) for action_id in remaining_ids
    }

    state.backtrack()

    assert state.goal is parent
    assert len(state.substitution.trail) == 1
    assert len(state.proof_sequence) == 1

@pytest.mark.skipif(pydical is None, reason = 'pydical is not installed')
def test_sat_backjump_two_can_skip_to_grandparent():
    env = ConnectionEnv(
        'tests/cnf_problems/SYN081+1.cnf',
        Settings(logic = 'ClassicalSAT'),
    )
    state = env.reset()

    grandparent = Tableau(Literal('grand', []), state.tableau)
    parent = Tableau(Literal('parent', []), grandparent)
    proven = Tableau(Literal('done', []), parent)
    sibling = Tableau(Literal('sibling', []), parent)
    current = Tableau(Literal('current', []), parent)
    proven.proven = True
    sibling.actions = {}
    current.actions = {}
    grandparent.actions = {'re0': Extension(id = 're0')}
    parent.actions = {'ex0': Extension(id = 'ex0')}
    parent.children = [proven, sibling, current]
    grandparent.children = [parent]
    state.tableau.children = [grandparent]
    state.goal = current

    state.substitution.trail = [[], [], []]
    state.proof_sequence = [
        Start(id = 'st0'),
        Extension(
            id = 'exg',
            principle_node = grandparent,
            clause_copy = [parent.literal],
            lit_idx = 0,
        ),
        Extension(
            id = 'exp',
            principle_node = parent,
            clause_copy = [proven.literal, sibling.literal, current.literal],
            lit_idx = 0,
        ),
    ]
    state._refresh_remaining_actions = lambda remaining_ids: {
        action_id: Extension(id = action_id) for action_id in remaining_ids
    }

    state.backtrack(steps = 2)

    assert state.goal is grandparent
    assert len(state.substitution.trail) == 1
    assert len(state.proof_sequence) == 1
