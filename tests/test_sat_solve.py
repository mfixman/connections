import json
import os
import pytest

import connections.calculi.classicalsat as classicalsat
import strategies.sat_solve_learn as sat_solve_learn_module
import strategies.sat_solve as sat_solve_module
from types import SimpleNamespace

from connections.env import ConnectionEnv, Settings
from connections.calculi.classicalsat import Backtrack, Extension, Reduction, Reset, SATConnectionState, SATSolver, ShadowSAT, Start, pydical
from connections.utils.primitives import Literal, Variable
from connections.utils.tableau import Tableau
from connections.utils.unification import Substitution
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
    assert 'SATCoPRust' in Strategy.strategies()
    assert 'SATSolve' in Strategy.strategies()
    assert 'SATSolveGuided' in Strategy.strategies()
    assert 'SATSolveGuidedProbeConflicts' in Strategy.strategies()
    assert 'SATSolveGuidedForcedTrue' in Strategy.strategies()
    assert 'SATSolveGuidedEVSIDS' in Strategy.strategies()
    assert 'SATSolveGuidedShort' in Strategy.strategies()
    assert 'SATSolveGuidedFast' in Strategy.strategies()
    assert 'SATSolveLearnInitial' in Strategy.strategies()
    assert 'SATSolveLearnTrain' in Strategy.strategies()
    assert 'SATSolveLearnEval' in Strategy.strategies()
    assert 'SATResetBandit' in Strategy.strategies()
    assert 'SATResetAtStaleBacktrack' in Strategy.strategies()
    assert 'SATResetAtStaleAction' in Strategy.strategies()
    assert 'SATResetShadow' in Strategy.strategies()
    assert 'SATResetNoCut' in Strategy.strategies()
    assert 'SATResetStepBudget' in Strategy.strategies()
    assert 'SATResetExtensionBudget' in Strategy.strategies()
    assert 'SATSolveLearnCombined' in Strategy.strategies()
    assert 'SATSolveLearnSATContext' in Strategy.strategies()
    assert 'SATSolveLiteral' in Strategy.strategies()
    assert 'SATSolveLiteralRandomTie' in Strategy.strategies()
    assert 'SATSolveLiteralShortTie' in Strategy.strategies()
    assert 'SATSolveLiteralLookaheadTie' in Strategy.strategies()
    assert 'SATSolveLiteralFrequencyTie' in Strategy.strategies()
    assert 'SATSolveLiteralFridayStartCut' in Strategy.strategies()
    assert 'SATSolveLiteralFridayRootReset' in Strategy.strategies()
    assert 'SATSolveLiteralFridayHybrid' in Strategy.strategies()
    assert 'SATSolveLiteralLowFrequencyTie' in Strategy.strategies()
    assert 'SATSolveLiteralOppositeSignedFrequencyTie' in Strategy.strategies()
    assert 'SATSolveLiteralFirstStart' in Strategy.strategies()
    assert 'SATSolveLiteralRareStart' in Strategy.strategies()
    assert 'SATSolveLiteralModelStart' in Strategy.strategies()
    assert 'SATSolvePropagation' in Strategy.strategies()
    assert 'SATSolveLiteralPropagation' in Strategy.strategies()

@pytest.mark.skipif(pydical is None, reason = 'pydical is not installed')
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

@pytest.mark.skipif(pydical is None, reason = 'pydical is not installed')
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

@pytest.mark.skipif(pydical is None, reason = 'pydical is not installed')
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

@pytest.mark.skipif(pydical is None, reason = 'pydical is not installed')
def test_sat_solve_handles_empty_start_clause_via_sat(tmp_path):
    problem = tmp_path / 'empty_start-1.p'
    problem.write_text('cnf(c1,axiom,$false).', encoding = 'utf-8')

    env = ConnectionEnv(str(problem), Settings(logic = 'ClassicalSAT'))
    Strategy.strategies()['SATSolve'](env)
    env.reset()

    assert env.state.is_terminal is True
    assert isinstance(env.state.info, str)
    assert env.state.info.startswith('Theorem (SAT)')

@pytest.mark.skipif(pydical is None, reason = 'pydical is not installed')
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

@pytest.mark.skipif(pydical is None, reason = 'pydical is not installed')
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

def test_sat_connection_state_can_use_cadical_backend(monkeypatch):
    calls = {'clauses': [], 'options': [], 'solvers': []}

    class FakeSolver:
        vars = 3

        def __init__(self):
            calls['solvers'].append(self)

        def add_clause(self, clause):
            calls['clauses'].append(clause)

        def set(self, name, value):
            calls['options'].append((name, value))
            return True

        def solve(self):
            return 10

        def val(self, var):
            return {1: -1, 2: 2, 3: 3}[var]

    monkeypatch.setattr(
        classicalsat,
        'pydical',
        SimpleNamespace(Solver = FakeSolver, SATISFIABLE = 10, UNSATISFIABLE = 20, UNSOLVED = 0),
    )

    state = object.__new__(SATConnectionState)
    state.sat_solver_backend = SATSolver.CaDiCaL
    state.shadow = ShadowSAT(ground_clauses = {(-1, 2)})

    state._solve_ground_clauses()
    solver = state.shadow.solver
    state.shadow.ground_clauses.add((3,))
    state._solve_ground_clauses()

    assert calls['solvers'] == [solver]
    assert calls['clauses'] == [[-1, 2], [3]]
    assert ('quiet', 1) in calls['options']
    assert state.shadow.model == {1: False, 2: True, 3: True}
    assert state.shadow.satisfiable is True

@pytest.mark.skipif(classicalsat.pydical is None, reason = 'pydical is not installed')
def test_sat_connection_state_can_use_real_pydical_backend():
    state = object.__new__(SATConnectionState)
    state.sat_solver_backend = SATSolver.CaDiCaL
    state.shadow = ShadowSAT(ground_clauses = {(1,), (-1, 2)})

    state._solve_ground_clauses()

    assert state.shadow.satisfiable is True
    assert state.shadow.model == {1: True, 2: True}

    state = object.__new__(SATConnectionState)
    state.sat_solver_backend = SATSolver.CaDiCaL
    state.shadow = ShadowSAT(ground_clauses = {(1,), (-1,)})

    state._solve_ground_clauses()

    assert state.shadow.satisfiable is False

def test_missing_cadical_disables_sat_backend(monkeypatch):
    monkeypatch.setattr(classicalsat, 'pydical', None)

    state = object.__new__(SATConnectionState)

    with pytest.raises(ModuleNotFoundError):
        state._require_sat_backend()

@pytest.mark.skipif(pydical is None, reason = 'pydical is not installed')
def test_unsat_shadow_shrinks_debug_reporting_to_irreducible_subset():
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

    state._solve_current_shadow_cnf_with_pydical = lambda: 'UNSAT'

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

@pytest.mark.parametrize(
    'strategy_name',
    [
        'SATSolve',
        'SATSolveGuided',
        'SATSolveLearnInitial',
        'SATResetBandit',
        'SATResetAtStaleBacktrack',
        'SATResetAtStaleAction',
        'SATResetShadow',
        'SATResetStepBudget',
        'SATResetExtensionBudget',
        'SATSolveLearnCombined',
        'SATSolveLiteral',
        'SATSolveLiteralRandomTie',
        'SATSolveLiteralShortTie',
        'SATSolveLiteralLookaheadTie',
        'SATSolveLiteralFrequencyTie',
        'SATSolveLiteralLowFrequencyTie',
        'SATSolveLiteralOppositeSignedFrequencyTie',
        'SATSolveLiteralFirstStart',
        'SATSolveLiteralRareStart',
        'SATSolveLiteralModelStart',
        'SATSolvePropagation',
        'SATSolveLiteralPropagation',
    ],
)
def test_sat_strategies_use_restricted_backtracking_on_forward_steps(strategy_name):
    class FakeSolver:
        def score(self, atom_id):
            return 0

    env = SimpleNamespace(
        legal_actions = lambda: [Extension(id = 'ex0')],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            shadow = SimpleNamespace(new_ground_clauses = set(), atom_map = {}, solver = FakeSolver()),
            sat_literal_value = lambda literal: None,
            enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled),
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
        ),
    )
    strategy = Strategy.strategies()[strategy_name](env)
    action = strategy.step()

    assert action.cut is True

def test_sat_reset_no_cut_leaves_forward_action_uncut():
    class FakeSolver:
        def score(self, atom_id):
            return 0

    action = Extension(id = 'ex0')
    env = SimpleNamespace(
        legal_actions = lambda: [action],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            shadow = SimpleNamespace(new_ground_clauses = set(), atom_map = {}, solver = FakeSolver()),
            sat_literal_value = lambda literal: None,
            enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled),
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
        ),
    )
    strategy = Strategy.strategies()['SATResetNoCut'](env)

    assert strategy.step() is action
    assert action.cut is False

def test_sat_solve_uses_first_randomised_forward_action():
    class FakeSolver:
        def score(self, atom_id):
            return 0

    first = Extension(id = 'ex0')
    second = Extension(id = 'ex1')
    env = SimpleNamespace(
        legal_actions = lambda: [first, second],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            shadow = SimpleNamespace(new_ground_clauses = set(), atom_map = {}, solver = FakeSolver()),
            sat_literal_value = lambda literal: None,
            enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled),
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
        ),
    )

    strategy = Strategy.strategies()['SATSolve'](env)

    assert strategy.step() is first

def test_satcop_rust_prefers_start_at_root_and_syncs_shadow():
    start = Start(id = 'st0', clause_copy = [Literal('p', [])])
    reset = Reset(id = 'rs', depth = 2)
    calls = []

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = None),
        max_depth = 1,
        shadow = SimpleNamespace(new_ground_clauses = set(), atom_map = {}, solver = None),
        sat_literal_value = lambda literal: None,
        _flush_new_ground_clauses = lambda: calls.append('flush'),
        _solve_ground_clauses = lambda: calls.append('solve'),
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [start, reset], state = state)
    strategy = Strategy.strategies()['SATCoPRust'](env)

    assert strategy.step() is start
    assert calls == ['flush', 'solve']

def test_satcop_rust_prefers_reduction_before_extension_with_cut():
    reduction = Reduction(id = 're0', path_lit = Literal('p', []))
    extension = Extension(id = 'ex0', lit_idx = 0, clause_copy = [Literal('q', []), Literal('r', [])])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(new_ground_clauses = set(), atom_map = {}, solver = None),
        sat_literal_value = lambda literal: None,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [extension, reduction], state = state)
    strategy = Strategy.strategies()['SATCoPRust'](env)
    action = strategy.step()

    assert action is reduction
    assert action.cut is True

def test_satcop_rust_reorders_true_promises_first_inside_extension():
    head = Literal('head', [])
    unknown = Literal('unknown', [])
    promising = Literal('promising', [])
    extension = Extension(
        id = 'ex0',
        lit_idx = 1,
        clause_copy = [unknown, head, promising],
        sub_updates = [],
    )

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(new_ground_clauses = set(), atom_map = {}, solver = None),
        substitution = Substitution(),
        sat_literal_value = lambda literal: True if literal.symbol == 'promising' else None,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [extension], state = state)
    strategy = Strategy.strategies()['SATCoPRust'](env)

    action = strategy.step()

    assert action is extension
    assert action.cut is True
    assert action.lit_idx == 0
    assert [literal.symbol for literal in action.clause_copy] == ['head', 'promising', 'unknown']

def test_satcop_rust_restarts_same_depth_when_shadow_grows_at_root():
    reset = Reset(id = 'rs', depth = 3)

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = None),
        max_depth = 2,
        shadow = SimpleNamespace(new_ground_clauses = {(1,)}, atom_map = {}, solver = None),
        sat_literal_value = lambda literal: None,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [reset], state = state)
    strategy = Strategy.strategies()['SATCoPRust'](env)

    action = strategy.step()

    assert isinstance(action, Reset)
    assert action.depth == 2

@pytest.mark.skipif(pydical is None, reason = 'pydical is not installed')
def test_sat_solve_guided_chooses_best_scored_extension():
    head = Literal('p', [])
    forced = Literal('forced', [])
    unforced = Literal('unforced', [])

    weak = Extension(id = 'ex0', lit_idx = 0, clause_copy = [head, unforced])
    strong = Extension(id = 'ex1', lit_idx = 0, clause_copy = [head, forced])

    solver = pydical.Solver()
    solver.add_clause([2])
    solver.solve()

    shadow = ShadowSAT(
        atom_map = {
            'p': 1,
            'forced': 2,
            'unforced': 3,
        },
        ground_clauses = {
            (1, 2),
            (1, 3),
        },
        solver = solver,
        model = {},
    )

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = shadow,
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [weak, strong], state = state)
    strategy = Strategy.strategies()['SATSolveGuided'](env)

    assert strategy.step() is strong
    assert strong.cut is True

def test_sat_solve_guided_uses_live_assumptions_instead_of_solver_copies(monkeypatch):
    monkeypatch.setattr(sat_solve_module, 'pydical', SimpleNamespace(UNSATISFIABLE = 20))

    head = Literal('p', [])
    weak_lit = Literal('weak', [])
    strong_lit = Literal('strong', [])

    weak = Extension(id = 'ex0', lit_idx = 0, clause_copy = [head, weak_lit])
    strong = Extension(id = 'ex1', lit_idx = 0, clause_copy = [head, strong_lit])

    class FakeSolver:
        def __init__(self):
            self.assumptions = []
            self.calls = []

        def fixed(self, sat_lit):
            self.calls.append(('fixed', sat_lit))
            return 0

        def score(self, atom_id):
            self.calls.append(('score', atom_id))
            return 0

        def assume(self, sat_lit):
            self.calls.append(('assume', sat_lit))
            self.assumptions.append(sat_lit)

        def solve(self):
            self.calls.append(('solve', tuple(self.assumptions)))
            return 20 if self.assumptions == [-3] else 10

        def reset_assumptions(self):
            self.calls.append(('reset_assumptions', tuple(self.assumptions)))
            self.assumptions = []

        def copy(self):
            raise AssertionError('SATSolveGuided should probe with live assumptions')

    solver = FakeSolver()
    shadow = SimpleNamespace(
        new_ground_clauses = set(),
        atom_map = {'p': 1, 'weak': 2, 'strong': 3},
        solver = solver,
    )
    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = shadow,
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [weak, strong], state = state)
    strategy = Strategy.strategies()['SATSolveGuided'](env)

    assert strategy.step() is strong
    assert ('assume', -2) in solver.calls
    assert ('assume', -3) in solver.calls
    assert ('reset_assumptions', (-2,)) in solver.calls
    assert ('reset_assumptions', (-3,)) in solver.calls

@pytest.mark.parametrize(
    ('strategy_name', 'expected'),
    [
        ('SATSolveGuidedProbeConflicts', 'probe'),
        ('SATSolveGuidedForcedTrue', 'forced'),
        ('SATSolveGuidedEVSIDS', 'evsids'),
        ('SATSolveGuidedFast', 'forced'),
    ],
)
def test_sat_solve_guided_signal_variants_choose_by_requested_signal(monkeypatch, strategy_name, expected):
    monkeypatch.setattr(sat_solve_module, 'pydical', SimpleNamespace(UNSATISFIABLE = 20))

    head = Literal('p', [])
    probe_lit = Literal('probe', [])
    forced_lit = Literal('forced', [])
    evsids_lit = Literal('evsids', [])

    probe = Extension(id = 'ex0', lit_idx = 0, clause_copy = [head, probe_lit])
    forced = Extension(id = 'ex1', lit_idx = 0, clause_copy = [head, forced_lit])
    evsids = Extension(id = 'ex2', lit_idx = 0, clause_copy = [head, evsids_lit])

    class FakeSolver:
        def __init__(self):
            self.assumptions = []

        def fixed(self, sat_lit):
            return 1 if sat_lit == 3 else 0

        def score(self, atom_id):
            return {2: 0.0, 3: 1.0, 4: 5.0}[atom_id]

        def assume(self, sat_lit):
            self.assumptions.append(sat_lit)

        def solve(self):
            return 20 if self.assumptions == [-2] else 10

        def reset_assumptions(self):
            self.assumptions = []

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            atom_map = {'p': 1, 'probe': 2, 'forced': 3, 'evsids': 4},
            solver = FakeSolver(),
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [probe, forced, evsids], state = state)
    strategy = Strategy.strategies()[strategy_name](env)

    assert strategy.step() is {'probe': probe, 'forced': forced, 'evsids': evsids}[expected]

def test_sat_solve_guided_short_chooses_fewer_remaining_subgoals_without_sat_calls():
    head = Literal('p', [])
    short = Extension(id = 'ex0', lit_idx = 0, clause_copy = [head, Literal('q', [])])
    long = Extension(
        id = 'ex1',
        lit_idx = 0,
        clause_copy = [head, Literal('r', []), Literal('s', [])],
    )

    class FakeSolver:
        def fixed(self, sat_lit):
            raise AssertionError('short scoring should not inspect SAT literals')

        def score(self, atom_id):
            raise AssertionError('short scoring should not inspect SAT literals')

        def assume(self, sat_lit):
            raise AssertionError('short scoring should not probe assumptions')

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(new_ground_clauses = set(), atom_map = {}, solver = FakeSolver()),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [long, short], state = state)
    strategy = Strategy.strategies()['SATSolveGuidedShort'](env)

    assert strategy.step() is short

def test_sat_solve_guided_fast_skips_assumption_probes(monkeypatch):
    monkeypatch.setattr(sat_solve_module, 'pydical', SimpleNamespace(UNSATISFIABLE = 20))

    head = Literal('p', [])
    weak = Extension(id = 'ex0', lit_idx = 0, clause_copy = [head, Literal('weak', [])])
    strong = Extension(id = 'ex1', lit_idx = 0, clause_copy = [head, Literal('strong', [])])

    class FakeSolver:
        def fixed(self, sat_lit):
            return 1 if sat_lit == 3 else 0

        def score(self, atom_id):
            return {2: 0.0, 3: 1.0}[atom_id]

        def assume(self, sat_lit):
            raise AssertionError('fast scoring should not probe assumptions')

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            atom_map = {'p': 1, 'weak': 2, 'strong': 3},
            solver = FakeSolver(),
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [weak, strong], state = state)
    strategy = Strategy.strategies()['SATSolveGuidedFast'](env)

    assert strategy.step() is strong

def test_sat_solve_literal_short_tie_prefers_reduction_before_extension():
    reduction = Reduction(id = 'rd0', path_lit = Literal('p', []))
    extension = Extension(id = 'ex0', lit_idx = 0, clause_copy = [Literal('q', []), Literal('r', [])])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(new_ground_clauses = set(), atom_map = {}, solver = None),
        sat_literal_value = lambda literal: None,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [extension, reduction], state = state)
    strategy = Strategy.strategies()['SATSolveLiteralShortTie'](env)

    assert strategy.step() is reduction

def test_sat_solve_literal_lookahead_tie_prefers_lookahead_literal():
    weak = Extension(id = 'ex0', lit_idx = 0, clause_copy = [Literal('weak', [])])
    strong = Extension(id = 'ex1', lit_idx = 0, clause_copy = [Literal('strong', [])])

    class FakeSolver:
        def lookahead(self):
            return 2

        def score(self, atom_id):
            return 0

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            atom_map = {'weak': 1, 'strong': 2},
            solver = FakeSolver(),
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [weak, strong], state = state)
    strategy = Strategy.strategies()['SATSolveLiteralLookaheadTie'](env)

    assert strategy.step() is strong

def test_sat_solve_learn_initial_prefers_extension_with_more_model_false_children():
    weak = Extension(
        id = 'ex0',
        lit_idx = 0,
        clause_copy = [Literal('head', []), Literal('open', [])],
    )
    strong = Extension(
        id = 'ex1',
        lit_idx = 0,
        clause_copy = [Literal('head', []), Literal('close', [])],
    )

    class FakeSolver:
        def score(self, atom_id):
            return 0

    goal = SimpleNamespace(parent = object(), depth = 4, path = lambda: [Literal('path', [])])
    state = SimpleNamespace(
        goal = goal,
        max_depth = 4,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {(2,), (2, 3), (3, 4)},
            atom_map = {'head': 1, 'open': 2, 'close': 3},
            solver = FakeSolver(),
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: False if literal.symbol == 'close' else None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [weak, strong], state = state)
    strategy = Strategy.strategies()['SATSolveLearnInitial'](env)

    assert strategy.step() is strong

    rows = strategy.training_rows()
    assert len(rows) == 1
    assert rows[0]['schema'] == 'sat_solve_learn_initial_bandit/v1'
    assert rows[0]['chosen_action_id'] == 'ex1'
    assert rows[0]['default_action_id'] == 'ex0'
    assert {candidate['action_id'] for candidate in rows[0]['candidates']} == {'ex0', 'ex1'}
    assert rows[0]['chosen_arm'] == 'whole'
    assert 'learn=schema:sat_solve_learn_initial_bandit/v1' in strategy.final_info()

def test_sat_solve_learn_initial_keeps_literal_score_primary():
    true_action = Extension(
        id = 'ex0',
        lit_idx = 0,
        clause_copy = [Literal('true_head', []), Literal('open', [])],
    )
    unknown_action = Extension(
        id = 'ex1',
        lit_idx = 0,
        clause_copy = [Literal('unknown_head', []), Literal('close', [])],
    )

    class FakeSolver:
        def score(self, atom_id):
            return 0

    goal = SimpleNamespace(parent = object(), depth = 2, path = lambda: [])
    state = SimpleNamespace(
        goal = goal,
        max_depth = 2,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {(1,), (2,), (3,)},
            atom_map = {'true_head': 1, 'unknown_head': 2, 'close': 3, 'open': 4},
            solver = FakeSolver(),
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: True if literal.symbol == 'true_head' else None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [unknown_action, true_action], state = state)
    strategy = Strategy.strategies()['SATSolveLearnInitial'](env)

    assert strategy.step() is true_action
    assert strategy.training_rows() == []

def test_sat_solve_learn_initial_rewards_ground_progress():
    weak = Extension(
        id = 'ex0',
        lit_idx = 0,
        clause_copy = [Literal('head', []), Literal('open', [])],
    )
    strong = Extension(
        id = 'ex1',
        lit_idx = 0,
        clause_copy = [Literal('head', []), Literal('close', [])],
    )

    class FakeSolver:
        def score(self, atom_id):
            return 0

    goal = SimpleNamespace(parent = object(), depth = 3, path = lambda: [Literal('path', [])])
    shadow = SimpleNamespace(
        new_ground_clauses = set(),
        ground_clauses = {(2,), (2, 3), (3, 4)},
        atom_map = {'head': 1, 'open': 2, 'close': 3},
        solver = FakeSolver(),
    )
    state = SimpleNamespace(
        goal = goal,
        max_depth = 3,
        shadow = shadow,
        substitution = Substitution(),
        sat_literal_value = lambda literal: False if literal.symbol == 'close' else None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
        proof_sequence = [],
        is_terminal = False,
        info = None,
    )
    env = SimpleNamespace(legal_actions = lambda: [weak, strong], state = state)
    strategy = Strategy.strategies()['SATSolveLearnInitial'](env)

    chosen = strategy.step()
    assert chosen is strong

    state.proof_sequence = [chosen]
    state.shadow.ground_clauses = {(2,), (2, 3), (3, 4), (5,)}
    strategy.step()

    rows = strategy.training_rows()
    assert rows[0]['reward'] == 1.0
    assert rows[0]['outcome'] == 'ground'

def test_sat_solve_learn_train_persists_and_eval_reads_store(tmp_path, monkeypatch):
    weak = Extension(
        id = 'ex0',
        lit_idx = 0,
        clause_copy = [Literal('head', []), Literal('open', [])],
    )
    strong = Extension(
        id = 'ex1',
        lit_idx = 0,
        clause_copy = [Literal('head', []), Literal('close', [])],
    )

    class FakeSolver:
        def score(self, atom_id):
            return 0

    def make_state():
        goal = SimpleNamespace(parent = object(), depth = 3, path = lambda: [Literal('path', [])])
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {(2,), (2, 3), (3, 4)},
            atom_map = {'head': 1, 'open': 2, 'close': 3},
            solver = FakeSolver(),
        )
        state = SimpleNamespace(
            goal = goal,
            max_depth = 3,
            shadow = shadow,
            substitution = Substitution(),
            sat_literal_value = lambda literal: False if literal.symbol == 'close' else None,
            _atom_key = lambda literal: literal.symbol,
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
            proof_sequence = [],
            is_terminal = False,
            info = None,
        )
        state.enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled)
        return state

    store_path = tmp_path / 'learn_store'
    monkeypatch.setattr(sat_solve_learn_module.SATSolveLearnTrain, '_learn_store_path', store_path)
    monkeypatch.setattr(sat_solve_learn_module.SATSolveLearnEval, '_learn_store_path', store_path)

    train_state = make_state()
    train_env = SimpleNamespace(legal_actions = lambda: [weak, strong], state = train_state)
    train_strategy = Strategy.strategies()['SATSolveLearnTrain'](train_env)

    chosen = train_strategy.step()
    assert chosen is strong

    train_state.proof_sequence = [chosen]
    train_state.shadow.ground_clauses = {(2,), (2, 3), (3, 4), (5,)}
    assert 'learn_store=mode:train' in train_strategy.final_info()

    store_files = list(store_path.glob('forward-*.json'))
    assert len(store_files) == 1
    stored = json.loads(store_files[0].read_text(encoding = 'utf-8'))
    assert stored['forward']['extensions_only']['whole']['pulls'] == 1
    assert stored['forward']['extensions_only']['whole']['reward'] == 1.0

    eval_state = make_state()
    eval_env = SimpleNamespace(legal_actions = lambda: [weak, strong], state = eval_state)
    eval_strategy = Strategy.strategies()['SATSolveLearnEval'](eval_env)

    assert eval_strategy.step() is strong
    assert eval_strategy.training_rows() == []
    assert 'learn_eval=schema:sat_solve_learn_initial_bandit/v1' in eval_strategy.final_info()

def test_sat_solve_learn_control_can_prefer_reset():
    class FakeSolver:
        def score(self, atom_id):
            return 0

    goal_parent = SimpleNamespace(parent = object())
    env = SimpleNamespace(
        legal_actions = lambda: [Backtrack(id = 'bt'), Reset(id = 'rs', depth = 5)],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = goal_parent),
            max_depth = 4,
            shadow = SimpleNamespace(
                new_ground_clauses = set(),
                atom_map = {},
                solver = FakeSolver(),
                ground_clauses = set(),
            ),
            sat_literal_value = lambda literal: None,
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
            proof_sequence = [],
            is_terminal = False,
            info = None,
        ),
    )
    env.state.enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled)
    strategy = Strategy.strategies()['SATResetBandit'](env)

    for arm in strategy._control_arms:
        strategy._control_pulls['shadow_stagnant'][arm] = 4
    strategy._control_rewards['shadow_stagnant']['bt1'] = 1.0
    strategy._control_rewards['shadow_stagnant']['bt2'] = 2.0
    strategy._control_rewards['shadow_stagnant']['reset'] = 5.0

    action = strategy.step()
    assert isinstance(action, Reset)
    assert action.depth == 5

def test_sat_solve_learn_control_can_emit_backjump_two():
    class FakeSolver:
        def score(self, atom_id):
            return 0

    goal_parent = SimpleNamespace(parent = object())
    env = SimpleNamespace(
        legal_actions = lambda: [Backtrack(id = 'bt'), Reset(id = 'rs', depth = 5)],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = goal_parent),
            max_depth = 4,
            shadow = SimpleNamespace(
                new_ground_clauses = set(),
                atom_map = {},
                solver = FakeSolver(),
                ground_clauses = set(),
            ),
            sat_literal_value = lambda literal: None,
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
            proof_sequence = [],
            is_terminal = False,
            info = None,
        ),
    )
    env.state.enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled)
    strategy = Strategy.strategies()['SATResetBandit'](env)

    for arm in strategy._control_arms:
        strategy._control_pulls['shadow_stagnant'][arm] = 4
    strategy._control_rewards['shadow_stagnant']['bt1'] = 1.0
    strategy._control_rewards['shadow_stagnant']['bt2'] = 5.0
    strategy._control_rewards['shadow_stagnant']['reset'] = 2.0

    action = strategy.step()
    assert isinstance(action, Backtrack)
    assert action.steps == 2

def test_sat_reset_shadow_prefers_same_depth_reset_while_shadow_grows():
    class FakeSolver:
        def score(self, atom_id):
            return 0

    env = SimpleNamespace(
        legal_actions = lambda: [Backtrack(id = 'bt'), Reset(id = 'rs', depth = 5)],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            max_depth = 4,
            shadow = SimpleNamespace(
                new_ground_clauses = {(1,), (2,)},
                atom_map = {},
                solver = FakeSolver(),
                ground_clauses = {(1,), (2,)},
            ),
            sat_literal_value = lambda literal: None,
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
            proof_sequence = [],
            is_terminal = False,
            info = None,
        ),
    )
    env.state.enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled)
    strategy = Strategy.strategies()['SATResetShadow'](env)

    action = strategy.step()
    assert isinstance(action, Reset)
    assert action.depth == 4
    assert action.reason == 'shadow_reset'

def test_sat_reset_at_stale_backtrack_deepens_once_shadow_stalls():
    class FakeSolver:
        def score(self, atom_id):
            return 0

    env = SimpleNamespace(
        legal_actions = lambda: [Backtrack(id = 'bt'), Reset(id = 'rs', depth = 5)],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            max_depth = 4,
            shadow = SimpleNamespace(
                new_ground_clauses = set(),
                atom_map = {},
                solver = FakeSolver(),
                ground_clauses = {(1,), (2,)},
            ),
            sat_literal_value = lambda literal: None,
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
            proof_sequence = [],
            is_terminal = False,
            info = None,
        ),
    )
    env.state.enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled)
    strategy = Strategy.strategies()['SATResetAtStaleBacktrack'](env)

    action = strategy.step()
    assert isinstance(action, Reset)
    assert action.depth == 5
    assert action.reason == 'stale_deepen'

def test_sat_reset_at_stale_action_can_proactively_deepen_after_budget():
    class FakeSolver:
        def score(self, atom_id):
            return 0

    action = Extension(id = 'ex0')
    env = SimpleNamespace(
        legal_actions = lambda: [action, Reset(id = 'rs', depth = 5)],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            max_depth = 4,
            shadow = SimpleNamespace(
                new_ground_clauses = set(),
                atom_map = {},
                solver = FakeSolver(),
                ground_clauses = set(),
            ),
            sat_literal_value = lambda literal: None,
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
        ),
    )
    env.state.enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled)
    strategy = Strategy.strategies()['SATResetAtStaleAction'](env)
    strategy._stale_action_count = strategy.stale_action_budget()

    reset = strategy.step()
    assert isinstance(reset, Reset)
    assert reset.depth == 5
    assert reset.reason == 'stale_action_reset'

def test_sat_reset_step_budget_can_proactively_reset():
    class FakeSolver:
        def score(self, atom_id):
            return 0

    action = Extension(id = 'ex0')
    env = SimpleNamespace(
        legal_actions = lambda: [action, Reset(id = 'rs', depth = 5)],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            max_depth = 4,
            shadow = SimpleNamespace(
                new_ground_clauses = set(),
                atom_map = {},
                solver = FakeSolver(),
                ground_clauses = set(),
            ),
            sat_literal_value = lambda literal: None,
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
        ),
    )
    env.state.enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled)
    strategy = Strategy.strategies()['SATResetStepBudget'](env)
    strategy._since_reset_steps = 8

    reset = strategy.step()
    assert isinstance(reset, Reset)
    assert reset.depth == 4
    assert reset.reason == 'step_budget'

def test_sat_reset_extension_budget_can_proactively_reset():
    class FakeSolver:
        def score(self, atom_id):
            return 0

    action = Extension(id = 'ex0')
    env = SimpleNamespace(
        legal_actions = lambda: [action, Reset(id = 'rs', depth = 5)],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            max_depth = 4,
            shadow = SimpleNamespace(
                new_ground_clauses = set(),
                atom_map = {},
                solver = FakeSolver(),
                ground_clauses = set(),
            ),
            sat_literal_value = lambda literal: None,
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
        ),
    )
    env.state.enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled)
    strategy = Strategy.strategies()['SATResetExtensionBudget'](env)
    strategy._since_reset_extensions = 4

    reset = strategy.step()
    assert isinstance(reset, Reset)
    assert reset.depth == 4
    assert reset.reason == 'extension_budget'

def test_sat_solve_learn_combined_uses_forward_tie_breaking():
    weak = Extension(
        id = 'ex0',
        lit_idx = 0,
        clause_copy = [Literal('head', []), Literal('open', [])],
    )
    strong = Extension(
        id = 'ex1',
        lit_idx = 0,
        clause_copy = [Literal('head', []), Literal('close', [])],
    )

    class FakeSolver:
        def score(self, atom_id):
            return 0

    goal = SimpleNamespace(parent = object(), depth = 4, path = lambda: [Literal('path', [])])
    state = SimpleNamespace(
        goal = goal,
        max_depth = 4,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {(2,), (2, 3), (3, 4)},
            atom_map = {'head': 1, 'open': 2, 'close': 3},
            solver = FakeSolver(),
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: False if literal.symbol == 'close' else None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        proof_sequence = [],
        is_terminal = False,
        info = None,
    )
    state.enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled)
    env = SimpleNamespace(legal_actions = lambda: [weak, strong], state = state)
    strategy = Strategy.strategies()['SATSolveLearnCombined'](env)

    assert strategy.step() is strong
    assert len(strategy.training_rows()) == 1

def test_sat_solve_learn_combined_uses_control_bandit():
    class FakeSolver:
        def score(self, atom_id):
            return 0

    goal_parent = SimpleNamespace(parent = object())
    env = SimpleNamespace(
        legal_actions = lambda: [Backtrack(id = 'bt'), Reset(id = 'rs', depth = 5)],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = goal_parent),
            max_depth = 4,
            shadow = SimpleNamespace(
                new_ground_clauses = set(),
                atom_map = {},
                solver = FakeSolver(),
                ground_clauses = set(),
            ),
            sat_literal_value = lambda literal: None,
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
            proof_sequence = [],
            is_terminal = False,
            info = None,
        ),
    )
    env.state.enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled)
    strategy = Strategy.strategies()['SATSolveLearnCombined'](env)

    for arm in strategy._control_arms:
        strategy._control_pulls['shadow_stagnant'][arm] = 4
    strategy._control_rewards['shadow_stagnant']['bt1'] = 1.0
    strategy._control_rewards['shadow_stagnant']['bt2'] = 2.0
    strategy._control_rewards['shadow_stagnant']['reset'] = 5.0

    assert isinstance(strategy.step(), Reset)

def test_sat_solve_learn_sat_context_uses_sat_aware_forward_context():
    weak = Extension(
        id = 'ex0',
        lit_idx = 0,
        clause_copy = [Literal('head', []), Literal('open', [])],
    )
    strong = Extension(
        id = 'ex1',
        lit_idx = 0,
        clause_copy = [Literal('head', []), Literal('close', [])],
    )

    class FakeSolver:
        def score(self, atom_id):
            return {1: 0.0, 2: 0.0, 3: 4.5}.get(atom_id, 0.0)

        def lookahead(self):
            return 3

    goal = SimpleNamespace(parent = object(), depth = 4, path = lambda: [Literal('path', [])])
    state = SimpleNamespace(
        goal = goal,
        max_depth = 4,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {(2,), (2, 3), (3, 4)},
            atom_map = {'head': 1, 'open': 2, 'close': 3},
            solver = FakeSolver(),
            model = {},
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        proof_sequence = [],
        is_terminal = False,
        info = None,
    )
    state.enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled)
    env = SimpleNamespace(legal_actions = lambda: [weak, strong], state = state)
    strategy = Strategy.strategies()['SATSolveLearnSATContext'](env)

    strategy._forward_explore = False
    for context in strategy._forward_contexts:
        for arm in strategy._forward_arms:
            strategy._forward_pulls[context][arm] = 3

    target = 'extensions_only|lookahead_hit|evsids'
    strategy._forward_rewards[target]['default'] = 0.0
    strategy._forward_rewards[target]['short'] = 0.0
    strategy._forward_rewards[target]['lookahead'] = 6.0
    strategy._forward_rewards[target]['whole'] = 1.0

    assert strategy.step() is strong
    assert 'learn=schema:sat_solve_learn_sat_context_forward_bandit/v1' in strategy.final_info()

def test_sat_solve_learn_sat_context_uses_sat_aware_control_context():
    class FakeSolver:
        def score(self, atom_id):
            return 0.0

    goal_parent = SimpleNamespace(parent = object())
    env = SimpleNamespace(
        legal_actions = lambda: [Backtrack(id = 'bt'), Reset(id = 'rs', depth = 5)],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = goal_parent),
            max_depth = 4,
            shadow = SimpleNamespace(
                new_ground_clauses = set(),
                atom_map = {},
                solver = FakeSolver(),
                ground_clauses = set(),
                model = {1: True},
            ),
            sat_literal_value = lambda literal: None,
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
            proof_sequence = [],
            is_terminal = False,
            info = None,
        ),
    )
    env.state.enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled)
    strategy = Strategy.strategies()['SATSolveLearnSATContext'](env)
    strategy._sat_nonzero_atom_score = 1.0

    strategy._control_explore = False if hasattr(strategy, '_control_explore') else False
    for context in strategy._control_contexts:
        for arm in strategy._control_arms:
            strategy._control_pulls[context][arm] = 4

    target = 'shadow_stagnant|model|scored'
    strategy._control_rewards[target]['bt1'] = 1.0
    strategy._control_rewards[target]['bt2'] = 2.0
    strategy._control_rewards[target]['reset'] = 5.0

    action = strategy.step()
    assert isinstance(action, Reset)
    assert action.depth == 5
    assert 'control=schema:sat_solve_learn_sat_context_control_bandit/v1' in strategy.final_info()

def test_sat_solve_literal_frequency_tie_prefers_frequent_unknown_atom():
    weak = Extension(id = 'ex0', lit_idx = 0, clause_copy = [Literal('weak', [])])
    strong = Extension(id = 'ex1', lit_idx = 0, clause_copy = [Literal('strong', [])])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {
                (1, 2),
                (-1, 2),
                (2, 3),
            },
            atom_map = {'weak': 1, 'strong': 2},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [weak, strong], state = state)
    strategy = Strategy.strategies()['SATSolveLiteralFrequencyTie'](env)

    assert strategy.step() is strong

    info = strategy.final_info()
    assert 'freq=mode:high' in info
    assert 'decisions:1' in info
    assert 'candidates:2' in info
    assert 'mapped:2' in info
    assert 'changed_default:1' in info

def test_sat_solve_literal_frequency_tie_keeps_model_truth_primary():
    true_action = Extension(id = 'ex0', lit_idx = 0, clause_copy = [Literal('model_true', [])])
    frequent_unknown = Extension(id = 'ex1', lit_idx = 0, clause_copy = [Literal('frequent', [])])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {
                (2,),
                (-2, 3),
                (2, 4),
            },
            atom_map = {'model_true': 1, 'frequent': 2},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: True if literal.symbol == 'model_true' else None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [true_action, frequent_unknown], state = state)
    strategy = Strategy.strategies()['SATSolveLiteralFrequencyTie'](env)

    assert strategy.step() is true_action
    assert 'freq=' not in strategy.final_info()

def test_sat_solve_literal_low_frequency_tie_prefers_rare_unknown_atom():
    rare = Extension(id = 'ex0', lit_idx = 0, clause_copy = [Literal('rare', [])])
    common = Extension(id = 'ex1', lit_idx = 0, clause_copy = [Literal('common', [])])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {
                (1, 2),
                (2, 3),
                (-2, 4),
            },
            atom_map = {'rare': 1, 'common': 2},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [common, rare], state = state)
    strategy = Strategy.strategies()['SATSolveLiteralLowFrequencyTie'](env)

    assert strategy.step() is rare
    info = strategy.final_info()
    assert 'mode:low' in info
    assert 'kind:atom' in info
    assert 'changed_default:1' in info

def test_sat_solve_literal_opposite_signed_frequency_tie_uses_opposite_polarity():
    positive = Extension(id = 'ex0', lit_idx = 0, clause_copy = [Literal('p', [])])
    negative = Extension(id = 'ex1', lit_idx = 0, clause_copy = [Literal('p', [], neg = True)])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {
                (1, 2),
                (1, 3),
                (-1, 4),
            },
            atom_map = {'p': 1},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )

    opposite_env = SimpleNamespace(legal_actions = lambda: [positive, negative], state = state)
    opposite_strategy = Strategy.strategies()['SATSolveLiteralOppositeSignedFrequencyTie'](opposite_env)
    assert opposite_strategy.step() is negative
    assert 'kind:opposite_signed' in opposite_strategy.final_info()

def test_sat_solve_literal_rare_start_prefers_clause_with_rare_atom():
    common_start = Start(id = 'st0', clause_copy = [Literal('common', []), Literal('common2', [])])
    rare_start = Start(id = 'st1', clause_copy = [Literal('rare', [])])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = None),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {
                (1, 2),
                (1, 3),
                (1, 4),
                (5,),
            },
            atom_map = {'common': 1, 'common2': 2, 'rare': 5},
            model = {},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(
        legal_actions = lambda: [common_start, rare_start, Reset(id = 'rs', depth = 2)],
        state = state,
    )
    strategy = Strategy.strategies()['SATSolveLiteralRareStart'](env)

    assert strategy.step() is rare_start
    info = strategy.final_info()
    assert 'start=mode:rare' in info
    assert 'decisions:1' in info
    assert 'candidates:2' in info
    assert 'mapped:2' in info
    assert 'changed_default:1' in info

def test_sat_solve_literal_model_start_prefers_model_false_goals():
    rare_start = Start(id = 'st0', clause_copy = [Literal('rare', [])])
    model_false_start = Start(id = 'st1', clause_copy = [Literal('model_false', [])])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = None),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {
                (1,),
                (2,),
                (2, 3),
            },
            atom_map = {'rare': 1, 'model_false': 2},
            model = {1: True, 2: False},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [rare_start, model_false_start], state = state)
    strategy = Strategy.strategies()['SATSolveLiteralModelStart'](env)

    assert strategy.step() is model_false_start
    info = strategy.final_info()
    assert 'start=mode:model' in info
    assert 'model_false:1' in info
    assert 'model_true:1' in info
    assert 'changed_default:1' in info

def test_sat_solve_literal_rare_start_does_not_change_extension_choice():
    common = Extension(id = 'ex0', lit_idx = 0, clause_copy = [Literal('common', [])])
    rare = Extension(id = 'ex1', lit_idx = 0, clause_copy = [Literal('rare', [])])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {
                (1,),
                (1, 2),
                (3,),
            },
            atom_map = {'common': 1, 'rare': 3},
            model = {},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [common, rare], state = state)
    strategy = Strategy.strategies()['SATSolveLiteralRareStart'](env)

    assert strategy.step() is common
    assert 'start=' not in strategy.final_info()

def test_sat_solve_literal_first_start_guides_only_fresh_root_list():
    rare_first = Start(id = 'st0', clause_copy = [Literal('rare', [])])
    common_second = Start(id = 'st1', clause_copy = [Literal('common', []), Literal('common2', [])])
    rare_remaining = Start(id = 'st0', clause_copy = [Literal('rare', [])])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = None),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {
                (1,),
                (2,),
                (2, 3),
            },
            atom_map = {'rare': 1, 'common': 2, 'common2': 3},
            model = {},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _start_clause_indices = lambda: [0, 1],
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )

    calls = {'count': 0}
    def legal_actions():
        calls['count'] += 1
        if calls['count'] == 1:
            return [common_second, rare_first, Reset(id = 'rs', depth = 2)]
        return [rare_remaining, Reset(id = 'rs', depth = 2)]

    env = SimpleNamespace(legal_actions = legal_actions, state = state)
    strategy = Strategy.strategies()['SATSolveLiteralFirstStart'](env)

    assert strategy.step() is rare_first
    assert strategy.step() is rare_remaining

    info = strategy.final_info()
    assert 'start=mode:model' in info
    assert 'decisions:1' in info
    assert 'candidates:2' in info

def test_sat_solve_literal_first_start_scores_only_bounded_prefix():
    starts = [
        Start(id = f'st{i}', clause_copy = [Literal(f'p{i}', [])])
        for i in range(40)
    ]

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = None),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {(i + 1,) for i in range(40)},
            atom_map = {f'p{i}': i + 1 for i in range(40)},
            model = {},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _start_clause_indices = lambda: list(range(40)),
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: starts, state = state)
    strategy = Strategy.strategies()['SATSolveLiteralFirstStart'](env)

    assert strategy.step() in starts[:32]
    assert 'candidates:32' in strategy.final_info()

def test_sat_solve_literal_friday_start_cut_resets_after_confident_root_choice_fails():
    weak = Start(id = 'st0', clause_copy = [Literal('weak', [])])
    strong = Start(id = 'st1', clause_copy = [Literal('strong', [])])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = None),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {(1,), (2,), (2, 3)},
            atom_map = {'weak': 1, 'strong': 2},
            model = {1: True, 2: False},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _start_clause_indices = lambda: [0, 1],
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [weak, strong, Reset(id = 'rs', depth = 2)], state = state)
    strategy = Strategy.strategies()['SATSolveLiteralFridayStartCut'](env)

    assert strategy.step() is strong

    env.legal_actions = lambda: [weak, Reset(id = 'rs', depth = 2)]
    action = strategy.step()

    assert isinstance(action, Reset)
    assert action.depth == 2
    info = strategy.final_info()
    assert 'friday=' in info
    assert 'confident_starts:1' in info
    assert 'commit_resets:1' in info

def test_sat_solve_literal_friday_root_reset_fires_after_stale_root_revisits():
    state = SimpleNamespace(
        goal = SimpleNamespace(parent = None),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {(1,), (2,), (3,)},
            atom_map = {'a': 1, 'b': 2},
            model = {},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _start_clause_indices = lambda: [0, 1],
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    start = Start(id = 'st0', clause_copy = [Literal('a', [])])
    env = SimpleNamespace(legal_actions = lambda: [start, Reset(id = 'rs', depth = 2)], state = state)
    strategy = Strategy.strategies()['SATSolveLiteralFridayRootReset'](env)

    for _ in range(strategy._friday_root_visit_limit - 1):
        assert strategy.step() is start

    action = strategy.step()

    assert isinstance(action, Reset)
    assert action.depth == 2
    assert 'stagnation_resets:1' in strategy.final_info()

def test_sat_solve_literal_friday_hybrid_guides_only_first_fresh_root_then_defaults():
    rare_first = Start(id = 'st0', clause_copy = [Literal('rare', [])])
    common_second = Start(id = 'st1', clause_copy = [Literal('common', []), Literal('common2', [])])
    rare_remaining = Start(id = 'st0', clause_copy = [Literal('rare', [])])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = None),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = {(1,), (2,), (3,)},
            atom_map = {'rare': 1, 'common': 2, 'common2': 3},
            model = {},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _start_clause_indices = lambda: [0, 1],
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )

    calls = {'count': 0}
    def legal_actions():
        calls['count'] += 1
        if calls['count'] == 1:
            return [common_second, rare_first, Reset(id = 'rs', depth = 2)]
        return [rare_remaining, Reset(id = 'rs', depth = 2)]

    env = SimpleNamespace(legal_actions = legal_actions, state = state)
    strategy = Strategy.strategies()['SATSolveLiteralFridayHybrid'](env)

    assert strategy.step() is rare_first
    assert strategy.step() is rare_remaining

def test_sat_solve_propagation_prefers_failed_assumption(monkeypatch):
    monkeypatch.setattr(sat_solve_module, 'pydical', SimpleNamespace(UNSATISFIABLE = 20))

    weak = Extension(id = 'ex0', lit_idx = 0, clause_copy = [Literal('weak', [])])
    strong = Extension(id = 'ex1', lit_idx = 0, clause_copy = [Literal('strong', [])])

    class FakeSolver:
        def __init__(self):
            self.assumptions = []
            self.calls = []

        def assume(self, sat_lit):
            self.calls.append(('assume', sat_lit))
            self.assumptions.append(sat_lit)

        def solve(self):
            self.calls.append(('solve', tuple(self.assumptions)))
            return 20 if self.assumptions == [2] else 10

        def reset_assumptions(self):
            self.calls.append(('reset_assumptions', tuple(self.assumptions)))
            self.assumptions = []

        def score(self, atom_id):
            return 0

    solver = FakeSolver()
    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = set(),
            atom_map = {'weak': 1, 'strong': 2},
            solver = solver,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [weak, strong], state = state)
    strategy = Strategy.strategies()['SATSolvePropagation'](env)
    monkeypatch.setattr(strategy, '_pysat_propagation', lambda sat_lit: None)

    assert strategy.step() is strong
    assert strong.cut is True
    assert ('assume', 1) in solver.calls
    assert ('assume', 2) in solver.calls
    assert 'prop=backend:cadical-fallback' in strategy.final_info()

def test_sat_solve_literal_propagation_keeps_literal_value_primary(monkeypatch):
    monkeypatch.setattr(sat_solve_module, 'pydical', SimpleNamespace(UNSATISFIABLE = 20))

    true_action = Extension(id = 'ex0', lit_idx = 0, clause_copy = [Literal('model_true', [])])
    false_action = Extension(id = 'ex1', lit_idx = 0, clause_copy = [Literal('model_false', [])])

    class FakeSolver:
        def __init__(self):
            self.assumptions = []

        def assume(self, sat_lit):
            self.assumptions.append(sat_lit)

        def solve(self):
            return 20 if self.assumptions == [2] else 10

        def reset_assumptions(self):
            self.assumptions = []

        def score(self, atom_id):
            return 0

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            ground_clauses = set(),
            atom_map = {'model_true': 1, 'model_false': 2},
            solver = FakeSolver(),
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: literal.symbol == 'model_true',
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [false_action, true_action], state = state)
    strategy = Strategy.strategies()['SATSolveLiteralPropagation'](env)
    monkeypatch.setattr(strategy, '_pysat_propagation', lambda sat_lit: None)

    assert strategy.step() is true_action

def test_sat_solve_literal_reports_choice_diagnostics():
    mapped = Extension(id = 'ex0', lit_idx = 0, clause_copy = [Literal('mapped', [])])
    unmapped = Extension(id = 'ex1', lit_idx = 0, clause_copy = [Literal('unmapped', [])])

    state = SimpleNamespace(
        goal = SimpleNamespace(parent = object()),
        max_depth = 1,
        shadow = SimpleNamespace(
            new_ground_clauses = set(),
            atom_map = {'mapped': 1},
            solver = None,
        ),
        substitution = Substitution(),
        sat_literal_value = lambda literal: None,
        _atom_key = lambda literal: literal.symbol,
        _flush_new_ground_clauses = lambda: None,
        _solve_ground_clauses = lambda: None,
        enable_paper_mode = lambda enabled = True: setattr(state, 'paper_mode', enabled),
    )
    env = SimpleNamespace(legal_actions = lambda: [mapped, unmapped], state = state)
    strategy = Strategy.strategies()['SATSolveLiteral'](env)

    assert strategy.step() is mapped

    info = strategy.final_info()
    assert 'literal=choices:1' in info
    assert 'forward:2' in info
    assert 'ties:1' in info
    assert 'avg_tie:2.00' in info
    assert 'best_unknown:1' in info
    assert 'tied_mappable:2' in info
    assert 'tied_mapped:1' in info
    assert 'chosen_extension:1' in info

def test_sat_solve_records_nonzero_atom_score():
    class FakeSolver:
        def score(self, atom_id):
            return {1: 0, 2: 3.5}[atom_id]

    action = Extension(id = 'ex0')
    env = SimpleNamespace(
        legal_actions = lambda: [action],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            shadow = SimpleNamespace(
                atom_map = {'p': 1, 'q': 2},
                new_ground_clauses = set(),
                solver = FakeSolver(),
            ),
            sat_literal_value = lambda literal: None,
            enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled),
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
        ),
    )
    strategy = Strategy.strategies()['SATSolve'](env)

    assert strategy.step() is action
    assert strategy.final_info() == 'score=3.5'

@pytest.mark.parametrize(
    'strategy_name',
    [
        'SATResetBandit',
        'SATSolve',
        'SATSolveGuided',
        'SATSolveLearnInitial',
        'SATSolveLearnCombined',
        'SATSolveLiteral',
        'SATSolveLiteralRandomTie',
        'SATSolveLiteralShortTie',
        'SATSolveLiteralLookaheadTie',
        'SATSolveLiteralFrequencyTie',
        'SATSolveLiteralLowFrequencyTie',
        'SATSolveLiteralOppositeSignedFrequencyTie',
        'SATSolveLiteralFirstStart',
        'SATSolveLiteralRareStart',
        'SATSolveLiteralModelStart',
        'SATSolvePropagation',
        'SATSolveLiteralPropagation',
    ],
)
def test_sat_strategies_only_reset_at_root_when_shadow_is_still_growing(strategy_name):
    class FakeSolver:
        def score(self, atom_id):
            return 0

    env = SimpleNamespace(
        legal_actions = lambda: [Backtrack(id = 'bt'), Reset(id = 'rs', depth = 3)],
        state = SimpleNamespace(
            goal = SimpleNamespace(parent = object()),
            shadow = SimpleNamespace(new_ground_clauses = {(1,)}, atom_map = {}, solver = FakeSolver()),
            sat_literal_value = lambda literal: None,
            enable_paper_mode = lambda enabled = True: setattr(env.state, 'paper_mode', enabled),
            _flush_new_ground_clauses = lambda: None,
            _solve_ground_clauses = lambda: None,
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

@pytest.mark.parametrize('strategy_name', ['SATSolve', 'SATSolveGuided', 'SATSolveLiteral'])
@pytest.mark.skipif(pydical is None, reason = 'pydical is not installed')
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
