import os
import pytest

import connections.calculi.classicalsat as classicalsat
import strategies.sat_solve as sat_solve_module
from types import SimpleNamespace

from connections.env import ConnectionEnv, Settings
from connections.calculi.classicalsat import Backtrack, Extension, Reduction, Reset, SATConnectionState, SATSolver, ShadowSAT, pydical
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
    assert 'SATSolve' in Strategy.strategies()
    assert 'SATSolveGuided' in Strategy.strategies()
    assert 'SATSolveGuidedProbeConflicts' in Strategy.strategies()
    assert 'SATSolveGuidedForcedTrue' in Strategy.strategies()
    assert 'SATSolveGuidedEVSIDS' in Strategy.strategies()
    assert 'SATSolveGuidedShort' in Strategy.strategies()
    assert 'SATSolveGuidedFast' in Strategy.strategies()
    assert 'SATSolveLiteral' in Strategy.strategies()
    assert 'SATSolveLiteralRandomTie' in Strategy.strategies()
    assert 'SATSolveLiteralShortTie' in Strategy.strategies()
    assert 'SATSolveLiteralLookaheadTie' in Strategy.strategies()
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

def test_unsat_shadow_records_current_ground_clauses_for_reporting():
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
    assert set(state.shadow.unsat_core) == {(1,), (-1,), (2, 3)}
    done, info = state.initial_status()
    assert done is True
    assert 'q(a)' in info
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
        'SATSolveLiteral',
        'SATSolveLiteralRandomTie',
        'SATSolveLiteralShortTie',
        'SATSolveLiteralLookaheadTie',
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
        'SATSolve',
        'SATSolveGuided',
        'SATSolveLiteral',
        'SATSolveLiteralRandomTie',
        'SATSolveLiteralShortTie',
        'SATSolveLiteralLookaheadTie',
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
