from pathlib import Path

import pytest

from cli import default_logic_for_strategy
from connections.calculi.classicalcut import Extension, Reduction, Reset, Start
from connections.calculi.leancop2 import Lemma
from connections.env import ConnectionEnv, Logic, Settings
from connections.utils.classical_problem_parsing import translate_tptp_problem
from strategies import Strategy
from strategies.custom import LeanCoP_2


def _write_matrix(tmp_path: Path, clauses: str) -> Path:
    path = tmp_path / 'problem.cnf'
    path.write_text(f'cnf(test,conjecture,{clauses}).', encoding='utf-8')
    return path


def _run_leancop2_to_completion(problem: Path, max_actions: int = 5_000) -> str:
    env = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2, step_limit=100_000),
    )
    strategy = LeanCoP_2(env)
    env.reset()

    for _ in range(max_actions):
        _, _, done, info = env.step(strategy.step())
        if done:
            assert isinstance(info, str)
            return info
    raise AssertionError(f'LeanCoP_2 did not terminate after {max_actions} actions')


def _assert_sound_closure_action(env: ConnectionEnv, action) -> None:
    goal = env.state.goal.literal
    if isinstance(action, Lemma):
        assert action.lemma_lit is not None
        assert goal.neg == action.lemma_lit.neg
        assert env.state.substitution.equal(goal, action.lemma_lit)
        return

    if isinstance(action, Reduction):
        other = action.path_lit
    elif isinstance(action, Extension):
        other = action.clause_copy[action.lit_idx]
    else:
        return

    assert other is not None
    assert goal.symbol == other.symbol
    assert goal.neg != other.neg
    env.state.substitution.update(action.sub_updates)
    try:
        assert env.state.substitution.equal(goal, other)
    finally:
        env.state.substitution.backtrack()


def _run_core_first_actions(problem: Path, *, cut: bool, depth: int = 3):
    env = ConnectionEnv(
        str(problem),
        Settings(
            logic=Logic.LeanCoP2,
            iterative_deepening=True,
            iterative_deepening_initial_depth=depth,
            leancop_cut=cut,
        ),
    )
    env.reset()

    for _ in range(100):
        assert len(env.state.substitution.trail) == len(env.state.proof_sequence)
        actions = env.legal_actions()
        if not actions:
            return env.state.info
        action = actions[0]
        if cut and isinstance(action, (Lemma, Reduction, Extension)):
            action.cut = True
        _, _, done, info = env.step(action)
        assert len(env.state.substitution.trail) == len(env.state.proof_sequence)
        if done:
            return info
    raise AssertionError('leanCoP core did not terminate after 100 actions')


def test_leancop2_is_public_and_uses_its_core_state():
    assert Strategy.strategies()['LeanCoP_2'] is LeanCoP_2
    assert default_logic_for_strategy('LeanCoP_2') is Logic.LeanCoP2


def test_scoped_lemma_closes_repeated_sibling(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p,p],[-(p)]]')
    env = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2, iterative_deepening=True),
    )
    env.reset()

    start = env.legal_actions()[0]
    assert isinstance(start, Start)
    env.step(start)

    extension = env.legal_actions()[0]
    assert isinstance(extension, Extension)
    env.step(extension)

    lemma = env.legal_actions()[0]
    assert isinstance(lemma, Lemma)
    _, _, done, info = env.step(lemma)
    assert done is True
    assert info == 'Theorem'


def test_scoped_lemma_does_not_close_opposite_polarity_sibling(tmp_path: Path):
    problem = _write_matrix(
        tmp_path,
        '[[p(_X),q(_X)],[-(p(a))],[-(p(_Y)),-(q(_Y))]]',
    )
    env = ConnectionEnv(
        str(problem),
        Settings(
            logic=Logic.LeanCoP2,
            iterative_deepening=True,
            iterative_deepening_initial_depth=3,
        ),
    )
    env.reset()

    env.step(env.legal_actions()[0])
    close_first_sibling = next(
        action
        for action in env.legal_actions()
        if isinstance(action, Extension) and len(action.clause_copy) == 1
    )
    env.step(close_first_sibling)
    extend_second_sibling = next(
        action
        for action in env.legal_actions()
        if isinstance(action, Extension) and len(action.clause_copy) == 2
    )
    env.step(extend_second_sibling)

    assert repr(env.state.goal.literal).startswith('-p(')
    assert not any(isinstance(action, Lemma) for action in env.legal_actions())


def test_lemma_requires_identity_and_does_not_unify_with_ground_sibling(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p(_X),p(a)],[-(p(_Y))]]')
    env = ConnectionEnv(
        str(problem),
        Settings(
            logic=Logic.LeanCoP2,
            iterative_deepening=True,
            iterative_deepening_initial_depth=3,
        ),
    )
    env.reset()

    env.step(env.legal_actions()[0])
    env.step(next(action for action in env.legal_actions() if isinstance(action, Extension)))

    assert repr(env.state.goal.literal) == 'p(a)'
    assert not any(isinstance(action, Lemma) for action in env.legal_actions())
    assert any(isinstance(action, Extension) for action in env.legal_actions())


def test_lemma_from_finished_descendant_does_not_escape_its_scope(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p,q],[-(p),q],[-(q)]]')
    env = ConnectionEnv(
        str(problem),
        Settings(
            logic=Logic.LeanCoP2,
            iterative_deepening=True,
            iterative_deepening_initial_depth=3,
        ),
    )
    env.reset()

    env.step(env.legal_actions()[0])
    env.step(
        next(
            action
            for action in env.legal_actions()
            if isinstance(action, Extension) and len(action.clause_copy) == 2
        )
    )
    env.step(
        next(
            action
            for action in env.legal_actions()
            if isinstance(action, Extension) and len(action.clause_copy) == 1
        )
    )

    assert repr(env.state.goal.literal) == 'q'
    assert [repr(literal) for literal in env.state._lemma_literals(env.state.goal)] == ['p']
    assert not any(isinstance(action, Lemma) for action in env.legal_actions())


def test_backtracking_unwinds_lemma_frame_before_earlier_extension(tmp_path: Path):
    problem = _write_matrix(
        tmp_path,
        '[[p(_X),p(_X),q(_X)],[-(p(b))],[-(p(a))],[-(q(a))]]',
    )

    # The first p-extension binds X=b, the second p closes by lemma, and q(b)
    # fails. Complete search must unwind both the empty lemma frame and X=b,
    # then try the p(a) extension and close q(a).
    assert _run_core_first_actions(problem, cut=False, depth=3) == 'Theorem'


def test_opposite_polarity_lemma_counterexample_is_not_proved(tmp_path: Path):
    problem = _write_matrix(
        tmp_path,
        '[[p(_X),q(_X)],[-(p(a))],[-(p(_Y)),-(q(_Y))]]',
    )
    assert _run_leancop2_to_completion(problem) == 'Non-Theorem: exhaustive complete search'


@pytest.mark.parametrize(
    'clauses',
    [
        '[[p]]',
        '[[p,q],[-(p),q]]',
        '[[p,q],[-(p),-(q)]]',
        '[[p],[-(p),p]]',
        '[[p(a),q(a)],[-(p(a)),q(a)],[p(a),-(q(a))]]',
        '[[r(a,b),s],[-(r(a,b)),s]]',
        '[[p(_X),q(_X)],[-(p(a))]]',
        '[[p(_X),q(_X)],[-(p(a))],[-(p(_Y)),-(q(_Y))]]',
    ],
    ids=[
        'single-positive-clause',
        'shared-positive-model',
        'exclusive-models',
        'tautological-cycle',
        'ground-arguments',
        'binary-predicate',
        'free-variable',
        'old-polarity-counterexample',
    ],
)
def test_complete_search_does_not_refute_satisfiable_micro_corpus(
    tmp_path: Path,
    clauses: str,
):
    problem = _write_matrix(tmp_path, clauses)

    assert _run_leancop2_to_completion(problem) == 'Non-Theorem: exhaustive complete search'


def test_reduction_requires_opposite_polarity(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p],[-(p),p]]')
    env = ConnectionEnv(
        str(problem),
        Settings(
            logic=Logic.LeanCoP2,
            iterative_deepening=True,
            iterative_deepening_initial_depth=3,
        ),
    )
    env.reset()

    env.step(env.legal_actions()[0])
    env.step(next(action for action in env.legal_actions() if isinstance(action, Extension)))

    assert repr(env.state.goal.literal) == 'p'
    assert repr(env.state.goal.path()[0]) == 'p'
    assert env.state.reductions() == []


def test_reduction_accepts_genuine_complement(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p],[-(p),-(p)]]')
    env = ConnectionEnv(
        str(problem),
        Settings(
            logic=Logic.LeanCoP2,
            iterative_deepening=True,
            iterative_deepening_initial_depth=3,
        ),
    )
    env.reset()

    env.step(env.legal_actions()[0])
    env.step(next(action for action in env.legal_actions() if isinstance(action, Extension)))

    reductions = env.state.reductions()
    assert len(reductions) == 1
    assert isinstance(reductions[0], Reduction)
    assert reductions[0].path_lit.neg != env.state.goal.literal.neg


@pytest.mark.parametrize(
    ('clauses', 'expected_closures'),
    [
        ('[[p,p],[-(p)]]', {Extension, Lemma}),
        ('[[p],[-(p),-(p)]]', {Extension, Reduction}),
    ],
    ids=['extension-and-lemma', 'extension-and-reduction'],
)
def test_every_action_in_small_proofs_obeys_signed_closure_invariants(
    tmp_path: Path,
    clauses: str,
    expected_closures: set[type],
):
    problem = _write_matrix(tmp_path, clauses)
    env = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2, step_limit=10_000),
    )
    strategy = LeanCoP_2(env)
    env.reset()
    seen_closures = set()

    for _ in range(100):
        action = strategy.step()
        _assert_sound_closure_action(env, action)
        if isinstance(action, (Lemma, Reduction, Extension)):
            seen_closures.add(type(action))
        _, _, done, info = env.step(action)
        if done:
            break
    else:
        raise AssertionError('small theorem search did not terminate')

    assert info == 'Theorem'
    assert seen_closures == expected_closures


def test_conjecture_phase_without_markers_falls_back_to_positive_starts(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p],[-(p)]]')
    env = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2, leancop_conjecture_start=True),
    )

    env.reset()

    starts = env.legal_actions()
    assert len(starts) == 1
    assert isinstance(starts[0], Start)
    assert repr(starts[0].clause_copy[0]) == 'p'

    _, _, done, _ = env.step(starts[0])
    assert done is False
    extension = next(
        action for action in env.legal_actions() if isinstance(action, Extension)
    )
    _, _, done, info = env.step(extension)
    assert done is True
    assert info == 'Theorem'


def test_negative_marker_switches_start_selection_to_conjecture_clauses(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p],[#,q],[-(#)]]')
    env = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2),
    )

    env.reset()

    starts = env.legal_actions()
    assert len(starts) == 1
    assert isinstance(starts[0], Start)
    assert [repr(literal) for literal in starts[0].clause_copy] == ['q']


def test_virtual_negative_marker_closes_positive_marker_subgoal(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[#,p],[-(#)],[-(p),#]]')
    env = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2),
    )
    env.reset()

    env.step(env.legal_actions()[0])
    env.step(next(action for action in env.legal_actions() if isinstance(action, Extension)))

    actions = env.legal_actions()
    assert isinstance(actions[0], Reduction)
    assert actions[0].id == 're_marker'
    _, _, done, info = env.step(actions[0])
    assert done is True
    assert info == 'Theorem'


def test_virtual_negative_marker_participates_in_regularity(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[#,p],[-(#)],[-(p),-(#)]]')
    env = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2),
    )
    env.reset()

    env.step(env.legal_actions()[0])
    env.step(next(action for action in env.legal_actions() if isinstance(action, Extension)))

    assert repr(env.state.goal.literal) == '-#'
    assert env.state.legal_forward_actions() == {}


def test_restricted_start_keeps_only_first_positive_clause(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p],[q],[-(p)],[-(q)]]')
    env = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2, leancop_scut=True),
    )
    state = env.reset()

    starts = [action for action in env.legal_actions() if isinstance(action, Start)]
    assert len(starts) == 1
    assert repr(starts[0].clause_copy[0]) == 'p'
    assert state.restricted_start is True


def test_restricted_start_matches_cut_sensitive_prolog_result(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p],[q],[-(q)]]')

    restricted = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2, leancop_scut=True),
    )
    restricted.reset()
    restricted.step(restricted.legal_actions()[0])
    restricted.step(restricted.legal_actions()[0])
    assert restricted.legal_actions() == []
    assert restricted.state.info is None

    assert _run_core_first_actions(problem, cut=False) == 'Theorem'


def test_ground_extension_is_allowed_at_path_limit(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p],[-(p)]]')
    env = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2, iterative_deepening=True),
    )
    state = env.reset()
    state.max_depth = 0
    env.step(env.legal_actions()[0])

    assert any(isinstance(action, Extension) for action in env.legal_actions())


def test_extensions_preserve_prolog_order_below_path_limit(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p(a)],[-(p(_X))],[-(p(a))]]')
    env = ConnectionEnv(
        str(problem),
        Settings(
            logic=Logic.LeanCoP2,
            iterative_deepening=True,
            iterative_deepening_initial_depth=3,
        ),
    )
    env.reset()
    env.step(env.legal_actions()[0])

    extensions = [
        action for action in env.legal_actions() if isinstance(action, Extension)
    ]

    assert len(extensions) == 2
    assert repr(extensions[0].clause_copy[0]).startswith('-p(_X')
    assert repr(extensions[1].clause_copy[0]) == '-p(a)'


def test_extension_order_matches_cut_sensitive_prolog_result(tmp_path: Path):
    problem = _write_matrix(
        tmp_path,
        '[[p(_X),q(_X)],[-(p(b)),r(_Y)],[-(r(_Z))],[-(p(a))],[-(q(a))]]',
    )

    # At path limit three, Prolog tries the first, nonground p-extension,
    # commits to X=b, and then cannot close q(b).  Without cut it backtracks to
    # the later ground p(a) extension and proves the matrix unsatisfiable.
    assert _run_core_first_actions(problem, cut=True, depth=3) is None
    assert _run_core_first_actions(problem, cut=False, depth=3) == 'Theorem'


def test_nonground_extension_records_path_limit_hit(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p(a)],[-(p(_X))]]')
    env = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2, iterative_deepening=True),
    )
    state = env.reset()
    env.step(env.legal_actions()[0])

    assert not any(isinstance(action, Extension) for action in env.legal_actions())
    assert state.path_limit_hit is True


def test_nonground_extension_becomes_available_at_next_prolog_depth(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p(a)],[-(p(_X)),q(_X)],[-(q(a))]]')
    env = ConnectionEnv(
        str(problem),
        Settings(
            logic=Logic.LeanCoP2,
            iterative_deepening=True,
            iterative_deepening_initial_depth=1,
        ),
    )
    env.reset()
    env.step(env.legal_actions()[0])

    assert not any(isinstance(action, Extension) for action in env.legal_actions())
    assert env.state.path_limit_hit is True

    env.step(env.state.resets()[0])
    env.step(env.legal_actions()[0])
    first_extension = next(
        action for action in env.legal_actions() if isinstance(action, Extension)
    )
    env.step(first_extension)
    second_extension = next(
        action for action in env.legal_actions() if isinstance(action, Extension)
    )
    _, _, done, info = env.step(second_extension)

    assert done is True
    assert info == 'Theorem'


def test_leancop21_schedule_matches_reference_shell():
    assert [(phase.share, phase.settings, phase.complete) for phase in LeanCoP_2.phases] == [
        (10, ('cut', 'comp(7)'), True),
        (15, ('conj', 'def', 'cut'), False),
        (15, ('nodef', 'scut', 'cut'), False),
        (10, ('scut',), False),
        (5, ('def', 'cut'), False),
        (4, ('conj', 'nodef', 'cut'), False),
        (2, ('def', 'scut', 'cut'), False),
        (2, ('scut', 'cut'), False),
        (1, ('conj', 'def'), False),
        (4, ('reo(40)', 'conj', 'nodef', 'scut', 'cut'), False),
        (4, ('reo(42)', 'def', 'scut', 'cut'), False),
        (2, ('reo(12)', 'def', 'scut', 'cut'), False),
        (2, ('reo(72)', 'def', 'scut', 'cut'), False),
        (2, ('reo(39)', 'nodef', 'cut'), False),
        (2, ('reo(38)', 'conj', 'def', 'cut'), False),
        (2, ('reo(15)', 'conj', 'def', 'cut'), False),
        (1, ('reo(73)', 'conj', 'def', 'cut'), False),
        (1, ('reo(57)', 'conj', 'def', 'cut'), False),
        (1, ('reo(13)', 'conj', 'nodef', 'scut', 'cut'), False),
        (1, ('reo(59)', 'conj', 'nodef', 'scut', 'cut'), False),
        (1, ('reo(75)', 'conj', 'nodef', 'scut', 'cut'), False),
        (1, ('reo(36)', 'conj', 'nodef', 'scut'), False),
        (1, ('reo(16)', 'conj', 'nodef', 'scut'), False),
        (1, ('reo(71)', 'def', 'scut'), False),
        (1, ('reo(58)', 'def', 'scut', 'cut'), False),
        (1, ('reo(76)', 'def', 'scut', 'cut'), False),
        (1, ('reo(74)', 'nodef', 'cut'), False),
        (1, ('reo(14)', 'nodef', 'cut'), False),
        (1, ('reo(37)', 'nodef', 'scut'), False),
        (99, ('def',), True),
    ]
    assert LeanCoP_2._schedule_weight == 194
    assert LeanCoP_2._schedule_denominator == 111


def test_raw_cnf_translation_adds_leancop_equality_axioms():
    matrix = translate_tptp_problem('tests/cnf_problems/GRP001-2.p')

    assert any(
        literal.symbol == 'equal___'
        for clause in matrix.clauses
        for literal in clause
    )
    assert len(matrix.clauses) > 8


def test_comp_restarts_complete_search_at_depth_one(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p],[-(p)]]')
    env = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2, step_limit=1000),
    )
    strategy = LeanCoP_2(env)

    reset = strategy._complete_comp_phase(Reset(id='rs', depth=8))

    assert reset.depth == 1
    assert env.state.restricted_backtracking is False
    assert env.state.restricted_start is False
    assert env.state.comp_limit is None
    assert strategy._phase_complete_search is True


def test_exhausted_complete_comp_search_finishes_without_later_phases(tmp_path: Path):
    problem = _write_matrix(tmp_path, '[[p]]')
    env = ConnectionEnv(
        str(problem),
        Settings(logic=Logic.LeanCoP2, step_limit=10_000),
    )
    strategy = LeanCoP_2(env)
    env.reset()

    for _ in range(100):
        _, _, done, info = env.step(strategy.step())
        if done:
            break
    else:
        raise AssertionError('complete comp search did not terminate')

    assert info == 'Non-Theorem: exhaustive complete search'
    assert strategy._phase_index == 0
