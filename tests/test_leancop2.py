from pathlib import Path

from cli import default_logic_for_strategy
from connections.calculi.classicalcut import Extension, Reset, Start
from connections.calculi.leancop2 import Lemma
from connections.env import ConnectionEnv, Logic, Settings
from connections.utils.classical_problem_parsing import translate_tptp_problem
from strategies import Strategy
from strategies.custom import LeanCoP_2


def _write_matrix(tmp_path: Path, clauses: str) -> Path:
    path = tmp_path / 'problem.cnf'
    path.write_text(f'cnf(test,conjecture,{clauses}).', encoding='utf-8')
    return path


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


def test_original_ten_phase_schedule_and_reordering_constants():
    assert [phase.share for phase in LeanCoP_2.phases] == [2, 60, 16, 4, 2, 2, 2, 2, 2, 8]
    assert [
        setting
        for phase in LeanCoP_2.phases
        for setting in phase.settings
        if setting.startswith('reo(')
    ] == ['reo(23)', 'reo(29)', 'reo(37)', 'reo(41)', 'reo(47)']


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
