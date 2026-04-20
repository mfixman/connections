import sys

from parse_args import parse_args
from connections.env import Logic, SATSolver, Settings

def test_parse_args_accepts_cadical_sat_solver(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', '--sat-solver', 'cadical', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_args()

    assert args.sat_solver == 'cadical'
    assert args.sat_solver is SATSolver.CaDiCaL

def test_parse_args_defaults_sat_solver_to_cadical(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_args()

    assert args.sat_solver is SATSolver.CaDiCaL

def test_settings_defaults_sat_solver_to_cadical():
    assert Settings().sat_solver is SATSolver.CaDiCaL

def test_settings_normalizes_string_configuration():
    settings = Settings(logic = 'ClassicalSAT', sat_solver = 'cadical')

    assert settings.logic is Logic.ClassicalSAT
    assert settings.sat_solver is SATSolver.CaDiCaL

def test_parse_args_accepts_debug(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', '--debug', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_args()

    assert args.debug is True

def test_parse_args_defaults_logic_to_none(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_args()

    assert args.logic is None
