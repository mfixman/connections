import sys

from parse_args import parse_args

def test_parse_args_accepts_sat_solver(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', '--sat-solver', 'pysat', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_args()

    assert args.sat_solver == 'pysat'

def test_parse_args_accepts_cadical_sat_solver(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', '--sat-solver', 'cadical', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_args()

    assert args.sat_solver == 'cadical'

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
