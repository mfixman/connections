import os
import sys
from pathlib import Path

import pytest

from cli import configure_tptp_root, parse_pycop_args
from connections.env import Logic, SATSolver, Settings

def test_parse_args_accepts_cadical_sat_solver(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', '--sat-solver', 'cadical', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_pycop_args()

    assert args.sat_solver == 'cadical'
    assert args.sat_solver is SATSolver.CaDiCaL

def test_parse_args_defaults_sat_solver_to_cadical(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_pycop_args()

    assert args.sat_solver is SATSolver.CaDiCaL

def test_parse_args_accepts_tptp_root(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', '--tptp', str(tmp_path), 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_pycop_args()

    assert args.tptp == str(tmp_path)

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

    args = parse_pycop_args()

    assert args.debug is True

def test_parse_args_defaults_print_trace_to_false(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_pycop_args()

    assert args.print_trace is False
    assert args.print_ratio == 0

def test_parse_args_print_trace_sets_print_ratio_to_one(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', '--print-trace', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_pycop_args()

    assert args.print_trace is True
    assert args.print_ratio == 1

def test_parse_args_rejects_logic_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', '--logic', 'ClassicalSAT', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    with pytest.raises(SystemExit) as excinfo:
        parse_pycop_args()

    assert excinfo.value.code == 2

def test_parse_args_accepts_split_and_part(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', '--split', '4', '--part', '2', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_pycop_args()

    assert args.split == 4
    assert args.part == 2

def test_parse_args_uses_slurm_array_when_split_and_part_are_absent(monkeypatch, capsys):
    monkeypatch.setenv('SLURM_ARRAY_TASK_COUNT', '4')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '2')
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_pycop_args()

    captured = capsys.readouterr()
    assert args.split == 4
    assert args.part == 2
    assert 'SLURM_ARRAY_TASK_COUNT=4' in captured.err
    assert 'SLURM_ARRAY_TASK_ID=2' in captured.err

def test_parse_args_prefers_explicit_split_and_part_over_slurm_array(monkeypatch, capsys):
    monkeypatch.setenv('SLURM_ARRAY_TASK_COUNT', '4')
    monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '2')
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', '--split', '3', '--part', '1', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    args = parse_pycop_args()

    captured = capsys.readouterr()
    assert args.split == 3
    assert args.part == 1
    assert 'SLURM_ARRAY_TASK_COUNT' not in captured.err

def test_parse_args_rejects_part_not_less_than_split(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', '--split', '4', '--part', '4', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    with pytest.raises(SystemExit) as excinfo:
        parse_pycop_args()

    assert excinfo.value.code == 2

def test_parse_args_rejects_strategy_groups(monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        ['pycop.py', '--strategy', 'SATControl', 'tests/cnf_problems/SYN081+1.cnf'],
    )

    with pytest.raises(SystemExit) as excinfo:
        parse_pycop_args()

    assert excinfo.value.code == 2

def test_pycop_help_lists_non_hidden_strategies_and_fof_cnf_inputs(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['pycop.py', '--help'])

    with pytest.raises(SystemExit) as excinfo:
        parse_pycop_args()

    captured = capsys.readouterr()
    help_text = ' '.join(captured.out.split())
    assert excinfo.value.code == 0
    assert 'Non-hidden strategies: PyCoP, LeanCoP, SATCoP_0, SATCoP, SATResetCoP' in help_text
    assert 'local FOF/CNF file' in help_text
    assert 'DIR_OR_FILE' in help_text

def test_configure_tptp_root_sets_environment(monkeypatch, tmp_path: Path):
    monkeypatch.delenv('TPTP', raising = False)

    configure_tptp_root(str(tmp_path))

    assert os.environ['TPTP'] == str(tmp_path.resolve())

def test_configure_tptp_root_rejects_missing_directory(monkeypatch, tmp_path: Path):
    monkeypatch.delenv('TPTP', raising = False)
    missing = tmp_path / 'missing'

    with pytest.raises(FileNotFoundError, match='--tptp does not point to a directory'):
        configure_tptp_root(str(missing))
