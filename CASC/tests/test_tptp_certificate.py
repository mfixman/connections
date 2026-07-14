from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


CASC = Path(__file__).resolve().parents[1]
PACKAGE = CASC / 'SATResetCoP'
PROBLEM = CASC / 'tests' / 'data' / 'proof_smoke+1.p'


def _run_prover(problem: Path = PROBLEM) -> str:
    env = os.environ.copy()
    env['TPTP'] = str(PACKAGE / 'tptp-root')
    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGE / 'starexec_prover.py'),
            '--strategy',
            'SATResetCoP',
            '--sat-solver',
            'PySAT',
            '--proof',
            '--cpu-limit',
            '30',
            str(problem),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stderr == ''
    return result.stdout


def test_satresetcop_emits_tptp_ground_refutation() -> None:
    output = _run_prover()

    assert '% SZS status Theorem for proof_smoke+1' in output
    assert '% SZS output start CNFRefutation for proof_smoke+1' in output
    assert '% SZS output end CNFRefutation for proof_smoke+1' in output
    assert '\nfof(satresetcop_input_1,axiom,' in output
    assert '\ncnf(satresetcop_false,plain,' in output
    assert 'inference(sat_refutation,[status(thm)]' in output
    assert re.search(r'cnf\(satresetcop_ground_\d+,plain,', output)

    # The derivation itself must be TPTP, not the old commented debug witness.
    assert '\n% fof(satresetcop_input_1' not in output
    assert '\n% cnf(satresetcop_false' not in output


def test_certificate_uses_standard_cnf_polarities() -> None:
    output = _run_prover()

    # These are the three clauses of axioms & negated conjecture. The internal
    # leanCoP connection matrix has the opposite polarity, so this guards the
    # certificate's required global complementation.
    assert '( ~ q(a) )' in output
    assert '( p(a) )' in output
    assert '( ~ p(X1) | q(X1) )' in output


def test_starexec_configuration_requests_proof_output() -> None:
    run_script = (PACKAGE / 'bin' / 'starexec_run_default').read_text()
    assert '--strategy SATResetCoP' in run_script
    assert '--proof' in run_script


def test_starexec_wrapper_accepts_standard_output_directory(tmp_path: Path) -> None:
    """StarExec passes an output directory as the second positional arg."""
    env = os.environ.copy()
    env['TPTP'] = str(PACKAGE / 'tptp-root')
    env['STAREXEC_CPU_LIMIT'] = '30'
    env['STAREXEC_WALLCLOCK_LIMIT'] = '30'
    result = subprocess.run(
        [
            str(PACKAGE / 'bin' / 'starexec_run_default'),
            str(PROBLEM),
            str(tmp_path / 'output_files'),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stderr == ''
    assert '% SZS status Theorem for proof_smoke+1' in result.stdout
    assert 'CNFRefutation' in result.stdout


def test_fixed_container_filename_uses_tptp_header_name(tmp_path: Path) -> None:
    problem = tmp_path / 'benchmark'
    problem.write_text(
        '% File : ZZZ999+1.005 : synthetic regression problem\n'
        + PROBLEM.read_text()
    )

    output = _run_prover(problem)

    assert '% SZS status Theorem for ZZZ999+1.005' in output
    assert "file('ZZZ999+1.005.p',rule)" in output


def test_include_is_resolved_beside_current_file_before_tptp(tmp_path: Path) -> None:
    local_dir = tmp_path / 'problem'
    tptp_dir = tmp_path / 'tptp'
    local_dir.mkdir()
    tptp_dir.mkdir()
    (local_dir / 'shared.ax').write_text(
        'fof(local_rule,axiom,![X]:(p(X)=>q(X))).\n'
        'fof(local_fact,axiom,p(a)).\n'
    )
    # A same-named but incompatible TPTP-root file proves that local wins.
    (tptp_dir / 'shared.ax').write_text('fof(wrong,axiom,r(a)).\n')
    problem = local_dir / 'local_include+1.p'
    problem.write_text("include('shared.ax').\nfof(goal,conjecture,q(a)).\n")

    env = os.environ.copy()
    env['TPTP'] = str(tptp_dir)
    result = subprocess.run(
        [sys.executable, str(PACKAGE / 'starexec_prover.py'), '--proof', str(problem)],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stderr == ''
    assert '% SZS status Theorem for local_include+1' in result.stdout


def test_include_falls_back_to_tptp_root(tmp_path: Path) -> None:
    problem_dir = tmp_path / 'problem'
    axiom_dir = tmp_path / 'tptp' / 'Axioms'
    problem_dir.mkdir()
    axiom_dir.mkdir(parents=True)
    (axiom_dir / 'root.ax').write_text(
        'fof(root_rule,axiom,![X]:(r(X)=>s(X))).\n'
        'fof(root_fact,axiom,r(b)).\n'
    )
    problem = problem_dir / 'root_include+1.p'
    problem.write_text("include('Axioms/root.ax').\nfof(goal,conjecture,s(b)).\n")

    env = os.environ.copy()
    env['TPTP'] = str(tmp_path / 'tptp')
    result = subprocess.run(
        [sys.executable, str(PACKAGE / 'starexec_prover.py'), '--proof', str(problem)],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stderr == ''
    assert '% SZS status Theorem for root_include+1' in result.stdout
