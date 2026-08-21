from __future__ import annotations

import os
import csv
import subprocess
import sys


def _env_without_logic_roots() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("TPTP", None)
    env.pop("ILTP", None)
    env.pop("QMLTP", None)
    return env


def test_pycop_console_help_runs():
    proc = subprocess.run(
        ["pycop", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=os.environ.copy(),
    )
    assert proc.returncode == 0
    assert "Connections Python connection-tableau prover" in proc.stdout


def test_policy_experiment_console_helps_run():
    for command, expected in (
        ("run-pycop", "Run one connections policy"),
        ("compare-strategies", "Compare connections policies"),
    ):
        proc = subprocess.run(
            [command, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=os.environ.copy(),
        )
        assert proc.returncode == 0
        assert expected in proc.stdout


def test_compare_strategies_lists_connections_policies():
    proc = subprocess.run(
        ["compare-strategies", "--print-all-strategies"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_env_without_logic_roots(),
    )

    assert proc.returncode == 0
    assert proc.stdout.splitlines() == [
        "FirstActionIDPolicy",
        "LeanCoPCon",
        "LeanCoPCon_2",
        "SATCoPCon",
        "SATResetCoP",
    ]


def test_run_pycop_accepts_raw_problem():
    proc = subprocess.run(
        [
            "run-pycop",
            "fof(c,conjecture,(p => p)).",
            "--strategy",
            "FirstActionIDPolicy",
            "--max-steps",
            "20",
            "--timeout",
            "2",
            "--num-workers",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_env_without_logic_roots(),
    )

    assert proc.returncode == 0
    payload = __import__("json").loads(proc.stdout)
    assert payload["status"] == "Theorem"
    assert payload["policy"] == "FirstActionIDPolicy"


def test_run_pycop_resolves_tptp_codename(tmp_path):
    problem_dir = tmp_path / "Problems" / "SYN"
    problem_dir.mkdir(parents=True)
    (problem_dir / "SYN001+1.p").write_text(
        "fof(c,conjecture,(p => p)).\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            "run-pycop",
            "SYN001+1.p",
            "--tptp",
            str(tmp_path),
            "--max-steps",
            "20",
            "--timeout",
            "2",
            "--num-workers",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_env_without_logic_roots(),
    )

    assert proc.returncode == 0
    assert '"status": "Theorem"' in proc.stdout


def test_run_pycop_expands_and_slices_directory(tmp_path):
    for name, atom in (("a.p", "p"), ("b.p", "q"), ("c.p", "r")):
        (tmp_path / name).write_text(
            f"fof(c,conjecture,({atom} => {atom})).\n",
            encoding="utf-8",
        )
    proc = subprocess.run(
        [
            "run-pycop",
            str(tmp_path),
            "--split",
            "2",
            "--part",
            "1",
            "--max-steps",
            "20",
            "--timeout",
            "2",
            "--num-workers",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_env_without_logic_roots(),
    )

    assert proc.returncode == 0
    payload = __import__("json").loads(proc.stdout)
    assert payload["problem"] == "b.p"
    assert payload["status"] == "Theorem"


def test_compare_strategies_writes_and_resumes_validated_results(tmp_path):
    problem = tmp_path / "tiny.p"
    problem.write_text("fof(c,conjecture,(p => p)).\n", encoding="utf-8")
    csv_path = tmp_path / "results.csv"
    partial_path = tmp_path / "partial.jsonl"
    command = [
        "compare-strategies",
        str(problem),
        "--strategies",
        "FirstActionID",
        "FirstActionIDPolicy",
        "--csv",
        str(csv_path),
        "--partial-file",
        str(partial_path),
        "--num-workers",
        "2",
        "--max-steps",
        "20",
        "--timeout",
        "2",
    ]

    first = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=_env_without_logic_roots(),
    )
    second = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=_env_without_logic_roots(),
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "completed=2  remaining=0" in second.stderr
    with csv_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    assert [row["policy"] for row in rows] == [
        "FirstActionID",
        "FirstActionIDPolicy",
    ]
    assert {row["status"] for row in rows} == {"Theorem"}

    mismatch = subprocess.run(
        command[:-4] + ["--max-steps", "21", "--timeout", "2"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_env_without_logic_roots(),
    )
    assert mismatch.returncode == 2
    assert "different code/configuration/problem set" in mismatch.stderr


def test_compare_strategies_runs_all_named_connections_policies(tmp_path):
    problem = tmp_path / "tiny.p"
    problem.write_text("fof(c,conjecture,(p => p)).\n", encoding="utf-8")
    csv_path = tmp_path / "policies.csv"
    partial_path = tmp_path / "policies.jsonl"
    proc = subprocess.run(
        [
            "compare-strategies",
            str(problem),
            "--strategies",
            "LeanCoPCon",
            "LeanCoPCon_2",
            "SATCoPCon",
            "SATResetCoP",
            "--csv",
            str(csv_path),
            "--partial-file",
            str(partial_path),
            "--num-workers",
            "4",
            "--max-steps",
            "100",
            "--timeout",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=_env_without_logic_roots(),
    )

    assert proc.returncode == 0, proc.stderr
    with csv_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    assert [row["policy"] for row in rows] == [
        "LeanCoPCon",
        "LeanCoPCon_2",
        "SATCoPCon",
        "SATResetCoP",
    ]
    assert {row["status"] for row in rows} == {"Theorem"}


def test_all_prover_helps_include_steps_option():
    proc = subprocess.run(
        ["pycop", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=os.environ.copy(),
    )
    assert proc.returncode == 0
    assert "--steps" in proc.stdout
    assert "--timeout" in proc.stdout
    assert "--source-dir" in proc.stdout
    assert "--trace-search" in proc.stdout
    assert "--trace-clausification" in proc.stdout


def test_pycop_accepts_explicit_source_dir_without_tptp_env(tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "axioms.ax").write_text("fof(a1,axiom,p).\n", encoding="utf-8")
    problem = tmp_path / "theorem.p"
    problem.write_text("include('axioms.ax').\nfof(c,conjecture,p).\n", encoding="utf-8")
    env = _env_without_logic_roots()

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "provers.pycop.cli",
            str(problem),
            "classical",
            "constant",
            "--source-dir",
            str(lib_dir),
            "--steps",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )

    assert proc.returncode == 0
    assert "Theorem" in proc.stdout


def test_pycop_reports_satisfiable_when_no_start_clause_exists(tmp_path):
    problem = tmp_path / "tiny.p"
    problem.write_text("fof(a1,axiom,~p).\n", encoding="utf-8")
    env = _env_without_logic_roots()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "provers.pycop.cli",
            str(problem),
            "classical",
            "constant",
            "--steps",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )
    assert proc.returncode == 0
    assert "Satisfiable" in proc.stdout


def test_pycop_supports_intuitionistic_matrix_construction(tmp_path):
    problem = tmp_path / "tiny.p"
    problem.write_text("fof(c,conjecture,(p => p)).\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "provers.pycop.cli",
            str(problem),
            "intuitionistic",
            "constant",
            "--steps",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_env_without_logic_roots(),
    )

    assert proc.returncode == 0
    assert "Theorem" in proc.stdout


def test_pycop_supports_modal_matrix_construction(tmp_path):
    problem = tmp_path / "tiny_modal.p"
    problem.write_text("qmf(c,conjecture,(#box:p => #box:p)).\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "provers.pycop.cli",
            str(problem),
            "D",
            "cumulative",
            "--steps",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_env_without_logic_roots(),
    )

    assert proc.returncode == 0
    assert "Theorem" in proc.stdout


def test_pycop_reports_counter_satisfiable_for_counterexample_problem(tmp_path):
    problem = tmp_path / "counterexample.p"
    problem.write_text("fof(a1,axiom,p).\nfof(c,conjecture,q).\n", encoding="utf-8")
    env = _env_without_logic_roots()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "provers.pycop.cli",
            str(problem),
            "classical",
            "constant",
            "--steps",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )
    assert proc.returncode == 0
    assert "CounterSatisfiable" in proc.stdout
    assert "GaveUp" not in proc.stdout


def test_pycop_reports_unsatisfiable_for_closed_cnf_with_negated_conjecture(
    tmp_path,
):
    problem = tmp_path / "unsat_cnf.p"
    problem.write_text(
        "cnf(c1,axiom,p).\ncnf(c2,negated_conjecture,~p).\n",
        encoding="utf-8",
    )
    env = _env_without_logic_roots()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "provers.pycop.cli",
            str(problem),
            "classical",
            "constant",
            "--steps",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )

    assert proc.returncode == 0
    assert "Unsatisfiable" in proc.stdout
    assert "Theorem" not in proc.stdout


def test_pycop_reports_satisfiable_for_exhausted_cnf_with_negated_conjecture(
    tmp_path,
):
    problem = tmp_path / "sat_cnf.p"
    problem.write_text(
        "cnf(c1,negated_conjecture,p).\n",
        encoding="utf-8",
    )
    env = _env_without_logic_roots()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "provers.pycop.cli",
            str(problem),
            "classical",
            "constant",
            "--steps",
            "20",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )

    assert proc.returncode == 0
    assert "Satisfiable" in proc.stdout
    assert "CounterSatisfiable" not in proc.stdout


def test_pycop_schedule_prints_strategy_status_without_trace(tmp_path):
    problem = tmp_path / "non_theorem.p"
    problem.write_text(
        "fof(a1,axiom,p).\nfof(c,conjecture,q).\n",
        encoding="utf-8",
    )
    schedule = tmp_path / "schedule.json"
    schedule.write_text('[{"settings":["cut"],"weight":1}]', encoding="utf-8")
    env = _env_without_logic_roots()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "provers.pycop.cli",
            str(problem),
            "classical",
            "constant",
            "--schedule",
            str(schedule),
            "--steps",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )
    assert proc.returncode == 0
    assert "strategy 1:" in proc.stdout
    # An exhausted step budget is the prover's own resource: ResourceOut.
    assert "ResourceOut" in proc.stdout


def test_pycop_default_cli_runs_single_strategy(tmp_path):
    problem = tmp_path / "non_theorem.p"
    problem.write_text(
        "fof(a1,axiom,p).\nfof(c,conjecture,q).\n",
        encoding="utf-8",
    )
    env = _env_without_logic_roots()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "provers.pycop.cli",
            str(problem),
            "classical",
            "constant",
            "--trace-search",
            "--steps",
            "60",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )
    action_lines = [
        line
        for line in proc.stdout.splitlines()
        if line in {"start", "extension", "reduction", "lemma", "backtrack"}
    ]
    assert proc.returncode == 0
    assert "strategy 1:" not in proc.stdout
    assert action_lines == ["start", "backtrack"]
    assert "CounterSatisfiable" in proc.stdout
