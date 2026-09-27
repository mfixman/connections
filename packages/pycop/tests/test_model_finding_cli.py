from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "run_pycop.py"


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(ROOT / "packages/pycop/src"), str(ROOT))
    )
    env.pop("TPTP", None)
    return env


def test_exact_root_command_supports_both_model_policies():
    problem = "fof(a,axiom, ! [X] : (f(X) != X))."
    for policy in ("mace", "sat-reset"):
        process = subprocess.run(
            [
                sys.executable,
                str(LAUNCHER),
                problem,
                "--model-policy",
                policy,
                "--model-max-domain",
                "2",
                "--num-workers",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=_environment(),
        )
        assert process.returncode == 0, process.stderr
        payload = json.loads(process.stdout)
        assert payload["policy"] == policy
        assert payload["status"] == "Satisfiable"
        assert payload["model"]["domain_size"] == 2


def test_installed_entry_point_runs_model_mode():
    process = subprocess.run(
        [
            "run-pycop",
            "cnf(a,axiom,p).",
            "--model-policy",
            "mace",
            "--model-max-domain",
            "1",
            "--num-workers",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_environment(),
    )

    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["status"] == "Satisfiable"


def test_model_cli_writes_jsonl_and_separate_tptp_models(tmp_path: Path):
    problems = tmp_path / "problems"
    problems.mkdir()
    for name, atom in (("a.p", "p"), ("b.p", "q")):
        (problems / name).write_text(f"cnf(a,axiom,{atom}).\n", encoding="utf-8")
    output = tmp_path / "results.jsonl"
    models = tmp_path / "models"

    process = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            str(problems),
            "--model-policy",
            "mace",
            "--split",
            "2",
            "--part",
            "1",
            "--num-workers",
            "2",
            "--out",
            str(output),
            "--model-tptp-dir",
            str(models),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_environment(),
    )

    assert process.returncode == 0, process.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["problem"] for row in rows] == ["b.p"]
    model_path = models / "b.model.p"
    assert model_path.is_file()
    assert "fi_domain" in model_path.read_text()


def test_model_cli_resolves_tptp_codename_and_custom_policy(tmp_path: Path):
    problem_dir = tmp_path / "Problems" / "SYN"
    problem_dir.mkdir(parents=True)
    (problem_dir / "SYN001+1.p").write_text("cnf(a,axiom,p).\n", encoding="utf-8")

    process = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "SYN001+1.p",
            "--tptp",
            str(tmp_path),
            "--model-policy",
            "experiment=tests.fixtures.model_policies:CustomModelPolicy",
            "--model-max-domain",
            "1",
            "--num-workers",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_environment(),
    )

    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert payload["policy"] == "experiment"
    assert payload["status"] == "Satisfiable"


def test_model_cli_parallelizes_independent_problem_files(tmp_path: Path):
    for name, atom in (("a.p", "p"), ("b.p", "q")):
        (tmp_path / name).write_text(f"cnf(a,axiom,{atom}).\n", encoding="utf-8")

    process = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            str(tmp_path),
            "--model-policy",
            "sat-reset",
            "--model-max-domain",
            "1",
            "--num-workers",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_environment(),
    )

    assert process.returncode == 0, process.stderr
    rows = [json.loads(line) for line in process.stdout.splitlines()]
    assert [row["problem"] for row in rows] == ["a.p", "b.p"]
    assert {row["status"] for row in rows} == {"Satisfiable"}


def test_model_cli_reports_input_errors_and_bounded_exhaustion(tmp_path: Path):
    unsupported = tmp_path / "typed.p"
    unsupported.write_text("tff(t,axiom,$true).\n", encoding="utf-8")
    impossible_at_one = tmp_path / "two.p"
    impossible_at_one.write_text("fof(a,axiom, ! [X] : (f(X) != X)).\n", encoding="utf-8")

    error = subprocess.run(
        [sys.executable, str(LAUNCHER), str(unsupported), "--model-policy", "mace"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_environment(),
    )
    bounded = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            str(impossible_at_one),
            "--model-policy",
            "mace",
            "--model-max-domain",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=_environment(),
    )

    error_payload = json.loads(error.stdout)
    bounded_payload = json.loads(bounded.stdout)
    assert error.returncode == 1
    assert error_payload["error_type"] == "InputError"
    assert bounded.returncode == 0
    assert bounded_payload["status"] == "GaveUp"
    assert bounded_payload["outcome"] == "NoModelWithinBound"
    assert "Unsatisfiable" not in bounded.stdout
