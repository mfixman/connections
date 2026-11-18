import signal
import time

import pytest

from axiom_prediction import collection
from axiom_prediction.cli import main
from axiom_prediction.collection import collect_axiom_dataset, collect_proof_example
from axiom_prediction.io import read_csv, write_csv
from axiom_prediction.problems_file import example_row, problem_fields

@pytest.fixture
def choices(tmp_path):
    problem = tmp_path / "choices.p"
    problem.write_text(
        "fof(a0,axiom,p(a0)).\n"
        "fof(a1,axiom,p(a1)).\n"
        "fof(a2,axiom,p(a2)).\n"
        "fof(c,conjecture,? [X]:p(X)).\n"
    )

    return problem

def test_parallel_collection_appends_and_resumes(tmp_path, tiny_problem_path):
    problems = [str(tiny_problem_path), str(tmp_path / "missing.p")]
    options = dict(output_dir = tmp_path / "dataset", num_workers = 2)
    summary = collect_axiom_dataset(problems, **options)
    path = options["output_dir"] / "problems.csv"
    original = path.read_bytes()
    assert (summary["problems_proved"], summary["problems_failed"]) == (1, 1)
    assert list(path.parent.iterdir()) == [path]
    assert {row["problem_name"]: row["outcome"] == "proved" for row in read_csv(path)} == {
        tiny_problem_path.name: True, "missing.p": False,
    }

    assert collect_axiom_dataset(problems, **options)["problems_reused"] == 2
    assert path.read_bytes() == original

    collect_axiom_dataset(problems + [str(tmp_path / "another.p")], **options)
    assert path.read_bytes().startswith(original)
    assert len(list(read_csv(path))) == 3

def test_seed_changes_labels_and_protects_resume(tmp_path, choices):
    common = ["collect", str(choices), "--num-workers", "1", "--timeout-seconds", "10"]
    labels = {}
    for name, seed_args in [("default", []), ("zero", ["--seed", "0"]), ("one", ["--seed", "1"])]:
        assert main([*common, "--dataset", str(tmp_path / name), *seed_args]) == 0
        row, = read_csv(tmp_path / name / "problems.csv")
        assert row["collection"]["seed"] == (1 if name == "one" else 0)
        assert sum(row["labels"]) == 1
        labels[name] = row["labels"]

    assert labels["default"] == labels["zero"] != labels["one"]

    # Results collected before seeds existed resume as seed 0.
    path = tmp_path / "default" / "problems.csv"
    rows = list(read_csv(path))
    rows[0]["collection"].pop("seed")
    write_csv(path, rows, problem_fields)
    assert main([*common, "--dataset", str(tmp_path / "default"), "--seed", "0"]) == 0
    assert len(list(read_csv(path))) == 1
    assert main([*common, "--dataset", str(tmp_path / "default"), "--seed", "1"]) == 2

def test_problems_from_a_file_are_collected_and_an_empty_file_is_refused(tmp_path, tiny_problem_path):
    listed = tmp_path / "problems.txt"
    listed.write_text(f"\n{tiny_problem_path}\n\n")
    output = tmp_path / "dataset"
    assert main(["collect", "--problems-from", str(listed), "--dataset", str(output), "--num-workers", "1"]) == 0
    assert [row["problem_name"] for row in read_csv(output / "problems.csv")] == [tiny_problem_path.name]

    listed.write_text("\n")
    assert main(["collect", "--problems-from", str(listed), "--dataset", str(output)]) == 2

def test_unsafe_targets_are_refused(tmp_path):
    with pytest.raises(ValueError, match = "share a file name"):
        collect_axiom_dataset(["one/same.p", "two/same.p"], output_dir = tmp_path / "names")

    (tmp_path / "cache.csv").touch()
    with pytest.raises(ValueError, match = "old collection results"):
        collect_axiom_dataset(["one.p"], output_dir = tmp_path)

    (tmp_path / "header").mkdir()
    (tmp_path / "header" / "problems.csv").write_text("schema,record,collection,summary,problem_path\n")
    with pytest.raises(ValueError, match = "old or unknown header"):
        collect_axiom_dataset(["one.p"], output_dir = tmp_path / "header")

def test_interrupted_append_preserves_completed_results(tmp_path, tiny_problem_path, monkeypatch):
    example, _ = collect_proof_example(str(tiny_problem_path), step_limit = 100)

    def interrupted(problems, **kwargs):
        yield collection.CollectedProblem(problems[0], "proved", True, True, example_row(problems[0], example))
        raise KeyboardInterrupt

    monkeypatch.setattr(collection, "collect_problems", interrupted)
    with pytest.raises(KeyboardInterrupt):
        collect_axiom_dataset(["one.p", "two.p"], output_dir = tmp_path)

    path = tmp_path / "problems.csv"
    original = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b"unfinished," + b"x" * 9000)

    pending = []

    def finish(problems, **kwargs):
        pending.extend(problems)
        yield collection.interrupted_problem(dict(problem = "two.p", outcome = "Timeout"))

    monkeypatch.setattr(collection, "collect_problems", finish)
    summary = collect_axiom_dataset(["one.p", "two.p"], output_dir = tmp_path)
    assert pending == ["two.p"]
    assert summary["problems_reused"] == 1
    assert path.read_bytes().startswith(original)
    assert len(list(read_csv(path))) == 2

    with pytest.raises(ValueError, match = "settings differ"):
        collect_axiom_dataset(["one.p"], output_dir = tmp_path, step_limit = 1)

    with path.open("a") as stream:
        stream.write("invalid,row\n")

    with pytest.raises(ValueError, match = "corrupt dataset CSV row"):
        collect_axiom_dataset([], output_dir = tmp_path)

@pytest.mark.parametrize("error_type", [RecursionError, MemoryError])
def test_problem_errors_are_recorded_and_skipped_on_resume(tmp_path, monkeypatch, error_type):
    calls = []

    def search(problem, **kwargs):
        calls.append(problem)
        if problem == "bad.p":
            raise error_type("problem exceeded a resource limit")

        return None, "Timeout"

    def sequential(problems, *, num_workers, **options):
        for problem in problems:
            yield collection.collect_one_problem(problem, *options.values())["collected"]

    monkeypatch.setattr(collection, "collect_proof_example", search)
    monkeypatch.setattr(collection, "collect_problems", sequential)
    options = dict(output_dir = tmp_path, tptp_root = tmp_path)
    assert collect_axiom_dataset(["bad.p", "next.p"], **options)["problems_failed"] == 2
    failure = next(row for row in read_csv(tmp_path / "problems.csv") if row["problem_name"] == "bad.p")
    assert failure["error"] is True
    assert failure["outcome"] == f"{error_type.__name__}: problem exceeded a resource limit"

    assert collect_axiom_dataset(["bad.p", "next.p"], **options)["problems_reused"] == 2
    assert calls == ["bad.p", "next.p"]

def test_budget_covers_parsing_and_restores_the_alarm(monkeypatch):
    previous = signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(collection, "resolve_tptp_problem", lambda *args, **kwargs: time.sleep(10))
    started = time.monotonic()
    assert collect_proof_example("unused.p", timeout_s = 0.02) == (None, "Timeout")
    assert time.monotonic() - started < 1
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)

def test_native_collection_skips_declared_non_refutable_problems(tmp_path, monkeypatch):
    from axiom_prediction import satcop

    monkeypatch.setattr(satcop, "collect_native_example", lambda *args, **kwargs: pytest.fail("searched"))
    problem = tmp_path / "declared.p"
    problem.write_text("% Status   : Satisfiable\nfof(a,axiom,p(a)).\n")
    assert collect_proof_example(problem, sat_policy = "satcop") == (None, "DeclaredSatisfiable")
