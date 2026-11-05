from dataclasses import replace
import hashlib

import pytest

from axiom_prediction import dataset
from axiom_prediction.data import axiom_training_example_to_json
from axiom_prediction.io import read_csv, write_json_atomic
from axiom_prediction.problems_file import example_row
from axiom_prediction.tptp import collect_proof_example

def test_parallel_collection_appends_and_resumes(tmp_path, tiny_problem_path):
    missing = str(tmp_path / "missing.p")
    problems = [str(tiny_problem_path), missing]
    options = dict(output_dir = tmp_path / "dataset", num_workers = 2)
    summary = dataset.collect_axiom_dataset(problems, **options)
    path = options["output_dir"] / "problems.csv"
    original = path.read_bytes()
    assert summary["problems_proved"] == 1
    assert summary["problems_failed"] == 1
    assert list(path.parent.iterdir()) == [path]

    examples, failures, _ = dataset.load_axiom_dataset(options["output_dir"])
    assert [example.problem_path for example in examples] == [tiny_problem_path.name]
    assert [row["problem"] for row in failures] == ["missing.p"]
    assert len(list(read_csv(path))) == 2

    resumed = dataset.collect_axiom_dataset(problems, **options)
    assert resumed["problems_reused"] == 2
    assert path.read_bytes() == original

    another = str(tmp_path / "another.p")
    dataset.collect_axiom_dataset(problems + [another], **options)
    assert path.read_bytes().startswith(original)
    assert len(list(read_csv(path))) == 3

def test_problems_sharing_a_file_name_are_rejected(tmp_path):
    with pytest.raises(ValueError, match = "share a file name"):
        dataset.collect_axiom_dataset(["one/same.p", "two/same.p"], output_dir = tmp_path)

@pytest.mark.parametrize("old", [".cache", "cache.csv", "metadata.json", "examples"])
def test_collect_refuses_directories_with_old_results(tmp_path, old):
    (tmp_path / old).touch()
    with pytest.raises(ValueError, match = "old collection results"):
        dataset.collect_axiom_dataset(["one.p"], output_dir = tmp_path)

    assert not (tmp_path / "problems.csv").exists()

def test_collect_refuses_a_problems_file_with_an_old_header(tmp_path):
    path = tmp_path / "problems.csv"
    path.write_text("schema,record,collection,summary,problem_path\n")
    with pytest.raises(ValueError, match = "old or unknown header"):
        dataset.collect_axiom_dataset(["one.p"], output_dir = tmp_path)

def test_legacy_cache_is_still_loaded(tmp_path, tiny_problem_path):
    example, _ = collect_proof_example(str(tiny_problem_path), step_limit = 100)
    assert example is not None
    cache = tmp_path / ".cache" / "TEST"
    (cache / "examples").mkdir(parents = True)
    (cache / "failures").mkdir()
    collection = dict(sat_policy = "satresetcop", step_limit = 100, timeout_seconds = 30, tptp_root = None)
    write_json_atomic(cache / "metadata.json", dict(schema = dataset.axiom_dataset_schema, collection = collection))
    key = hashlib.sha256(str(tiny_problem_path).encode()).hexdigest()
    write_json_atomic(cache / "examples" / f"{key}.json", axiom_training_example_to_json(example))

    failure = dict(schema = dataset.axiom_dataset_schema, problem = "bad.p", outcome = "TPTPParseError: bad input")
    key = hashlib.sha256(b"bad.p").hexdigest()
    write_json_atomic(cache / "failures" / f"{key}.json", failure)
    alternative = replace(example, labels = [0.0 for _ in example.labels])
    write_json_atomic(cache / "examples" / "alternative.json", axiom_training_example_to_json(alternative))

    examples, failures, _ = dataset.load_axiom_dataset(cache)
    assert len(examples) == 2
    assert len(failures) == 1

def test_interrupted_append_preserves_completed_results(tmp_path, tiny_problem_path, monkeypatch):
    example, _ = collect_proof_example(str(tiny_problem_path), step_limit = 100)
    assert example is not None

    def interrupted(problems, **kwargs):
        payload = example_row(problems[0], example)
        yield dataset.CollectedAxiomRecord(problems[0], "proved", True, True, payload)
        raise KeyboardInterrupt

    monkeypatch.setattr(dataset, "collect_problem_records_parallel", interrupted)
    with pytest.raises(KeyboardInterrupt):
        dataset.collect_axiom_dataset(["one.p", "two.p"], output_dir = tmp_path)

    path = tmp_path / "problems.csv"
    original = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b"unfinished," + b"x" * 9000)

    pending = []

    def finish(problems, **kwargs):
        pending.extend(problems)
        yield dataset.interrupted_record(dict(problem = "two.p", outcome = "Timeout"))

    monkeypatch.setattr(dataset, "collect_problem_records_parallel", finish)
    summary = dataset.collect_axiom_dataset(["one.p", "two.p"], output_dir = tmp_path)
    assert pending == ["two.p"]
    assert summary["problems_reused"] == 1
    assert path.read_bytes().startswith(original)
    assert len(list(read_csv(path))) == 2
    assert len(dataset.load_axiom_dataset(tmp_path)[0]) == 1

    with pytest.raises(ValueError, match = "settings differ"):
        dataset.collect_axiom_dataset(["one.p"], output_dir = tmp_path, step_limit = 1)

def test_corrupt_complete_row_is_rejected(tmp_path):
    dataset.collect_axiom_dataset([], output_dir = tmp_path)
    with (tmp_path / "problems.csv").open("a") as stream:
        stream.write("invalid,row\n")

    with pytest.raises(ValueError, match = "corrupt dataset CSV row"):
        dataset.collect_axiom_dataset([], output_dir = tmp_path)
