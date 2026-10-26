from dataclasses import replace
import hashlib

import pytest

from axiom_prediction import dataset
from axiom_prediction.data import axiom_training_example_to_json
from axiom_prediction.io import read_csv, write_json_atomic
from axiom_prediction.tptp import collect_proof_example

def test_parallel_collection_appends_and_resumes(tmp_path, tiny_problem_path):
    missing = str(tmp_path / "missing.p")
    problems = [str(tiny_problem_path), missing]
    options = dict(output_dir = tmp_path / "dataset", shard_name = "TEST", num_workers = 2)
    shard, summary = dataset.collect_axiom_dataset_shard(problems, **options)
    cache = options["output_dir"] / ".cache" / "TEST" / "cache.csv"
    original = cache.read_bytes()
    assert summary["problems_proved"] == 1
    assert summary["problems_failed"] == 1
    assert list(cache.parent.iterdir()) == [cache]

    examples, failures, _ = dataset.load_axiom_dataset(options["output_dir"])
    assert [example.problem_path for example in examples] == [str(tiny_problem_path)]
    assert [row["problem"] for row in failures] == [missing]
    assert len(list(read_csv(shard))) == 3

    _, resumed = dataset.collect_axiom_dataset_shard(problems, **options)
    assert resumed["problems_reused"] == 2
    assert cache.read_bytes() == original

    another = str(tmp_path / "another.p")
    dataset.collect_axiom_dataset_shard(problems + [another], **options)
    assert cache.read_bytes().startswith(original)
    assert len(list(read_csv(cache))) == 4

@pytest.mark.parametrize("partial", [False, True])
def test_legacy_cache_is_loaded_and_resumed_implicitly(tmp_path, tiny_problem_path, monkeypatch, partial):
    example, _ = collect_proof_example(str(tiny_problem_path), step_limit = 100)
    assert example is not None
    cache = tmp_path / ".cache" / "TEST"
    old = cache / "partial" if partial else cache
    (old / "examples").mkdir(parents = True)
    (old / "failures").mkdir()
    collection = dict(sat_policy = "satresetcop", step_limit = 100, timeout_seconds = 30, tptp_root = None)
    write_json_atomic(cache / "metadata.json", dict(schema = dataset.axiom_dataset_schema, collection = collection))
    key = hashlib.sha256(str(tiny_problem_path).encode()).hexdigest()
    write_json_atomic(old / "examples" / f"{key}.json", axiom_training_example_to_json(example))

    failure = dict(schema = dataset.axiom_dataset_schema, problem = "bad.p", outcome = "TPTPParseError: bad input")
    key = hashlib.sha256(b"bad.p").hexdigest()
    write_json_atomic(old / "failures" / f"{key}.json", failure)
    alternative = replace(example, labels = [0.0 for _ in example.labels])
    write_json_atomic(old / "examples" / "alternative.json", axiom_training_example_to_json(alternative))
    original = {path: path.read_bytes() for path in cache.rglob("*.json")}

    if not partial:
        examples, failures, _ = dataset.load_axiom_dataset(cache)
        assert len(examples) == 2
        assert len(failures) == 1

    def no_search(*args, **kwargs):
        pytest.fail("legacy cached problems must not be searched again")

    monkeypatch.setattr(dataset, "supervised_collection", no_search)
    shard, summary = dataset.collect_axiom_dataset_shard(
        [str(tiny_problem_path), "bad.p"],
        output_dir = tmp_path,
        shard_name = "TEST",
        step_limit = 100,
        timeout_s = 30,
    )

    assert summary["problems_reused"] == 2
    assert summary["problems_unparseable"] == 1
    assert len(list(read_csv(shard))) == 4
    assert len(dataset.load_axiom_dataset(cache)[0]) == 2
    assert all(path.read_bytes() == contents for path, contents in original.items())

def test_interrupted_append_preserves_completed_results(tmp_path, tiny_problem_path, monkeypatch):
    example, _ = collect_proof_example(str(tiny_problem_path), step_limit = 100)
    assert example is not None

    def interrupted(problems, **kwargs):
        payload = axiom_training_example_to_json(replace(example, problem_path = problems[0]))
        yield dataset.CollectedAxiomRecord(problems[0], "proved", True, True, payload)
        raise KeyboardInterrupt

    monkeypatch.setattr(dataset, "collect_problem_records_parallel", interrupted)
    with pytest.raises(KeyboardInterrupt):
        dataset.collect_axiom_dataset(["one.p", "two.p"], output_dir = tmp_path)

    cache = tmp_path / "cache.csv"
    original = cache.read_bytes()
    with cache.open("ab") as stream:
        stream.write(b"unfinished," + b"x" * 9000)

    pending = []

    def finish(problems, **kwargs):
        pending.extend(problems)
        yield dataset.interrupted_record(dict(problem = "two.p", outcome = "Timeout"))

    monkeypatch.setattr(dataset, "collect_problem_records_parallel", finish)
    summary = dataset.collect_axiom_dataset(["one.p", "two.p"], output_dir = tmp_path)
    assert pending == ["two.p"]
    assert summary["problems_reused"] == 1
    assert cache.read_bytes().startswith(original)
    assert len(list(read_csv(cache))) == 3
    assert len(dataset.load_axiom_dataset(tmp_path)[0]) == 1

    with pytest.raises(ValueError, match = "settings differ"):
        dataset.collect_axiom_dataset(["one.p"], output_dir = tmp_path, step_limit = 1)

def test_corrupt_complete_cache_row_is_rejected(tmp_path):
    dataset.collect_axiom_dataset([], output_dir = tmp_path)
    with (tmp_path / "cache.csv").open("a") as stream:
        stream.write("invalid,row\n")

    with pytest.raises(ValueError, match = "corrupt dataset CSV row"):
        dataset.collect_axiom_dataset([], output_dir = tmp_path)
