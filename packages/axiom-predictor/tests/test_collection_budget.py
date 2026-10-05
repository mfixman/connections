import json
import signal
import sys
import time

import pytest

from axiom_prediction.tptp import collect_proof_example

@pytest.mark.parametrize("error_type", [RecursionError, MemoryError, OverflowError])
def test_collection_records_problem_errors_and_resumes_past_them(tmp_path, monkeypatch, error_type):
    import axiom_prediction.dataset as dataset

    calls = []

    def search(problem, **kwargs):
        calls.append(problem)
        if problem == "bad.p":
            raise error_type("problem exceeded a resource limit")

        return None, "Timeout"

    def sequential_collection(problems, **kwargs):
        workers = kwargs.pop("num_workers")
        assert workers == (1 if problems else 0)
        for problem in problems:
            yield dataset.collect_one_problem_to_partial(problem, **kwargs)

    monkeypatch.setattr(dataset, "collect_proof_example", search)
    monkeypatch.setattr(dataset, "collect_problem_records_parallel", sequential_collection)
    options = dict(output_dir = tmp_path, shard_name = "TEST", num_workers = 1, tptp_root = tmp_path)
    shard, summary = dataset.collect_axiom_dataset_shard(["bad.p", "next.p"], **options)
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    failure = next(row for row in rows[1:] if row["problem"] == "bad.p")
    assert failure["error"] is True
    assert failure["outcome"] == f"{error_type.__name__}: problem exceeded a resource limit"
    assert calls == ["bad.p", "next.p"]
    assert summary["problems_failed"] == 2

    _, resumed = dataset.collect_axiom_dataset_shard(["bad.p", "next.p"], **options)
    assert resumed["problems_reused"] == 2
    assert calls == ["bad.p", "next.p"]

@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason = "POSIX alarm required")
def test_collection_budget_covers_parsing_and_restores_handler(monkeypatch):
    previous = signal.getsignal(signal.SIGALRM)
    previous_limit = sys.getrecursionlimit()

    def slow_parse(*args, **kwargs):
        time.sleep(10)

    monkeypatch.setattr("axiom_prediction.tptp.resolve_tptp_problem", slow_parse)
    started = time.monotonic()
    assert collect_proof_example("unused.p", timeout_s = 0.02) == (None, "Timeout")
    assert time.monotonic() - started < 1
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
    assert sys.getrecursionlimit() == previous_limit
