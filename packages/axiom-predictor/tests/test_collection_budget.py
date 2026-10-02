import signal
import sys
import time
from types import SimpleNamespace

import pytest

from axiom_prediction.tptp import collect_proof_example
from axiom_prediction.run import RunConfig, run_problem
from connections.constraints.term import TermSubstitution
from connections.syntax.formula import Function, Variable

def test_collection_supports_deep_search_terms_and_restores_limit(monkeypatch, tiny_problem_path):
    previous = sys.getrecursionlimit()
    term = Variable("X")
    for _ in range(1200):
        term = Function("f", (term,))

    def search(*args, **kwargs):
        resolved = TermSubstitution().substitute_term(term)
        depth = 0
        while isinstance(resolved, Function):
            depth += 1
            resolved = resolved.args[0]

        assert depth == 1200
        return SimpleNamespace(szs_status = None)

    monkeypatch.setattr("axiom_prediction.tptp.run_schedule", search)
    assert collect_proof_example(tiny_problem_path, tptp_root = tiny_problem_path.parent) == (None, "unknown")
    assert sys.getrecursionlimit() == previous

@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason = "POSIX alarm required")
def test_collection_budget_covers_parsing_and_restores_handler(monkeypatch):
    previous = signal.getsignal(signal.SIGALRM)
    previous_limit = sys.getrecursionlimit()

    def slow_parse(*args, **kwargs):
        time.sleep(10)

    monkeypatch.setattr("axiom_prediction.tptp.resolve_tptp_problem", slow_parse)
    started = time.monotonic()
    assert collect_proof_example("unused.p", timeout_seconds = 0.02) == (None, "Timeout")
    assert time.monotonic() - started < 1
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
    assert sys.getrecursionlimit() == previous_limit

@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason = "POSIX alarm required")
def test_search_budget_includes_checkpoint_loading(monkeypatch):
    previous = signal.getsignal(signal.SIGALRM)

    def slow_load(*args, **kwargs):
        time.sleep(10)

    monkeypatch.setattr("axiom_prediction.run.cached_predictor", slow_load)
    config = RunConfig(checkpoint = "unused.pt", device = "cpu", timeout_seconds = 0.02)
    started = time.monotonic()
    result = run_problem("unused.p", tptp_root = None, config = config)
    assert result["outcome"] == "Timeout"
    assert time.monotonic() - started < 1

    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
