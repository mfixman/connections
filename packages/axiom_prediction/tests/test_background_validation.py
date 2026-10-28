import time
from types import SimpleNamespace

from axiom_prediction import background_validation

class FakeRun:
    def __init__(self):
        self.logs = []
        self.summary = {}

    def log(self, payload):
        self.logs.append(payload)

def slow(metrics):
    def run(*args, **kwargs):
        time.sleep(0.5)
        return metrics

    return run

def test_busy_validation_keeps_only_the_newest_epoch(monkeypatch):
    monkeypatch.setattr(background_validation, "run_baseline", slow({"test_base_x_solved_num": 1}))
    monkeypatch.setattr(background_validation, "run_validation", slow({"test_x_solved_num": 2}))
    run = FakeRun()
    background = background_validation.BackgroundValidation(
        ["a.p"],
        [],
        SimpleNamespace(run = run),
        count = 1,
        tptp_root = None,
    )

    for epoch in (1, 2, 3):
        background.submit(epoch, f"epoch-{epoch}.ts")

    background.finish()
    assert run.summary == {"evaluation/test_base_x_solved_num": 1}
    assert run.logs == [{
        "epoch": 3,
        "evaluation/test_x_solved_num": 2,
        "evaluation/test_base_x_solved_num": 1,
    }]
