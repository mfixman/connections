from pathlib import Path
import random
from types import SimpleNamespace

import pytest

@pytest.fixture
def tiny_problem_path():
    return Path(__file__).parent / "fixtures/problems/tiny_theorem.p"

# Chains p0(a), p_i(X) => p_{i+1}(f(X)) with unrelated clauses around them; the names spread over the split's parts.
@pytest.fixture(scope = "session")
def chain_problems(tmp_path_factory):
    directory = tmp_path_factory.mktemp("problems")
    rnd = random.Random(7)
    for number in range(24):
        length = rnd.randint(2, 5)
        steps = [f"cnf(step{i}, axiom, ~p{i}(X) | p{i + 1}(f(X)))." for i in range(length)]
        noise = [f"cnf(noise{i}, axiom, ~q{rnd.randint(0, 5)}(X, Y) | q{rnd.randint(0, 5)}(g(Y), X))." for i in range(rnd.randint(1, 6))]
        goal = f"cnf(goal, negated_conjecture, ~p{length}({'f(' * length}a{')' * length}))."
        lines = ["% Status : Unsatisfiable", "cnf(base, axiom, p0(a)).", *steps, *noise, goal]
        (directory / f"SYN{900 + number}-1.p").write_text("\n".join(lines) + "\n")

    return directory

# A joined dataset with the SATResetCoP proof of every chain problem.
@pytest.fixture(scope = "session")
def joined_dataset(chain_problems, tmp_path_factory):
    from axiom_prediction.collection import collect_axiom_dataset
    from axiom_prediction.joined import join_datasets

    root = tmp_path_factory.mktemp("dataset")
    problems = sorted(map(str, chain_problems.glob("*.p")))
    summary = collect_axiom_dataset(problems, output_dir = root / "collected", timeout_s = 30, num_workers = 2)
    assert summary["problems_proved"] == len(problems)
    join_datasets([root / "collected"], root / "joined", 1)
    return root / "joined"

class FakeRun:
    def __init__(self):
        self.logs = []
        self.summary = {}
        self.models = []
        self.finished = []

    def log(self, payload, step = None):
        self.logs.append(payload)

    def log_model(self, *, path, name):
        self.models.append((path, name))

    def finish(self, *, exit_code = 0):
        self.finished.append(exit_code)

# Stands in for the wandb module; init arguments land in run.init.
@pytest.fixture
def wandb_run(monkeypatch):
    import sys

    run = FakeRun()
    run.init = {}

    def init(**kwargs):
        run.init.update(kwargs)
        return run

    module = SimpleNamespace(
        __name__ = "wandb",
        login = lambda **kwargs: True,
        init = init,
        Histogram = lambda values, num_bins: ("histogram", list(values), num_bins),
        Table = lambda data, columns: ("table", data, columns),
        plot = SimpleNamespace(line = lambda table, x, y, title: ("line", table, x, y, title)),
    )

    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    return run
