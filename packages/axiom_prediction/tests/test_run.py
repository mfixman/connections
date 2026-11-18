import csv
import io
import multiprocessing as mp
from pathlib import Path
import signal
import time

import pytest

from axiom_prediction import guided
from axiom_prediction.cli import main
from axiom_prediction.run import RunConfig, run_problem
from axiom_prediction.search_workers import supervised_results
from axiom_prediction.split import ProblemSplit
from axiom_prediction.tptp import load_tptp_problem
from axiom_prediction.training import AxiomTrainingConfig, train_axiom_predictor
from axiom_prediction.wandb_tracking import WandbConfig

def ignores_alarm(problem, *args):
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    Path(problem).touch()
    time.sleep(30)
    return {"problem": problem, "proved": True}

def test_supervisor_terminates_unresponsive_workers(tmp_path):
    before = {child.pid for child in mp.active_children()}
    started = time.monotonic()
    requests = [(str(tmp_path / "one"),), (str(tmp_path / "two"),)]
    results = list(supervised_results(ignores_alarm, requests, workers = 1, timeout = 0.5))
    assert [result["outcome"] for result in results] == ["Timeout", "Timeout"]
    assert time.monotonic() - started < 10
    assert {child.pid for child in mp.active_children()} == before

def rows(text):
    return list(csv.DictReader(io.StringIO(text)))

def test_guided_search_through_the_cli_resumes_its_output(tmp_path, chain_problems, joined_dataset, capsys):
    config = AxiomTrainingConfig(model = "SmallFull", epochs = 2, device = "cpu", num_workers = 1)
    train_axiom_predictor(dataset = joined_dataset, output_dir = tmp_path / "model", config = config, wandb_config = WandbConfig(enabled = False))
    held_out = ProblemSplit(10, [8, 9]).select(sorted(map(str, chain_problems.glob("*.p"))))
    common = ["run", str(chain_problems), "--split", "10", "--parts", "8", "9", "--cpu", "--num-workers", "2", "--no-wandb", "--seed", "1"]
    output = tmp_path / "guided.csv"
    guided_run = [*common, "--model", str(tmp_path / "model"), "--output", str(output)]
    assert main(guided_run) == 0
    guided_rows = rows(output.read_text())
    assert sorted(row["problem"] for row in guided_rows) == sorted(Path(problem).name for problem in held_out)
    assert all(row["proved"] == "true" and row["mode"] == "weighted" and int(row["network_size"]) > 0 for row in guided_rows)
    assert {row["part"] for row in guided_rows} <= {"8", "9"}

    # A finished output is left alone; a torn one resumes only what is missing; other settings need a new file.
    contents = output.read_text()
    assert main(guided_run) == 0
    assert output.read_text() == contents
    output.write_text("".join(contents.splitlines(keepends = True)[:2]) + "torn")
    assert main(guided_run) == 0
    assert sorted(row["problem"] for row in rows(output.read_text())) == sorted(row["problem"] for row in guided_rows)
    assert main([*guided_run[:-2], "--seed", "2", "--output", str(output)]) == 2

    capsys.readouterr()
    for arguments, mode in [([], "base"), (["--sine"], "sine")]:
        assert main([*common, *arguments]) == 0
        printed = rows(capsys.readouterr().out)
        assert len(printed) == len(held_out)
        assert all(row["proved"] == "true" and row["mode"] == mode for row in printed)

def test_sine_weighs_selected_clauses_and_floors_the_rest(tmp_path, tiny_problem_path, monkeypatch):
    path = tmp_path / "search.p"
    path.write_text(tiny_problem_path.read_text() + "\nfof(unused,axiom,unrelated(z)).\n")
    original = guided.AxiomGuided.__init__
    captured = {}

    def capture(self, **kwargs):
        captured.update(kwargs)
        original(self, **kwargs)
        captured["retained"] = self.weights

    monkeypatch.setattr(guided.AxiomGuided, "__init__", capture)
    result = run_problem(str(path), tptp_root = None, config = RunConfig(mode = "sine", timeout_s = 10))
    assert result["proved"] and result["mode"] == "sine", result
    assert set(captured["clause_weights"].values()) == {0.0, 1.0}
    assert set(captured["retained"].values()) == {1e-6, 1.0}
    assert len(captured["retained"]) == len(load_tptp_problem(str(path)).matrix.clauses)
    assert captured["matrix_digest"]

def test_unparseable_problems_are_error_rows(tmp_path, tiny_problem_path):
    problem = tmp_path / "broken.p"
    problem.write_text("fof(broken, axiom, ).\n")
    output = tmp_path / "results"
    common = ["--num-workers", "1", "--no-wandb", "--output", str(output)]
    assert main(["run", str(problem), *common]) == 2
    row, = rows(output.read_text())
    assert (row["error"], row["proved"]) == ("true", "false")

    assert main(["run", str(problem), str(tiny_problem_path), "--num-workers", "1", "--no-wandb"]) == 0
    with pytest.raises(ValueError, match = "do not go together"):
        RunConfig(mode = "weighted")
