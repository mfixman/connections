import csv
import json
from dataclasses import fields
from pathlib import Path
import sys

import pytest

from axiom_prediction import cli, sine, training
from axiom_prediction.choices import GuidanceMode
from axiom_prediction.graph import build_axiom_graph
from axiom_prediction.representation.schema import GraphInput
from axiom_prediction.output import output_fields
from axiom_prediction.output_resume import OutputJournal, resume_identity
from axiom_prediction.run import RunConfig, run_problem, run_problems
from axiom_prediction.tptp import load_tptp_problem, collect_proof_example
from axiom_prediction.wandb_tracking import WandbConfig
from test_axiom_prediction_wandb import _FakeRun, fake_wandb

@pytest.mark.parametrize("conjecture, expected", [
    ({"common"}, [0, 0, 0, 0, 0]),
    ({"rare"}, [1, 1, 1, 0, 0]),
    ({"other"}, [0, 0, 0, 1, 0]),
    (set(), [0, 0, 0, 0, 0]),
])
def test_fixed_point_and_rare_triggers(conjecture, expected):
    axioms = [
        {"rare", "bridge", "common"}, {"bridge", "end", "common"},
        {"end", "common"}, {"other"}, set(),
    ]
    assert sine.select(axioms, conjecture) == expected

def test_frequency_ties():
    axioms = [{"a", "b"}, {"a", "c"}, {"b", "d"}]
    assert sine.select(axioms, {"a"}) == [1, 0, 0]
    assert sine.select(axioms, {"b"}) == [1, 0, 0]
    assert sine.select(axioms, {"c"}) == [1, 1, 0]

def test_saved_graph_and_matrix_agree(tmp_path):
    path = tmp_path / "nested.p"
    path.write_text(
        "cnf(a,axiom,p(f(a),X) | p(f(a),X) | q(X)).\n"
        "cnf(b,axiom,q(f(a))).\n"
        "cnf(c,axiom,disconnected(g(b))).\n"
        "cnf(d,negated_conjecture,~p(f(a),Y)).\n"
    )
    loaded = load_tptp_problem(str(path))
    graph = build_axiom_graph(
        loaded.matrix, axiom_clause_ids = loaded.axiom_clause_ids,
        conjecture_clause_ids = loaded.conjecture_clause_ids,
    )
    symbols = sine.matrix_symbols(loaded.matrix)
    assert symbols[0] == {("p", 0), ("q", 0), ("f", 1), ("a", 2)}
    expected = sine.score_matrix(
        loaded.matrix, loaded.axiom_clause_ids, loaded.conjecture_clause_ids,
    )
    saved = GraphInput.from_dict(json.loads(json.dumps(graph.graph.to_dict())))
    path.unlink()
    assert sine.graph_symbols(saved) == sine.graph_symbols(graph.graph)
    assert expected == [1.0, 1.0, 0.0]
    assert sine.clause_scores(
        sine.graph_symbols(graph.graph), graph.axiom_clause_ids, graph.conjecture_clause_ids,
    ) == expected
    assert sine.clause_scores(sine.graph_symbols(graph.graph), [2, 0, 1], []) == [0, 0, 0]
    assert sine.clause_scores(
        sine.graph_symbols(graph.graph), [2, 0, 1], graph.conjecture_clause_ids,
    ) == [0, 1, 1]

def forbidden(*args, **kwargs):
    raise AssertionError("SInE must not construct neural inference or resolve a device")

@pytest.fixture
def no_neural(monkeypatch):
    from axiom_prediction import model, multiprocess

    monkeypatch.setattr(model.AxiomPredictor, "load", forbidden)
    monkeypatch.setattr(model, "resolve_device", forbidden)
    monkeypatch.setattr(multiprocess, "inference_service", forbidden)
    monkeypatch.setattr(cli, "print_device", forbidden)
    monkeypatch.setattr(training.torch.cuda, "is_available", forbidden)

@pytest.mark.parametrize("policy", ["satcop", "satresetcop"])
def test_search_binary_weights_and_floor(tmp_path, tiny_problem_path, monkeypatch, no_neural, policy):
    from axiom_prediction import guided

    path = tmp_path / "search.p"
    path.write_text(tiny_problem_path.read_text() + "\nfof(unused,axiom,unrelated(z)).\n")
    original = guided.AxiomGuided.__init__
    captured = {}

    def capture(self, **kwargs):
        captured.update(kwargs)
        original(self, **kwargs)
        captured["retained"] = self.weights

    monkeypatch.setattr(guided.AxiomGuided, "__init__", capture)
    config = RunConfig(mode = "sine", policy = policy, timeout_s = 10)
    result = run_problem(str(path), tptp_root = None, config = config)
    assert result["proved"], result
    assert result["mode"] == "sine"
    assert result["prediction_seconds"] > 0
    assert set(captured["clause_weights"].values()) == {0.0, 1.0}
    assert set(captured["retained"].values()) == {1e-6, 1.0}
    assert len(captured["retained"]) == len(load_tptp_problem(str(path)).matrix.clauses)
    assert captured["mode"] == GuidanceMode.Weighted
    assert captured["temperature"] == 1
    assert captured["matrix_digest"]

def test_multiprocess_without_inference(tiny_problem_path, no_neural):
    results = list(run_problems(
        [str(tiny_problem_path)] * 2, tptp_root = None, num_workers = 2,
        config = RunConfig(mode = "sine", multiprocess = True, timeout_s = 15),
    ))
    assert len(results) == 2
    assert all(row["proved"] and row["mode"] == "sine" for row in results)

@pytest.mark.parametrize("command", ["run", "evaluate"])
def test_cli_conflicts(command, capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([command, "--sine", "--model", "missing"])

    assert cli.main([command, "--sine", "--model-name", "SmallFull"]) == 2
    assert "incompatible with --model-name" in capsys.readouterr().err
    assert cli.main(["run", "--sine", "--data-dir", "missing"]) == 2

@pytest.mark.parametrize("use_csv", [False, True])
@pytest.mark.parametrize("command", ["run", "evaluate"])
def test_cli_fresh_results_and_resume(tmp_path, tiny_problem_path, no_neural, use_csv, command):
    output = tmp_path / "results"
    args = [
        command, "--sine", str(tiny_problem_path), "--output", str(output),
        "--num-workers", "1", "--no-wandb", "--multiprocess",
    ] + (["--csv"] if use_csv else [])
    assert cli.main(args) == 0
    content = output.read_text()
    rows = list(csv.DictReader(content.splitlines())) if use_csv else list(map(json.loads, content.splitlines()))
    rows = [row for row in rows if row.get("problem")]
    assert len(rows) == 1
    assert int(rows[0]["network_size"]) == 0
    if command == "run":
        assert rows[0]["mode"] == "sine"

    directory = Path(str(output) + ".resume")
    completed = json.loads((directory / "completion.json").read_text())
    assert completed["metrics"]["sine_settings"] == sine.settings()["sine_settings"]
    assert "model_config" not in completed["metrics"]
    assert cli.main(args) == 0
    assert output.read_text() == content
    assert cli.main([arg for arg in args if arg != "--sine"]) == 2
    assert cli.main([*args[:1], "--model", "missing", *args[2:]]) == 2

def test_saved_dataset_policy_and_wandb(tmp_path, tiny_problem_path, monkeypatch, no_neural):
    example, _ = collect_proof_example(str(tiny_problem_path), step_limit = 100)
    monkeypatch.setattr(training, "load_axiom_dataset", lambda _: (
        [example], [], {"collection": {"sat_policy": "satcop"}},
    ))
    monkeypatch.setattr(training, "collect_examples", forbidden)
    run = _FakeRun()
    init = {}
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb(run, init))
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    metrics = training.evaluate_axiom_predictor(None, sine = True, dataset = tmp_path)
    assert metrics["sat_policy"] == "satcop"
    assert metrics["network_size"] == 0
    assert init["config"]["sine_settings"] == sine.settings()["sine_settings"]
    assert "model_config" not in init["config"]
    assert "training_split" not in metrics
    assert run.finished == [0]
    assert cli.main([
        "run", "--sine", str(tiny_problem_path), "--num-workers", "1",
    ]) == 0
    assert init["config"]["mode"] == "sine"
    assert init["config"]["sine_settings"] == sine.settings()["sine_settings"]
    for source in ("--dataset", "--data-dir"):
        assert cli.main(["evaluate", "--sine", source, str(tmp_path), "--no-wandb"]) == 0

    with pytest.raises(ValueError, match = "dataset labels"):
        training.evaluate_axiom_predictor(
            None, sine = True, dataset = tmp_path,
            config = training.AxiomTrainingConfig(sat_policy = "satresetcop"),
            wandb_config = WandbConfig(enabled = False),
        )

def test_prediction_cache_resumes(tmp_path, tiny_problem_path, monkeypatch):
    example, _ = collect_proof_example(str(tiny_problem_path), step_limit = 100)
    journal = OutputJournal(tmp_path / "results", "evaluate", False, output_fields("evaluate"), sine.settings())
    try:
        first = training.example_outputs(None, [example], sine = True, batch_size = 1, resume = journal)
        assert set(first[1]) <= {0.0, 1.0}
        monkeypatch.setattr(sine, "graph_symbols", forbidden)
        assert training.example_outputs(None, [example], sine = True, batch_size = 1, resume = journal) == first
    finally:
        journal.close()

def test_configuration_compatibility():
    assert resume_identity({"command": "run", "sine": False}) == {"command": "run"}
    config = RunConfig(mode = "sine")
    assert RunConfig.from_state(config.to_dict()) == config
    assert RunConfig.from_state([getattr(config, field.name) for field in fields(config)]) == config
    with pytest.raises(ValueError, match = "checkpoint"):
        RunConfig(mode = "sine", checkpoint = "model.pt")
