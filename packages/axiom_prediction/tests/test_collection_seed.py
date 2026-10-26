import csv
import io
import json

import pytest

from axiom_prediction.cli import main
from axiom_prediction.dataset import load_axiom_dataset
from axiom_prediction.tptp import collect_proof_example

@pytest.mark.parametrize("policy", ["satresetcop", "satcopcon"])
def test_collection_seed_changes_labels_and_protects_resume(tmp_path, policy, capsys):
    problem = tmp_path / "choices.p"
    problem.write_text(
        "fof(a0,axiom,p(a0)).\n"
        "fof(a1,axiom,p(a1)).\n"
        "fof(a2,axiom,p(a2)).\n"
        "fof(c,conjecture,? [X]:p(X)).\n"
    )
    common = [
        "collect", str(problem), "--policy", policy,
        "--num-workers", "1", "--step-limit", "1000", "--timeout-seconds", "10",
    ]
    labels = []
    for name, seed_args in [("default", []), ("zero", ["--seed", "0"]), ("one", ["--seed", "1"])]:
        output = tmp_path / name
        assert main([*common, "--dataset", str(output), *seed_args]) == 0
        examples, failures, metadata = load_axiom_dataset(output)
        assert failures == []
        assert metadata["collection"]["seed"] == (1 if name == "one" else 0)
        assert sum(examples[0].labels) == 1
        labels.append(examples[0].labels)

    assert labels[0] == labels[1]
    assert labels[0] != labels[2]
    capsys.readouterr()

    output = tmp_path / "default"
    metadata_path = output / ".cache/problems/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["collection"].pop("seed")
    metadata_path.write_text(json.dumps(metadata))

    assert main([*common, "--dataset", str(output), "--seed", "0"]) == 0
    summary = next(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert summary["problems_reused"] == "1"
    assert main([*common, "--dataset", str(output), "--seed", "1"]) == 2
    assert "collection settings differ" in capsys.readouterr().err

def test_collection_still_passes_seed_to_native_policy(monkeypatch):
    from axiom_prediction import satcop

    calls = []

    def collect(problem, **kwargs):
        calls.append(kwargs)
        return None, "Timeout"

    monkeypatch.setattr(satcop, "collect_native_example", collect)
    assert collect_proof_example("unused.p", sat_policy = "satcop", seed = 7) == (None, "Timeout")
    assert calls[0]["seed"] == 7

def test_training_visits_each_distinct_core_once(tmp_path, monkeypatch):
    from axiom_prediction import training
    from axiom_prediction.data import axiom_training_example_to_json
    from axiom_prediction.dataset import axiom_dataset_shard_schema, dataset_fields
    from axiom_prediction.io import write_csv
    from axiom_prediction.split import ProblemSplit
    from axiom_prediction.wandb_tracking import WandbConfig

    problem = tmp_path / "choices.p"
    problem.write_text(
        "fof(a0,axiom,p(a0)).\nfof(a1,axiom,p(a1)).\nfof(a2,axiom,p(a2)).\n"
        "fof(c,conjecture,? [X]:p(X)).\n"
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    expected = []
    for seed in (0, 1):
        example, outcome = collect_proof_example(problem, seed = seed)
        assert outcome == "proved"
        expected.append(tuple(example.labels))
        row = axiom_training_example_to_json(example)
        header = {
            "schema": axiom_dataset_shard_schema, "record": "metadata",
            "collection": {"sat_policy": "satresetcop", "seed": seed},
        }
        write_csv(dataset / f"seed{seed}.csv", [header, row, row], dataset_fields)

    seen = []
    collate = training.collate_axiom_graphs

    def record_batch(chunk):
        seen.extend(tuple(labels) for graph, labels in chunk)
        return collate(chunk)

    monkeypatch.setattr(training, "collate_axiom_graphs", record_batch)
    part = ProblemSplit(10).part(problem)
    metrics = training.train_axiom_predictor(
        dataset = dataset,
        output_dir = tmp_path / "model",
        config = training.AxiomTrainingConfig(
            epochs = 1, hidden_dim = 8, message_rounds = 1,
            batch_size = 1, device = "cpu", num_workers = 1,
        ),
        split = ProblemSplit(10, [part]),
        wandb_config = WandbConfig(enabled = False),
    )

    assert expected[0] != expected[1]
    assert sorted(seen) == sorted(expected)
    assert metrics["problems_proved"] == 2
