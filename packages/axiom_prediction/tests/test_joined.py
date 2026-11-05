import json

import numpy as np
import pytest

from axiom_prediction.cli import main
from axiom_prediction.dataset import axiom_dataset_schema, axiom_dataset_shard_schema, dataset_fields, load_axiom_dataset
from axiom_prediction.data import axiom_training_example_to_json
from axiom_prediction.io import write_csv
from axiom_prediction.joined import join_datasets, one_way_complements
from axiom_prediction.tptp import collect_proof_example

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

def collected(tmp_path, problem, seed):
    output = tmp_path / f"seed-{seed}"
    arguments = ["collect", str(problem), "--dataset", str(output), "--seed", str(seed), "--num-workers", "1"]
    assert main(arguments) == 0
    return output

def table_rows(tables):
    return sorted(
        (name, sorted(map(tuple, np.asarray(rows).tolist())))
        for name, rows in tables.items()
    )

def contents(examples):
    return sorted(
        (
            example.problem_path,
            example.labels,
            table_rows(example.graph.graph.nodes),
            table_rows(example.graph.graph.edges),
        )
        for example in examples
    )

def old_shard(path, problem, seed):
    example, _ = collect_proof_example(problem, seed = seed)
    header = {
        "schema": axiom_dataset_shard_schema,
        "record": "metadata",
        "collection": {"sat_policy": "satresetcop", "seed": seed},
    }

    failure = {"schema": axiom_dataset_schema, "problem": "/old/failed.p", "outcome": "Timeout"}
    path.parent.mkdir(parents = True)
    write_csv(path, [header, axiom_training_example_to_json(example), failure], dataset_fields)

def test_joined_dataset_matches_the_csv_loader(tmp_path, choices):
    inputs = [collected(tmp_path, choices, seed) for seed in (0, 0, 1, 2)]
    old_shard(tmp_path / "old" / ".cache/problems/cache.csv", choices, 1)
    (tmp_path / "old/problems.csv").write_text("schema\n")
    joined = tmp_path / "joined"

    arguments = ["join", *map(str, inputs), str(tmp_path / "old"), "--dataset", str(joined), "--num-workers", "2"]
    assert main(arguments) == 0

    examples, failures, metadata = load_axiom_dataset(joined)
    assert len({id(example.graph) for example in examples}) == 1
    assert failures == [{"problem": "failed.p", "outcome": "Timeout"}]
    assert len(metadata["shards"]) == 4

    described = json.loads((joined / "metadata.json").read_text())
    assert described["seeds"] == [0, 1, 2]
    assert described["provers"] == ["satresetcop"]
    assert described["examples"] == len(examples)

    rows = {}
    for directory in inputs:
        csv_examples, _, _ = load_axiom_dataset(directory)
        rows.update({tuple(example.labels): example for example in csv_examples})

    assert contents(examples) == contents(rows.values())
    assert len(examples) > 1

def test_joined_dataset_trains_and_filters_large_records(tmp_path, choices):
    from axiom_prediction import training
    from axiom_prediction.split import ProblemSplit
    from axiom_prediction.wandb_tracking import WandbConfig

    joined = tmp_path / "joined"
    join_datasets([collected(tmp_path, choices, seed) for seed in (0, 1)], joined, 1)
    part = ProblemSplit(10).part(choices)
    metrics = training.train_axiom_predictor(
        dataset = joined,
        output_dir = tmp_path / "model",
        config = training.AxiomTrainingConfig(
            epochs = 1, hidden_dim = 8, message_rounds = 1,
            batch_size = 2, device = "cpu", num_workers = 1,
        ),
        split = ProblemSplit(10, [part]),
        wandb_config = WandbConfig(enabled = False),
    )

    assert metrics["problems_proved"] == 2
    with pytest.raises(RuntimeError, match = "no training examples"):
        load_axiom_dataset(joined, max_example_mb = 0.0001)

def test_loader_rejects_mixed_old_and_new_results(tmp_path, choices):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "new").symlink_to(collected(tmp_path, choices, 0))
    old_shard(tmp_path / "old" / "cache.csv", choices, 1)
    (mixed / "old").symlink_to(tmp_path / "old")
    with pytest.raises(ValueError, match = "mixes old and new"):
        load_axiom_dataset(mixed)

def test_join_refuses_stubs_and_existing_output(tmp_path, choices):
    stub = tmp_path / "stub.csv"
    header = {"schema": axiom_dataset_shard_schema, "record": "metadata", "collection": {}}
    write_csv(stub, [header], dataset_fields)
    with pytest.raises(ValueError, match = "has no results"):
        join_datasets([stub], tmp_path / "from-stub", 1)

    joined = tmp_path / "joined"
    join_datasets([collected(tmp_path, choices, 0)], joined, 1)
    with pytest.raises(FileExistsError):
        join_datasets([tmp_path / "seed-0"], joined, 1)

def test_unmirrored_complements_are_rejected():
    rows = np.asarray([[0, 1], [1, 0], [2, 3]], dtype = np.int32)
    with pytest.raises(ValueError, match = "not mirrored"):
        one_way_complements(rows, "bad.p")

    assert one_way_complements(rows[:2], "good.p").tolist() == [[0, 1]]
