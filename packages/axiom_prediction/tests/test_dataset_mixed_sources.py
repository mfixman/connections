from axiom_prediction.io import read_csv, write_csv

import json

import pytest

from axiom_prediction.data import axiom_example_schema
from axiom_prediction.dataset import axiom_dataset_schema, axiom_dataset_shard_schema, load_axiom_dataset, dataset_fields


def write_shard(path, *, step_limit, timeout_seconds, sat_policy = "satresetcop"):
    records = [
        {
            "schema": axiom_dataset_shard_schema,
            "record": "metadata",
            "collection": {
                "sat_policy": sat_policy,
                "step_limit": step_limit,
                "timeout_seconds": timeout_seconds,
                "tptp_root": f"/data/{path.stem}",
            },
        },
        {
            "schema": axiom_example_schema,
            "problem_path": f"{path.stem}.p",
            "graph": {
                "nodes": {},
                "edges": {},
                "actions": [],
                "preprocessor": "axiom_prediction",
                "version": "1",
            },
            "axiom_clause_ids": [0],
            "conjecture_clause_ids": [],
            "labels": [1.0],
        },
    ]
    write_csv(path, records, dataset_fields)


def test_combined_dataset_retains_per_shard_collection_settings(tmp_path):
    write_shard(tmp_path / "Mizar40.csv", step_limit = 1_000_000_000, timeout_seconds = 900)
    write_shard(tmp_path / "TPTP.csv", step_limit = 1_000_000, timeout_seconds = 120)

    examples, failures, metadata = load_axiom_dataset(tmp_path)

    assert len(examples) == 2
    assert failures == []
    assert metadata["collection"]["sat_policy"] == "satresetcop"
    assert metadata["collection"]["source_shards"]["Mizar40.csv"]["timeout_seconds"] == 900
    assert metadata["collection"]["source_shards"]["TPTP.csv"]["step_limit"] == 1_000_000


def test_combined_dataset_rejects_different_label_policies(tmp_path):
    write_shard(tmp_path / "Mizar40.csv", step_limit = 1_000_000_000, timeout_seconds = 900)
    write_shard(
        tmp_path / "TPTP.csv",
        step_limit = 1_000_000,
        timeout_seconds = 120,
        sat_policy = "satcop",
    )

    with pytest.raises(ValueError, match="incompatible collection settings"):
        load_axiom_dataset(tmp_path)

def test_combined_dataset_keeps_distinct_cores_across_seeds(tmp_path):
    for seed, label in enumerate([1.0, 0.0, 1.0, 0.0]):
        path = tmp_path / f"seed{seed}.csv"
        write_shard(path, step_limit = 100, timeout_seconds = 10)
        rows = list(read_csv(path))
        rows[0]["collection"]["seed"] = seed
        rows[0]["collection"]["predictor"] = f"model{seed}"
        rows[1].update(problem_path = "shared.p", labels = [label])
        rows[1]["graph"]["metadata"] = {"predictor": f"model{seed}"}

        rows.append(rows[1])
        rows.append({"schema": axiom_dataset_schema, "problem": "shared.p", "outcome": "Timeout"})
        write_csv(path, rows, dataset_fields)

    examples, failures, metadata = load_axiom_dataset(tmp_path)
    assert [example.labels for example in examples] == [[1.0], [0.0]]
    assert all(example.problem_path == "shared.p" for example in examples)
    assert failures == []
    sources = metadata["collection"]["source_shards"]
    assert [sources[f"seed{seed}.csv"]["seed"] for seed in range(4)] == list(range(4))

def test_legacy_dataset_keeps_each_distinct_core_once(tmp_path):
    shard = tmp_path / "source.csv"
    write_shard(shard, step_limit = 100, timeout_seconds = 10)
    header, example = read_csv(shard)
    dataset = tmp_path / "legacy"
    records = dataset / "examples"
    records.mkdir(parents = True)
    metadata = {"schema": axiom_dataset_schema, "collection": header["collection"]}
    (dataset / "metadata.json").write_text(json.dumps(metadata))

    for index, label in enumerate([1.0, 0.0, 1.0]):
        example["labels"] = [label]
        (records / f"{index}.json").write_text(json.dumps(example))

    examples, _, _ = load_axiom_dataset(dataset)
    assert [example.labels for example in examples] == [[1.0], [0.0]]

@pytest.mark.parametrize("change", ["graph", "native_matrix"])
def test_repeated_core_rejects_conflicting_matrices(tmp_path, change):
    for index in range(2):
        path = tmp_path / f"run{index}.csv"
        write_shard(path, step_limit = 100, timeout_seconds = 10)
        rows = list(read_csv(path))
        rows[1]["problem_path"] = "shared.p"
        if change == "graph":
            rows[1]["graph"]["nodes"] = {"clause": [[index]]}
        else:
            rows[1]["graph"]["metadata"] = {"satcop": {"matrix_sha256": str(index)}}

        write_csv(path, rows, dataset_fields)

    with pytest.raises(ValueError, match = "conflicting matrices"):
        load_axiom_dataset(tmp_path)

def test_dataset_csv_preserves_large_nested_records_and_empty_values(tmp_path):
    path = tmp_path / "records.csv"
    records = [
        {"problem_path": 'null', "axiom_clause_texts": ['a,b\n"c"' * 30_000]},
        {"problem_path": "", "labels": [], "graph": {"nodes": {}, "extra": None}},
        {"problem": "true", "parseable": False, "error": True, "outcome": None},
    ]
    write_csv(path, records, dataset_fields)
    assert list(read_csv(path)) == records

@pytest.mark.parametrize("content", ['a,b\n1\n', 'a,b\n1,2,3\n'])
def test_dataset_csv_rejects_missing_or_extra_cells(tmp_path, content):
    path = tmp_path / "broken.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match = "corrupt dataset CSV"):
        list(read_csv(path))
