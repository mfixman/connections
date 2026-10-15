from axiom_prediction.io import read_csv, write_csv

import pytest

from axiom_prediction.data import axiom_example_schema
from axiom_prediction.dataset import axiom_dataset_shard_schema, load_axiom_dataset, dataset_fields


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
