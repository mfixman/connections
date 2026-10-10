import json

import pytest

from axiom_prediction.data import axiom_example_schema
from axiom_prediction.dataset import axiom_dataset_shard_schema, load_axiom_dataset


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
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_combined_dataset_retains_per_shard_collection_settings(tmp_path):
    write_shard(tmp_path / "Mizar40.jsonl", step_limit = 1_000_000_000, timeout_seconds = 900)
    write_shard(tmp_path / "TPTP.jsonl", step_limit = 1_000_000, timeout_seconds = 120)

    examples, failures, metadata = load_axiom_dataset(tmp_path)

    assert len(examples) == 2
    assert failures == []
    assert metadata["collection"]["sat_policy"] == "satresetcop"
    assert metadata["collection"]["source_shards"]["Mizar40.jsonl"]["timeout_seconds"] == 900
    assert metadata["collection"]["source_shards"]["TPTP.jsonl"]["step_limit"] == 1_000_000


def test_combined_dataset_rejects_different_label_policies(tmp_path):
    write_shard(tmp_path / "Mizar40.jsonl", step_limit = 1_000_000_000, timeout_seconds = 900)
    write_shard(
        tmp_path / "TPTP.jsonl",
        step_limit = 1_000_000,
        timeout_seconds = 120,
        sat_policy = "satcop",
    )

    with pytest.raises(ValueError, match="incompatible collection settings"):
        load_axiom_dataset(tmp_path)
