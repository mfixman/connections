import json

import pytest

from axiom_prediction.data import axiom_training_example_from_json
from axiom_prediction.dataset import collection_settings, load_axiom_dataset
from axiom_prediction.satcop import convert_example, convert_exports
from axiom_prediction.training import label_policy
from axiom_prediction.choices import ProverPolicy

@pytest.fixture
def native_export():
    def atom(symbol, *args):
        return {"symbol": symbol, "args": list(args)}

    source = {"kind": "input", "path": "'example.p'", "name": "a"}
    variable = {"variable": 0}
    return {
        "schema": "satcop.collection.v1",
        "problem_path": "example.p",
        "status": "Theorem",
        "prover_version": "0.1.0",
        "prover_revision": "abc123",
        "seed": 3,
        "threads": 1,
        "timeout_seconds": 10,
        "core_clause_ids": [0, 2],
        "matrix": {
            "have_conjecture": True,
            "start_clause_ids": [2],
            "grounding_constant": 2,
            "symbols": [
                {"name": "p", "arity": 1, "predicate": True},
                {"name": "p", "arity": 1, "predicate": True},
                {"name": "a", "arity": 0, "predicate": False},
            ],
            "clauses": [
                {
                    "is_goal": False, "variables": 1, "source": source,
                    "literals": [{"positive": True, "atom": atom(0, variable)}],
                    "disequations": [[variable, atom(2)]],
                },
                {
                    "is_goal": False, "variables": 1, "source": source,
                    "literals": [{"positive": True, "atom": atom(1, variable)}],
                    "disequations": [],
                },
                {
                    "is_goal": True, "variables": 0, "source": {**source, "name": "c"},
                    "literals": [{"positive": False, "atom": atom(0, atom(2))}],
                    "disequations": [],
                },
            ],
        },
    }

def test_native_labels_use_matrix_ids_and_preserve_symbol_identity(native_export, tmp_path):
    row = convert_example(native_export, tmp_path / "export.json")
    example = axiom_training_example_from_json(row)
    assert example.labels == [1, 0]
    assert example.graph.axiom_clause_ids == [0, 1]
    assert example.graph.conjecture_clause_ids == [2]
    assert len(example.graph.graph.nodes["var"]) == 2
    assert len(example.graph.graph.nodes["symbol"]) == 3
    assert len(example.graph.graph.edges["complement"]) == 2
    provenance = example.graph.graph.metadata["satcop"]
    assert provenance["seed"] == 3
    assert provenance["start_clause_ids"] == [2]
    assert provenance["disequations"]["0"] == native_export["matrix"]["clauses"][0]["disequations"]
    assert len(provenance["matrix_sha256"]) == 64

def test_native_csv_is_trainable_and_duplicate_exports_fail_atomically(native_export, tmp_path):
    export = tmp_path / "proof.json"
    export.write_text(json.dumps(native_export))
    output = tmp_path / "dataset.csv"
    assert convert_exports([export], output) == 1
    examples, failures, metadata = load_axiom_dataset(output)
    assert examples[0].labels == [1, 0]
    assert failures == []
    assert metadata["collection"]["prover_backend"] == "rust-satcop"
    assert label_policy(None, metadata) == ProverPolicy.SatCoP
    with pytest.raises(FileExistsError):
        convert_exports([export], output)

    duplicate = tmp_path / "duplicate.csv"
    with pytest.raises(ValueError, match = "multiple exports"):
        convert_exports([export, export], duplicate)

    assert not duplicate.exists()

def test_historical_satcop_metadata_selects_python_policy():
    assert label_policy(None, {"collection": {"sat_policy": "satcop"}}) == ProverPolicy.SatCoPCon
    assert collection_settings({"sat_policy": "satcop"}) == {"sat_policy": "satcopcon"}

def test_historical_python_checkpoint_can_resume_with_renamed_policy(monkeypatch):
    import random
    import torch
    from axiom_prediction import resume

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    rnd = random.Random(0)
    checkpoint = {
        "training_config": {"sat_policy": "satcop", "epochs": 1},
        "training_state": resume.snapshot_training(optimizer, rnd, "data"),
        "model_state_dict": model.state_dict(),
        "epoch": 1,
    }
    monkeypatch.setattr(resume, "load_checkpoint", lambda path: checkpoint)
    config = {"sat_policy": "satcopcon", "epochs": 2}
    assert resume.restore_training("model.ts", model, optimizer, rnd, config, "data") == 2

    checkpoint["training_config"]["collection"] = {"prover_backend": "rust-satcop"}
    with pytest.raises(ValueError, match = "same data and configuration"):
        resume.restore_training("model.ts", model, optimizer, rnd, config, "data")

def test_native_export_rejects_unproved_or_wrong_matrix_core(native_export, tmp_path):
    native_export["status"] = "TimeOut"
    with pytest.raises(ValueError, match = "successful refutation"):
        convert_example(native_export, tmp_path / "export.json")

    native_export["status"] = "Theorem"
    native_export["core_clause_ids"] = [999]
    with pytest.raises(ValueError, match = "out-of-matrix"):
        convert_example(native_export, tmp_path / "export.json")
