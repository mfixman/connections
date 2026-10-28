import json

import pytest

from axiom_prediction.data import axiom_training_example_from_json
from axiom_prediction.dataset import collection_settings, dataset_fields, load_axiom_dataset
from axiom_prediction.io import read_csv, write_csv
from axiom_prediction.satcop import convert_example, convert_exports, decode_matrix, expand_matrix
from axiom_prediction.satcop_dump import parse_dump, parse_matrix, rust_string, symbol_name
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

def test_native_csv_is_trainable_and_repeated_exports_have_no_extra_weight(native_export, tmp_path):
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
    assert convert_exports([export, export], duplicate) == 2
    examples, failures, _ = load_axiom_dataset(duplicate)
    assert len(examples) == 1
    assert examples[0].labels == [1, 0]
    assert failures == []

    alternative = tmp_path / "alternative.json"
    native_export.update(seed = 9, core_clause_ids = [1, 2])
    alternative.write_text(json.dumps(native_export))
    mixed = tmp_path / "mixed.csv"
    assert convert_exports([export, export, alternative], mixed) == 3
    examples, _, _ = load_axiom_dataset(mixed)
    assert [example.labels for example in examples] == [[1, 0], [0, 1]]

    for path, seed in [(output, 3), (mixed, 9)]:
        rows = list(read_csv(path))
        rows[0]["collection"]["seed"] = seed
        write_csv(path, rows, dataset_fields)

    examples, _, metadata = load_axiom_dataset(tmp_path)
    assert len(examples) == 2
    assert metadata["collection"]["prover_backend"] == "rust-satcop"
    assert label_policy(None, metadata) == ProverPolicy.SatCoP

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

def test_compact_tables_reconstruct_terms_scopes_and_sources(native_export, tmp_path):
    tables = {
        "symbols": [["sG", 0, False], ["sPE", 2, True], ["p", 1, True], ["f", 1, False], ["a", 0, False]],
        "terms": [0, 4, 3, 0, 2, 2, 3, 1, 2, 6, -1, 1, 10, 10],
        "literals": [[1, 4], [0, 8], [1, 11]],
        "disequations": [[0, 1]],
        "clauses": [[1, 0, 1, 0, 1, 0], [0, 1, 2, 1, 1, 1], [2, 2, 3, 1, 1, 0]],
        "sources": [["'example.p'", "a"], ["'example.p'", "c"], None],
        "start_clause_ids": [1],
        "have_conjecture": True,
        "grounding_constant": 4,
    }
    matrix = expand_matrix(tables)
    variable = {"variable": 0}
    constant = {"symbol": 4, "args": []}
    assert matrix["clauses"][0]["literals"][0]["atom"] == {
        "symbol": 2, "args": [{"symbol": 3, "args": [variable]}],
    }
    assert matrix["clauses"][0]["disequations"] == [[variable, constant]]
    assert matrix["clauses"][1]["literals"][0]["atom"] == {
        "symbol": 2, "args": [{"symbol": 3, "args": [constant]}],
    }
    assert matrix["clauses"][2]["source"] == {"kind": "equality"}
    assert matrix["clauses"][2]["literals"][0]["atom"]["args"] == [{"variable": 1}] * 2
    payload = {**native_export, "schema": "satcop.collection.v2", "matrix": tables, "core_clause_ids": [0, 1]}
    example = convert_example(payload, tmp_path / "export.json")
    assert example["axiom_clause_ids"] == [0, 2]
    assert example["conjecture_clause_ids"] == [1]
    assert example["labels"] == [1, 0]

@pytest.fixture
def native_stdout():
    return '''% SATCoP matrix (true, #0, "example.p", 3, Some(5))
% SATCoP terms Block([2])
% SATCoP literals Block([#0, ¬#0])
% SATCoP disequations Block([])
% SATCoP start [#1]
% SATCoP symbol #0 0 false Grounding
% SATCoP symbol #1 2 true Equality
% SATCoP symbol #2 0 true Atom("p")
% SATCoP clause #0 Clause { vars: #0, literals: #0-#1, disequations: #0-#0 } Info { is_goal: false, source: Axiom { path: "'example.p'", name: "a1" } }
% SATCoP clause #1 Clause { vars: #0, literals: #1-#2, disequations: #0-#0 } Info { is_goal: true, source: Axiom { path: "'example.p'", name: "c" } }
% SZS status Theorem for example
% SZS output start ListOfCNF for example
% SATCoP core 1
% SATCoP core 0
% SATCoP core 0
% SZS output end ListOfCNF for example
'''

def test_debug_dump_can_be_converted_directly_to_training_csv(native_stdout, tmp_path):
    path = tmp_path / "example.out"
    path.write_text(native_stdout)
    payload = parse_dump(native_stdout)
    assert payload["seed"] == 3
    assert payload["core_clause_ids"] == [0, 1]
    assert payload["problem_path"] == "example.p"
    assert payload["matrix"]["symbols"][2] == ["p", 0, True]
    output = tmp_path / "dataset.csv"
    assert convert_exports([path], output) == 1
    examples, _, _ = load_axiom_dataset(output)
    assert examples[0].labels == [1]

def test_debug_dump_rejects_truncated_or_changed_output(native_stdout):
    with pytest.raises(ValueError, match = "complete successful proof"):
        parse_dump(native_stdout.replace("% SZS output end ListOfCNF", "% truncated"))

    with pytest.raises(ValueError, match = "literal format"):
        parse_dump(native_stdout.replace("Block([#0, ¬#0])", "Block([broken])"))

# `satcop tiny.p --collect --clausify` for p(a), p(X) => q(X) |- q(a).
clausified_stdout = '''% SATCoP terms Block([3, 2, 0, 0, 2, 3, 4, 3, 4, 0])
% SATCoP literals Block([#1, ¬#4, #6, ¬#8])
% SATCoP symbol #0 0
% SATCoP symbol #1 2
% SATCoP symbol #2 1
% SATCoP symbol #3 0
% SATCoP symbol #4 1
% SATCoP clause #0 #0 #0-#1 false
% SATCoP clause #1 #1 #1-#3 false
% SATCoP clause #2 #0 #3-#4 true
cnf(c0, axiom, p(a)). % 
cnf(c1, axiom, ~p(X0) | q(X0)). % 
cnf(c2, negated_conjecture, ~q(a)). % 
'''

def test_matrix_dump_before_search_matches_the_proof_dump():
    proof = "\n".join([
        *clausified_stdout.splitlines()[:10],
        "% SZS status Theorem for tiny",
        "% SZS output start ListOfCNF for tiny",
        "% SATCoP core 0",
        "% SZS output end ListOfCNF for tiny",
    ])

    before = decode_matrix(expand_matrix(parse_matrix(clausified_stdout)))
    after = decode_matrix(expand_matrix(parse_dump(proof, problem = "tiny.p")["matrix"]))
    assert list(map(str, before.clauses)) == list(map(str, after.clauses))
    assert before.conjecture_clauses == after.conjecture_clauses == (2,)

def test_rust_debug_strings_and_generated_names():
    assert rust_string(r'"\u{1f}\0\n\r\t\"\\ café🦀"') == '\x1f\0\n\r\t"\\ café🦀'
    assert rust_string(r'"\\u{1f}\\0"') == r'\u{1f}\0'
    assert symbol_name('Quoted("p")') == "'p'"
    assert symbol_name('Distinct("a")') == '"a"'
    assert symbol_name('Skolem(#3)') == 'sK3'
    assert symbol_name('Definition(#5)') == 'sP5'

def test_native_wrapper_reaps_a_panicking_search_process(tmp_path):
    import os
    import sys
    import time
    from axiom_prediction.satcop import run_process

    log_path = tmp_path / "panic.log"
    code = 'import time; print("thread satcop panicked at failure", flush=True); time.sleep(30)'
    started = time.monotonic()
    with log_path.open("w") as log_file:
        with pytest.raises(RuntimeError, match = "search panicked"):
            run_process([sys.executable, "-c", code], dict(os.environ), log_file, log_path, 10)

    assert time.monotonic() - started < 3
