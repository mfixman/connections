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
