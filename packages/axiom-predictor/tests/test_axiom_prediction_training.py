"""Training-level checks for the axiom predictor."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest
from typing import Callable, NoReturn, cast

from connections.parsing.tptp import TPTPParseError

import torch

import axiom_prediction.training as training_module
import axiom_prediction.dataset as dataset_module
from axiom_prediction.data import axiom_training_example_to_json
from axiom_prediction.dataset import NoParseableProblemsError, collect_axiom_dataset, collect_axiom_dataset_shard, load_axiom_dataset
from axiom_prediction.model import AxiomPredictor
from axiom_prediction.tptp import collect_proof_example, load_tptp_problem
from axiom_prediction.training import AxiomTrainingConfig, collect_examples, train_axiom_predictor
from axiom_prediction.wandb_tracking import WandbConfig
from axiom_prediction.io import write_json_atomic

fail = cast(Callable[..., NoReturn], pytest.fail)
skip = cast(Callable[..., NoReturn], pytest.skip)

pytestmark = pytest.mark.training


def test_fixture_proof_labels_training_and_checkpoint_round_trip(tmp_path):
    problem = "packages/axiom-predictor/tests/fixtures/problems/t11_wellord1.p"
    example, outcome = collect_proof_example(
        problem, step_limit=2_000, timeout_seconds=30, sat_policy="satcop"
    )
    assert outcome == "proved"
    assert example is not None
    assert 0 < sum(example.labels) < len(example.labels)

    output = tmp_path / "checkpoint"
    metrics = train_axiom_predictor(
        [problem],
        output_dir=output,
        config=AxiomTrainingConfig(
            epochs=120,
            hidden_dim=32,
            message_rounds=2,
            num_hidden_layers=1,
            step_limit=2_000,
            timeout_seconds=30,
            device="cpu",
            sat_policy="satcop",
        ),
        wandb_config=WandbConfig(enabled=False),
    )
    assert metrics["roc_auc"] >= 0.9
    assert "overfit" in metrics["evaluation_kind"]
    assert (output / "model.pt").is_file()
    assert json.loads((output / "metrics.json").read_text())["problems_proved"] == 1

    loaded = load_tptp_problem(problem)
    first = AxiomPredictor.load(output, device="cpu").predict(
        loaded.matrix,
        axiom_clause_ids=loaded.axiom_clause_ids,
        conjecture_clause_ids=loaded.conjecture_clause_ids,
    )
    second = AxiomPredictor.load(output / "model.pt", device="cpu").predict(
        loaded.matrix,
        axiom_clause_ids=loaded.axiom_clause_ids,
        conjecture_clause_ids=loaded.conjecture_clause_ids,
    )
    assert first == second
    assert [prediction.rank for prediction in first] == list(range(1, len(first) + 1))
    assert all(0.0 <= prediction.probability <= 1.0 for prediction in first)


def test_cuda_inference_smoke_when_available(tmp_path, tiny_problem_path):
    if not torch.cuda.is_available():
        skip(reason="CUDA is unavailable")
    train_axiom_predictor(
        [str(tiny_problem_path)],
        output_dir=tmp_path,
        config=AxiomTrainingConfig(epochs=1, hidden_dim=8, message_rounds=1, device="cpu"),
        wandb_config=WandbConfig(enabled=False),
    )
    loaded = load_tptp_problem(tiny_problem_path)
    predictions = AxiomPredictor.load(tmp_path, device="cuda").predict(
        loaded.matrix,
        axiom_clause_ids=loaded.axiom_clause_ids,
        conjecture_clause_ids=loaded.conjecture_clause_ids,
    )
    assert predictions


def test_failed_proof_is_reported_and_skipped(tiny_problem_path):
    example, outcome = collect_proof_example(tiny_problem_path, step_limit=0, timeout_seconds=30)
    assert example is None
    assert outcome == "ResourceOut"


@pytest.mark.parametrize(
    ("declared_status", "expected_outcome"),
    [
        ("Satisfiable", "DeclaredSatisfiable"),
        ("CounterSatisfiable", "DeclaredCounterSatisfiable"),
    ],
)
def test_declared_non_refutable_problem_skips_proof_search(
    tmp_path, monkeypatch, declared_status, expected_outcome
):
    problem = tmp_path / "declared-sat.p"
    problem.write_text(
        f"% Status : {declared_status}\nfof(a,axiom,p).\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "axiom_prediction.tptp.run_schedule",
        lambda *args, **kwargs: fail(reason="declared satisfiable problem was searched"),
    )

    example, outcome = collect_proof_example(problem)

    assert example is None
    assert outcome == expected_outcome


def test_declared_satisfiable_typed_problem_remains_unparseable(tmp_path):
    problem = tmp_path / "typed.p"
    problem.write_text(
        "% Status : Satisfiable\ntff(a,axiom,p).\n",
        encoding="utf-8",
    )

    with pytest.raises(TPTPParseError, match="Unsupported annotated formula 'tff'"):
        collect_proof_example(problem)


def test_declared_satisfiable_problem_is_a_parseable_dataset_failure(tmp_path):
    problem = tmp_path / "declared-sat.p"
    problem.write_text(
        "% Status : Satisfiable\ncnf(a,axiom,p).\n",
        encoding="utf-8",
    )

    summary = collect_axiom_dataset(
        [str(problem)],
        output_dir=tmp_path / "dataset",
        num_workers=1,
    )

    assert summary["problems_failed"] == 1
    assert summary["problems_unparseable"] == 0


def test_training_updates_once_per_minibatch(tmp_path, tiny_problem_path, monkeypatch):
    example, outcome = collect_proof_example(
        tiny_problem_path, step_limit=2_000, timeout_seconds=30
    )
    assert outcome == "proved"
    assert example is not None
    monkeypatch.setattr(
        training_module,
        "collect_examples",
        lambda *args, **kwargs: ([example, example, example], []),
    )
    steps = 0
    original_step = torch.optim.Adam.step

    def counted_step(optimizer, *args, **kwargs):
        nonlocal steps
        steps += 1
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    train_axiom_predictor(
        ["unused-by-mock"],
        output_dir=tmp_path,
        config=AxiomTrainingConfig(
            epochs=2,
            batch_size=2,
            hidden_dim=8,
            message_rounds=1,
            num_hidden_layers=1,
            device="cpu",
        ),
        wandb_config=WandbConfig(enabled=False),
    )
    assert steps == 4


def test_unsupported_tptp_dialect_is_reported_and_skipped(tmp_path, tiny_problem_path):
    thf_problem = tmp_path / "ABC001^1.p"
    thf_problem.write_text(
        "thf(a_type,type,a:$o).\nthf(a,axiom,a).\n",
        encoding="utf-8",
    )
    examples, skipped = collect_examples(
        [str(thf_problem), str(tiny_problem_path)],
        tptp_root=None,
        config=AxiomTrainingConfig(step_limit=2_000, timeout_seconds=30),
    )
    assert len(examples) == 1
    assert len(skipped) == 1
    assert skipped[0]["problem"] == str(thf_problem)
    assert "E_NON_FOF_TOPLEVEL" in skipped[0]["outcome"]


def test_directory_collection_skips_unparseable_files_but_publishes_shard(
    tmp_path, tiny_problem_path
):
    thf_problem = tmp_path / "ABC001^1.p"
    thf_problem.write_text(
        "thf(a_type,type,a:$o).\nthf(a,axiom,a).\n",
        encoding="utf-8",
    )
    shard, summary = collect_axiom_dataset_shard(
        [str(thf_problem), str(tiny_problem_path)],
        output_dir=tmp_path / "dataset",
        shard_name="ABC",
        tptp_root=tmp_path,
        step_limit=2_000,
        timeout_seconds=30,
        num_workers=1,
    )

    assert shard.is_file()
    assert summary["problems_requested"] == 2
    assert summary["problems_unparseable"] == 1
    assert summary["problems_proved"] == 1


def test_all_unparseable_directory_does_not_publish_shard(tmp_path):
    first = tmp_path / "ABC001^1.p"
    second = tmp_path / "ABC002^1.p"
    for problem in (first, second):
        problem.write_text("thf(a,axiom,a).\n", encoding="utf-8")

    with pytest.raises(NoParseableProblemsError, match="could be parsed"):
        collect_axiom_dataset_shard(
            [str(first), str(second)],
            output_dir=tmp_path / "dataset",
            shard_name="ABC",
            tptp_root=tmp_path,
            num_workers=1,
        )

    assert not (tmp_path / "dataset" / "ABC.jsonl").exists()


def test_training_collection_reports_all_unparseable_inputs(tmp_path):
    problem = tmp_path / "ABC001^1.p"
    problem.write_text("thf(a,axiom,a).\n", encoding="utf-8")
    with pytest.raises(NoParseableProblemsError, match="parsed as FOF or CNF"):
        collect_examples(
            [str(problem)],
            tptp_root=tmp_path,
            config=AxiomTrainingConfig(num_workers=1),
        )


def test_collection_fails_when_no_problem_has_a_usable_sat_core(monkeypatch):
    monkeypatch.setattr(
        dataset_module,
        "collect_proof_example",
        lambda *args, **kwargs: (None, "invalid SAT core: missing provenance"),
    )
    with pytest.raises(RuntimeError, match="usable SAT core"):
        collect_examples(
            ["bad.p"],
            tptp_root=None,
            config=AxiomTrainingConfig(),
        )


def test_dataset_collection_resumes_and_trains_without_reproving(
    tmp_path, tiny_problem_path, monkeypatch
):
    example, outcome = collect_proof_example(
        tiny_problem_path, step_limit=2_000, timeout_seconds=30
    )
    assert outcome == "proved"
    assert example is not None
    calls = 0

    def fake_collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        return example, "proved"

    monkeypatch.setattr(dataset_module, "collect_proof_example", fake_collect)
    dataset = tmp_path / "dataset"
    problem = str(tiny_problem_path)
    first = collect_axiom_dataset(
        [problem],
        output_dir=dataset,
        step_limit=2_000,
        timeout_seconds=30,
    )
    second = collect_axiom_dataset(
        [problem],
        output_dir=dataset,
        step_limit=2_000,
        timeout_seconds=30,
    )
    assert calls == 1
    assert first["problems_reused"] == 0
    assert second["problems_reused"] == 1
    assert (dataset / "partial").is_dir()
    assert not list((dataset / "partial").rglob("*.json"))

    examples, failures, metadata = load_axiom_dataset(dataset)
    assert len(examples) == 1
    assert examples[0].matrix is None
    assert not failures
    assert metadata["collection"]["sat_policy"] == "satresetcop"

    monkeypatch.setattr(
        training_module,
        "collect_examples",
        lambda *args, **kwargs: fail(reason="dataset training repeated proof search"),
    )
    output = tmp_path / "model"
    metrics = train_axiom_predictor(
        dataset=dataset,
        output_dir=output,
        config=AxiomTrainingConfig(
            epochs=1,
            hidden_dim=8,
            message_rounds=1,
            num_hidden_layers=1,
            device="cpu",
        ),
        wandb_config=WandbConfig(enabled=False),
    )
    assert metrics["dataset"] == str(dataset)
    assert metrics["problems_proved"] == 1
    assert (output / "model.pt").is_file()


def test_dataset_recovers_worker_result_left_in_partial(tmp_path, tiny_problem_path, monkeypatch):
    example, outcome = collect_proof_example(
        tiny_problem_path, step_limit=2_000, timeout_seconds=30
    )
    assert outcome == "proved"
    assert example is not None

    dataset = tmp_path / "dataset"
    partial_examples = dataset / "partial" / "examples"
    partial_examples.mkdir(parents=True)
    problem = str(tiny_problem_path)
    key = dataset_module._problem_key(problem)
    write_json_atomic(
        partial_examples / f"{key}.json",
        axiom_training_example_to_json(example),
    )
    monkeypatch.setattr(
        dataset_module,
        "collect_proof_example",
        lambda *args, **kwargs: fail(reason="partial result was recomputed"),
    )

    summary = collect_axiom_dataset(
        [problem],
        output_dir=dataset,
        step_limit=2_000,
        timeout_seconds=30,
    )

    assert summary["problems_proved"] == 1
    assert summary["problems_reused"] == 1
    assert (dataset / "examples" / f"{key}.json").is_file()
    assert not (partial_examples / f"{key}.json").exists()


def test_directory_shards_combine_and_are_directly_trainable(
    tmp_path, tiny_problem_path, monkeypatch
):
    example, outcome = collect_proof_example(
        tiny_problem_path, step_limit=2_000, timeout_seconds=30
    )
    assert outcome == "proved"
    assert example is not None
    monkeypatch.setattr(
        dataset_module,
        "collect_proof_example",
        lambda problem, **kwargs: (
            replace(example, problem_path=str(problem)),
            "proved",
        ),
    )
    dataset = tmp_path / "dataset"
    syn_problem = str(tmp_path / "SYN001+1.p")
    puz_problem = str(tmp_path / "PUZ001-1.p")
    shard, summary = collect_axiom_dataset_shard(
        [syn_problem],
        output_dir=dataset,
        shard_name="SYN",
        step_limit=2_000,
        timeout_seconds=30,
    )
    second_shard, _ = collect_axiom_dataset_shard(
        [puz_problem],
        output_dir=dataset,
        shard_name="PUZ",
        step_limit=2_000,
        timeout_seconds=30,
    )

    assert shard == dataset / "SYN.jsonl"
    assert shard.is_file()
    assert second_shard == dataset / "PUZ.jsonl"
    assert summary["problems_proved"] == 1
    examples, failures, metadata = load_axiom_dataset(dataset)
    assert {item.problem_path for item in examples} == {syn_problem, puz_problem}
    assert not failures
    assert metadata["collection"]["sat_policy"] == "satresetcop"

    examples, failures, _ = load_axiom_dataset(shard)
    assert [item.problem_path for item in examples] == [syn_problem]
    assert not failures


def test_real_multiprocess_collection():
    problems = [
        "packages/axiom-predictor/tests/fixtures/problems/tiny_theorem.p",
        "packages/axiom-predictor/tests/fixtures/problems/t11_wellord1.p",
    ]
    results = list(
        dataset_module.collect_problems_parallel(
            problems,
            tptp_root=None,
            step_limit=2_000,
            timeout_seconds=30,
            sat_policy="satcop",
            num_workers=2,
        )
    )
    assert {result.problem for result in results} == set(problems)
    assert all(result.example is not None for result in results)
    assert all(result.outcome == "proved" for result in results)


@pytest.mark.parametrize("sat_policy", ["satresetcop", "satcop"])
def test_sat_core_labels_use_the_proved_clausification(sat_policy):
    from connections.clausification import matrix_from_file

    problem = "packages/axiom-predictor/tests/fixtures/problems/marked_conjecture_clausification.p"
    unmarked = matrix_from_file(problem, mark_conjecture=False)
    marked = matrix_from_file(problem, mark_conjecture=True)
    assert [str(clause) for clause in unmarked.clauses] != [
        str(clause) for clause in marked.clauses
    ]

    example, outcome = collect_proof_example(problem, timeout_seconds=30, sat_policy=sat_policy)

    assert outcome == "proved"
    assert example is not None
    positives = {
        text
        for text, label in zip(example.axiom_clause_texts, example.labels, strict=True)
        if label
    }
    assert "[-p(a)]" in positives
    assert "[p(V1),-q(V1)]" in positives or "[p(V1),-r(V1)]" in positives


def test_sat_core_clause_check_rejects_a_different_matrix():
    from connections.syntax.formula import Atom
    from connections.syntax.matrix import Clause, Literal, Matrix
    from axiom_prediction.tptp import check_sat_core_clauses

    matrix = Matrix((Clause((Literal(Atom("p")),)), Clause((Literal(Atom("q")),))))
    check_sat_core_clauses(matrix, (1,), ["[q]"])
    with pytest.raises(RuntimeError, match="labelled matrix"):
        check_sat_core_clauses(matrix, (1,), ["[p]"])
    with pytest.raises(RuntimeError, match="missing"):
        check_sat_core_clauses(matrix, (1,), None)


def test_outdated_dataset_schema_asks_for_recollection(tmp_path):
    from axiom_prediction.dataset import collect_axiom_dataset

    (tmp_path / "metadata.json").write_text(
        json.dumps({"schema": "learncop.axiom_prediction.dataset.v1", "collection": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="re-collect"):
        collect_axiom_dataset([], output_dir=tmp_path)


def test_dataset_training_uses_only_the_requested_split(tmp_path):
    from axiom_prediction.split import ProblemSplit

    problems = [
        "packages/axiom-predictor/tests/fixtures/problems/tiny_theorem.p",
        "packages/axiom-predictor/tests/fixtures/problems/marked_conjecture_clausification.p",
        "packages/axiom-predictor/tests/fixtures/problems/t11_wellord1.p",
    ]
    dataset = tmp_path / "dataset"
    collect_axiom_dataset(problems, output_dir=dataset, timeout_seconds=30, num_workers=1)
    examples, _, _ = load_axiom_dataset(dataset)
    split = next(
        ProblemSplit(4, (part,))
        for part in range(4)
        if 0
        < len(ProblemSplit(4, (part,)).select([example.problem_path for example in examples]))
        < len(examples)
    )
    expected = split.select([example.problem_path for example in examples])

    metrics = train_axiom_predictor(
        dataset=dataset,
        output_dir=tmp_path / "model",
        config=AxiomTrainingConfig(
            epochs=1, hidden_dim=8, message_rounds=1, num_hidden_layers=1, device="cpu"
        ),
        wandb_config=WandbConfig(enabled=False),
        split=split,
    )

    assert metrics["problems_proved"] == len(expected)


def test_evaluate_scores_a_dataset_split_without_proof_search(tmp_path, monkeypatch, capsys):
    from axiom_prediction.split import ProblemSplit
    from axiom_prediction.training import evaluate_axiom_predictor

    problems = [
        "packages/axiom-predictor/tests/fixtures/problems/tiny_theorem.p",
        "packages/axiom-predictor/tests/fixtures/problems/marked_conjecture_clausification.p",
        "packages/axiom-predictor/tests/fixtures/problems/t11_wellord1.p",
    ]
    dataset = tmp_path / "dataset"
    collect_axiom_dataset(problems, output_dir=dataset, timeout_seconds=30, num_workers=1)
    examples, _, _ = load_axiom_dataset(dataset)
    names = [example.problem_path for example in examples]
    train_part = next(
        part for part in range(4) if 0 < len(ProblemSplit(4, (part,)).select(names)) < len(names)
    )
    held_out = ProblemSplit(4, tuple(part for part in range(4) if part != train_part))
    train_axiom_predictor(
        dataset=dataset,
        output_dir=tmp_path / "model",
        config=AxiomTrainingConfig(
            epochs=1, hidden_dim=8, message_rounds=1, num_hidden_layers=1, device="cpu"
        ),
        wandb_config=WandbConfig(enabled=False),
        split=ProblemSplit(4, (train_part,)),
    )
    monkeypatch.setattr(
        training_module,
        "collect_examples",
        lambda *args, **kwargs: fail(reason="dataset evaluation repeated proof search"),
    )
    capsys.readouterr()

    metrics = evaluate_axiom_predictor(
        tmp_path / "model",
        dataset=dataset,
        device="cpu",
        split=held_out,
        wandb_config=WandbConfig(enabled=False),
    )

    assert metrics["problems_proved"] == len(held_out.select(names))
    assert metrics["split"] == held_out.to_dict()
    assert metrics["training_split"] == ProblemSplit(4, (train_part,)).to_dict()
    assert "not held-out" not in capsys.readouterr().err

    evaluate_axiom_predictor(
        tmp_path / "model",
        dataset=dataset,
        device="cpu",
        split=ProblemSplit(4, (train_part,)),
        wandb_config=WandbConfig(enabled=False),
    )
    assert "not held-out" in capsys.readouterr().err


def test_evaluate_rejects_problems_together_with_a_dataset(tmp_path):
    from axiom_prediction.cli import main

    assert (
        main(
            [
                "evaluate",
                str(tmp_path / "model"),
                "x.p",
                "--data-dir",
                str(tmp_path),
                "--no-wandb",
            ]
        )
        == 2
    )


def test_evaluate_warns_about_checkpoints_split_with_an_older_hash(capsys):
    from axiom_prediction.split import ProblemSplit

    training_module._warn_on_training_overlap(
        {"split": {"split": 4, "parts": [0, 1, 2], "by": "family"}},
        ProblemSplit(4, (3,)),
    )
    assert "older split hash" in capsys.readouterr().err
    training_module._warn_on_training_overlap(
        {"split": ProblemSplit(4, (0, 1, 2)).to_dict()}, ProblemSplit(4, (3,))
    )
    assert "warning" not in capsys.readouterr().err
