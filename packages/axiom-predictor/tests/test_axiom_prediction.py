from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
from typing import Callable, NoReturn, cast


import torch

from connections.clausification import matrix_from_file

import axiom_prediction.cli as cli_module
import axiom_prediction.training as training_module
from axiom_prediction.graph import build_axiom_graph, collate_axiom_graphs
from axiom_prediction.data import axiom_training_example_from_json, axiom_training_example_to_json, sat_core_clause_ids, training_example_from_sat_core
from axiom_prediction.metrics import prediction_metrics
from axiom_prediction.cli import _cli_properties, build_parser, main as axiom_predictor_main
from axiom_prediction.model import AxiomModelConfig, AxiomPredictionNetwork, AxiomPredictor, device_description
from axiom_prediction.tptp import NoProblemFilesError, expand_problem_inputs, find_tptp_root, problem_input_directory, resolve_tptp_problem


fail = cast(Callable[..., NoReturn], pytest.fail)


def _problem(path):
    matrix = matrix_from_file(path, mark_conjecture=True)
    conjectures = tuple(matrix.conjecture_clauses)
    axioms = tuple(index for index in range(len(matrix)) if index not in conjectures)
    return matrix, axioms, conjectures


def test_explicit_clause_ids_are_validated_and_drive_roles(tiny_problem_path):
    matrix, axioms, conjectures = _problem(tiny_problem_path)
    graph = build_axiom_graph(matrix, axiom_clause_ids=axioms, conjecture_clause_ids=conjectures)
    assert graph.graph.nodes["clause"][axioms[0]][1:] == [0, 1, 0]
    assert graph.graph.nodes["clause"][conjectures[0]][1:] == [1, 1, 1]

    for axiom_ids, conjecture_ids, match in (
        ((), conjectures, "axiom_clause_ids"),
        (axioms, (), "conjecture_clause_ids"),
        ((99,), conjectures, "out of range"),
        (axioms, axioms, "disjoint"),
        ((axioms[0], axioms[0]), conjectures, "duplicates"),
    ):
        with pytest.raises(ValueError, match=match):
            build_axiom_graph(
                matrix,
                axiom_clause_ids=axiom_ids,
                conjecture_clause_ids=conjecture_ids,
            )


def test_sat_core_labels_only_candidate_axioms(tiny_problem_path):
    matrix, axioms, conjectures = _problem(tiny_problem_path)
    core = tuple(sorted((axioms[0], conjectures[0])))
    example = training_example_from_sat_core(
        matrix,
        problem_path=str(tiny_problem_path),
        axiom_clause_ids=axioms,
        conjecture_clause_ids=conjectures,
        core_clause_ids=core,
    )
    assert example.labels[0] == 1
    assert len(example.labels) == len(axioms)


def test_axiom_training_example_json_is_self_contained(tiny_problem_path):
    matrix, axioms, conjectures = _problem(tiny_problem_path)
    example = training_example_from_sat_core(
        matrix,
        problem_path=str(tiny_problem_path),
        axiom_clause_ids=axioms,
        conjecture_clause_ids=conjectures,
        core_clause_ids=(axioms[0], conjectures[0]),
    )
    restored = axiom_training_example_from_json(axiom_training_example_to_json(example))
    assert restored.problem_path == example.problem_path
    assert restored.matrix is None
    assert restored.graph == example.graph
    assert restored.labels == example.labels
    assert restored.axiom_clause_texts == example.axiom_clause_texts


@pytest.mark.parametrize(
    "diagnostics,match",
    [
        ({}, "missing"),
        ({"sat_core_clause_ids": [True]}, "integers"),
        ({"sat_core_clause_ids": [1, 0]}, "sorted"),
        ({"sat_core_clause_ids": [99]}, "out-of-range"),
    ],
)
def test_invalid_sat_core_diagnostics_are_rejected(diagnostics, match):
    with pytest.raises(ValueError, match=match):
        sat_core_clause_ids(diagnostics, matrix_size=2)


def test_batched_axiom_logits_match_individual_graphs(tiny_problem_path):
    matrix, axioms, conjectures = _problem(tiny_problem_path)
    example = build_axiom_graph(matrix, axiom_clause_ids=axioms, conjecture_clause_ids=conjectures)
    torch.manual_seed(0)
    model = AxiomPredictionNetwork(
        AxiomModelConfig(hidden_dim=8, message_rounds=2, num_hidden_layers=1)
    )
    model.eval()
    with torch.no_grad():
        individual = model(collate_axiom_graphs([(example, None)]))
        batched = model(collate_axiom_graphs([(example, None), (example, None)]))
    assert torch.allclose(batched[: len(axioms)], individual, atol=1e-6)
    assert torch.allclose(batched[len(axioms) :], individual, atol=1e-6)


def test_metrics_cover_ranking_and_degenerate_classes():
    metrics = prediction_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], problem_sizes=[4])
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics["macro_recall_at_1"] == 0.5
    assert metrics["macro_recall_at_3"] == 1.0
    assert metrics["bce"] is not None
    assert math.isfinite(metrics["bce"])

    no_positives = prediction_metrics([0, 0], [0.2, 0.1])
    assert no_positives["roc_auc"] is None
    assert no_positives["average_precision"] is None


def test_tptp_root_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    environment = tmp_path / "environment"
    explicit.mkdir()
    environment.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TPTP", str(environment))
    assert find_tptp_root(explicit) == explicit.resolve()
    assert find_tptp_root() == environment.resolve()
    monkeypatch.delenv("TPTP")
    assert find_tptp_root() is None
    with pytest.raises(FileNotFoundError, match="not a directory"):
        find_tptp_root(tmp_path / "missing")


def test_problem_input_detection_modes(tmp_path, monkeypatch):
    category = tmp_path / "Problems" / "ABC"
    nested = category / "nested"
    nested.mkdir(parents=True)
    first = category / "ABC001+1.p"
    second = category / "ABC002-1.p"
    ignored = nested / "ABC003-1.p"
    for path in (first, second, ignored):
        path.write_text("fof(a,conjecture,a).\n", encoding="utf-8")
    monkeypatch.setenv("TPTP", str(tmp_path))

    assert expand_problem_inputs([str(first)]) == (str(first.resolve()),)
    assert expand_problem_inputs([str(category)]) == (
        str(first.resolve()),
        str(second.resolve()),
    )
    assert expand_problem_inputs([first.name]) == (str(first.resolve()),)
    assert expand_problem_inputs(["ABC"]) == (
        str(first.resolve()),
        str(second.resolve()),
    )
    assert problem_input_directory("ABC") == category.resolve()
    assert resolve_tptp_problem(first.name)[0] == first.resolve()


def test_filename_resolution_requires_tptp_root(tmp_path, monkeypatch):
    direct = tmp_path / "ABC001+1.p"
    direct.write_text("fof(a,conjecture,a).\n", encoding="utf-8")
    monkeypatch.delenv("TPTP", raising=False)
    monkeypatch.chdir(tmp_path)

    assert expand_problem_inputs([str(direct)]) == (str(direct.resolve()),)
    with pytest.raises(FileNotFoundError, match="without a TPTP root"):
        expand_problem_inputs(["ZZZ001+1.p"])


def test_empty_problem_directory_has_specific_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(NoProblemFilesError, match="no .p problem files"):
        expand_problem_inputs([str(empty)], tptp_root=tmp_path)


def test_missing_and_malformed_checkpoints_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        AxiomPredictor.load(tmp_path / "missing.pt")
    malformed = tmp_path / "model.pt"
    torch.save({"format": "something-else"}, malformed)
    with pytest.raises(ValueError, match="not an axiom predictor checkpoint"):
        AxiomPredictor.load(malformed)


def test_cli_prints_compute_device_before_checkpoint_error(tmp_path, tiny_problem_path, capsys):
    assert (
        axiom_predictor_main(["predict", str(tmp_path / "missing.pt"), str(tiny_problem_path)]) == 2
    )
    stderr = capsys.readouterr().err
    assert re.match(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} axiom-predictor: COMPUTE DEVICE: ", stderr
    )
    assert "checkpoint not found" in stderr
    assert device_description("cpu") == "CPU"


def test_split_parts_are_disjoint_complete_and_path_independent(tmp_path):
    from axiom_prediction.split import ProblemSplit

    names = [
        f"ABC{index:03d}{sign}{version}.p"
        for index in range(60)
        for sign, version in (("+", 1), ("+", 2), ("-", 1))
    ]
    parts = [ProblemSplit(4, (part,)) for part in range(4)]
    for name in names:
        owners = [part.parts[0] for part in parts if part.contains(name)]
        assert len(owners) == 1
        assert ProblemSplit(4, (owners[0],)).contains(f"/rds/TPTP/Problems/ABC/{name}")
        assert ProblemSplit(4, (owners[0],)).contains(f"Problems/ABC/{name}")
    assert all(
        parts[0].part(f"ABC{index:03d}+1.p") == parts[0].part(f"ABC{index:03d}-1.p")
        for index in range(60)
    )
    assert len({parts[0].part(name) for name in names}) == 4
    assert ProblemSplit(4, (0, 1, 2, 3)).select(names) == tuple(names)
    assert ProblemSplit().select(names) == tuple(names)


def test_split_sizes_partition_independently():
    from axiom_prediction.split import ProblemSplit

    names = [f"ABC{index:03d}+{version}.p" for index in range(1000) for version in range(1, 5)]
    subset = ProblemSplit(40, (0,), "problem").select(names)
    quarters = [len(ProblemSplit(4, (part,), "problem").select(subset)) for part in range(4)]
    assert all(count > len(subset) / 8 for count in quarters)
    assert ProblemSplit(4, (0,)).to_dict()["scheme"] == 2


def test_split_by_problem_and_domain():
    from axiom_prediction.split import ProblemSplit, split_key

    assert split_key("Problems/AGT/AGT001+2.p") == "AGT001"
    assert split_key("AGT001+2.p", by="problem") == "AGT001+2"
    assert split_key("AGT001+2.p", by="domain") == "AGT"
    assert (
        split_key("packages/axiom-predictor/tests/fixtures/problems/t11_wellord1.p")
        == "t11_wellord1"
    )
    domain = ProblemSplit(3, (0,), "domain")
    assert domain.contains("AGT001+1.p") == domain.contains("AGT999-7.p")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--split", "4", "--parts", "4"],
        ["--split", "4", "--parts", "1", "1"],
        ["--split", "0"],
    ],
)
def test_invalid_split_arguments_are_rejected(tmp_path, arguments):
    with pytest.raises((SystemExit, ValueError)):
        args = build_parser().parse_args(
            ["collect", "x.p", "--data-dir", str(tmp_path), *arguments]
        )
        cli_module._split(args)


def test_selected_problems_without_inputs_use_every_tptp_problem(tmp_path):
    problems = tmp_path / "Problems" / "ABC"
    problems.mkdir(parents=True)
    for index in range(20):
        (problems / f"ABC{index:03d}-1.p").write_text("", encoding="utf-8")
    (problems / "ignored.p").write_text("", encoding="utf-8")
    parser = build_parser()

    everything = cli_module._selected_problems(
        parser.parse_args(["collect", "--data-dir", str(tmp_path), "--tptp", str(tmp_path)])
    )
    assert len(everything) == 20
    halves = [
        cli_module._selected_problems(
            parser.parse_args(
                [
                    "collect",
                    "--data-dir",
                    str(tmp_path),
                    "--tptp",
                    str(tmp_path),
                    "--split",
                    "2",
                    "--part",
                    str(part),
                ]
            )
        )
        for part in (0, 1)
    ]
    assert sorted(halves[0] + halves[1]) == sorted(everything)
    assert not set(halves[0]) & set(halves[1])


def test_problem_directory_input_expands_immediate_p_files_in_order(tmp_path):
    problems = tmp_path / "Problems" / "SYN"
    nested = problems / "nested"
    nested.mkdir(parents=True)
    second = problems / "SYN002-1.p"
    first = problems / "SYN001+1.p"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    (problems / "README").write_text("", encoding="utf-8")
    (nested / "SYN003-1.p").write_text("", encoding="utf-8")

    assert expand_problem_inputs([str(problems)]) == (
        str(first.resolve()),
        str(second.resolve()),
    )


def test_problem_directory_input_can_be_relative_to_tptp_root(tmp_path):
    problems = tmp_path / "Problems" / "SYN"
    problems.mkdir(parents=True)
    problem = problems / "SYN001+1.p"
    problem.write_text("", encoding="utf-8")

    assert expand_problem_inputs(["Problems/SYN"], tptp_root=tmp_path) == (str(problem.resolve()),)


def test_training_cli_properties_include_every_parsed_argument(tmp_path):
    args = build_parser().parse_args(
        [
            "train",
            "--data-dir",
            str(tmp_path / "run"),
            "--tptp",
            str(tmp_path),
            "--split",
            "4",
            "--parts",
            "0",
            "2",
            "--seed",
            "17",
            "--batch-size",
            "321",
            "--num-workers",
            "7",
            "--device",
            "cpu",
            "--no-wandb",
            "--wandb-name",
            "sample-test",
        ]
    )
    properties = _cli_properties(args)
    assert set(properties) == set(vars(args))
    assert properties["split"] == 4
    assert properties["parts"] == [0, 2]
    assert properties["seed"] == 17
    assert properties["batch_size"] == 321
    assert properties["num_workers"] == 7
    assert properties["wandb"] is False
    assert properties["data_dir"] == str(tmp_path / "run")
    assert properties["sat_policy"] == "satresetcop"


def test_cli_sat_policy_choices_do_not_apply_to_predict(tmp_path):
    parser = build_parser()
    train = parser.parse_args(
        ["train", "x.p", "--data-dir", str(tmp_path), "--sat-policy", "satcop"]
    )
    evaluate = parser.parse_args(["evaluate", "model.pt", "x.p", "--sat-policy", "satcop"])
    predict = parser.parse_args(["predict", "model.pt", "x.p"])
    assert train.sat_policy == evaluate.sat_policy == "satcop"
    assert not hasattr(predict, "sat_policy")


def test_cli_supports_separate_collection_and_dataset_training(tmp_path):
    parser = build_parser()
    collect = parser.parse_args(
        [
            "collect",
            "--split",
            "10",
            "--part",
            "3",
            "--data-dir",
            str(tmp_path),
        ]
    )
    train = parser.parse_args(
        [
            "train",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert collect.command == "collect"
    assert collect.step_limit == 1_000_000
    assert collect.timeout_seconds == 120.0
    assert train.step_limit == 1_000_000
    assert train.timeout_seconds == 120.0
    assert train.data_dir == tmp_path
    assert collect.num_workers is None
    problems, dataset = cli_module._training_request(train)
    assert problems == ()
    assert dataset == tmp_path / "dataset"


def test_train_resolves_dataset_below_data_directory(tmp_path):
    args = build_parser().parse_args(["train", "--data-dir", str(tmp_path)])

    problems, resolved_dataset = cli_module._training_request(args)

    assert problems == ()
    assert resolved_dataset == tmp_path / "dataset"


def test_collect_cli_names_directory_shard_and_expands_problems(tmp_path, monkeypatch):
    category = tmp_path / "Problems" / "SYN"
    category.mkdir(parents=True)
    first = category / "SYN001+1.p"
    second = category / "SYN002-1.p"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    data_dir = tmp_path / "nested" / "run"
    output = data_dir / "dataset"
    captured = {}

    def fake_collect(problems, **kwargs):
        captured["problems"] = problems
        captured.update(kwargs)
        return output / "SYN.jsonl", {"problems_proved": 2}

    monkeypatch.setattr(cli_module, "collect_axiom_dataset_shard", fake_collect)
    monkeypatch.setattr(
        cli_module,
        "collect_axiom_dataset",
        lambda *args, **kwargs: fail(reason="directory collection was not sharded"),
    )

    assert (
        axiom_predictor_main(
            [
                "collect",
                str(category),
                "--data-dir",
                str(data_dir),
                "--num-workers",
                "1",
            ]
        )
        == 0
    )
    assert data_dir.is_dir()
    assert captured["problems"] == (str(first.resolve()), str(second.resolve()))
    assert captured["shard_name"] == "SYN"
    assert captured["output_dir"] == output


def test_collect_cli_returns_one_for_empty_directory(tmp_path):
    category = tmp_path / "Problems" / "ABC"
    category.mkdir(parents=True)
    assert (
        axiom_predictor_main(
            [
                "collect",
                str(category),
                "--tptp",
                str(tmp_path),
                "--data-dir",
                str(tmp_path / "run"),
            ]
        )
        == 1
    )


def test_collect_cli_returns_one_when_directory_is_all_unparseable(tmp_path):
    category = tmp_path / "Problems" / "ABC"
    category.mkdir(parents=True)
    (category / "ABC001^1.p").write_text("thf(a,axiom,a).\n", encoding="utf-8")
    assert (
        axiom_predictor_main(
            [
                "collect",
                str(category),
                "--tptp",
                str(tmp_path),
                "--data-dir",
                str(tmp_path / "run"),
                "--num-workers",
                "1",
            ]
        )
        == 1
    )


def test_evaluate_expands_tptp_category(tmp_path, monkeypatch):
    category = tmp_path / "Problems" / "ABC"
    category.mkdir(parents=True)
    first = category / "ABC001+1.p"
    second = category / "ABC002-1.p"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    captured = {}

    def fake_evaluate(checkpoint, problems, **kwargs):
        captured["checkpoint"] = checkpoint
        captured["problems"] = problems
        return {"ok": True}

    monkeypatch.setattr(training_module, "evaluate_axiom_predictor", fake_evaluate)
    assert (
        axiom_predictor_main(
            [
                "evaluate",
                "model.pt",
                "ABC",
                "--tptp",
                str(tmp_path),
                "--device",
                "cpu",
                "--no-wandb",
            ]
        )
        == 0
    )
    assert str(captured["checkpoint"]) == "model.pt"
    assert captured["problems"] == [str(first.resolve()), str(second.resolve())]


def test_predict_expands_category_and_skips_unparseable_file(
    tmp_path, tiny_problem_path, monkeypatch
):
    category = tmp_path / "Problems" / "ABC"
    category.mkdir(parents=True)
    valid = category / "ABC001+1.p"
    valid.symlink_to(tiny_problem_path.resolve())
    (category / "ABC002^1.p").write_text("thf(a,axiom,a).\n", encoding="utf-8")
    seen = []

    class FakePredictor:
        def predict(self, matrix, **kwargs):
            seen.append(len(matrix))
            return ()

    monkeypatch.setattr(
        AxiomPredictor,
        "load",
        lambda *args, **kwargs: FakePredictor(),
    )
    assert (
        axiom_predictor_main(
            [
                "predict",
                "model.pt",
                "ABC",
                "--tptp",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert len(seen) == 1


def test_model_name_selects_a_model_directory_for_every_command(tmp_path):
    parser = build_parser()
    data = tmp_path / "run"
    assert cli_module.model_directory(data, None) == data / "model"
    assert cli_module.model_directory(data, "split4-a") == data / "models" / "split4-a"
    with pytest.raises(ValueError):
        cli_module.model_directory(data, "../escape")

    evaluate = parser.parse_args(
        [
            "evaluate",
            "--data-dir",
            str(data),
            "--model-name",
            "split4-a",
            "--split",
            "4",
            "--parts",
            "3",
        ]
    )
    assert cli_module._checkpoint(evaluate) == data / "models" / "split4-a"
    explicit = parser.parse_args(["evaluate", "some/model.pt", "--data-dir", str(data)])
    assert cli_module._checkpoint(explicit) == Path("some/model.pt")
    with pytest.raises(ValueError):
        cli_module._checkpoint(
            parser.parse_args(
                [
                    "evaluate",
                    "some/model.pt",
                    "--data-dir",
                    str(data),
                    "--model-name",
                    "x",
                ]
            )
        )
    with pytest.raises(ValueError):
        cli_module._checkpoint(parser.parse_args(["evaluate", "--split", "4"]))
    assert (
        parser.parse_args(["train", "--data-dir", str(data), "--model-name", "split4-a"]).model_name
        == "split4-a"
    )


def test_predict_with_data_dir_treats_every_positional_as_a_problem(tmp_path, monkeypatch):
    seen = {}

    class FakePredictor:
        @staticmethod
        def load(checkpoint, device):
            seen["checkpoint"] = checkpoint
            raise FileNotFoundError("stop")

    import axiom_prediction.model as model_module

    monkeypatch.setattr(model_module, "AxiomPredictor", FakePredictor)
    assert (
        cli_module.main(["predict", "a.p", "b.p", "--data-dir", str(tmp_path), "--model-name", "m"])
        == 2
    )
    assert seen["checkpoint"] == tmp_path / "models" / "m"
