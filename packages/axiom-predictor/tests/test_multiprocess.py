from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace

import pytest
import torch

from axiom_prediction.model import AxiomModelConfig, AxiomPredictionNetwork, AxiomPredictor, save_checkpoint
from axiom_prediction.multiprocess import AdaptiveBatches, adaptive_predictions, inference_service, shared_predictor
from axiom_prediction.run import RunConfig, run_problem, run_problems
from axiom_prediction.tptp import load_tptp_problem

def test_adaptive_oom_retries_preserve_order(monkeypatch):
    import axiom_prediction.multiprocess as module

    attempts = []

    def predict(model, graphs):
        attempts.append(len(graphs))
        if len(graphs) > 10:
            raise torch.cuda.OutOfMemoryError("simulated capacity")

        return [[graph] for graph in graphs]

    monkeypatch.setattr(module, "predict_graphs", predict)
    actual = list(adaptive_predictions(None, list(range(80)), AdaptiveBatches(32)))
    assert actual == [[i] for i in range(80)]
    assert any(size > 10 for size in attempts)

def test_single_graph_oom_is_reported(monkeypatch):
    import axiom_prediction.multiprocess as module

    def fail(*args):
        raise torch.cuda.OutOfMemoryError("too large")

    monkeypatch.setattr(module, "predict_graphs", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        list(adaptive_predictions(None, [1], AdaptiveBatches(8)))

def test_large_singleton_does_not_discard_other_results(monkeypatch):
    import axiom_prediction.multiprocess as module

    def predict(model, graphs):
        if 3 in graphs:
            raise torch.cuda.OutOfMemoryError("large graph")

        return [[graph] for graph in graphs]

    monkeypatch.setattr(module, "predict_graphs", predict)
    results = list(
        adaptive_predictions(
            None,
            list(range(8)),
            AdaptiveBatches(8),
            tolerate_singleton_oom = True,
        )
    )

    assert len(results) == 8
    assert isinstance(results[3], RuntimeError)
    assert [value for value in results if not isinstance(value, Exception)] == [
        [i] for i in range(8) if i != 3
    ]

@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks = pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason = "CUDA unavailable",
            ),
        ),
    ],
)
def test_shared_inference_and_parallel_search(tmp_path, tiny_problem_path, device):
    torch.manual_seed(7)
    model = AxiomPredictionNetwork(AxiomModelConfig(hidden_dim = 8, message_rounds = 1))
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, training_config = {"sat_policy": "satresetcop"})
    loaded = load_tptp_problem(str(tiny_problem_path))
    kwargs = dict(
        axiom_clause_ids = loaded.axiom_clause_ids,
        conjecture_clause_ids = loaded.conjecture_clause_ids,
    )

    expected = AxiomPredictor.load(checkpoint, device = device).predict(
        loaded.matrix,
        **kwargs,
    )

    with inference_service(str(checkpoint), device, 4) as (address, key):
        predictor = shared_predictor(address, key)
        with ThreadPoolExecutor(max_workers = 4) as pool:
            results = list(
                pool.map(lambda _: predictor.predict(loaded.matrix, **kwargs), range(16))
            )

        for result in results:
            assert [p.clause_index for p in result] == [p.clause_index for p in expected]
            assert [p.probability for p in result] == pytest.approx(
                [p.probability for p in expected],
                abs = 1e-6,
            )

    paths = []
    for i in range(4):
        path = tmp_path / f"theorem{i}.p"
        path.write_text(tiny_problem_path.read_text())
        paths.append(str(path))

    config = RunConfig(checkpoint = str(checkpoint), device = device, timeout_seconds = 40)
    expected_result = run_problem(paths[0], tptp_root = None, config = config)
    results = list(
        run_problems(
            paths,
            tptp_root = None,
            config = replace(config, multiprocess = True),
            num_workers = 2,
        )
    )

    assert len(results) == len(paths)
    assert {r["problem"] for r in results} == set(paths)
    assert all(r["proved"] == expected_result["proved"] for r in results)
    assert all(r["outcome"] == expected_result["outcome"] for r in results)

def test_run_config_reads_old_worker_state_without_shifting_fields():
    import pickle

    old = ["base", "satresetcop", None, "cpu", 0.7, None, 42, 500, 12.5]
    config = object.__new__(RunConfig)
    config.__setstate__(old)

    assert config.timeout_seconds == 12.5
    assert config.seed == 42
    assert config.temperature == 0.7
    assert config.multiprocess is False
    assert config.inference_address is None

    assert pickle.loads(pickle.dumps(config)) == config
    current_positional = [
        getattr(config, item.name)
        for item in fields(config)
    ]

    restored = object.__new__(RunConfig)
    restored.__setstate__(current_positional)
    assert restored == config
