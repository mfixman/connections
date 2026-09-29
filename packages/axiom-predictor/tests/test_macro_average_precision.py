import pytest

from axiom_prediction.metrics import prediction_metrics

def test_macro_average_precision_weights_problems_equally():
    metrics = prediction_metrics(
        [1, 0, 0, 0, 1, 0, 0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        problem_sizes = [1, 4, 2],
    )

    assert metrics["macro_average_precision"] == pytest.approx((1 + 1 / 4) / 2)
    assert metrics["average_precision"] == pytest.approx((1 + 2 / 5) / 2)

def test_macro_average_precision_undefined_and_tied_scores():
    empty_core = prediction_metrics([0, 0], [0.2, 0.1], problem_sizes = [1, 1])
    assert empty_core["macro_average_precision"] is None
    tied = prediction_metrics([1, 0, 1], [0.5, 0.5, 0.5], problem_sizes = [2, 1])
    assert tied["macro_average_precision"] == pytest.approx(0.75)
