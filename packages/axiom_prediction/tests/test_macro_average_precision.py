import pytest

from axiom_prediction.metrics import prediction_metrics

def test_empty_axiom_sets_have_no_metric_contribution():
    empty = prediction_metrics([], [], problem_sizes = [0, 0])
    assert all(value is None for value in empty.values())

    expected = prediction_metrics([1, 0], [0.9, 0.1], problem_sizes = [2])
    mixed = prediction_metrics([1, 0], [0.9, 0.1], problem_sizes = [0, 2, 0])
    assert mixed == expected

    with pytest.raises(ValueError, match = "counts differ"):
        prediction_metrics([], [0.5], problem_sizes = [0])

def test_macro_average_precision_weights_problems_equally():
    metrics = prediction_metrics(
        [1, 0, 0, 0, 1, 0, 0],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        problem_sizes = [1, 4, 2],
    )

    assert metrics["macro_average_precision"] == pytest.approx((1 + 1 / 4) / 2)
    assert metrics["average_precision"] == pytest.approx((1 + 2 / 5) / 2)
