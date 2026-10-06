from __future__ import annotations

import math
from itertools import groupby

def prediction_metrics(
    labels: list[int],
    probabilities: list[float],
    *,
    problem_sizes: list[int] | None = None,
) -> dict[str, float | int | None]:
    if len(labels) != len(probabilities):
        raise ValueError("label and prediction counts differ")

    eps = 1e-7
    clipped = [min(1.0 - eps, max(eps, float(value))) for value in probabilities]
    bce = -sum(
        label * math.log(probability) + (1 - label) * math.log(1 - probability)
        for label, probability in zip(labels, clipped, strict = True)
    ) / len(labels) if labels else None

    result: dict[str, float | int | None] = {
        "bce": bce,

        "roc_auc": roc_auc(labels, probabilities),
        "average_precision": average_precision(labels, probabilities),
    }

    if problem_sizes is not None:
        if sum(problem_sizes) != len(labels) or any(size < 0 for size in problem_sizes):
            raise ValueError("problem_sizes must be nonnegative and sum to the label count")

        offset = 0
        per_problem: list[tuple[list[int], list[float]]] = []
        for size in problem_sizes:
            per_problem.append(
                (
                    labels[offset : offset + size],
                    probabilities[offset : offset + size],
                )
            )

            offset += size

        result["macro_average_precision"] = mean_defined(
            average_precision(y, p) for y, p in per_problem
        )

        for k in (1, 3, 5, 10):
            result[f"macro_recall_at_{k}"] = mean_defined(
                recall_at_k(y, p, k) for y, p in per_problem
            )

    return result

def mean_defined(values):
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None

def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict = True) if label]
    negatives = [
        score
        for label, score in zip(labels, scores, strict = True)
        if not label
    ]

    if not positives or not negatives:
        return None

    ranked = sorted(scores)
    first: dict[float, int] = {}
    last: dict[float, int] = {}
    for rank, score in enumerate(ranked, start = 1):
        first.setdefault(score, rank)
        last[score] = rank

    positive_ranks = sum((first[score] + last[score]) / 2 for score in positives)
    return (positive_ranks - len(positives) * (len(positives) + 1) / 2) / (
        len(positives) * len(negatives)
    )

def average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None

    ranked = sorted(
        zip(scores, labels, strict = True),
        key = lambda item: item[0],
        reverse = True,
    )

    hits = 0
    count = 0
    total = 0.0
    for _, group in groupby(ranked, key = lambda item: item[0]):
        labels_at_score = [label for _, label in group]
        positives_at_score = sum(labels_at_score)
        hits += positives_at_score
        count += len(labels_at_score)
        total += positives_at_score * hits / count

    return total / positives

def recall_at_k(labels: list[int], scores: list[float], k: int) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None

    indices = sorted(range(len(labels)), key = lambda index: (-scores[index], index))[:k]
    return sum(labels[index] for index in indices) / positives
