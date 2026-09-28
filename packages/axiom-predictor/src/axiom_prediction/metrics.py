from __future__ import annotations

import math

def prediction_metrics(
    labels: list[int],
    probabilities: list[float],
    *,
    problem_sizes: list[int] | None = None,
) -> dict[str, float | int | None]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must have equal, nonzero length")

    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be binary")

    eps = 1e-7
    clipped = [min(1.0 - eps, max(eps, float(value))) for value in probabilities]
    bce = -sum(
        label * math.log(probability) + (1 - label) * math.log(1 - probability)
        for label, probability in zip(labels, clipped, strict = True)
    ) / len(labels)

    predicted = [int(value >= 0.5) for value in probabilities]
    tp = sum(p == 1 and y == 1 for p, y in zip(predicted, labels, strict = True))
    fp = sum(p == 1 and y == 0 for p, y in zip(predicted, labels, strict = True))
    fn = sum(p == 0 and y == 1 for p, y in zip(predicted, labels, strict = True))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    result: dict[str, float | int | None] = {
        "examples": len(labels),
        "positives": sum(labels),
        "bce": bce,

        "roc_auc": roc_auc(labels, probabilities),
        "average_precision": average_precision(labels, probabilities),

        "precision_at_0.5": precision,
        "recall_at_0.5": recall,
        "f1_at_0.5": f1,
    }

    if problem_sizes is not None:
        if sum(problem_sizes) != len(labels) or any(size <= 0 for size in problem_sizes):
            raise ValueError("problem_sizes must be positive and sum to the label count")

        offset = 0
        per_problem: list[tuple[list[int], list[float]]] = []
        for size in problem_sizes:
            per_problem.append(
                (
                    list(labels[offset : offset + size]),
                    list(probabilities[offset : offset + size]),
                )
            )

            offset += size

        for k in (1, 3, 5, 10):
            recalls = [recall_at_k(y, p, k) for y, p in per_problem]
            defined = [value for value in recalls if value is not None]
            result[f"macro_recall_at_{k}"] = sum(
                defined
            ) / len(defined) if defined else None

    return result

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
    total = 0.0
    for rank, (_score, label) in enumerate(ranked, start = 1):
        if label:
            hits += 1
            total += hits / rank

    return total / positives

def recall_at_k(labels: list[int], scores: list[float], k: int) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None

    indices = sorted(range(len(labels)), key = lambda index: (-scores[index], index))[:k]
    return sum(labels[index] for index in indices) / positives

__all__ = ["prediction_metrics"]
