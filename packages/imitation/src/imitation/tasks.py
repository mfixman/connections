"""Assign corpus problems to episodes and split training from holdout data."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from connections.interaction.run import Problem


@dataclass(frozen=True, slots=True)
class EpisodeTask:
    """One problem attempt, with its budgets, round, and trajectory index."""

    problem: Problem
    round_index: int
    step_limit: int | None = None
    timeout_seconds: float | None = None
    trajectory_index: int = 0


def holdout_split(
    problems: Sequence[Problem],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[tuple[Problem, ...], tuple[Problem, ...]]:
    """Deterministic train/holdout partition of a corpus."""

    if not 0.0 <= holdout_fraction <= 1.0:
        raise ValueError("holdout_fraction must be within [0, 1]")
    shuffled = list(problems)
    random.Random(seed).shuffle(shuffled)
    held = round(len(shuffled) * holdout_fraction)
    return tuple(shuffled[held:]), tuple(shuffled[:held])


__all__ = ["EpisodeTask", "holdout_split"]
