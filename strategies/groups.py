"""Strategy-group expansion helpers for CLI entrypoints."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import strategies.sat_solve as sat_solve_module
from strategies.base import Strategy


STRATEGY_GROUPS: dict[str, tuple[str, ...]] = {
    'SATControl': ('FirstAction', 'LeanCopCut'),
}


def _sat_test_strategies() -> tuple[str, ...]:
    return tuple(
        cls.__name__
        for cls in sat_solve_module.__dict__.values()
        if (
            isinstance(cls, type)
            and cls.__module__ == sat_solve_module.__name__
            and issubclass(cls, Strategy)
            and not cls.__name__.startswith('_')
        )
    )


def strategy_groups() -> dict[str, tuple[str, ...]]:
    return STRATEGY_GROUPS | {'SATTest': _sat_test_strategies()}


def strategy_group_names() -> list[str]:
    return list(strategy_groups())


def strategy_choices(available_strategies: Sequence[str] | None = None) -> list[str]:
    if available_strategies is None:
        available_strategies = list(Strategy.strategies().keys())
    return list(available_strategies) + strategy_group_names()


def expand_strategy_groups(names: Iterable[str]) -> list[str]:
    groups = strategy_groups()
    available = Strategy.strategies()
    expanded: list[str] = []

    for name in names:
        if name in groups:
            expanded.extend(groups[name])
        elif name in available:
            expanded.append(name)
        else:
            known = ', '.join(strategy_choices(list(available)))
            raise ValueError(f'unknown strategy or group {name!r}; choose from: {known}')

    return list(dict.fromkeys(expanded))
