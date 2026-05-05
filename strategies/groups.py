"""Strategy-group expansion helpers for CLI entrypoints."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from strategies.base import Strategy


STRATEGY_GROUPS: dict[str, tuple[str, ...]] = {
    'SATControl': ('FirstAction', 'LeanCopCut'),
}


def _sat_test_strategies() -> tuple[str, ...]:
    ordered_names = (
        'SATSolve',
        'SATCoPRust',
        'SATSolveGuided',
        'SATSolveGuidedProbeConflicts',
        'SATSolveGuidedForcedTrue',
        'SATSolveGuidedEVSIDS',
        'SATSolveGuidedShort',
        'SATSolveGuidedFast',
        'SATSolveLiteral',
        'SATSolveLiteralRandomTie',
        'SATSolveLiteralShortTie',
        'SATSolveLiteralLookaheadTie',
        'SATSolveLearnInitial',
        'SATResetBandit',
        'SATResetAtStaleBacktrack',
        'SATResetAtStaleBacktrackTableau',
        'SATResetAtStaleAction',
        'SATResetAtTwo',
        'SATResetAtTwoTableau',
        'SATResetStart',
        'SATResetShadow',
        'SATResetNoCut',
        'SATResetStepBudget',
        'SATResetExtensionBudget',
        'SATSolveLearnCombined',
        'SATSolveLearnSATContext',
        'SATSolveLiteralFrequencyTie',
        'SATSolveLiteralLowFrequencyTie',
        'SATSolveLiteralOppositeSignedFrequencyTie',
        'SATSolveLiteralRareStart',
        'SATSolveLiteralModelStart',
        'SATSolveLiteralFirstStart',
        'SATSolveLiteralFridayStartCut',
        'SATSolveLiteralFridayRootReset',
        'SATSolveLiteralFridayHybrid',
        'SATSolvePropagation',
        'SATSolveLiteralPropagation',
    )
    available = Strategy.strategies()
    return tuple(name for name in ordered_names if name in available)


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
