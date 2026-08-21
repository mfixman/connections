from __future__ import annotations

import pytest

from connections.policy import (
    FirstActionIDPolicy,
    LeanCoPCon,
    LeanCoPCon_2,
    SATCoPCon,
    SATResetCoP,
)
from provers.pycop.policy_resolution import (
    builtin_policy_names,
    resolve_policies,
    resolve_policy,
    schedule_for_policy,
    strategy_for_policy,
)


def test_resolve_builtin_policy_alias():
    selection = resolve_policy("baseline=FirstActionID")

    assert selection.label == "baseline"
    assert selection.policy_class is FirstActionIDPolicy


@pytest.mark.parametrize(
    ("name", "policy_class"),
    [
        ("LeanCoPCon", LeanCoPCon),
        ("LeanCoPCon_2", LeanCoPCon_2),
        ("SATCoPCon", SATCoPCon),
        ("SATResetCoP", SATResetCoP),
    ],
)
def test_resolve_named_connections_policies(name, policy_class):
    selection = resolve_policy(name)

    assert selection.label == name
    assert selection.policy_class is policy_class
    assert name in builtin_policy_names()


def test_resolve_importable_policy():
    selection = resolve_policy("tests.fixtures.cli_policies:BlockingPolicy")

    assert selection.label == "BlockingPolicy"
    assert selection.policy_class.__name__ == "BlockingPolicy"


def test_reject_non_policy_import():
    with pytest.raises(TypeError, match="does not resolve to a Policy subclass"):
        resolve_policy("tests.fixtures.cli_policies:NotAPolicy")


def test_reject_duplicate_policy_labels():
    with pytest.raises(ValueError, match="duplicate policy labels"):
        resolve_policies(
            ["same=FirstActionID", "same=FirstActionIDPolicy"]
        )


def test_strategy_replaces_policy_but_keeps_pycop_settings():
    selection = resolve_policy("FirstActionIDPolicy")
    strategy = strategy_for_policy(
        selection,
        settings=["cut", "comp(7)"],
        backtrack="maximal",
    )

    assert strategy.policy.policy_class is FirstActionIDPolicy
    assert strategy.policy.args == {
        "cut": True,
        "scut": False,
        "comp": 7,
        "factorization": "equal",
        "backtrack": "maximal",
    }


def test_leancop_named_strategy_uses_cut_comp_defaults():
    strategy = strategy_for_policy(
        resolve_policy("LeanCoPCon"),
        settings=[],
        backtrack="step",
    )

    assert strategy.policy.policy_class is LeanCoPCon
    assert strategy.policy.args["cut"] is True
    assert strategy.policy.args["comp"] == 7


@pytest.mark.parametrize("name", ["SATCoPCon", "SATResetCoP"])
def test_sat_named_strategies_use_all_clause_starts(name):
    strategy = strategy_for_policy(
        resolve_policy(name),
        settings=[],
        backtrack="step",
    )

    assert strategy.matrix.start_clauses == "all"


def test_leancop2_named_policy_builds_published_schedule():
    schedule = schedule_for_policy(
        resolve_policy("LeanCoPCon_2"),
        settings=[],
        backtrack="step",
        steps=194,
        timeout=111.0,
    )

    assert len(schedule.entries) == 30
    assert all(
        entry.strategy.policy.policy_class is LeanCoPCon_2
        for entry in schedule.entries
    )
    assert sum(entry.step_limit or 0 for entry in schedule.entries) == 194
    assert sum(entry.timeout_seconds or 0.0 for entry in schedule.entries) == 111.0


def test_leancop2_schedule_rejects_global_settings():
    with pytest.raises(ValueError, match="owns its 30 phase settings"):
        schedule_for_policy(
            resolve_policy("LeanCoPCon_2"),
            settings=["cut"],
            backtrack="step",
            steps=100,
            timeout=10.0,
        )
