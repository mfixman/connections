from __future__ import annotations

import pytest

from connections.policy import FirstActionIDPolicy
from provers.pycop.policy_resolution import (
    resolve_policies,
    resolve_policy,
    strategy_for_policy,
)


def test_resolve_builtin_policy_alias():
    selection = resolve_policy("baseline=FirstActionID")

    assert selection.label == "baseline"
    assert selection.policy_class is FirstActionIDPolicy


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
