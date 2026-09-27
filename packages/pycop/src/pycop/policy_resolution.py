from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import importlib
import inspect

from connections.agent import Agent, AgentOptions
from connections.interaction.strategy import PolicyOptions, StrategySchedule
from pycop.leancop import leancop_agent
from pycop.settings_codec import LeancopSettingsCodec


@dataclass(frozen=True, slots=True)
class PolicySelection:
    label: str
    reference: str
    policy_class: Callable[..., Agent]


def builtin_policy_names():
    return (
        "FirstActionIDPolicy",
        "leancop_con",
        "LeanCoPCon_2",
        "satcop_con",
        "satresetcop",
    )


def resolve_policy(specification):
    label, separator, reference = specification.partition("=")
    if not separator:
        reference, label = label, ""
    if not reference or (separator and not label):
        raise ValueError("labeled policies use LABEL=ALIAS or LABEL=package.module:Class")
    if reference in {"satcop", "satcop_con", "SATCoPCon", "satresetcop", "SATResetCoP"}:
        from connections.agent.sat import SATCoPCon, SATResetCoP

        factory = SATResetCoP if reference.lower() == "satresetcop" else SATCoPCon
    elif reference in {
        "FirstActionID",
        "FirstActionIDPolicy",
        "leancop_con",
        "LeanCoPCon",
        "LeanCoPCon_2",
    }:
        factory = leancop_agent
    else:
        module, separator, name = reference.partition(":")
        if not separator or not name:
            raise ValueError(
                f"unknown policy {reference!r}; choose from {', '.join(builtin_policy_names())}"
            )
        factory = importlib.import_module(module)
        for component in name.split("."):
            factory = getattr(factory, component)
        if not inspect.isclass(factory) or not issubclass(factory, Agent):
            raise TypeError(f"{reference!r} must name an Agent subclass")
    return PolicySelection(label or reference.rsplit(":", 1)[-1], reference, factory)


def resolve_policies(specifications):
    selections = tuple(resolve_policy(specification) for specification in specifications)
    labels = [selection.label for selection in selections]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate policy labels")
    return selections


def strategy_for_policy(selection, *, settings, backtrack, debug_sat_core=False):
    is_sat = selection.policy_class.__module__.startswith("connections.agent.sat")
    defaults = ["cut"] if is_sat else []
    if selection.reference in {"leancop_con", "LeanCoPCon"}:
        defaults = ["cut", "comp(7)"]
    tokens = [token for text in settings for token in LeancopSettingsCodec.split_token_list(text)]
    base = LeancopSettingsCodec.from_tokens([*defaults, *tokens])
    args = dict(base.policy.args)
    args["backtrack"] = backtrack
    if selection.policy_class is not leancop_agent:
        if is_sat:
            args["start"] = "all"
        args = {"options": AgentOptions(**args)}
        if is_sat:
            args["debug_sat_core"] = debug_sat_core
    inspect.signature(selection.policy_class).bind(**args)
    return replace(base, policy=PolicyOptions(selection.policy_class, args))


def schedule_for_policy(selection, *, settings, backtrack, steps, timeout, debug_sat_core=False):
    if selection.reference == "LeanCoPCon_2":
        if settings:
            raise ValueError("LeanCoPCon_2 owns its phase settings; do not pass --settings")
        from pycop.schedule import load_schedule_entries

        entries = []
        for entry in load_schedule_entries("classical"):
            args = {**entry.strategy.policy.args, "backtrack": backtrack}
            strategy = replace(entry.strategy, policy=PolicyOptions(leancop_agent, args))
            entries.append(replace(entry, strategy=strategy))
        return StrategySchedule.from_weighted(entries, steps=steps, timeout_seconds=timeout)
    strategy = strategy_for_policy(
        selection,
        settings=settings,
        backtrack=backtrack,
        debug_sat_core=debug_sat_core,
    )
    return StrategySchedule.single(strategy, steps=steps, timeout_seconds=timeout)
