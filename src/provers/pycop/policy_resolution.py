from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import inspect

from connections.policy import FirstActionIDPolicy, Policy
from connections.prover.strategy import PolicyOptions, Strategy
from provers.pycop.settings_codec import LeancopSettingsCodec


_BUILTIN_POLICIES: dict[str, type[Policy]] = {
    "FirstActionID": FirstActionIDPolicy,
    "FirstActionIDPolicy": FirstActionIDPolicy,
}
_CANONICAL_POLICY_NAMES = ("FirstActionIDPolicy",)


@dataclass(frozen=True, slots=True)
class PolicySelection:
    label: str
    reference: str
    policy_class: type[Policy]


def builtin_policy_names() -> tuple[str, ...]:
    return _CANONICAL_POLICY_NAMES


def resolve_policy(specification: str) -> PolicySelection:
    label, reference = _split_label(specification)
    policy_class = _BUILTIN_POLICIES.get(reference)
    if policy_class is None:
        policy_class = _import_policy(reference)
    return PolicySelection(
        label=label or reference.rsplit(":", 1)[-1],
        reference=reference,
        policy_class=policy_class,
    )


def resolve_policies(specifications: list[str]) -> tuple[PolicySelection, ...]:
    selections = tuple(resolve_policy(specification) for specification in specifications)
    labels = [selection.label for selection in selections]
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        raise ValueError(f"duplicate policy labels: {', '.join(duplicate_labels)}")
    return selections


def strategy_for_policy(
    selection: PolicySelection,
    *,
    settings: list[str],
    backtrack: str,
) -> Strategy:
    base = LeancopSettingsCodec.from_tokens(settings)
    policy_args = dict(base.policy.args or {})
    policy_args["backtrack"] = backtrack
    _validate_constructor(selection, policy_args)
    return replace(
        base,
        policy=PolicyOptions(
            policy_class=selection.policy_class,
            args=policy_args,
        ),
    )


def _split_label(specification: str) -> tuple[str | None, str]:
    if "=" not in specification:
        return None, specification
    label, reference = specification.split("=", 1)
    if not label or not reference:
        raise ValueError(
            "labeled policies must use LABEL=ALIAS or LABEL=package.module:PolicyClass"
        )
    return label, reference


def _import_policy(reference: str) -> type[Policy]:
    if ":" not in reference:
        known = ", ".join(_BUILTIN_POLICIES)
        raise ValueError(
            f"unknown policy {reference!r}; choose from {known}, or use package.module:PolicyClass"
        )
    module_name, qualname = reference.split(":", 1)
    if not module_name or not qualname:
        raise ValueError("policy import paths must use package.module:PolicyClass")
    module = importlib.import_module(module_name)
    candidate: object = module
    for component in qualname.split("."):
        candidate = getattr(candidate, component)
    if not inspect.isclass(candidate) or not issubclass(candidate, Policy):
        raise TypeError(f"{reference!r} does not resolve to a Policy subclass")
    return candidate


def _validate_constructor(
    selection: PolicySelection,
    policy_args: dict[str, object],
) -> None:
    try:
        inspect.signature(selection.policy_class).bind(**policy_args)
    except TypeError as exc:
        raise ValueError(
            f"policy {selection.label!r} does not accept the selected pycop settings: {exc}"
        ) from exc


__all__ = [
    "PolicySelection",
    "builtin_policy_names",
    "resolve_policies",
    "resolve_policy",
    "strategy_for_policy",
]
