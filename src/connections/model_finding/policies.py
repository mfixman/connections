from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
import inspect
import time

import pydical  # type: ignore[unresolved-import]

from connections.model_finding.core import FiniteModel, ModelProblem, ModelSearchBudget, ModelSearchEvent, ModelSearchTimeout
from connections.model_finding.encoding import FixedDomainEncoding, GroundInstance


@dataclass(frozen=True, slots=True)
class BoundSearchResult:
    model: FiniteModel | None
    event: ModelSearchEvent


class ModelPolicy(ABC):
    """One fixed-domain model-search policy.

    A policy may report only a model for this domain or exhaustion of this
    domain. It cannot turn fixed-domain UNSAT into global unsatisfiability.
    """

    alias = "model-policy"

    @abstractmethod
    def search_bound(
        self,
        problem: ModelProblem,
        domain_size: int,
        budget: ModelSearchBudget,
    ) -> BoundSearchResult:
        raise NotImplementedError

    def complete_candidate(self, encoding: FixedDomainEncoding) -> FiniteModel:
        """Complete a partial SAT assignment into a total interpretation."""

        return encoding.extract_model()

    def select_counterexample(
        self,
        encoding: FixedDomainEncoding,
        candidate: FiniteModel,
    ) -> GroundInstance | None:
        """Select one falsified universal instance for a CEGAR refinement."""

        return encoding.first_falsified_instance(candidate)


class MaceModelPolicy(ModelPolicy):
    alias = "mace"

    def search_bound(
        self,
        problem: ModelProblem,
        domain_size: int,
        budget: ModelSearchBudget,
    ) -> BoundSearchResult:
        started = time.monotonic()
        encoding = FixedDomainEncoding(
            problem,
            domain_size,
            static_constant_symmetry=True,
            budget=budget,
        )
        for instance in encoding.all_instances():
            if budget.timed_out:
                raise ModelSearchTimeout
            encoding.encode_instance(instance)
        budget.consume_step()
        status = encoding.solve()
        if status == pydical.UNSATISFIABLE:
            model = None
            satisfiable = False
        elif status == pydical.SATISFIABLE:
            model = self.complete_candidate(encoding)
            satisfiable = True
        else:
            raise RuntimeError(f"CaDiCaL returned unexpected status {status}")
        return BoundSearchResult(
            model=model,
            event=ModelSearchEvent(
                domain_size=domain_size,
                sat_calls=1,
                variables=encoding.variable_count,
                clauses=encoding.clause_count,
                refinements=0,
                elapsed_seconds=time.monotonic() - started,
                satisfiable=satisfiable,
            ),
        )


class SatResetModelPolicy(ModelPolicy):
    alias = "sat-reset"

    def search_bound(
        self,
        problem: ModelProblem,
        domain_size: int,
        budget: ModelSearchBudget,
    ) -> BoundSearchResult:
        started = time.monotonic()
        encoding = FixedDomainEncoding(problem, domain_size, budget=budget)
        sat_calls = 0
        refinements = 0
        while True:
            budget.consume_step()
            sat_calls += 1
            status = encoding.solve()
            if status == pydical.UNSATISFIABLE:
                return BoundSearchResult(
                    model=None,
                    event=ModelSearchEvent(
                        domain_size=domain_size,
                        sat_calls=sat_calls,
                        variables=encoding.variable_count,
                        clauses=encoding.clause_count,
                        refinements=refinements,
                        elapsed_seconds=time.monotonic() - started,
                        satisfiable=False,
                    ),
                )
            if status != pydical.SATISFIABLE:
                raise RuntimeError(f"CaDiCaL returned unexpected status {status}")
            candidate = self.complete_candidate(encoding)
            counterexample = self.select_counterexample(encoding, candidate)
            if counterexample is None:
                return BoundSearchResult(
                    model=candidate,
                    event=ModelSearchEvent(
                        domain_size=domain_size,
                        sat_calls=sat_calls,
                        variables=encoding.variable_count,
                        clauses=encoding.clause_count,
                        refinements=refinements,
                        elapsed_seconds=time.monotonic() - started,
                        satisfiable=True,
                    ),
                )
            encoding.encode_instance(counterexample)
            refinements += 1


@dataclass(frozen=True, slots=True)
class ModelPolicySelection:
    label: str
    reference: str
    policy_class: type[ModelPolicy]

    def create(self) -> ModelPolicy:
        return self.policy_class()


_BUILTINS: dict[str, type[ModelPolicy]] = {
    "mace": MaceModelPolicy,
    "sat-reset": SatResetModelPolicy,
}


def builtin_model_policy_names() -> tuple[str, ...]:
    return tuple(_BUILTINS)


def resolve_model_policy(specification: str) -> ModelPolicySelection:
    label, reference = _split_label(specification)
    policy_class = _BUILTINS.get(reference)
    if policy_class is None:
        policy_class = _import_policy(reference)
    try:
        inspect.signature(policy_class).bind()
    except TypeError as exc:
        raise ValueError(f"model policy {reference!r} must have a no-argument constructor") from exc
    return ModelPolicySelection(
        label=label or reference.rsplit(":", 1)[-1],
        reference=reference,
        policy_class=policy_class,
    )


def _split_label(specification: str) -> tuple[str | None, str]:
    if "=" not in specification:
        return None, specification
    label, reference = specification.split("=", 1)
    if not label or not reference:
        raise ValueError("labeled model policies use LABEL=ALIAS or LABEL=package.module:Class")
    return label, reference


def _import_policy(reference: str) -> type[ModelPolicy]:
    if ":" not in reference:
        known = ", ".join(_BUILTINS)
        raise ValueError(
            f"unknown model policy {reference!r}; choose from {known}, or use package.module:Class"
        )
    module_name, qualname = reference.split(":", 1)
    if not module_name or not qualname:
        raise ValueError("model policy import paths must use package.module:Class")
    module = importlib.import_module(module_name)
    candidate: object = module
    for component in qualname.split("."):
        candidate = getattr(candidate, component)
    if not inspect.isclass(candidate) or not issubclass(candidate, ModelPolicy):
        raise TypeError(f"{reference!r} does not resolve to a ModelPolicy subclass")
    return candidate


__all__ = [
    "BoundSearchResult",
    "MaceModelPolicy",
    "ModelPolicy",
    "ModelPolicySelection",
    "SatResetModelPolicy",
    "builtin_model_policy_names",
    "resolve_model_policy",
]
