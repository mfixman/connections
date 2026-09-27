from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from connections.model_finding.core import (
    ModelProblem,
    ModelSearchBudget,
    ModelSearchEvent,
    ModelSearchResult,
    ModelSearchStepLimit,
    ModelSearchTimeout,
    ModelValidationError,
)
from connections.model_finding.policies import ModelPolicy, resolve_model_policy
from connections.model_finding.preprocessing import preprocess_model_problem
from connections.model_finding.rendering import render_tptp_model
from connections.model_finding.validation import validate_model


class ModelFinder:
    def __init__(
        self,
        policy: str | ModelPolicy | type[ModelPolicy] = "mace",
    ):
        if isinstance(policy, str):
            selection = resolve_model_policy(policy)
            self.policy_label = selection.label
            self.policy = selection.create()
        elif isinstance(policy, ModelPolicy):
            self.policy = policy
            self.policy_label = policy.alias
        elif isinstance(policy, type) and issubclass(policy, ModelPolicy):
            self.policy = policy()
            self.policy_label = self.policy.alias
        else:
            raise TypeError("policy must be an alias, ModelPolicy, or ModelPolicy class")

    def find_file(
        self,
        path: str | Path,
        *,
        source_roots: Iterable[str | Path] = (),
        budget: ModelSearchBudget | None = None,
    ) -> ModelSearchResult:
        problem = preprocess_model_problem(path, source_roots=source_roots)
        return self.find(problem, budget=budget)

    def find(
        self,
        problem: ModelProblem,
        *,
        budget: ModelSearchBudget | None = None,
    ) -> ModelSearchResult:
        active_budget = budget or ModelSearchBudget()
        events: list[ModelSearchEvent] = []
        domain_size = 1
        try:
            while True:
                if active_budget.timed_out:
                    raise ModelSearchTimeout
                if active_budget.max_domain is not None and domain_size > active_budget.max_domain:
                    return self._limited_result(
                        active_budget,
                        events,
                        status="GaveUp",
                        outcome="NoModelWithinBound",
                    )
                bound = self.policy.search_bound(
                    problem,
                    domain_size,
                    active_budget,
                )
                if active_budget.timed_out:
                    raise ModelSearchTimeout
                if bound.model is None:
                    events.append(bound.event)
                    domain_size += 1
                    continue
                try:
                    validate_model(problem, bound.model)
                    if active_budget.timed_out:
                        raise ModelSearchTimeout
                except ModelValidationError as exc:
                    events.append(replace(bound.event, validation_passed=False))
                    return ModelSearchResult(
                        policy=self.policy_label,
                        status="Error",
                        outcome="ModelValidationError",
                        solved=False,
                        steps=active_budget.steps,
                        elapsed_seconds=active_budget.elapsed_seconds,
                        events=tuple(events),
                        error_type="ModelValidationError",
                        error_message=str(exc),
                    )
                events.append(replace(bound.event, validation_passed=True))
                public_model = bound.model.restricted_to(problem)
                return ModelSearchResult(
                    policy=self.policy_label,
                    status=problem.result_status,
                    outcome="ModelFound",
                    solved=True,
                    steps=active_budget.steps,
                    elapsed_seconds=active_budget.elapsed_seconds,
                    model=public_model,
                    tptp_model=render_tptp_model(public_model),
                    events=tuple(events),
                )
        except ModelSearchTimeout:
            return self._limited_result(
                active_budget,
                events,
                status="Timeout",
                outcome="Timeout",
            )
        except ModelSearchStepLimit:
            return self._limited_result(
                active_budget,
                events,
                status="GaveUp",
                outcome="StepBudget",
            )

    def _limited_result(
        self,
        budget: ModelSearchBudget,
        events: list[ModelSearchEvent],
        *,
        status: str,
        outcome: str,
    ) -> ModelSearchResult:
        return ModelSearchResult(
            policy=self.policy_label,
            status=status,
            outcome=outcome,
            solved=False,
            steps=budget.steps,
            elapsed_seconds=budget.elapsed_seconds,
            events=tuple(events),
        )


__all__ = ["ModelFinder"]
