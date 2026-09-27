from __future__ import annotations

from pathlib import Path

import pytest

from connections.model_finding import (
    FiniteModel,
    ModelFinder,
    ModelSearchBudget,
    ModelValidationError,
    preprocess_model_problem,
    validate_model,
)
from connections.model_finding.validation import table_is_total
from connections.parsing.tptp.parser import parse_tptp


def _problem(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "problem.p"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("policy", ("mace", "sat-reset"))
def test_both_policies_reject_domain_one_and_find_valid_domain_two(
    tmp_path: Path,
    policy: str,
):
    path = _problem(tmp_path, "fof(a,axiom, ! [X] : (f(X) != X)).\n")
    problem = preprocess_model_problem(path)

    result = ModelFinder(policy).find(
        problem,
        budget=ModelSearchBudget(max_domain=2, max_steps=50, timeout_seconds=2),
    )

    assert result.status == "Satisfiable"
    assert result.outcome == "ModelFound"
    assert result.solved
    assert result.model is not None
    assert result.model.domain_size == 2
    assert [event.satisfiable for event in result.events] == [False, True]
    assert result.events[-1].validation_passed is True
    validate_model(problem, result.model)
    assert table_is_total(result.model, "f", 1, predicate=False)


@pytest.mark.parametrize("policy", ("mace", "sat-reset"))
def test_mixed_quantifiers_equality_and_nested_functions(tmp_path: Path, policy: str):
    path = _problem(
        tmp_path,
        "fof(a,axiom, ! [X] : ? [Y] : (X != Y)).\n"
        "fof(b,axiom, ! [X] : ((f(f(X)) = X) & (f(X) != X))).\n",
    )
    problem = preprocess_model_problem(path)

    result = ModelFinder(policy).find(
        problem,
        budget=ModelSearchBudget(max_domain=2, max_steps=100, timeout_seconds=2),
    )

    assert result.status == "Satisfiable"
    assert result.model is not None
    assert result.model.domain_size == 2
    assert not (set(result.model.functions) & problem.hidden_function_symbols)
    validate_model(problem, result.model)


@pytest.mark.parametrize("policy", ("mace", "sat-reset"))
def test_fof_countermodel_status_and_total_predicate_table(tmp_path: Path, policy: str):
    path = _problem(
        tmp_path,
        "fof(a,axiom,p(a)).\nfof(c,conjecture, ! [X] : p(X)).\n",
    )
    problem = preprocess_model_problem(path)

    result = ModelFinder(policy).find(
        problem,
        budget=ModelSearchBudget(max_domain=2, max_steps=100, timeout_seconds=2),
    )

    assert result.status == "CounterSatisfiable"
    assert result.model is not None
    assert result.model.domain_size == 2
    assert table_is_total(result.model, "p", 1, predicate=True)


def test_sat_reset_exercises_multiple_cegar_refinements(tmp_path: Path):
    path = _problem(tmp_path, "cnf(a,axiom,p).\ncnf(b,axiom,q).\n")

    result = ModelFinder("sat-reset").find_file(
        path,
        budget=ModelSearchBudget(max_domain=1, max_steps=20),
    )

    assert result.solved
    assert result.events[0].refinements >= 2
    assert result.events[0].sat_calls >= 3


def test_explicit_limits_never_report_global_unsatisfiable(tmp_path: Path):
    path = _problem(tmp_path, "fof(a,axiom, ! [X] : (f(X) != X)).\n")
    problem = preprocess_model_problem(path)

    bounded = ModelFinder("mace").find(
        problem, budget=ModelSearchBudget(max_domain=1, max_steps=10)
    )
    step_limited = ModelFinder("sat-reset").find(
        problem, budget=ModelSearchBudget(max_domain=2, max_steps=0)
    )
    timed_out = ModelFinder("mace").find(
        problem, budget=ModelSearchBudget(max_domain=2, timeout_seconds=0)
    )

    assert (bounded.status, bounded.outcome) == ("GaveUp", "NoModelWithinBound")
    assert (step_limited.status, step_limited.outcome) == ("GaveUp", "StepBudget")
    assert (timed_out.status, timed_out.outcome) == ("Timeout", "Timeout")
    assert "Unsatisfiable" not in {
        bounded.status,
        bounded.outcome,
        step_limited.status,
        step_limited.outcome,
        timed_out.status,
        timed_out.outcome,
    }


def test_independent_validator_detects_deliberate_corruption(tmp_path: Path):
    path = _problem(tmp_path, "fof(a,axiom,p).\n")
    problem = preprocess_model_problem(path)
    result = ModelFinder("mace").find_file(path, budget=ModelSearchBudget(max_domain=1))
    assert result.model is not None
    corrupted = FiniteModel(
        domain_size=1,
        functions=result.model.functions,
        predicates={"p": {(): False}},
    )

    with pytest.raises(ModelValidationError, match="asserted formula"):
        validate_model(problem, corrupted)


def test_tptp_rendering_has_standard_total_interpretation_roles(tmp_path: Path):
    path = _problem(tmp_path, "fof(a,axiom,p(f(a))).\n")
    result = ModelFinder("mace").find_file(path, budget=ModelSearchBudget(max_domain=1))

    assert result.tptp_model is not None
    rendered = result.tptp_model
    assert "fi_domain" in rendered
    assert "fi_functors" in rendered
    assert "fi_predicates" in rendered
    assert len(parse_tptp(rendered).items) == 3


@pytest.mark.parametrize("policy", ("mace", "sat-reset"))
def test_empty_signature_truth_is_a_valid_one_element_model(tmp_path: Path, policy: str):
    path = _problem(tmp_path, "cnf(a,axiom,$true).\n")
    problem = preprocess_model_problem(path)

    result = ModelFinder(policy).find(problem, budget=ModelSearchBudget(max_domain=1, max_steps=10))

    assert result.status == "Satisfiable"
    assert result.model is not None
    assert result.model.domain_size == 1
    assert result.model.functions == {}
    assert result.model.predicates == {}
