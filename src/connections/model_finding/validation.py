from __future__ import annotations

from itertools import product
from typing import Mapping

from connections.model_finding.core import FiniteModel, ModelProblem, ModelValidationError
from connections.syntax.formula import (
    And,
    Atom,
    Eq,
    Exists,
    Forall,
    Formula,
    Function,
    Iff,
    Impl,
    Not,
    Or,
    Term,
    Variable,
)


def validate_model(problem: ModelProblem, model: FiniteModel):
    """Exhaustively validate a candidate against the original parsed formulas."""

    try:
        for index, formula in enumerate(problem.asserted_formulas):
            if not evaluate_formula(formula, model):
                raise ModelValidationError(
                    f"asserted formula {index + 1} is false in the candidate model"
                )
        if problem.conjecture_formula is not None and evaluate_formula(
            problem.conjecture_formula, model
        ):
            raise ModelValidationError(
                "the conjecture conjunction is true; the candidate is not a countermodel"
            )
    except ModelValidationError:
        raise
    except (KeyError, ValueError) as exc:
        raise ModelValidationError(
            f"candidate interpretation is incomplete or malformed: {exc}"
        ) from exc


def evaluate_formula(
    formula: Formula,
    model: FiniteModel,
    environment: Mapping[str, int] | None = None,
) -> bool:
    env = dict(environment or {})
    if (
        isinstance(formula, Impl)
        and isinstance(formula.left, Atom)
        and formula.left == formula.right
        and formula.left.symbol == "true___"
    ):
        return True
    if (
        isinstance(formula, And)
        and isinstance(formula.left, Atom)
        and formula.left.symbol == "false___"
        and isinstance(formula.right, Not)
        and formula.right.formula == formula.left
    ):
        return False
    if isinstance(formula, Atom):
        arguments = tuple(_evaluate_term(arg, model, env) for arg in formula.args)
        return model.predicates[formula.symbol][arguments]
    if isinstance(formula, Eq):
        return _evaluate_term(formula.left, model, env) == _evaluate_term(formula.right, model, env)
    if isinstance(formula, Not):
        return not evaluate_formula(formula.formula, model, env)
    if isinstance(formula, And):
        return evaluate_formula(formula.left, model, env) and evaluate_formula(
            formula.right, model, env
        )
    if isinstance(formula, Or):
        return evaluate_formula(formula.left, model, env) or evaluate_formula(
            formula.right, model, env
        )
    if isinstance(formula, Impl):
        return not evaluate_formula(formula.left, model, env) or evaluate_formula(
            formula.right, model, env
        )
    if isinstance(formula, Iff):
        return evaluate_formula(formula.left, model, env) == evaluate_formula(
            formula.right, model, env
        )
    if isinstance(formula, Forall):
        return all(
            evaluate_formula(
                formula.body,
                model,
                {**env, formula.variable.symbol: value},
            )
            for value in model.domain
        )
    if isinstance(formula, Exists):
        return any(
            evaluate_formula(
                formula.body,
                model,
                {**env, formula.variable.symbol: value},
            )
            for value in model.domain
        )
    raise ModelValidationError(
        f"unsupported formula reached independent validation: {type(formula).__name__}"
    )


def _evaluate_term(
    term: Term,
    model: FiniteModel,
    environment: Mapping[str, int],
) -> int:
    if isinstance(term, Variable):
        return environment[term.symbol]
    if isinstance(term, Function):
        arguments = tuple(_evaluate_term(arg, model, environment) for arg in term.args)
        return model.functions[term.symbol][arguments]
    raise TypeError(f"unsupported term: {type(term)!r}")


def table_is_total(model: FiniteModel, symbol: str, arity: int, *, predicate: bool) -> bool:
    table = model.predicates[symbol] if predicate else model.functions[symbol]
    return set(table) == set(product(model.domain, repeat=arity))


__all__ = ["evaluate_formula", "table_is_total", "validate_model"]
