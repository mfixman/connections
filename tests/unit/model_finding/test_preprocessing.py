from __future__ import annotations

from pathlib import Path

import pytest

from connections.model_finding import InputError, preprocess_model_problem


def _problem(tmp_path: Path, text: str, name: str = "problem.p") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_axiom_only_fof_and_cnf_are_asserted_without_negation(tmp_path: Path):
    path = _problem(
        tmp_path,
        "fof(a,axiom, ! [X] : (p(X) => q(X))).\ncnf(b,negated_conjecture,(p(a) | ~q(a))).\n",
    )

    problem = preprocess_model_problem(path)

    assert problem.result_status == "Satisfiable"
    assert not problem.has_conjecture
    assert len(problem.asserted_formulas) == 2
    assert problem.user_function_symbols == {"a"}
    assert problem.user_predicate_symbols == {"p", "q"}
    assert problem.clauses[0].source == "b"


def test_fof_conjectures_are_combined_and_negated_for_countermodels(tmp_path: Path):
    path = _problem(
        tmp_path,
        "fof(a,axiom,p(a)).\nfof(c1,conjecture, ! [X] : p(X)).\nfof(c2,conjecture,q).\n",
    )

    problem = preprocess_model_problem(path)

    assert problem.result_status == "CounterSatisfiable"
    assert problem.conjecture_formula is not None
    assert {clause.source for clause in problem.clauses} >= {
        "a",
        "model_negated_conjectures",
    }


def test_mixed_quantifiers_introduce_hidden_skolem_but_not_public_symbol(
    tmp_path: Path,
):
    path = _problem(tmp_path, "fof(a,axiom, ! [X] : ? [Y] : (X != Y)).\n")

    problem = preprocess_model_problem(path)

    assert problem.hidden_function_symbols
    assert not problem.user_function_symbols
    assert problem.hidden_function_symbols <= set(problem.function_signatures)


def test_includes_retain_statement_roles_for_model_preprocessing(tmp_path: Path):
    included = _problem(tmp_path, "cnf(i,axiom,p(a)).\n", "included.ax")
    main = _problem(
        tmp_path,
        f"include('{included.name}').\nfof(c,conjecture,? [X] : ~p(X)).\n",
    )

    problem = preprocess_model_problem(main, source_roots=(tmp_path,))

    assert [statement.name for statement in problem.statements] == ["i", "c"]
    assert problem.result_status == "CounterSatisfiable"


@pytest.mark.parametrize(
    "text, fragment",
    (
        ("qmf(m,axiom,#box:p).", "modal QMF"),
        ("tff(t,axiom,$true).", "Unsupported annotated formula"),
        ('fof(a,axiom,p("object")).', "distinct-object"),
        ("fof(a,axiom,p(42)).", "numeric and arithmetic"),
        ("fof(a,axiom,p($sum(a,b))).", "defined/system symbol"),
        ("fof(a,axiom,p($$system)).", "defined/system symbol"),
        ("fof(a,unknown,p).", "not an axiom-like"),
        ("cnf(c,conjecture,p).", "CNF conjecture"),
    ),
)
def test_unsupported_inputs_fail_with_clear_input_error(
    tmp_path: Path,
    text: str,
    fragment: str,
):
    with pytest.raises(InputError, match=fragment):
        preprocess_model_problem(_problem(tmp_path, text))


def test_metadata_status_comments_are_not_part_of_model_semantics(tmp_path: Path):
    path = _problem(
        tmp_path,
        "% Status   : Unsatisfiable\nfof(a,axiom,p).\n",
    )

    problem = preprocess_model_problem(path)

    assert problem.result_status == "Satisfiable"
