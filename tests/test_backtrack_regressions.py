from pathlib import Path

from connections.env import ConnectionEnv, Settings
from strategies import FirstAction


CAT_AXIOMS = """\
%--------------------------------------------------------------------------
cnf(equivalence_implies_existence1,axiom,
    ( ~ equivalent(X,Y)
    | there_exists(X) ) ).

cnf(equivalence_implies_existence2,axiom,
    ( ~ equivalent(X,Y)
    | X = Y ) ).

cnf(existence_and_equality_implies_equivalence1,axiom,
    ( ~ there_exists(X)
    | X != Y
    | equivalent(X,Y) ) ).

cnf(domain_has_elements,axiom,
    ( ~ there_exists(domain(X))
    | there_exists(X) ) ).

cnf(codomain_has_elements,axiom,
    ( ~ there_exists(codomain(X))
    | there_exists(X) ) ).

cnf(composition_implies_domain,axiom,
    ( ~ there_exists(compose(X,Y))
    | there_exists(domain(X)) ) ).

cnf(domain_codomain_composition1,axiom,
    ( ~ there_exists(compose(X,Y))
    | domain(X) = codomain(Y) ) ).

cnf(domain_codomain_composition2,axiom,
    ( ~ there_exists(domain(X))
    | domain(X) != codomain(Y)
    | there_exists(compose(X,Y)) ) ).

cnf(associativity_of_compose,axiom,
    compose(X,compose(Y,Z)) = compose(compose(X,Y),Z) ).

cnf(compose_domain,axiom,
    compose(X,domain(X)) = X ).

cnf(compose_codomain,axiom,
    compose(codomain(X),X) = X ).

cnf(equivalence_implies_existence3,axiom,
    ( ~ equivalent(X,Y)
    | there_exists(Y) ) ).

cnf(existence_and_equality_implies_equivalence2,axiom,
    ( ~ there_exists(X)
    | ~ there_exists(Y)
    | X != Y
    | equivalent(X,Y) ) ).

cnf(composition_implies_codomain,axiom,
    ( ~ there_exists(compose(X,Y))
    | there_exists(codomain(X)) ) ).

cnf(indiscernibles1,axiom,
    ( there_exists(f1(X,Y))
    | X = Y ) ).

cnf(indiscernibles2,axiom,
    ( X = f1(X,Y)
    | Y = f1(X,Y)
    | X = Y ) ).

cnf(indiscernibles3,axiom,
    ( X != f1(X,Y)
    | Y != f1(X,Y)
    | X = Y ) ).
"""


CAT001_PROBLEM = """\
include('../../Axioms/CAT003-0.ax').

cnf(assume_ab_exists,hypothesis,
    there_exists(compose(a,b)) ).

cnf(monomorphism,hypothesis,
    ( compose(compose(a,b),X) != Y
    | compose(compose(a,b),Z) != Y
    | X = Z ) ).

cnf(assume_bh_exists,hypothesis,
    there_exists(compose(b,h)) ).

cnf(bh_equals_bg,hypothesis,
    compose(b,h) = compose(b,g) ).

cnf(prove_h_equals_g,negated_conjecture,
    h != g ).
"""


CAT003_PROBLEM = """\
include('../../Axioms/CAT003-0.ax').

cnf(assume_ab_exists,hypothesis,
    there_exists(compose(a,b)) ).

cnf(epimorphism,hypothesis,
    ( compose(X,compose(a,b)) != Y
    | compose(Z,compose(a,b)) != Y
    | X = Z ) ).

cnf(assume_h_exists,hypothesis,
    there_exists(h) ).

cnf(ha_equals_ga,hypothesis,
    compose(h,a) = compose(g,a) ).

cnf(prove_g_equals_h,negated_conjecture,
    g != h ).
"""


def _write_cat_problem(tmp_path: Path, filename: str, problem_text: str) -> Path:
    axioms_dir = tmp_path / "Axioms"
    problems_dir = tmp_path / "Problems" / "CAT"
    axioms_dir.mkdir(parents = True)
    problems_dir.mkdir(parents = True)
    (axioms_dir / "CAT003-0.ax").write_text(CAT_AXIOMS, encoding = "utf-8")
    problem_path = problems_dir / filename
    problem_path.write_text(problem_text, encoding = "utf-8")
    return problem_path


def _run_steps(problem_path: Path, steps: int = 2000) -> None:
    env = ConnectionEnv(str(problem_path), Settings(logic = "ClassicalCut"))
    strategy = FirstAction(env)
    env.reset()
    for _ in range(steps):
        action = strategy.step()
        _, _, done, _ = env.step(action)
        if done:
            return


class TestBacktrackRegression:
    def test_cat001_backtracking_recomputes_fresh_actions(self, tmp_path: Path):
        problem_path = _write_cat_problem(tmp_path, "CAT001-3.p", CAT001_PROBLEM)
        _run_steps(problem_path)

    def test_cat003_backtracking_recomputes_fresh_actions(self, tmp_path: Path):
        problem_path = _write_cat_problem(tmp_path, "CAT003-3.p", CAT003_PROBLEM)
        _run_steps(problem_path)
