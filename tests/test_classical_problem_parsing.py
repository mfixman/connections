from pathlib import Path
from unittest.mock import patch

from connections.utils.classical_problem_parsing import file2cnf


class TestClassicalProblemParsing:
    def test_connections_cnf_passthrough(self):
        matrix = file2cnf('tests/cnf_problems/SYN726+1.cnf')

        assert len(matrix.clauses) == 6
        assert str(matrix.clauses[0][0]) == 'q(f_skolem(3), f_skolem(4))'

    def test_raw_minus_problem_uses_tptp_translation(self, tmp_path: Path):
        axiom = tmp_path / 'AX001-0.ax'
        axiom.write_text("cnf(ax,axiom,p(a)).")
        problem = tmp_path / 'FLD071-2.p'
        problem.write_text("include('AX001-0.ax').\ncnf(goal,negated_conjecture,~p(a)).")

        matrix = file2cnf(str(problem))

        assert len(matrix.clauses) == 2
        assert str(matrix.clauses[0][0]) == 'p(a)'
        assert str(matrix.clauses[1][0]) == '-p(a)'

    def test_raw_plus_problem_uses_tptp_translation(self, tmp_path: Path):
        problem = tmp_path / 'GEO205+1.p'
        problem.write_text("fof(a,axiom,p).")

        sentinel = object()
        with patch(
            'connections.utils.classical_problem_parsing._translate_tptp_problem',
            return_value=sentinel,
        ) as translate:
            assert file2cnf(str(problem)) is sentinel

        translate.assert_called_once_with(problem)
