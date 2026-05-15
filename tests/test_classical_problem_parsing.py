from pathlib import Path
from unittest.mock import patch

import pytest

from connections.utils.classical_problem_parsing import file2cnf, problem_metadata

class TestClassicalProblemParsing:
    def test_connections_cnf_passthrough(self):
        matrix = file2cnf('tests/cnf_problems/SYN726+1.cnf')

        assert len(matrix.clauses) == 6
        assert str(matrix.clauses[0][0]) == 'q(f_skolem(3), f_skolem(4))'

    def test_connections_cnf_is_detected_by_content_not_suffix(self, tmp_path: Path):
        problem = tmp_path / 'problem.txt'
        problem.write_text("cnf(test,axiom,[p(a)]).\n")

        matrix = file2cnf(str(problem))

        assert len(matrix.clauses) == 1
        assert str(matrix.clauses[0][0]) == 'p(a)'

    def test_raw_minus_problem_uses_tptp_translation(self, tmp_path: Path):
        axiom = tmp_path / 'AX001-0.ax'
        axiom.write_text("cnf(ax,axiom,p(a)).")
        problem = tmp_path / 'FLD071-2.p'
        problem.write_text("include('AX001-0.ax').\ncnf(goal,negated_conjecture,~p(a)).")

        matrix = file2cnf(str(problem))

        assert len(matrix.clauses) == 2
        assert str(matrix.clauses[0][0]) == 'p(a)'
        assert str(matrix.clauses[1][0]) == '-p(a)'

    def test_raw_minus_problem_resolves_axioms_include_from_tptp_root(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        tptp_root = tmp_path / 'tptp'
        axiom = tptp_root / 'Axioms' / 'AX001-0.ax'
        problem = tptp_root / 'Problems' / 'FLD' / 'FLD071-2.p'
        axiom.parent.mkdir(parents = True)
        problem.parent.mkdir(parents = True)
        axiom.write_text("cnf(ax,axiom,p(a)).")
        problem.write_text("include('Axioms/AX001-0.ax').\ncnf(goal,negated_conjecture,~p(a)).")
        monkeypatch.setenv('TPTP', str(tptp_root))

        matrix = file2cnf(str(problem))

        assert len(matrix.clauses) == 2
        assert str(matrix.clauses[0][0]) == 'p(a)'
        assert str(matrix.clauses[1][0]) == '-p(a)'

    def test_raw_minus_problem_include_without_tptp_fails_loudly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        problem = tmp_path / 'FLD071-2.p'
        problem.write_text("include('Axioms/AX001-0.ax').\ncnf(goal,negated_conjecture,~p(a)).")
        monkeypatch.delenv('TPTP', raising = False)

        with pytest.raises(FileNotFoundError, match='pass --tptp ROOT or set the TPTP environment variable'):
            file2cnf(str(problem))

    def test_raw_plus_problem_uses_tptp_translation(self, tmp_path: Path):
        problem = tmp_path / 'GEO205+1.p'
        problem.write_text("fof(a,axiom,p).")

        sentinel = object()
        with patch(
            'connections.utils.classical_problem_parsing._translate_tptp_problem',
            return_value=sentinel,
        ) as translate:
            assert file2cnf(str(problem)) is sentinel

        translate.assert_called_once()
        assert translate.call_args.args[0] == problem

    def test_mptp_problem_filename_uses_tptp_translation(self, tmp_path: Path):
        problem = tmp_path / 'compts_1__t10_compts_1.p'
        problem.write_text("fof(t10_compts_1,conjecture,p).")

        sentinel = object()
        with patch(
            'connections.utils.classical_problem_parsing._translate_tptp_problem',
            return_value=sentinel,
        ) as translate:
            assert file2cnf(str(problem)) is sentinel

        translate.assert_called_once()
        assert translate.call_args.args[0] == problem

    def test_unsupported_file_warns_and_fails_fast(self, tmp_path: Path):
        problem = tmp_path / 'problem.txt'
        problem.write_text("fof(a,axiom,p).\n")

        with pytest.warns(UserWarning, match='unsupported problem file'):
            with pytest.raises(ValueError, match='unsupported problem file'):
                file2cnf(str(problem))

    def test_raw_minus_problem_parses_typed_cnf(self, tmp_path: Path):
        problem = tmp_path / 'SYN000-3.p'
        problem.write_text("tcf(a,axiom, ! [X: $i] : ( p(X) | ~ q(a) )).\n")

        matrix = file2cnf(str(problem))

        assert len(matrix.clauses) == 1
        assert str(matrix.clauses[0][0]) == 'p(X)'
        assert str(matrix.clauses[0][1]) == '-q(a)'

    def test_problem_metadata_collects_transitive_spc_tags(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        tptp_root = tmp_path / 'tptp'
        axiom = tptp_root / 'Axioms' / 'AX001-0.ax'
        problem = tptp_root / 'Problems' / 'FLD' / 'FLD071-2.p'
        axiom.parent.mkdir(parents = True)
        problem.parent.mkdir(parents = True)
        axiom.write_text("% SPC      : TH0_SAT_EQU_NAR\ncnf(ax,axiom,p(a)).")
        problem.write_text("% Status   : Satisfiable\n% SPC      : CNF_SAT_RFO\ninclude('Axioms/AX001-0.ax').\ncnf(goal,negated_conjecture,~p(a)).")
        monkeypatch.setenv('TPTP', str(tptp_root))

        metadata = problem_metadata(problem)

        assert metadata.typed is True
        assert metadata.declared_status == 'Satisfiable'
        assert metadata.declared_status_kind == 'sat'
        assert metadata.spc_tags == frozenset({'CNF_SAT_RFO', 'TH0_SAT_EQU_NAR'})

    def test_file2cnf_rejects_problem_marked_typed_by_spc(self, tmp_path: Path):
        problem = tmp_path / 'SYN000-3.p'
        problem.write_text(
            "% Status   : Satisfiable\n"
            "% SPC      : TCF_SAT_RFO_EQU_NUE\n"
            "tcf(a,axiom, ! [X: $i] : ( p(X) | ~ q(a) )).\n"
        )

        with pytest.warns(UserWarning, match='unsupported typed problem file'):
            with pytest.raises(ValueError, match='unsupported typed problem file'):
                file2cnf(str(problem))

    @pytest.mark.parametrize(
        ('status', 'expected_kind'),
        [
            ('Theorem', 'unsat'),
            ('Unsatisfiable', 'unsat'),
            ('ContradictoryAxioms', 'unsat'),
            ('Satisfiable', 'sat'),
            ('CounterSatisfiable', 'sat'),
            ('Open', 'sat'),
            ('Unknown', 'unknown'),
            ('GaveUp', 'other'),
        ],
    )
    def test_problem_metadata_normalizes_declared_status_kind(
        self,
        tmp_path: Path,
        status: str,
        expected_kind: str,
    ):
        problem = tmp_path / 'FLD071-2.p'
        problem.write_text(f"% Status   : {status}\n% SPC      : CNF_UNS_RFO\ncnf(a,axiom,p(a)).\n")

        metadata = problem_metadata(problem)

        assert metadata.declared_status == status
        assert metadata.declared_status_kind == expected_kind
