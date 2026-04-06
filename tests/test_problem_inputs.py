from pathlib import Path

import problem_inputs

class TestProblemInputsDirectoryFiltering:
    def test_expand_problem_inputs_skips_unsupported_directory_entries(self, tmp_path: Path):
        problems = tmp_path / 'Problems'
        supported = problems / 'AAA001-1.p'
        unsupported_hat = problems / 'AAA001^1.p'
        unsupported_underscore = problems / 'AAA001_1.p'
        problems.mkdir()
        supported.write_text('supported')
        unsupported_hat.write_text('unsupported')
        unsupported_underscore.write_text('unsupported')

        result = problem_inputs.expand_problem_inputs([str(problems)])

        assert result == [supported.resolve()]

    def test_expand_problem_inputs_filters_default_tptp_listing(self, tmp_path: Path, monkeypatch):
        tptp_root = tmp_path / 'tptp'
        supported = tptp_root / 'Problems' / 'SYN' / 'SYN001+1.p'
        unsupported = tptp_root / 'Problems' / 'SYN' / 'SYN001^1.p'
        supported.parent.mkdir(parents = True)
        supported.write_text('supported')
        unsupported.write_text('unsupported')

        monkeypatch.setenv('TPTP', str(tptp_root))

        result = problem_inputs.expand_problem_inputs([])

        assert result == [supported.resolve()]
