from pathlib import Path

import pytest

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

class TestProblemInputSplitting:
    def test_split_problem_paths_keeps_matching_zero_based_indices(self):
        paths = ['a.p', 'b.p', 'c.p', 'd.p', 'e.p']

        assert problem_inputs.split_problem_paths(paths, 2, 0) == ['a.p', 'c.p', 'e.p']
        assert problem_inputs.split_problem_paths(paths, 2, 1) == ['b.p', 'd.p']
        assert problem_inputs.split_problem_paths(paths, 3, 2) == ['c.p']

    @pytest.mark.parametrize(
        ('split', 'part'),
        [
            (0, 0),
            (-1, 0),
            (2, -1),
            (2, 2),
        ],
    )
    def test_validate_problem_split_rejects_invalid_values(self, split: int, part: int):
        with pytest.raises(ValueError):
            problem_inputs.validate_problem_split(split, part)

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
