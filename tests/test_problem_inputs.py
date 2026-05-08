from pathlib import Path

import pytest

import cli as problem_inputs

class TestProblemInputsDirectoryFiltering:
    def test_expand_problem_inputs_skips_unsupported_directory_entries(self, tmp_path: Path):
        problems = tmp_path / 'Problems'
        supported = problems / 'AAA001-1.p'
        typed_supported_shape = problems / 'AAA002-1.p'
        unsupported_hat = problems / 'AAA001^1.p'
        unsupported_underscore = problems / 'AAA001_1.p'
        problems.mkdir()
        supported.write_text('% SPC      : CNF_UNS_RFO\nsupported')
        typed_supported_shape.write_text('% SPC      : TCF_SAT_RFO_EQU_NUE\ntyped')
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

    def test_split_problem_paths_parts_partition_expanded_directory_inputs(self, tmp_path: Path):
        problems_dir = tmp_path / 'Problems'
        problems_dir.mkdir()
        for name in ['AAA003-1.p', 'AAA001-1.p', 'AAA004-1.p', 'AAA002-1.p', 'AAA005-1.p']:
            (problems_dir / name).write_text(name)

        problems = problem_inputs.expand_problem_inputs([str(problems_dir)])
        part0 = problem_inputs.split_problem_paths(problems, 2, 0)
        part1 = problem_inputs.split_problem_paths(problems, 2, 1)

        assert set(part0).isdisjoint(part1)
        assert set(part0) | set(part1) == set(problems)
        assert part0 == problems[0::2]
        assert part1 == problems[1::2]

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
        typed_supported_shape = tptp_root / 'Problems' / 'SYN' / 'SYN002-1.p'
        unsupported = tptp_root / 'Problems' / 'SYN' / 'SYN001^1.p'
        supported.parent.mkdir(parents = True)
        supported.write_text('% SPC      : FOF_THM_RFO\nsupported')
        typed_supported_shape.write_text('% SPC      : TCF_SAT_RFO_EQU_NUE\ntyped')
        unsupported.write_text('unsupported')

        monkeypatch.setenv('TPTP', str(tptp_root))

        result = problem_inputs.expand_problem_inputs([])

        assert result == [supported.resolve()]

    def test_expand_problem_inputs_without_tptp_fails_loudly(self, monkeypatch):
        monkeypatch.delenv('TPTP', raising = False)

        with pytest.raises(FileNotFoundError, match='pass --tptp ROOT or set the TPTP environment variable'):
            problem_inputs.expand_problem_inputs([])

    def test_resolve_problem_path_without_tptp_fails_loudly(self, monkeypatch):
        monkeypatch.delenv('TPTP', raising = False)

        with pytest.raises(FileNotFoundError, match='pass --tptp ROOT or set the TPTP environment variable'):
            problem_inputs.resolve_problem_path('SYN001+1.p')
