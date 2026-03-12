from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


TRACE_PATH = Path(__file__).resolve().parents[1] / 'trace.py'
TRACE_SPEC = spec_from_file_location('mypycop_trace', TRACE_PATH)
trace = module_from_spec(TRACE_SPEC)
TRACE_SPEC.loader.exec_module(trace)


class TestTraceProblemResolution:
    def test_resolve_problem_path_prefers_local_file(self, tmp_path: Path, monkeypatch):
        local_problem = tmp_path / 'SEU257+1.p'
        local_problem.write_text('local')

        tptp_root = tmp_path / 'tptp'
        remote_problem = tptp_root / 'Problems' / 'SEU' / 'SEU257+1.p'
        remote_problem.parent.mkdir(parents=True)
        remote_problem.write_text('remote')

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('TPTP', str(tptp_root))

        resolved = trace.resolve_problem_path('SEU257+1.p')

        assert Path(resolved).resolve() == local_problem.resolve()

    def test_resolve_problem_path_finds_codename_under_tptp(self, tmp_path: Path, monkeypatch):
        tptp_root = tmp_path / 'tptp'
        problem = tptp_root / 'Problems' / 'SEU' / 'SEU257+1.p'
        problem.parent.mkdir(parents=True)
        problem.write_text('remote')

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('TPTP', str(tptp_root))

        resolved = trace.resolve_problem_path('SEU257+1')

        assert Path(resolved).resolve() == problem.resolve()

    def test_resolve_problem_path_finds_filename_under_tptp(self, tmp_path: Path, monkeypatch):
        tptp_root = tmp_path / 'tptp'
        problem = tptp_root / 'Problems' / 'SEU' / 'SEU257+1.p'
        problem.parent.mkdir(parents=True)
        problem.write_text('remote')

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('TPTP', str(tptp_root))

        resolved = trace.resolve_problem_path('SEU257+1.p')

        assert Path(resolved).resolve() == problem.resolve()

    def test_resolve_problem_path_rejects_ambiguous_tptp_match(self, tmp_path: Path, monkeypatch):
        tptp_root = tmp_path / 'tptp'
        first = tptp_root / 'Problems' / 'SEU' / 'SEU257+1.p'
        second = tptp_root / 'Problems' / 'ALT' / 'SEU257+1.p'
        first.parent.mkdir(parents=True)
        second.parent.mkdir(parents=True)
        first.write_text('first')
        second.write_text('second')

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('TPTP', str(tptp_root))

        with pytest.raises(FileNotFoundError, match='ambiguous'):
            trace.resolve_problem_path('SEU257+1')
