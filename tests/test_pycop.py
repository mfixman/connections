from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

PYCOP_PATH = Path(__file__).resolve().parents[1] / 'pycop.py'
PYCOP_SPEC = spec_from_file_location('mypycop_pycop', PYCOP_PATH)
pycop = module_from_spec(PYCOP_SPEC)
PYCOP_SPEC.loader.exec_module(pycop)

class TestPycopProblemResolution:
    def test_resolved_problem_paths_prefers_local_file(self, tmp_path: Path, monkeypatch):
        local_problem = tmp_path / 'SEU257+1.p'
        local_problem.write_text('local')

        tptp_root = tmp_path / 'tptp'
        remote_problem = tptp_root / 'Problems' / 'SEU' / 'SEU257+1.p'
        remote_problem.parent.mkdir(parents = True)
        remote_problem.write_text('remote')

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('TPTP', str(tptp_root))

        with pycop.resolved_problem_paths('SEU257+1.p') as paths:
            assert [Path(path).resolve() for path in paths] == [local_problem.resolve()]

    def test_resolved_problem_paths_expands_directory(self, tmp_path: Path, monkeypatch):
        problems = tmp_path / 'Problems'
        first = problems / 'AAA001-1.p'
        second = problems / 'AAA002-1.p'
        problems.mkdir()
        first.write_text('first')
        second.write_text('second')

        monkeypatch.chdir(tmp_path)

        with pycop.resolved_problem_paths(str(problems)) as paths:
            assert [Path(path).resolve() for path in paths] == [first.resolve(), second.resolve()]

    def test_resolved_problem_paths_finds_codename_under_tptp(self, tmp_path: Path, monkeypatch):
        tptp_root = tmp_path / 'tptp'
        problem = tptp_root / 'Problems' / 'SEU' / 'SEU257+1.p'
        problem.parent.mkdir(parents = True)
        problem.write_text('remote')

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('TPTP', str(tptp_root))

        with pycop.resolved_problem_paths('SEU257+1') as paths:
            assert [Path(path).resolve() for path in paths] == [problem.resolve()]

    def test_resolved_problem_paths_materializes_raw_tptp_text(self):
        raw_problem = "cnf(test,axiom,p(a)).\n"

        with pycop.resolved_problem_paths(raw_problem) as paths:
            assert len(paths) == 1
            temp_path = Path(paths[0])
            assert temp_path.exists()
            assert temp_path.name.endswith('-1.p')
            assert temp_path.read_text(encoding = 'utf-8') == raw_problem

        assert not temp_path.exists()

class TestPycopDebugOutput:
    def test_run_problem_shows_closed_tree_for_theorem_when_debug_enabled(self, monkeypatch):
        class FakeNode:
            def __str__(self):
                return '└── p(a)'

        class FakeEnv:
            def __init__(self, path, settings):
                self.state = SimpleNamespace(
                    initial_status = lambda: (False, None),
                    tableau = SimpleNamespace(children = [FakeNode()], literal = None),
                )

            def reset(self):
                return None

            def step(self, action):
                return None, 1, True, 'Theorem'

        class FakeStrategy:
            def __init__(self, env):
                self.env = env

            def step(self):
                return 'stub-action'

        class FakeStrategyRegistry:
            @staticmethod
            def strategies():
                return {'FirstAction': FakeStrategy}

        monkeypatch.setattr(pycop, 'ConnectionEnv', FakeEnv)
        monkeypatch.setattr(pycop, 'Strategy', FakeStrategyRegistry)

        result = pycop._run_problem(
            'dummy.p',
            SimpleNamespace(
                logic = 'Classical',
                domain = 'Constant',
                sat_solver = 'pycosat',
                debug = True,
                strategy = 'FirstAction',
                max_steps = 10,
                print_ratio = 0,
            ),
        )

        assert result['info'] == 'Theorem\n└── p(a)'
        assert result['steps'] == 1

    def test_run_problem_keeps_plain_theorem_without_debug(self, monkeypatch):
        class FakeNode:
            def __str__(self):
                return '└── p(a)'

        class FakeEnv:
            def __init__(self, path, settings):
                self.state = SimpleNamespace(
                    initial_status = lambda: (False, None),
                    tableau = SimpleNamespace(children = [FakeNode()], literal = None),
                )

            def reset(self):
                return None

            def step(self, action):
                return None, 1, True, 'Theorem'

        class FakeStrategy:
            def __init__(self, env):
                self.env = env

            def step(self):
                return 'stub-action'

        class FakeStrategyRegistry:
            @staticmethod
            def strategies():
                return {'FirstAction': FakeStrategy}

        monkeypatch.setattr(pycop, 'ConnectionEnv', FakeEnv)
        monkeypatch.setattr(pycop, 'Strategy', FakeStrategyRegistry)

        result = pycop._run_problem(
            'dummy.p',
            SimpleNamespace(
                logic = 'Classical',
                domain = 'Constant',
                sat_solver = 'pycosat',
                debug = False,
                strategy = 'FirstAction',
                max_steps = 10,
                print_ratio = 0,
            ),
        )

        assert result['info'] == 'Theorem'
        assert result['steps'] == 1
