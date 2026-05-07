from pathlib import Path
from types import SimpleNamespace

import pycop
from pycop import Logic, SATSolver

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
                logic = Logic.Classical,
                sat_solver = SATSolver.CaDiCaL,
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
                logic = Logic.Classical,
                sat_solver = SATSolver.CaDiCaL,
                debug = False,
                strategy = 'FirstAction',
                max_steps = 10,
                print_ratio = 0,
            ),
        )

        assert result['info'] == 'Theorem'
        assert result['steps'] == 1

class TestPycopLogicSelection:
    def test_default_logic_for_strategy_matches_strategy_family(self):
        assert pycop.default_logic_for_strategy('FirstAction') is Logic.Classical
        assert pycop.default_logic_for_strategy('LeanCopCut') is Logic.ClassicalCut
        assert pycop.default_logic_for_strategy('SATSolve') is Logic.ClassicalSAT
        assert pycop.default_logic_for_strategy('SATSolveLiteral') is Logic.ClassicalSAT
        assert pycop.default_logic_for_strategy('SATResetAtStaleAction') is Logic.ClassicalSAT
        assert pycop.default_logic_for_strategy('SATResetStepBudget') is Logic.ClassicalSAT

    def test_resolve_logic_for_strategy_ignores_explicit_override(self):
        assert pycop.resolve_logic_for_strategy('LeanCopCut', Logic.Classical) is Logic.ClassicalCut
        assert pycop.resolve_logic_for_strategy('FirstAction', Logic.ClassicalSAT) is Logic.Classical

    def test_run_problem_uses_strategy_default_logic(self, monkeypatch):
        seen = {}

        class FakeEnv:
            def __init__(self, path, settings):
                seen['logic'] = settings.logic
                self.state = SimpleNamespace(
                    initial_status = lambda: (True, 'Theorem'),
                    tableau = SimpleNamespace(children = [], literal = None),
                )

            def reset(self):
                return None

        class FakeStrategy:
            def __init__(self, env):
                self.env = env

        class FakeStrategyRegistry:
            @staticmethod
            def strategies():
                return {'LeanCopCut': FakeStrategy}

        monkeypatch.setattr(pycop, 'ConnectionEnv', FakeEnv)
        monkeypatch.setattr(pycop, 'Strategy', FakeStrategyRegistry)

        result = pycop._run_problem(
            'dummy.p',
            SimpleNamespace(
                logic = None,
                sat_solver = SATSolver.CaDiCaL,
                debug = False,
                strategy = 'LeanCopCut',
                max_steps = 10,
                print_ratio = 0,
            ),
        )

        assert seen['logic'] is Logic.ClassicalCut
        assert result['info'] == 'Theorem'

    def test_run_problem_ignores_explicit_logic_override(self, monkeypatch):
        seen = {}

        class FakeEnv:
            def __init__(self, path, settings):
                seen['logic'] = settings.logic
                self.state = SimpleNamespace(
                    initial_status = lambda: (True, 'Theorem'),
                    tableau = SimpleNamespace(children = [], literal = None),
                )

            def reset(self):
                return None

        class FakeStrategy:
            def __init__(self, env):
                self.env = env

        class FakeStrategyRegistry:
            @staticmethod
            def strategies():
                return {'LeanCopCut': FakeStrategy}

        monkeypatch.setattr(pycop, 'ConnectionEnv', FakeEnv)
        monkeypatch.setattr(pycop, 'Strategy', FakeStrategyRegistry)

        result = pycop._run_problem(
            'dummy.p',
            SimpleNamespace(
                logic = Logic.Classical,
                sat_solver = SATSolver.CaDiCaL,
                debug = False,
                strategy = 'LeanCopCut',
                max_steps = 10,
                print_ratio = 0,
            ),
        )

        assert seen['logic'] is Logic.ClassicalCut
        assert result['info'] == 'Theorem'

class TestPycopPrinting:
    def test_print_result_keeps_single_line_info_inline(self, capsys):
        pycop._print_result({'info': 'Theorem', 'steps': 1})

        captured = capsys.readouterr()

        assert captured.out == "{'info': 'Theorem', 'steps': 1}\n"

    def test_print_result_renders_multiline_info_as_block(self, capsys):
        pycop._print_result({'info': 'Theorem\n└── p(a)', 'steps': 1})

        captured = capsys.readouterr()

        assert captured.out == "{'steps': 1}\nTheorem\n└── p(a)\n"

    def test_print_result_keeps_problem_label_for_multiline_info(self, capsys):
        pycop._print_result({'info': 'Theorem\n└── p(a)', 'steps': 1}, 'SWV904-1.p')

        captured = capsys.readouterr()

        assert captured.out == "{'problem': 'SWV904-1.p', 'steps': 1}\nTheorem\n└── p(a)\n"
