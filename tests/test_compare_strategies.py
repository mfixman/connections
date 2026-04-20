from pathlib import Path
import sys

import pytest

import compare_strategies
from compare_strategies import Logic, ReportWriter, SATSolver

class TestCompareStrategiesPartialReport:
    def test_load_partial_report_recovers_colored_rows(self, tmp_path: Path):
        partial = tmp_path / 'partial.txt'
        partial.write_text(
            'Problem     Strategy     Result\n'
            '--------------------------------\n'
            f'SET027+3.p  FirstAction  {compare_strategies.green("1,234 / 56ms")}\n'
            f'SET027+3.p  SmartCut     {compare_strategies.bold("max-steps")}\n',
            encoding='utf-8',
        )

        problem = Path('SET027+3.p')
        state = compare_strategies.load_partial_report(
            partial,
            [problem],
            ['FirstAction', 'SmartCut'],
        )

        assert state.has_header is True
        assert state.has_summary is False
        assert state.has_csv_line is False
        assert state.results[problem]['FirstAction'] == {
            'status': 'solved',
            'solved': True,
            'steps': 1234,
            'time': 0.056,
        }
        assert state.results[problem]['SmartCut'] == {
            'status': 'max_steps',
            'solved': False,
        }

    def test_report_writer_strips_color_from_partial_by_default(self, tmp_path: Path, capsys):
        partial = tmp_path / 'partial.txt'
        writer = ReportWriter(partial, partial_color=False)

        writer.emit(compare_strategies.green('solved'))
        writer.close()

        captured = capsys.readouterr()
        assert '\033[32m' in captured.out
        assert partial.read_text(encoding='utf-8') == 'solved\n'

    def test_resumed_partial_matches_single_pass_output(self, tmp_path: Path, capsys):
        problem = Path('SET027+3.p')
        strategies = ['FirstAction', 'SmartCut']
        result1 = {'status': 'solved', 'solved': True, 'steps': 10, 'time': 0.012}
        result2 = {'status': 'max_steps', 'solved': False}
        problem_width, strategy_width, header, sep = compare_strategies.report_layout([problem], strategies)

        full_partial = tmp_path / 'full.txt'
        full_writer = ReportWriter(full_partial, partial_color=False)
        full_writer.emit(header)
        full_writer.emit(sep)
        full_writer.emit(compare_strategies.format_task_result(problem, 'FirstAction', result1, True, True, problem_width, strategy_width))
        full_writer.emit(compare_strategies.format_task_result(problem, 'SmartCut', result2, True, True, problem_width, strategy_width))
        full_writer.emit(sep)
        full_writer.emit('Solved: FirstAction=1/1  SmartCut=0/1  |  step wins: FirstAction=1  SmartCut=0  ties=0')
        full_writer.emit('CSV: /tmp/results.csv')
        full_writer.close()
        capsys.readouterr()

        resumed_partial = tmp_path / 'resumed.txt'
        resumed_partial.write_text(
            '\n'.join(
                [
                    header,
                    sep,
                    compare_strategies.strip_ansi(
                        compare_strategies.format_task_result(
                            problem, 'FirstAction', result1, True, True, problem_width, strategy_width
                        )
                    ),
                ]
            ) + '\n',
            encoding='utf-8',
        )

        state = compare_strategies.load_partial_report(resumed_partial, [problem], strategies)
        resumed_writer = ReportWriter(
            resumed_partial,
            partial_color=False,
            last_line=state.last_line,
        )
        resumed_writer.emit(compare_strategies.format_task_result(problem, 'SmartCut', result2, True, True, problem_width, strategy_width))
        if resumed_writer.last_line != sep:
            resumed_writer.emit(sep)
        resumed_writer.emit('Solved: FirstAction=1/1  SmartCut=0/1  |  step wins: FirstAction=1  SmartCut=0  ties=0')
        resumed_writer.emit('CSV: /tmp/results.csv')
        resumed_writer.close()
        capsys.readouterr()

        assert resumed_partial.read_text(encoding='utf-8') == full_partial.read_text(encoding='utf-8')

class TestCompareStrategiesLogicSelection:
    def test_default_logic_is_classical_cut_for_non_sat_strategies(self):
        assert compare_strategies.default_logic_for_strategy('FirstAction') is Logic.ClassicalCut
        assert compare_strategies.default_logic_for_strategy('SmartCut') is Logic.ClassicalCut

    def test_default_logic_is_classical_sat_for_sat_solve(self):
        assert compare_strategies.default_logic_for_strategy('SATSolve') is Logic.ClassicalSAT
        assert compare_strategies.default_logic_for_strategy('SATSolveGuided') is Logic.ClassicalSAT
        assert compare_strategies.default_logic_for_strategy('SATSolveGuidedFast') is Logic.ClassicalSAT

    def test_explicit_logic_override_wins(self):
        assert (
            compare_strategies.resolve_logic_for_strategy('SATSolve', Logic.ClassicalCut)
            is Logic.ClassicalCut
        )
        assert (
            compare_strategies.resolve_logic_for_strategy('FirstAction', Logic.ClassicalSAT)
            is Logic.ClassicalSAT
        )

    def test_parse_args_accepts_cadical_sat_solver(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['compare_strategies.py', '--sat-solver', 'cadical'])
        args = compare_strategies.parse_args()

        assert args.sat_solver == 'cadical'
        assert args.sat_solver is SATSolver.CaDiCaL

    def test_parse_args_defaults_sat_solver_to_cadical(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['compare_strategies.py'])
        args = compare_strategies.parse_args()

        assert args.sat_solver is SATSolver.CaDiCaL

    def test_parse_args_accepts_logic_enum(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['compare_strategies.py', '--logic', 'ClassicalSAT'])
        args = compare_strategies.parse_args()

        assert args.logic is Logic.ClassicalSAT

    def test_parse_args_accepts_split_and_part(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', '--split', '3', '--part', '1'],
        )
        args = compare_strategies.parse_args()

        assert args.split == 3
        assert args.part == 1

    def test_parse_args_rejects_zero_split(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['compare_strategies.py', '--split', '0'])

        with pytest.raises(SystemExit) as excinfo:
            compare_strategies.parse_args()

        assert excinfo.value.code == 2

class TestCompareStrategiesSatStatus:
    def test_theorem_sat_with_core_suffix_counts_as_success(self):
        assert compare_strategies._is_success_info('Theorem (SAT); (1, 2); (-1,)') is True

    def test_non_theorem_does_not_count_as_success(self):
        assert compare_strategies._is_success_info('Non-Theorem: no positive start clauses') is False

class TestCompareStrategiesInitialStatus:
    def _run_fake_problem(
        self,
        monkeypatch,
        *,
        initial_status = (False, None),
        step_info = 'Theorem (SAT)',
        strategy_info = None,
        max_steps = 100,
    ):
        sent = {}

        class FakeConn:
            def send(self, result):
                sent.update(result)

            def close(self):
                pass

        class FakeEnv:
            def __init__(self, path, settings):
                self.state = type(
                    'State',
                    (),
                    {
                        'initial_status': lambda self: initial_status,
                    },
                )()

            def reset(self):
                return None

            def step(self, action):
                return self.state, 1, True, step_info

        class FakeStrategy:
            def __init__(self, env):
                self.env = env

            def step(self):
                return object()

            def final_info(self):
                return strategy_info

        class FakeStrategyRegistry:
            @staticmethod
            def strategies():
                return {'SATSolve': FakeStrategy}

        monkeypatch.setattr(compare_strategies, 'ConnectionEnv', FakeEnv)
        monkeypatch.setattr(compare_strategies, 'Strategy', FakeStrategyRegistry)

        compare_strategies._run_problem(
            FakeConn(),
            'dummy.p',
            'SATSolve',
            Logic.ClassicalSAT,
            SATSolver.CaDiCaL,
            max_steps,
        )

        return sent

    def test_run_problem_counts_initial_status_as_zero_steps(self, monkeypatch):
        sent = self._run_fake_problem(
            monkeypatch,
            initial_status = (True, 'Theorem (SAT)'),
        )

        assert sent['status'] == 'solved'
        assert sent['solved'] is True
        assert sent['steps'] == 0
        assert sent['info'] == 'Theorem (SAT)'

    def test_run_problem_appends_strategy_final_info(self, monkeypatch):
        sent = self._run_fake_problem(
            monkeypatch,
            strategy_info = 'score=3.5',
        )

        assert sent['info'] == 'Theorem (SAT); score=3.5'

    def test_run_problem_keeps_strategy_final_info_on_max_steps(self, monkeypatch):
        sent = self._run_fake_problem(
            monkeypatch,
            strategy_info = 'score=3.5',
            max_steps = 0,
        )

        assert sent['status'] == 'max_steps'
        assert sent['info'] == 'score=3.5'
