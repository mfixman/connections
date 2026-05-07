import csv
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
            f'SET027+3.p  LeanCopCut  {compare_strategies.bold("max-steps")}\n',
            encoding='utf-8',
        )

        problem = Path('SET027+3.p')
        state = compare_strategies.load_partial_report(
            partial,
            [problem],
            ['FirstAction', 'LeanCopCut'],
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
        assert state.results[problem]['LeanCopCut'] == {
            'status': 'max_steps',
            'solved': False,
        }

    def test_load_partial_report_prefers_full_metadata_for_final_csv_fields(self, tmp_path: Path):
        partial = tmp_path / 'partial.txt'
        problem = Path('SET027+3.p')
        result = {
            'status': 'error',
            'solved': False,
            'error': 'ValueError: boom',
            'info': 'score=3.5',
            'steps': 17,
            'time': 0.25,
        }
        problem_width, strategy_width, header, sep = compare_strategies.report_layout([problem], ['LeanCopCut'])
        partial.write_text(
            '\n'.join(
                [
                    header,
                    sep,
                    compare_strategies.strip_ansi(
                        compare_strategies.format_task_result(
                            problem, 'LeanCopCut', result, True, True, problem_width, strategy_width
                        )
                    ),
                    compare_strategies._partial_result_line(problem, 'LeanCopCut', result),
                ]
            ) + '\n',
            encoding='utf-8',
        )

        state = compare_strategies.load_partial_report(partial, [problem], ['LeanCopCut'])

        assert state.results[problem]['LeanCopCut'] == result
        assert state.last_line == compare_strategies.strip_ansi(
            compare_strategies.format_task_result(
                problem, 'LeanCopCut', result, True, True, problem_width, strategy_width
            )
        )

    def test_load_partial_report_reads_inline_metadata_on_same_line(self, tmp_path: Path):
        partial = tmp_path / 'partial.txt'
        problem = Path('SET027+3.p')
        result = {
            'status': 'error',
            'solved': False,
            'error': 'ValueError: boom',
            'info': 'score=3.5',
            'steps': 17,
            'time': 0.25,
        }
        problem_width, strategy_width, header, sep = compare_strategies.report_layout([problem], ['LeanCopCut'])
        row = compare_strategies.strip_ansi(
            compare_strategies.format_task_result(
                problem, 'LeanCopCut', result, True, True, problem_width, strategy_width
            )
        )
        partial.write_text(
            '\n'.join(
                [
                    header,
                    sep,
                    compare_strategies._task_result_with_inline_metadata(problem, 'LeanCopCut', result, row),
                ]
            ) + '\n',
            encoding='utf-8',
        )

        state = compare_strategies.load_partial_report(partial, [problem], ['LeanCopCut'])

        assert state.results[problem]['LeanCopCut'] == result
        assert state.last_line == row

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
        strategies = ['FirstAction', 'LeanCopCut']
        result1 = {'status': 'solved', 'solved': True, 'steps': 10, 'time': 0.012}
        result2 = {'status': 'max_steps', 'solved': False}
        problem_width, strategy_width, header, sep = compare_strategies.report_layout([problem], strategies)

        full_partial = tmp_path / 'full.txt'
        full_writer = ReportWriter(full_partial, partial_color=False)
        full_writer.emit(header)
        full_writer.emit(sep)
        full_writer.emit_task_result(
            problem,
            'FirstAction',
            result1,
            compare_strategies.format_task_result(problem, 'FirstAction', result1, True, True, problem_width, strategy_width),
        )
        full_writer.emit_task_result(
            problem,
            'LeanCopCut',
            result2,
            compare_strategies.format_task_result(problem, 'LeanCopCut', result2, True, True, problem_width, strategy_width),
        )
        full_writer.emit(sep)
        full_writer.emit('Solved: FirstAction=1/1  LeanCopCut=0/1  |  step wins: FirstAction=1  LeanCopCut=0  ties=0')
        full_writer.emit('CSV: /tmp/results.csv')
        full_writer.close()
        capsys.readouterr()

        resumed_partial = tmp_path / 'resumed.txt'
        resumed_partial.write_text(
            '\n'.join(
                [
                    header,
                    sep,
                    compare_strategies._task_result_with_inline_metadata(
                        problem,
                        'FirstAction',
                        result1,
                        compare_strategies.strip_ansi(
                            compare_strategies.format_task_result(
                                problem, 'FirstAction', result1, True, True, problem_width, strategy_width
                            )
                        ),
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
        resumed_writer.emit_task_result(
            problem,
            'LeanCopCut',
            result2,
            compare_strategies.format_task_result(problem, 'LeanCopCut', result2, True, True, problem_width, strategy_width),
        )
        if resumed_writer.last_line != sep:
            resumed_writer.emit(sep)
        resumed_writer.emit('Solved: FirstAction=1/1  LeanCopCut=0/1  |  step wins: FirstAction=1  LeanCopCut=0  ties=0')
        resumed_writer.emit('CSV: /tmp/results.csv')
        resumed_writer.close()
        capsys.readouterr()

        assert resumed_partial.read_text(encoding='utf-8') == full_partial.read_text(encoding='utf-8')

    def test_cleanup_task_process_kills_worker_if_terminate_does_not_finish(self):
        class FakeConn:
            def poll(self):
                return False

        class FakeProcess:
            exitcode = None

            def __init__(self):
                self.terminate_calls = 0
                self.kill_calls = 0
                self.join_timeouts = []

            def is_alive(self):
                return self.kill_calls == 0

            def terminate(self):
                self.terminate_calls += 1

            def kill(self):
                self.kill_calls += 1

            def join(self, timeout=None):
                self.join_timeouts.append(timeout)

        proc = FakeProcess()
        result = compare_strategies._cleanup_task_process(FakeConn(), proc)

        assert result == {'status': 'timeout', 'solved': False}
        assert proc.terminate_calls == 1
        assert proc.kill_calls == 1
        assert proc.join_timeouts == [1.0, 1.0]

    def test_print_run_status_includes_paths_and_progress(self, capsys):
        compare_strategies.print_run_status(
            Path('partial/SYN_SYN_1'),
            'results/SYN/SYN_1',
            total_tasks=2990,
            completed_tasks=241,
            max_workers=4,
            timeout=120,
        )

        captured = capsys.readouterr()
        assert captured.out == ''
        assert 'partial=partial/SYN_SYN_1' in captured.err
        assert 'csv=results/SYN/SYN_1' in captured.err
        assert 'tasks=2990' in captured.err
        assert 'completed=241' in captured.err
        assert 'remaining=2749' in captured.err
        assert 'workers=4' in captured.err
        assert 'timeout=120s' in captured.err

    def test_write_results_csv_reduces_error_field_to_status_marker(self, tmp_path: Path):
        csv_path = tmp_path / 'results.csv'
        problem = Path('SET027+3.p')
        results = {
            problem: {
                'LeanCopCut': {
                    'status': 'error',
                    'solved': False,
                    'error': 'ValueError: boom',
                    'time': 0.25,
                }
            }
        }

        compare_strategies.write_results_csv(str(csv_path), [problem], ['LeanCopCut'], results)

        with open(csv_path, newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))

        assert rows == [{
            'problem': 'SET027+3.p',
            'strategy': 'LeanCopCut',
            'status': 'error',
            'result': '',
            'solved': 'False',
            'steps': '',
            'time_ms': '250',
            'problem_spc_tags': '',
            'problem_typed': '',
            'problem_declared_status': '',
            'problem_declared_status_kind': '',
            'info': '',
            'error': 'error',
        }]

    def test_load_partial_report_backfills_result_from_old_inline_info(self, tmp_path: Path):
        partial = tmp_path / 'partial.txt'
        problem = Path('SET027+3.p')
        old_result = {
            'status': 'solved',
            'solved': True,
            'steps': 17,
            'time': 0.25,
            'info': 'Theorem (SAT); score=3.5',
        }
        partial.write_text(
            compare_strategies._partial_result_line(problem, 'LeanCopCut', old_result) + '\n',
            encoding='utf-8',
        )

        state = compare_strategies.load_partial_report(partial, [problem], ['LeanCopCut'])

        assert state.results[problem]['LeanCopCut']['result'] == 'Theorem (SAT)'
        assert state.results[problem]['LeanCopCut']['info'] == 'score=3.5'

class TestCompareStrategiesLogicSelection:
    def test_default_logic_matches_strategy_family(self):
        assert compare_strategies.default_logic_for_strategy('FirstAction') is Logic.Classical
        assert compare_strategies.default_logic_for_strategy('LeanCopCut') is Logic.ClassicalCut
        assert compare_strategies.default_logic_for_strategy('SATSolve') is Logic.ClassicalSAT
        assert compare_strategies.default_logic_for_strategy('SATSolveLiteral') is Logic.ClassicalSAT
        assert compare_strategies.default_logic_for_strategy('SATResetAtTwoFixedStep') is Logic.ClassicalSAT
        assert compare_strategies.default_logic_for_strategy('SATResetShadow') is Logic.ClassicalSAT

    def test_default_sat_solver_is_inferred_from_strategy(self):
        assert compare_strategies.default_sat_solver_for_strategy('SATSolve') is SATSolver.CaDiCaL
        assert compare_strategies.default_sat_solver_for_strategy('SATSolveLiteral') is SATSolver.CaDiCaL

    def test_explicit_sat_solver_override_wins(self):
        assert (
            compare_strategies.resolve_sat_solver_for_strategy('SATSolveLiteral', SATSolver.CaDiCaL)
            is SATSolver.CaDiCaL
        )

    def test_explicit_logic_override_is_ignored(self):
        assert (
            compare_strategies.resolve_logic_for_strategy('SATSolve', Logic.ClassicalCut)
            is Logic.ClassicalSAT
        )
        assert (
            compare_strategies.resolve_logic_for_strategy('FirstAction', Logic.ClassicalSAT)
            is Logic.Classical
        )

    def test_determine_worker_count_uses_override_instead_of_cpu_detection(self, monkeypatch):
        monkeypatch.setenv('SLURM_CPUS_PER_TASK', '32')
        monkeypatch.setattr(compare_strategies, '_process_cpu_count', lambda: 64)

        assert compare_strategies.determine_worker_count(10, requested_workers=3) == 3

    def test_determine_worker_count_caps_override_at_total_tasks(self):
        assert compare_strategies.determine_worker_count(2, requested_workers=8) == 2

    def test_parse_args_accepts_cadical_sat_solver(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', '--sat-solver', 'cadical', '--csv', '/tmp/results.csv'],
        )
        args = compare_strategies.parse_args()

        assert args.sat_solver == 'cadical'
        assert args.sat_solver is SATSolver.CaDiCaL

    def test_parse_args_leaves_sat_solver_unset_for_implicit_strategy_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['compare_strategies.py', '--csv', '/tmp/results.csv'])
        args = compare_strategies.parse_args()

        assert args.sat_solver is None

    def test_parse_args_uses_large_compare_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['compare_strategies.py', '--csv', '/tmp/results.csv'])
        args = compare_strategies.parse_args()

        assert args.max_steps == 1000000
        assert args.timeout == 90.0
        assert args.num_workers is None

    def test_parse_args_accepts_num_workers_override(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', '--num-workers', '7', '--csv', '/tmp/results.csv'],
        )
        args = compare_strategies.parse_args()

        assert args.num_workers == 7

    def test_parse_args_rejects_non_positive_num_workers(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', '--num-workers', '0', '--csv', '/tmp/results.csv'],
        )

        with pytest.raises(SystemExit) as excinfo:
            compare_strategies.parse_args()

        assert excinfo.value.code == 2

    def test_parse_args_accepts_logic_enum(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', '--logic', 'ClassicalSAT', '--csv', '/tmp/results.csv'],
        )
        args = compare_strategies.parse_args()

        assert args.logic is Logic.ClassicalSAT

    def test_parse_args_expands_sat_control_group(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', '--strategies', 'SATControl', '--csv', '/tmp/results.csv'],
        )
        args = compare_strategies.parse_args()

        assert args.strategies == ['FirstAction', 'LeanCopCut']

    def test_parse_args_expands_sat_test_group(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', '--strategies', 'SATTest', '--csv', '/tmp/results.csv'],
        )
        args = compare_strategies.parse_args()

        assert args.strategies == [
            'SATSolve',
            'SATSolveLiteral',
            'SATResetAtStaleBacktrack',
            'SATResetAtStaleAction',
            'SATResetAtTwo',
            'SATResetAtTwoFixedStep',
            'SATResetStart',
            'SATResetShadow',
            'SATResetNoCut',
            'SATResetStepBudget',
        ]

    def test_parse_args_deduplicates_expanded_groups(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', '--strategies', 'SATSolve', 'SATTest', '--csv', '/tmp/results.csv'],
        )
        args = compare_strategies.parse_args()

        assert args.strategies.count('SATSolve') == 1

    def test_parse_args_accepts_split_and_part(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', '--split', '3', '--part', '1', '--csv', '/tmp/results.csv'],
        )
        args = compare_strategies.parse_args()

        assert args.split == 3
        assert args.part == 1

    def test_parse_args_requires_csv_without_name(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['compare_strategies.py'])

        with pytest.raises(SystemExit) as excinfo:
            compare_strategies.parse_args()

        assert excinfo.value.code == 2

    def test_parse_args_defaults_partial_file_without_name(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', 'tests/cnf_problems', '--csv', '/tmp/results.csv'],
        )
        args = compare_strategies.parse_args()

        assert args.partial_file == str(Path('partial') / 'cnf_problems')
        assert args.csv == '/tmp/results.csv'

    def test_parse_args_name_sets_partial_file_and_csv_for_nonzero_part(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', 'tests/cnf_problems', '--name', 'run', '--split', '4', '--part', '2'],
        )
        args = compare_strategies.parse_args()

        assert args.partial_file == str(Path('partial') / 'run_cnf_problems_2')
        assert args.csv == str(Path('results') / 'run' / 'cnf_problems_2.csv')

    def test_parse_args_name_sets_partial_file_and_csv_with_part_zero_suffix_when_split(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', 'tests/cnf_problems', '--name', 'run', '--split', '4', '--part', '0'],
        )
        args = compare_strategies.parse_args()

        assert args.partial_file == str(Path('partial') / 'run_cnf_problems_0')
        assert args.csv == str(Path('results') / 'run' / 'cnf_problems_0.csv')

    def test_parse_args_name_omits_part_suffix_when_unsplit(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            ['compare_strategies.py', 'tests/cnf_problems', '--name', 'run'],
        )
        args = compare_strategies.parse_args()

        assert args.partial_file == str(Path('partial') / 'run_cnf_problems')
        assert args.csv == str(Path('results') / 'run' / 'cnf_problems.csv')

    def test_parse_args_name_does_not_override_explicit_partial_file_or_csv(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'compare_strategies.py',
                'tests/cnf_problems',
                '--name',
                'run',
                '--partial-file',
                '/tmp/partial.txt',
                '--csv',
                '/tmp/results.csv',
            ],
        )
        args = compare_strategies.parse_args()

        assert args.partial_file == '/tmp/partial.txt'
        assert args.csv == '/tmp/results.csv'

    def test_parse_args_uses_slurm_array_when_split_and_part_are_absent(self, monkeypatch, capsys):
        monkeypatch.setenv('SLURM_ARRAY_TASK_COUNT', '3')
        monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '1')
        monkeypatch.setattr(sys, 'argv', ['compare_strategies.py', '--csv', '/tmp/results.csv'])

        args = compare_strategies.parse_args()

        captured = capsys.readouterr()
        assert args.split == 3
        assert args.part == 1
        assert 'SLURM_ARRAY_TASK_COUNT=3' in captured.err
        assert 'SLURM_ARRAY_TASK_ID=1' in captured.err

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
        path = 'dummy.p',
        initial_status = (False, None),
        step_info = 'Theorem (SAT)',
        strategy_info = None,
        max_steps = 100,
        timeout = None,
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
            path,
            'SATSolve',
            Logic.ClassicalSAT,
            SATSolver.CaDiCaL,
            max_steps,
            timeout,
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
        assert sent['result'] == 'Theorem (SAT)'
        assert 'info' not in sent

    def test_run_problem_appends_strategy_final_info(self, monkeypatch):
        sent = self._run_fake_problem(
            monkeypatch,
            strategy_info = 'score=3.5',
        )

        assert sent['result'] == 'Theorem (SAT)'
        assert sent['info'] == 'score=3.5'

    def test_run_problem_keeps_strategy_final_info_on_max_steps(self, monkeypatch):
        sent = self._run_fake_problem(
            monkeypatch,
            strategy_info = 'score=3.5',
            max_steps = 0,
        )

        assert sent['status'] == 'max_steps'
        assert sent['info'] == 'score=3.5'

    def test_run_problem_keeps_strategy_final_info_on_soft_timeout(self, monkeypatch):
        class Clock:
            def __init__(self):
                self.values = iter([0.0, 2.0, 2.0])

            def __call__(self):
                return next(self.values)

        monkeypatch.setattr(compare_strategies.time, 'perf_counter', Clock())
        sent = self._run_fake_problem(
            monkeypatch,
            strategy_info = 'literal=choices:1',
            step_info = None,
            timeout = 1.0,
        )

        assert sent['status'] == 'timeout'
        assert sent['steps'] == 0
        assert sent['info'] == 'literal=choices:1'

    def test_run_problem_warns_when_declared_satisfiable_is_proved_unsat(self, monkeypatch, tmp_path: Path):
        problem = tmp_path / 'SYN000-3.p'
        problem.write_text("% Status   : Satisfiable\n% SPC      : CNF_SAT_RFO\ncnf(a,axiom,p(a)).\n")

        sent = self._run_fake_problem(
            monkeypatch,
            path = str(problem),
            initial_status = (True, 'Theorem'),
        )

        assert sent['result'] == 'Theorem'
        assert 'WARNING: declared status=Satisfiable but prover found Theorem' in sent['info']

    def test_run_problem_infers_sat_solver_when_none_is_selected(self, monkeypatch):
        captured = {}

        class FakeConn:
            def send(self, result):
                captured['result'] = result

            def close(self):
                pass

        class FakeEnv:
            def __init__(self, path, settings):
                captured['sat_solver'] = settings.sat_solver
                self.state = type(
                    'State',
                    (),
                    {
                        'initial_status': lambda self: (True, 'Theorem (SAT)'),
                    },
                )()

            def reset(self):
                return None

        class FakeStrategy:
            sat_solver_backend = SATSolver.CaDiCaL

            def __init__(self, env):
                pass

        class FakeStrategyRegistry:
            @staticmethod
            def strategies():
                return {'SATSolveLiteral': FakeStrategy}

        monkeypatch.setattr(compare_strategies, 'ConnectionEnv', FakeEnv)
        monkeypatch.setattr(compare_strategies, 'Strategy', FakeStrategyRegistry)

        compare_strategies._run_problem(
            FakeConn(),
            'dummy.p',
            'SATSolveLiteral',
            Logic.ClassicalSAT,
            None,
            100,
            None,
        )

        assert captured['sat_solver'] is SATSolver.CaDiCaL
        assert captured['result']['status'] == 'solved'
