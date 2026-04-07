from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

COMPARE_PATH = Path(__file__).resolve().parents[1] / 'compare_strategies.py'
COMPARE_SPEC = spec_from_file_location('mypycop_compare_strategies', COMPARE_PATH)
compare_strategies = module_from_spec(COMPARE_SPEC)
COMPARE_SPEC.loader.exec_module(compare_strategies)

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
        writer = compare_strategies.ReportWriter(partial, partial_color=False)

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
        full_writer = compare_strategies.ReportWriter(full_partial, partial_color=False)
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
        resumed_writer = compare_strategies.ReportWriter(
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
        assert compare_strategies.default_logic_for_strategy('FirstAction') == 'ClassicalCut'
        assert compare_strategies.default_logic_for_strategy('SmartCut') == 'ClassicalCut'

    def test_default_logic_is_classical_sat_for_sat_solve(self):
        assert compare_strategies.default_logic_for_strategy('SATSolve') == 'ClassicalSAT'

    def test_explicit_logic_override_wins(self):
        assert compare_strategies.resolve_logic_for_strategy('SATSolve', 'ClassicalCut') == 'ClassicalCut'
        assert compare_strategies.resolve_logic_for_strategy('FirstAction', 'ClassicalSAT') == 'ClassicalSAT'

    def test_parse_args_accepts_sat_solver(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['compare_strategies.py', '--sat-solver', 'pysat'])
        args = compare_strategies.parse_args()

        assert args.sat_solver == 'pysat'
