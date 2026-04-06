from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

MODULE_PATH = Path(__file__).resolve().parents[1] / "schedule_from_table.py"
MODULE_SPEC = spec_from_file_location("mypycop_schedule_from_table", MODULE_PATH)
schedule_from_table = module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = schedule_from_table
MODULE_SPEC.loader.exec_module(schedule_from_table)

def write_table(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

class TestScheduleFromTable:
    def test_load_table_treats_non_numeric_as_unsolved(self, tmp_path: Path):
        table_path = tmp_path / "table.csv"
        write_table(
            table_path,
            [
                "problem,A,B",
                "P1,12,timeout",
                "P2,max_steps,7",
            ],
        )

        table = schedule_from_table.load_table(str(table_path))

        assert table.strategies == ("A", "B")
        assert table.problems == ("P1", "P2")
        assert table.steps_by_strategy["A"] == (12, None)
        assert table.steps_by_strategy["B"] == (None, 7)

    def test_find_best_schedule_can_use_three_experimental_slots(self, tmp_path: Path):
        table_path = tmp_path / "table.csv"
        write_table(
            table_path,
            [
                "problem,A,B,C",
                "P1,5,timeout,timeout",
                "P2,timeout,5,timeout",
                "P3,timeout,timeout,5",
                "P4,200,200,200",
            ],
        )

        table = schedule_from_table.load_table(str(table_path))
        result = schedule_from_table.find_best_schedule(
            table,
            total_budget=100,
            max_slots=3,
            shares=(50,),
            allow_repeat=False,
        )

        assert result.score.solved == 3
        assert {slot.strategy for slot in result.schedule} == {"A", "B", "C"}
        assert [slot.cap for slot in result.schedule] == [50, 25, 25]

    def test_include_and_exclude_filter_strategies(self, tmp_path: Path):
        table_path = tmp_path / "table.csv"
        write_table(
            table_path,
            [
                "problem,Control,Exp1,Exp2",
                "P1,3,10,timeout",
                "P2,timeout,4,timeout",
            ],
        )

        table = schedule_from_table.load_table(
            str(table_path),
            include=("Control", "Exp1", "Exp2"),
            exclude=("Control",),
        )

        assert table.strategies == ("Exp1", "Exp2")
