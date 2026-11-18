import csv
import io

import pytest

from axiom_prediction.output import output_fields, output_format, write_record
from axiom_prediction.output_resume import OutputJournal

def open_journal(path, identity = None):
    return OutputJournal(path, output_fields("run"), identity or {"command": "run"})

def test_journal_repairs_torn_tail_and_preserves_multiline(tmp_path, capsys):
    path = tmp_path / "results"
    row = {"problem": 'a,b\n"c"', "proved": False, "seconds": 1.25, "outcome": "Timeout"}
    with output_format("run", path, {"command": "run"}):
        write_record(row)

    committed = path.read_bytes()
    with path.open("a") as f:
        f.write('"unfinished\nquoted')

    journal = open_journal(path)
    assert journal.records[0]["problem"] == row["problem"]
    assert journal.records[0]["proved"] is False
    assert journal.records[0]["seconds"] == 1.25
    assert journal.records[0]["outcome"] == "Timeout"
    journal.write(row)
    journal.close()

    assert path.read_bytes() == committed
    assert capsys.readouterr().out == ""

def test_journal_refuses_mixed_configuration_and_concurrent_writers(tmp_path):
    path = tmp_path / "results"
    first = open_journal(path)
    with pytest.raises(BlockingIOError):
        open_journal(path)

    first.close()
    with pytest.raises(ValueError, match = "different command"):
        open_journal(path, identity = {"command": "different"})

def test_non_csv_output_is_rejected_without_changes(tmp_path):
    path = tmp_path / "results"
    original = '{"problem": "a.p", "proved": true}\n'
    path.write_text(original)
    with pytest.raises(ValueError, match = "CSV header"):
        open_journal(path)

    assert path.read_text() == original

def test_completion_marker_rejects_truncated_output(tmp_path):
    path = tmp_path / "results"
    journal = open_journal(path)
    journal.write({"problem": "a.p", "proved": True})
    journal.metrics = {"problems": 1}
    journal.finish(0)
    assert journal.complete()
    journal.close()
    path.write_text("")

    journal = open_journal(path)
    assert not journal.complete()
    journal.close()

def test_problem_names_preserve_distinct_paths_for_resume(tmp_path):
    path = tmp_path / "results"
    problems = ["/first/SYN001-1.p", "/second/SYN001-1.p"]
    journal = open_journal(path)
    for problem in problems:
        journal.write({"problem": problem, "proved": True})

    journal.close()
    contents = path.read_text()
    rows = list(csv.DictReader(io.StringIO(contents)))

    assert [row["problem"] for row in rows] == ["SYN001-1.p", "SYN001-1.p"]
    journal = open_journal(path)
    assert [row["problem"] for row in journal.records] == problems
    for problem in problems:
        journal.write({"problem": problem, "proved": True})

    journal.close()
    assert path.read_text() == contents

def test_run_resume_rejects_changed_model(tmp_path):
    path = tmp_path / "results"
    identity = {"command": "run", "model": "epoch-0001.pt", "policy": "SatResetCoP"}
    first = open_journal(path, identity = identity)
    first.close()
    with pytest.raises(ValueError, match = "different command/configuration"):
        open_journal(path, identity = {**identity, "model": "epoch-0002.pt"})

@pytest.mark.parametrize("errors, expected", [([True, False], 0), ([True, True], 2), ([False], 0)])
def test_run_exit_status_includes_saved_results(tmp_path, errors, expected):
    path = tmp_path / "results.csv"
    journal = open_journal(path)
    for index, error in enumerate(errors):
        journal.write({"problem": f"{index}.p", "error": error, "proved": False})

    assert journal.exit_code() == expected
    journal.finish(expected)
    journal.close()
    journal = open_journal(path)
    assert journal.complete()
    assert journal.exit_code() == expected
    journal.close()
