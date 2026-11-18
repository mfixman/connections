# Result rows: CSV on stdout, or the resumable journal that --output selects.

import csv
from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import sys

journal = ContextVar("output_journal", default = None)
writer = ContextVar("output_writer", default = None)

def output_fields(command):
    fields = {
        "collect": "schema problems_requested problems_proved problems_failed problems_reused problems_unparseable dataset_file",
        "join": "inputs problems graphs examples failures data_bytes dataset",
        "run": "problem part tptp_status proved seconds mode policy seed steps proof_size axioms prediction_seconds error network_size",
    }

    return fields[command].split()

def write_record(record):
    if journal.get() is not None:
        journal.get().write(record)
        return

    record = display_record(record)
    output = writer.get()
    if output is None:
        print(record, file = sys.stderr, flush = True)
    else:
        output.writerow({key: csv_value(value) for key, value in record.items()})
        sys.stdout.flush()

def display_record(record):
    record = dict(record)
    if record.get("problem_path"):
        record["problem_path"] = Path(str(record["problem_path"])).name

    if not record.get("problem"):
        return record

    problem = record["problem"]
    record.pop("outcome", None)
    record["problem"] = Path(str(problem)).name
    record["tptp_status"] = problem_status(problem)
    return record

def problem_status(problem):
    from .tptp import declared_tptp_status

    try:
        return declared_tptp_status(problem) or ""
    except OSError:
        return ""

def report_metrics(metrics):
    session = journal.get()
    if session is not None:
        session.metrics = metrics
    else:
        print(json.dumps(metrics, sort_keys = True, default = str), file = sys.stderr)

def csv_value(value):
    return value if isinstance(value, str) else json.dumps(value, default = str)

@contextmanager
def output_format(command, path = None, identity = None):
    if path is not None:
        from .output_resume import OutputJournal
        fields = output_fields(command)
        session = OutputJournal(path, fields, identity)
        token = journal.set(session)
        try:
            yield session
        finally:
            journal.reset(token)
            session.close()

        return

    fields = output_fields(command)
    output = csv.DictWriter(sys.stdout, fieldnames = fields, lineterminator = "\n")
    output.writeheader()
    sys.stdout.flush()

    token = writer.set(output)
    try:
        yield
    finally:
        writer.reset(token)
