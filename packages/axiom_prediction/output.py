import csv
from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import sys

journal = ContextVar("output_journal", default = None)
writer = ContextVar("output_writer", default = None)

def output_fields(command):
    metric_fields = (
        "bce roc_auc average_precision macro_average_precision macro_recall_at_1 "
        "macro_recall_at_3 macro_recall_at_5 macro_recall_at_10"
    ).split()

    dataset_fields = (
        "problems duplicate_problems problems_skipped skipped_outcomes batches_per_epoch "
        "axiom_clauses core_clauses positive_rate axioms_per_problem core_per_problem "
        "core_fraction_per_problem problems_without_axioms problems_with_empty_core "
        "problems_with_full_core graph_nodes_per_problem graph_edges_per_problem largest_problems"
    ).split()

    result_fields = (
        "evaluation_kind model_config network_size label_semantics sat_policy dataset split "
        "training_split problems_proved problems_skipped skipped"
    ).split()

    command_fields = {
        "predict": "problem_path clause_index clause_text probability rank network_size".split(),
        "collect": (
            "schema problems_requested problems_proved problems_failed problems_reused "
            "problems_unparseable dataset_shard"
        ).split(),
        "run": (
            "problem part tptp_status proved seconds mode policy seed steps proof_size "
            "axioms prediction_seconds error network_size"
        ).split(),
        "train": ["event", *dataset_fields, *metric_fields, *result_fields, *(
            "examples shards collection failures parameters device config epoch epochs batch batches "
            "loss seconds labels eval_seconds grad_norm_mean grad_norm_max metrics"
        ).split()],
        "evaluate": ["event", "problem", "part", "tptp_status", *dataset_fields, *metric_fields, *result_fields],
    }

    return list(dict.fromkeys(command_fields[command]))

def write_record(record):
    if journal.get() is not None:
        journal.get().write(record)
        return

    record = display_record(record)
    output = writer.get()
    if output is None:
        print(json.dumps(record, sort_keys = True, default = str), flush = True)
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
def output_format(command, use_csv, path = None, identity = None):
    if path is not None:
        from .output_resume import OutputJournal
        fields = output_fields(command)
        session = OutputJournal(path, command, use_csv, fields, identity)
        token = journal.set(session)
        try:
            yield session
        finally:
            journal.reset(token)
            session.close()

        return

    output = None
    if use_csv:
        fields = output_fields(command)
        output = csv.DictWriter(sys.stdout, fieldnames = fields, lineterminator = "\n")
        output.writeheader()
        sys.stdout.flush()

    token = writer.set(output)
    try:
        yield
    finally:
        writer.reset(token)
