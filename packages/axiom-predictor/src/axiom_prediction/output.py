import csv
from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import sys

METRIC_FIELDS = (
    "examples positives bce roc_auc average_precision precision_at_0.5 "
    "recall_at_0.5 f1_at_0.5 macro_average_precision macro_recall_at_1 "
    "macro_recall_at_3 macro_recall_at_5 macro_recall_at_10"
).split()

DATASET_FIELDS = (
    "problems duplicate_problems problems_skipped skipped_outcomes batches_per_epoch "
    "axiom_clauses core_clauses positive_rate axioms_per_problem core_per_problem "
    "core_fraction_per_problem problems_without_axioms problems_with_empty_core "
    "problems_with_full_core graph_nodes_per_problem graph_edges_per_problem largest_problems"
).split()

RESULT_FIELDS = (
    "evaluation_kind model_config label_semantics sat_policy dataset split "
    "training_split problems_proved problems_skipped skipped"
).split()

COMMAND_FIELDS = {
    "predict": "problem_path clause_index clause_text probability rank".split(),
    "collect": (
        "schema problems_requested problems_proved problems_failed problems_reused "
        "problems_unparseable dataset_shard"
    ).split(),
    "run": (
        "problem part tptp_status proved seconds mode policy seed steps proof_size "
        "axioms kept_axioms prediction_seconds guidance_fallback parseable error"
    ).split(),
    "train": ["event", *DATASET_FIELDS, *METRIC_FIELDS, *RESULT_FIELDS, *(
        "shards collection failures parameters device config epoch epochs batch batches "
        "loss seconds labels eval_seconds grad_norm_mean grad_norm_max metrics"
    ).split()],
    "evaluate": ["event", "problem", "part", "tptp_status", *DATASET_FIELDS, *METRIC_FIELDS, *RESULT_FIELDS],
}

journal = ContextVar("output_journal", default = None)
writer = ContextVar("output_writer", default = None)

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
    if not record.get("problem"):
        return record

    record = dict(record)
    problem = record["problem"]
    record.pop("outcome", None)
    record["problem"] = Path(str(problem)).stem
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
        fields = list(dict.fromkeys(COMMAND_FIELDS[command]))
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
        fields = list(dict.fromkeys(COMMAND_FIELDS[command]))
        output = csv.DictWriter(sys.stdout, fieldnames = fields, lineterminator = "\n")
        output.writeheader()
        sys.stdout.flush()

    token = writer.set(output)
    try:
        yield
    finally:
        writer.reset(token)
