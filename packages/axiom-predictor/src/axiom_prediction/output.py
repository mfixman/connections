import csv
from contextlib import contextmanager
from contextvars import ContextVar
import json
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
        "event problem part outcome proved seconds mode policy seed steps proof_size "
        "axioms kept_axioms prediction_seconds guidance_fallback parseable error "
        "problems proved_seconds_total proved_seconds_mean"
    ).split(),
    "train": ["event", *DATASET_FIELDS, *METRIC_FIELDS, *RESULT_FIELDS, *(
        "shards collection failures parameters device config epoch epochs batch batches "
        "loss seconds labels eval_seconds grad_norm_mean grad_norm_max metrics"
    ).split()],
    "evaluate": ["event", "problem", "part", "outcome", *DATASET_FIELDS, *METRIC_FIELDS, *RESULT_FIELDS],
}

writer = ContextVar("output_writer", default = None)

def write_record(record):
    output = writer.get()
    if output is None:
        print(json.dumps(record, sort_keys = True, default = str), flush = True)
    else:
        output.writerow({key: csv_value(value) for key, value in record.items()})
        sys.stdout.flush()

def csv_value(value):
    return value if isinstance(value, str) else json.dumps(value, default = str)

@contextmanager
def output_format(command, use_csv):
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
