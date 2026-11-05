# The append-only problems.csv that collect writes: one header, then one row per problem.

import csv
import json
import os
from pathlib import Path

from axiom_prediction.representation.schema import GraphInput

from .data import AxiomTrainingExample
from .graph import AxiomGraph

problem_schema = "learncop.axiom_prediction.problem.v3"
problem_fields = (
    "schema problem_name collection outcome parseable error "
    "graph axiom_clause_ids conjecture_clause_ids labels"
).split()

def problem_name(problem):
    return Path(problem).name

def failure_row(problem, outcome, *, parseable = True, error = False):
    return {
        "schema": problem_schema,
        "problem_name": problem_name(problem),
        "outcome": outcome,
        "parseable": parseable,
        "error": error,
    }

def example_row(problem, example):
    graph = example.graph
    return {
        "schema": problem_schema,
        "problem_name": problem_name(problem),
        "outcome": "proved",
        "graph": {"nodes": graph.graph.nodes, "edges": graph.graph.edges},
        "axiom_clause_ids": graph.axiom_clause_ids,
        "conjecture_clause_ids": graph.conjecture_clause_ids,
        "labels": example.labels,
    }

def example_from_row(row):
    axioms = row["axiom_clause_ids"]
    labels = list(map(float, row["labels"]))
    if len(labels) != len(axioms):
        raise ValueError("label count must match axiom clause count")

    graph = GraphInput(row["graph"]["nodes"], row["graph"]["edges"], [], "axiom_prediction", "1")
    return AxiomTrainingExample(
        problem_path = row["problem_name"],
        matrix = None,
        graph = AxiomGraph(graph, axioms, row["conjecture_clause_ids"]),
        labels = labels,
    )

def append_row(path, row):
    with path.open("a", encoding = "utf-8", newline = "") as stream:
        writer = csv.DictWriter(stream, fieldnames = problem_fields, lineterminator = "\n")
        writer.writerow({key: json.dumps(value, sort_keys = True) for key, value in row.items()})
        stream.flush()
        os.fsync(stream.fileno())

def check_header(path):
    with path.open(encoding = "utf-8", newline = "") as stream:
        header = next(csv.reader([stream.readline()]))

    if header != problem_fields:
        raise ValueError(f"{path} has an old or unknown header; collect into a new directory")

def repair_tail(path):
    # JSON-encoded cells keep every CSV record on one physical line.
    with path.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if end == 0:
            raise ValueError(f"empty problems file: {path}")

        stream.seek(end - 1)
        if stream.read(1) == b"\n":
            return

        while end > 0:
            start = max(0, end - 8192)
            stream.seek(start)
            chunk = stream.read(end - start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                stream.truncate(start + newline + 1)
                return

            end = start

        raise ValueError(f"problems file has no complete header: {path}")
