from __future__ import annotations

from .choices import ProverPolicy

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import gc
import hashlib
from itertools import chain
import json
import time
from pathlib import Path
from typing import Any

from axiom_prediction.io import read_csv, write_csv
from .problems_file import (
    append_row,
    check_header,
    example_from_row,
    example_row,
    failure_row,
    problem_fields,
    problem_name,
    problem_schema,
    repair_tail,
)
from connections.parsing.tptp import TPTPParseError

from .data import AxiomTrainingExample, axiom_training_example_from_json, axiom_training_example_to_json
from .logs import log
from .logs import progress as track_progress
from .parallel import determine_worker_count
from .search_workers import supervised_results
from .limits import default_collection_step_limit, default_collection_timeout_s
from .tptp import collect_proof_example, find_tptp_root

axiom_dataset_schema = "learncop.axiom_prediction.dataset.v2"
axiom_dataset_shard_schema = "learncop.axiom_prediction.dataset-shard.v2"
dataset_fields = (
    "schema record collection summary problem_path graph axiom_clause_ids "
    "conjecture_clause_ids labels axiom_clause_texts problem outcome parseable error"
).split()

outdated_dataset_hint = "datasets collected before schema v2 label SAT cores against a different clausification; delete and re-collect them"

class NoParseableProblemsError(RuntimeError):
    # Every problem in a requested directory failed TPTP parsing.
    pass

@dataclass(frozen = True, slots = True)
class CollectedAxiomProblem:
    problem: str
    example: AxiomTrainingExample | None
    outcome: str
    parseable: bool = True

@dataclass(frozen = True, slots = True)
class CollectedAxiomRecord:
    problem: str
    outcome: str
    proved: bool
    parseable: bool
    payload: dict[str, Any]

def requested_collection(
    output: Path,
    *,
    tptp_root: str | Path | None,
    step_limit: int,
    timeout_s: float,
    sat_policy: str,
    satcop_location: str | Path | None,
    seed: int,
) -> tuple[dict[str, Any], dict | None]:
    resolved_root = find_tptp_root(tptp_root)
    collection = {
        "sat_policy": ProverPolicy(sat_policy).wire_value(),
        "seed": seed,
        "step_limit": step_limit,
        "timeout_seconds": timeout_s,
        "tptp_root": None if resolved_root is None else str(resolved_root),
    }

    if ProverPolicy(sat_policy) != ProverPolicy.SatCoP:
        if satcop_location is not None:
            raise ValueError("--satcop-location requires --policy SatCoP")

        return collection, None

    from .satcop import native_binary, binary_identity, label_semantics

    binary = native_binary(satcop_location)
    collection.update(
        prover_backend = "rust-satcop",
        binary_sha256 = binary_identity(binary),
        label_semantics = label_semantics,
    )

    collection.pop("step_limit")
    log("native SATCoP uses the wall-clock timeout; --step-limit applies to Python policies only")
    return collection, {"binary": str(binary), "output_dir": str(output / "native")}

def finished_problems(path, collection):
    # Reading only these columns skips decoding the graphs of earlier results.
    columns = {"problem_name", "collection", "outcome", "parseable"}
    finished = {}
    for row in track_progress(read_csv(path, columns = columns), "reading finished problems"):
        if not finished and collection_settings(row["collection"]) != collection_settings(collection):
            raise ValueError(
                "dataset collection settings differ from the existing dataset: "
                f"{row['collection']!r} != {collection!r}"
            )

        finished[row["problem_name"]] = (row["outcome"], row.get("parseable", True))

    return finished

def open_problems_file(output, collection):
    for old in ("cache.csv", ".cache", "metadata.json", "examples"):
        if (output / old).exists():
            raise ValueError(f"{output} holds old collection results ({old}); collect into a new directory")

    path = output / "problems.csv"
    if not path.exists():
        write_csv(path, [], problem_fields)
        return path, {}

    repair_tail(path)
    check_header(path)
    return path, finished_problems(path, collection)

def require_unique_names(problems):
    counts = Counter(map(problem_name, problems))
    repeated = sorted(name for name, count in counts.items() if count > 1)
    if repeated:
        raise ValueError(f"problems in different directories share a file name: {repeated[:5]}")

def collect_axiom_dataset(
    problems: Sequence[str],
    *,
    output_dir: str | Path,

    tptp_root: str | Path | None = None,
    step_limit: int = default_collection_step_limit,
    timeout_s: float = default_collection_timeout_s,
    sat_policy: str = ProverPolicy.SatResetCoP,

    num_workers: int | None = None,
    progress: Callable[[int, int, int, int, str, str], None] | None = None,
    satcop_location: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    # Only the parent appends results; workers never write the shared file.
    log(f"preparing collection of {len(problems)} problems into {output_dir}")
    output = Path(output_dir)
    output.mkdir(parents = True, exist_ok = True)
    require_unique_names(problems)

    collection, native_options = requested_collection(
        output,
        tptp_root = tptp_root,
        step_limit = step_limit,
        timeout_s = timeout_s,
        sat_policy = sat_policy,
        satcop_location = satcop_location,
        seed = seed,
    )

    path, finished = open_problems_file(output, collection)

    proved = 0
    failed = 0
    reused = 0
    unparseable = 0
    total = len(problems)
    pending: list[str] = []
    last_report = time.monotonic()

    def report(problem: str, outcome: str):
        nonlocal last_report
        processed = proved + failed
        now = time.monotonic()
        if processed in (1, total) or now - last_report >= 60:
            log(
                f"[{processed}/{total}] {problem}: {outcome} "
                f"({proved} proved, {failed} failed, {reused} reused)"
            )

            last_report = now

        if progress is not None:
            progress(processed, total, proved, failed, problem, outcome)

    for problem in track_progress(problems, "checking finished problems", len(problems)):
        record = finished.get(problem_name(problem))
        if record is None:
            pending.append(problem)
            continue

        outcome, parseable = record
        proved += int(outcome == "proved")
        failed += int(outcome != "proved")
        unparseable += int(not parseable)
        reused += 1
        report(problem, f"{outcome} (reused)")

    workers = determine_worker_count(len(pending), num_workers)
    if pending:
        log(
            f"axiom-predictor: using {workers} worker "
            f"{'process' if workers == 1 else 'processes'} for "
            f"{len(pending)} unfinished problems"
        )

    for result in collect_problem_records_parallel(
        pending,
        tptp_root = tptp_root,

        step_limit = step_limit,
        timeout_s = timeout_s,
        sat_policy = sat_policy,
        num_workers = workers,
        native_options = native_options,
        seed = seed,
    ):
        append_row(path, {**result.payload, "collection": collection})
        if result.proved:
            proved += 1
        else:
            failed += 1

        unparseable += int(not result.parseable)
        report(result.problem, result.outcome)

    if total > 0 and unparseable == total:
        raise NoParseableProblemsError(f"none of the {total} .p files could be parsed as FOF or CNF")

    return {
        "schema": axiom_dataset_schema,
        "problems_requested": total,
        "problems_proved": proved,
        "problems_failed": failed,
        "problems_reused": reused,
        "problems_unparseable": unparseable,
        "dataset_file": str(path),
    }

def legacy_rows(directory, metadata, max_bytes = None):
    yield {
        "schema": axiom_dataset_shard_schema,
        "record": "metadata",
        "collection": metadata.get("collection", {}),
    }

    seen = set()
    for stage in ("examples", "failures", "partial/examples", "partial/failures"):
        for record in sorted((directory / stage).glob("*.json")):
            if max_bytes is not None and record.stat().st_size > max_bytes:
                continue

            payload = read_object(record)
            if record.name not in seen:
                seen.add(record.name)
                yield payload

def collect_problems_parallel(
    problems: Sequence[str],
    *,
    tptp_root: str | Path | None,
    step_limit: int,

    timeout_s: float,
    sat_policy: str,
    num_workers: int | None = None,
) -> Iterator[CollectedAxiomProblem]:
    workers = determine_worker_count(len(problems), num_workers)
    if workers == 0:
        return

    arguments = (
        (problem, tptp_root, step_limit, timeout_s, sat_policy)
        for problem in problems
    )

    for result in supervised_collection(
        collect_one_problem,
        arguments,
        workers = workers,
        timeout = timeout_s,
    ):
        yield result.get("collected") or CollectedAxiomProblem(
            result["problem"],
            None,
            result["outcome"],
        )

def collect_problem_records_parallel(
    problems: Sequence[str],
    *,
    tptp_root: str | Path | None,
    step_limit: int,

    timeout_s: float,
    sat_policy: str,
    num_workers: int | None = None,
    native_options: dict | None = None,
    seed: int = 0,
) -> Iterator[CollectedAxiomRecord]:
    workers = determine_worker_count(len(problems), num_workers)
    if workers == 0:
        return

    arguments = (
        (
            problem,
            tptp_root,
            step_limit,
            timeout_s,
            sat_policy,
            native_options,
            seed,
        )
        for problem in problems
    )

    for result in supervised_collection(
        collect_one_problem_record,
        arguments,
        workers = workers,
        timeout = timeout_s,
    ):
        yield result.get("collected") or interrupted_record(result)

def supervised_collection(function, arguments, *, workers, timeout):
    requests = ((args[0], function, args[1:]) for args in arguments)
    yield from supervised_results(
        collection_result,
        requests,
        workers = workers,
        timeout = timeout,
    )

def collection_result(problem, function, arguments):
    return {"collected": function(problem, *arguments)}

def interrupted_record(result):
    payload = failure_row(result["problem"], result["outcome"], error = bool(result.get("error", False)))
    return CollectedAxiomRecord(result["problem"], result["outcome"], False, True, payload)

def collect_one_problem_record(
    problem: str,
    tptp_root: str | Path | None,
    step_limit: int,
    timeout_s: float,
    sat_policy: str,
    native_options: dict | None = None,
    seed: int = 0,
) -> CollectedAxiomRecord:
    parseable = True
    collection_error = False
    try:
        options = {} if native_options is None else {"native_options": native_options}
        example, outcome = collect_proof_example(
            problem,
            tptp_root = tptp_root,
            step_limit = step_limit,
            timeout_s = timeout_s,
            sat_policy = sat_policy,
            seed = seed,
            **options,
        )
    except TPTPParseError as error:
        example = None
        parseable = False
        collection_error = True
        message = " ".join(str(error).split())
        outcome = f"{type(error).__name__}: {message}"
    except Exception as error:
        example = None
        collection_error = True
        message = " ".join(str(error).split())
        outcome = f"{type(error).__name__}: {message}"

    if example is None:
        payload = failure_row(problem, outcome, parseable = parseable, error = collection_error)
    else:
        payload = example_row(problem, example)

    return CollectedAxiomRecord(problem, outcome, example is not None, parseable, payload)

def collect_one_problem(
    problem: str,
    tptp_root: str | Path | None,
    step_limit: int,
    timeout_s: float,
    sat_policy: str,
) -> CollectedAxiomProblem:
    parseable = True
    try:
        example, outcome = collect_proof_example(
            problem,
            tptp_root = tptp_root,
            step_limit = step_limit,
            timeout_s = timeout_s,
            sat_policy = sat_policy,
        )
    except TPTPParseError as error:
        example = None
        parseable = False
        message = " ".join(str(error).split())
        outcome = f"{type(error).__name__}: {message}"
    except Exception as error:
        example = None
        message = " ".join(str(error).split())
        outcome = f"{type(error).__name__}: {message}"

    if example is not None:
        example = axiom_training_example_from_json(
            axiom_training_example_to_json(example)
        )

    return CollectedAxiomProblem(problem, example, outcome, parseable)

def load_axiom_dataset(
    path: str | Path,
    max_example_mb: float | None = None,
) -> tuple[list[AxiomTrainingExample], list[dict[str, str]], dict[str, Any]]:
    root = Path(path)
    log(f"discovering dataset records in {root}")
    max_bytes = None if max_example_mb is None else int(max_example_mb * 1e6)
    if max_bytes is not None:
        log(f"skipping dataset records larger than {max_example_mb} MB")

    if root.suffix == ".npz" or (root / "data.npz").exists():
        from .joined import load_joined

        return load_joined(root.parent if root.suffix == ".npz" else root, max_bytes)

    if root.is_file():
        return load_axiom_dataset_shards((root,), dataset_path = root, max_bytes = max_bytes)

    shards = sorted(root.glob("*.csv")) or sorted(root.glob("*.jsonl")) or collection_caches(root)
    if shards:
        legacy = (root / "metadata.json").exists() or (root / "examples").exists()
        if legacy and shards != [root / "cache.csv"]:
            raise ValueError(
                f"mixed dataset formats in {root}: separate CSV shards from legacy records"
            )

        return load_axiom_dataset_shards(shards, dataset_path = root, max_bytes = max_bytes)

    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"axiom dataset metadata not found: {metadata_path}")

    metadata = read_object(metadata_path)
    if metadata.get("schema") != axiom_dataset_schema:
        raise ValueError(
            f"unsupported axiom dataset {root} (schema {metadata.get('schema')!r}); {outdated_dataset_hint}"
        )

    examples_by_input = {}
    for record in track_progress(sorted((root / "examples").glob("*.json")), "loading examples"):
        add_core_example(
            examples_by_input,
            axiom_training_example_from_json(read_object(record)),
        )

    examples = list(core_examples(examples_by_input))

    failures: list[dict[str, str]] = []
    for record in track_progress(sorted((root / "failures").glob("*.json")), "loading failure records"):
        payload = read_object(record)
        failures.append(
            {
                "problem": str(payload.get("problem", "")),
                "outcome": str(payload.get("outcome", "failed")),
            }
        )

    if not examples:
        raise RuntimeError(f"axiom dataset contains no training examples: {root}")

    return examples, failures, metadata

def collection_caches(root):
    # Links to collection caches let training read shards before they finish.
    return sorted(
        path
        for path in root.iterdir()
        if any((path / name).exists() for name in ("cache.csv", "problems.csv", "metadata.json"))
    )

def read_dataset_rows(path, max_bytes = None):
    for name in ("cache.csv", "problems.csv"):
        if path.is_dir() and (path / name).exists():
            yield from read_csv(path / name, max_bytes)
            return

    if path.is_dir():
        yield from legacy_rows(path, read_object(path / "metadata.json"), max_bytes)
        return

    if path.suffix == ".jsonl":
        with path.open() as stream:
            lines = filter(str.strip, stream)
            if max_bytes is not None:
                lines = (line for line in lines if len(line) <= max_bytes)

            yield from map(json.loads, lines)

        return

    yield from read_csv(path, max_bytes)

def example_matrix(example):
    graph = example.graph
    return (
        graph.axiom_clause_ids,
        graph.conjecture_clause_ids,
        example.axiom_clause_texts,

        graph.graph.nodes,
        graph.graph.edges,
        graph.graph.actions,
        graph.graph.metadata.get("satcop", {}).get("matrix_sha256"),
    )

def matrix_digest(example):
    text = json.dumps(example_matrix(example), sort_keys = True)
    return hashlib.sha256(text.encode()).hexdigest()

def add_core_example(examples_by_input, example):
    # Distinct matrices of one problem, such as from different SAT policies, are separate inputs.
    cores = examples_by_input.setdefault((example.problem_path, matrix_digest(example)), {})
    core = frozenset(
        clause_id
        for clause_id, label in zip(example.graph.axiom_clause_ids, example.labels)
        if label
    )
    cores.setdefault(core, example)

def core_examples(examples_by_input):
    for cores in examples_by_input.values():
        yield from cores.values()

@contextmanager
def paused_gc():
    # Scanning the new objects takes over half of a load; freezing them spares training too.
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.freeze()
        if enabled:
            gc.enable()

@paused_gc()
def load_axiom_dataset_shards(
    shards: Sequence[Path],
    *,
    dataset_path: Path,
    max_bytes: int | None = None,
) -> tuple[list[AxiomTrainingExample], list[dict[str, str]], dict[str, Any]]:
    examples_by_input: dict[tuple[str, str], dict[frozenset[int], AxiomTrainingExample]] = {}
    failures_by_problem: dict[str, dict[str, str]] = {}
    proved: set[str] = set()
    formats: set[bool] = set()
    shard_collections: dict[str, dict[str, Any]] = {}

    for shard in track_progress(shards, "loading dataset shards", len(shards)):
        rows = iter(read_dataset_rows(shard, max_bytes))
        header = next(rows, None)
        if header is None:
            log(f"no results yet in {shard}")
            continue

        # Old rows hash their clause texts, so they would not deduplicate against new ones.
        formats.add(header.get("schema") == problem_schema)
        if len(formats) > 1:
            raise ValueError(f"{dataset_path} mixes old and new collect results; combine them with join")

        if header.get("schema") == problem_schema:
            rows = chain([header], rows)
        elif (
            header.get("schema") != axiom_dataset_shard_schema
            or header.get("record") != "metadata"
            or not isinstance(header.get("collection"), Mapping)
        ):
            raise ValueError(
                f"unsupported axiom dataset shard {shard} (schema {header.get('schema')!r}); {outdated_dataset_hint}"
            )

        shard_collections[shard.name] = dict(header["collection"])
        for row in track_progress(rows, f"reading {shard.name}"):
            problem = failed_problem(row)
            if problem is not None:
                if problem and problem not in proved:
                    failures_by_problem.setdefault(
                        problem,
                        {
                            "problem": problem,
                            "outcome": str(row.get("outcome", "failed")),
                        },
                    )

                continue

            example = row_example(row)
            proved.add(example.problem_path)
            failures_by_problem.pop(example.problem_path, None)
            add_core_example(examples_by_input, example)

    if not examples_by_input:
        raise RuntimeError(f"axiom dataset contains no training examples: {dataset_path}")

    metadata = {
        "schema": axiom_dataset_schema,
        "collection": merged_collection(shard_collections),
        "shards": list(map(str, shards)),
    }

    return (
        list(core_examples(examples_by_input)),
        list(failures_by_problem.values()),
        metadata,
    )

def failed_problem(row):
    if row.get("schema") == axiom_dataset_schema:
        return str(row.get("problem", ""))

    if row.get("schema") == problem_schema and "graph" not in row:
        return row["problem_name"]

    return None

def row_example(row):
    if row.get("schema") == problem_schema:
        return example_from_row(row)

    return axiom_training_example_from_json(row)

def merged_collection(collections):
    first = next(filter(None, collections.values()), {})
    settings = {
        json.dumps(collection_settings(collection), sort_keys = True)
        for collection in collections.values()
    }

    if len(settings) <= 1:
        return first

    return {**dataset_compatibility_settings(first), "source_shards": collections}

def collection_settings(collection):
    settings = {key: value for key, value in collection.items() if key != "tptp_root"}
    if settings.get("sat_policy") == "satcop" and settings.get("prover_backend") != "rust-satcop":
        settings["sat_policy"] = "satcopcon"

    if settings.get("seed") == 0:
        settings.pop("seed")

    return settings

def dataset_compatibility_settings(collection):
    return {
        key: value
        for key, value in collection_settings(collection).items()
        if key in {"sat_policy", "prover_backend", "label_semantics"}
    }

def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding = "utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON record must be an object: {path}")

    return dict(value)
