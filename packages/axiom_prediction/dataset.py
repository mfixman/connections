from __future__ import annotations

from .choices import ProverPolicy

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from axiom_prediction.io import read_csv, write_csv
from .collection_cache import append_cache, repair_cache
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
    # Only the parent appends results; workers never write the shared cache.
    log(f"preparing collection of {len(problems)} problems into {output_dir}")
    output = Path(output_dir)
    output.mkdir(parents = True, exist_ok = True)
    cache = output / "cache.csv"

    collection = {
        "sat_policy": ProverPolicy(sat_policy).wire_value(),
        "seed": seed,
        "step_limit": step_limit,
        "timeout_seconds": timeout_s,
        "tptp_root": (
            None if (resolved_root := find_tptp_root(tptp_root)) is None else str(
                resolved_root
            )
        ),
    }

    native_options = None
    if ProverPolicy(sat_policy) == ProverPolicy.SatCoP:
        from .satcop import native_binary, binary_identity, label_semantics

        binary = native_binary(satcop_location)
        native_options = {"binary": str(binary), "output_dir": str(output / "native")}
        collection.update(
            prover_backend = "rust-satcop",
            binary_sha256 = binary_identity(binary),
            label_semantics = label_semantics,
        )
        collection.pop("step_limit")
        log("native SATCoP uses the wall-clock timeout; --step-limit applies to Python policies only")
    elif satcop_location is not None:
        raise ValueError("--satcop-location requires --policy SatCoP")

    if cache.exists():
        repair_cache(cache)
        metadata = next(read_csv(cache))
        if metadata.get("schema") != axiom_dataset_shard_schema:
            raise ValueError(f"unsupported collection cache: {cache}")
    elif (output / "metadata.json").exists():
        metadata = read_object(output / "metadata.json")
        if metadata.get("schema") != axiom_dataset_schema:
            raise ValueError(
                f"unsupported axiom dataset {output}; {outdated_dataset_hint}"
            )
    else:
        metadata = {"collection": collection}

    if collection_settings(metadata.get("collection", {})) != collection_settings(collection):
        raise ValueError(
            "dataset collection settings differ from the existing dataset: "
            f"{metadata.get('collection')!r} != {collection!r}"
        )

    if not cache.exists():
        write_csv(
            cache,
            shard_rows(output, metadata = metadata, summary = {}),
            dataset_fields,
        )

    cached = {}
    rows = iter(read_csv(cache))
    next(rows)
    for row in track_progress(rows, "reading collection cache"):
        if row.get("schema") == axiom_dataset_schema:
            cached.setdefault(row["problem"], (False, failure_is_parseable(row), row["outcome"]))
        else:
            example = axiom_training_example_from_json(row)
            cached[example.problem_path] = (True, True, "proved")

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
                f"({proved} proved, {failed} failed, {reused} cached)"
            )

            last_report = now

        if progress is not None:
            progress(processed, total, proved, failed, problem, outcome)

    for problem in track_progress(problems, "checking collection cache", len(problems)):
        record = cached.get(problem)
        if record is None:
            pending.append(problem)
            continue

        success, parseable, outcome = record
        proved += int(success)
        failed += int(not success)
        unparseable += int(not parseable)
        reused += 1
        outcome = f"{outcome} (cached)"
        report(problem, outcome)

    workers = determine_worker_count(len(pending), num_workers)
    if pending:
        log(
            f"axiom-predictor: using {workers} worker "
            f"{'process' if workers == 1 else 'processes'} for "
            f"{len(pending)} uncached problems"
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
        append_cache(cache, result.payload, dataset_fields)
        if result.proved:
            proved += 1
        else:
            failed += 1

        unparseable += int(not result.parseable)
        report(result.problem, result.outcome)

    summary: dict[str, Any] = {
        "schema": axiom_dataset_schema,
        "problems_requested": total,
        "problems_proved": proved,
        "problems_failed": failed,
        "problems_reused": reused,
        "problems_unparseable": unparseable,
    }

    return summary

def collect_axiom_dataset_shard(
    problems: Sequence[str],
    *,
    output_dir: str | Path,
    shard_name: str,
    tptp_root: str | Path | None = None,

    step_limit: int = default_collection_step_limit,
    timeout_s: float = default_collection_timeout_s,
    sat_policy: str = ProverPolicy.SatResetCoP,
    num_workers: int | None = None,
    satcop_location: str | Path | None = None,
    seed: int = 0,
) -> tuple[Path, dict[str, Any]]:
    # Resume from the append-only cache and publish the finished shard atomically.

    if not shard_name or Path(shard_name).name != shard_name:
        raise ValueError(f"invalid axiom dataset shard name: {shard_name!r}")

    output = Path(output_dir)
    cache = output / ".cache" / shard_name
    summary = collect_axiom_dataset(
        problems,
        output_dir = cache,
        tptp_root = tptp_root,

        step_limit = step_limit,
        timeout_s = timeout_s,
        sat_policy = sat_policy,
        num_workers = num_workers,
        satcop_location = satcop_location,
        seed = seed,
    )

    if summary.get(
        "problems_requested",
        0,
    ) > 0 and summary.get("problems_unparseable") == summary.get("problems_requested"):
        raise NoParseableProblemsError(
            f"none of the {summary['problems_requested']} .p files in "
            f"{shard_name!r} could be parsed as FOF or CNF"
        )

    metadata = next(read_csv(cache / "cache.csv"))
    shard = output / f"{shard_name}.csv"
    log(f"writing dataset shard {shard}")
    write_csv(shard, shard_rows(cache, metadata = metadata, summary = summary), dataset_fields)
    log(f"saved dataset shard {shard}")
    return shard, summary

def shard_rows(
    cache: Path,
    *,
    metadata: Mapping[str, Any],
    summary: Mapping[str, object],
    max_bytes: int | None = None,
) -> Iterator[Mapping[str, Any]]:
    yield {
        "schema": axiom_dataset_shard_schema,
        "record": "metadata",
        "collection": metadata.get("collection", {}),
        "summary": dict(summary),
    }

    if (cache / "cache.csv").exists():
        rows = iter(read_csv(cache / "cache.csv", max_bytes))
        next(rows)
        yield from rows
        return

    seen = set()
    for stage in ("examples", "failures", "partial/examples", "partial/failures"):
        for record in sorted((cache / stage).glob("*.json")):
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
    payload = {
        "schema": axiom_dataset_schema,
        "problem": result["problem"],
        "outcome": result["outcome"],
        "parseable": True,
        "error": bool(result.get("error", False)),
    }

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
        payload = {
            "schema": axiom_dataset_schema,
            "problem": problem,
            "outcome": outcome,
            "parseable": parseable,
            "error": collection_error,
        }
    else:
        payload = axiom_training_example_to_json(example)

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
        if (path / "cache.csv").exists() or (path / "metadata.json").exists()
    )

def read_dataset_rows(path, max_bytes = None):
    if path.is_dir() and (path / "cache.csv").exists():
        yield from read_csv(path / "cache.csv", max_bytes)
        return

    if path.is_dir():
        metadata = read_object(path / "metadata.json")
        yield from shard_rows(path, metadata = metadata, summary = {}, max_bytes = max_bytes)
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

def load_axiom_dataset_shards(
    shards: Sequence[Path],
    *,
    dataset_path: Path,
    max_bytes: int | None = None,
) -> tuple[list[AxiomTrainingExample], list[dict[str, str]], dict[str, Any]]:
    examples_by_input: dict[tuple[str, str], dict[frozenset[int], AxiomTrainingExample]] = {}
    failures_by_problem: dict[str, dict[str, str]] = {}
    proved: set[str] = set()
    collection: dict[str, Any] | None = None
    shard_collections: dict[str, dict[str, Any]] = {}

    for shard in track_progress(shards, "loading dataset shards", len(shards)):
        rows = iter(read_dataset_rows(shard, max_bytes))
        try:
            header = next(rows)
        except StopIteration:
            raise ValueError(f"empty axiom dataset shard: {shard}") from None

        if (
            header.get("schema") != axiom_dataset_shard_schema
            or header.get("record") != "metadata"
            or not isinstance(header.get("collection"), Mapping)
        ):
            raise ValueError(
                f"unsupported axiom dataset shard {shard} (schema {header.get('schema')!r}); {outdated_dataset_hint}"
            )

        shard_collection = dict(header["collection"])
        shard_collections[shard.name] = shard_collection
        collection = collection or shard_collection
        for row in track_progress(rows, f"reading {shard.name}"):
            if row.get("schema") == axiom_dataset_schema:
                problem = str(row.get("problem", ""))
                if problem and problem not in proved:
                    failures_by_problem.setdefault(
                        problem,
                        {
                            "problem": problem,
                            "outcome": str(row.get("outcome", "failed")),
                        },
                    )

                continue

            example = axiom_training_example_from_json(row)
            proved.add(example.problem_path)
            failures_by_problem.pop(example.problem_path, None)
            add_core_example(examples_by_input, example)

    if not examples_by_input:
        raise RuntimeError(f"axiom dataset contains no training examples: {dataset_path}")

    dataset_collection = collection or {}
    if len(
        {
            json.dumps(collection_settings(settings), sort_keys = True)
            for settings in shard_collections.values()
        }
    ) > 1:
        dataset_collection = {
            **dataset_compatibility_settings(dataset_collection),
            "source_shards": shard_collections,
        }

    metadata = {
        "schema": axiom_dataset_schema,
        "collection": dataset_collection,
        "shards": list(map(str, shards)),
    }

    return (
        list(core_examples(examples_by_input)),
        list(failures_by_problem.values()),
        metadata,
    )

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

def failure_is_parseable(payload: Mapping[str, Any]) -> bool:
    value = payload.get("parseable")
    if isinstance(value, bool):
        return value
    # Older failure records omit parseability.

    return not str(payload.get("outcome", "")).startswith("TPTPParseError:")
