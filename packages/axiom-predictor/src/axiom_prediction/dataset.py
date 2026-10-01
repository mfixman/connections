from __future__ import annotations

from .choices import ProverPolicy

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from axiom_prediction.io import read_jsonl, write_json_atomic, write_jsonl
from connections.parsing.tptp import TPTPParseError

from .data import AxiomTrainingExample, axiom_training_example_from_json, axiom_training_example_to_json
from .logs import log
from .logs import progress as track_progress
from .parallel import determine_worker_count
from .search_workers import supervised_results
from .tptp import DEFAULT_STEP_LIMIT, DEFAULT_TIMEOUT_SECONDS, collect_proof_example, find_tptp_root

AXIOM_DATASET_SCHEMA = "learncop.axiom_prediction.dataset.v2"
AXIOM_DATASET_SHARD_SCHEMA = "learncop.axiom_prediction.dataset-shard.v2"
OUTDATED_DATASET_HINT = "datasets collected before schema v2 label SAT cores against a different clausification; delete and re-collect them"

class NoParseableProblemsError(RuntimeError):
    """Every problem in a requested directory failed TPTP parsing."""

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
    parseable: bool = True

def collect_axiom_dataset(
    problems: Sequence[str],
    *,
    output_dir: str | Path,

    tptp_root: str | Path | None = None,
    step_limit: int = DEFAULT_STEP_LIMIT,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    sat_policy: str = ProverPolicy.SatResetCoP,

    num_workers: int | None = None,
    progress: Callable[[int, int, int, int, str, str], None] | None = None,
) -> dict[str, Any]:
    """Collect model-ready SAT-core examples, committing each problem atomically.

    Existing success and failure records are left untouched, so rerunning the
    same command resumes rather than repeating proof search.
    """

    log(f"preparing collection of {len(problems)} problems into {output_dir}")
    output = Path(output_dir)
    examples_dir = output / "examples"
    failures_dir = output / "failures"

    partial_dir = output / "partial"
    partial_examples_dir = partial_dir / "examples"
    partial_failures_dir = partial_dir / "failures"

    examples_dir.mkdir(parents = True, exist_ok = True)
    failures_dir.mkdir(parents = True, exist_ok = True)
    partial_examples_dir.mkdir(parents = True, exist_ok = True)
    partial_failures_dir.mkdir(parents = True, exist_ok = True)

    collection = {
        "sat_policy": ProverPolicy(sat_policy).wire_value,
        "step_limit": step_limit,
        "timeout_seconds": timeout_seconds,
        "tptp_root": (
            None if (resolved_root := find_tptp_root(tptp_root)) is None else str(
                resolved_root
            )
        ),
    }

    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        metadata = read_object(metadata_path)
        if metadata.get("schema") != AXIOM_DATASET_SCHEMA:
            raise ValueError(
                f"unsupported axiom dataset {output} (schema {metadata.get('schema')!r}); {OUTDATED_DATASET_HINT}"
            )

        if collection_settings(metadata.get("collection", {})) != collection_settings(collection):
            raise ValueError(
                "dataset collection settings differ from the existing dataset: "
                f"{metadata.get('collection')!r} != {collection!r}"
            )
    else:
        write_json_atomic(
            metadata_path,
            {"schema": AXIOM_DATASET_SCHEMA, "collection": collection},
        )

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
        key = problem_key(problem)
        example_path = examples_dir / f"{key}.json"
        failure_path = failures_dir / f"{key}.json"
        partial_example_path = partial_examples_dir / f"{key}.json"
        partial_failure_path = partial_failures_dir / f"{key}.json"
        if example_path.exists():
            validate_problem_record(example_path, problem)
            proved += 1
            reused += 1
            outcome = "proved (cached)"
        elif failure_path.exists():
            failure = read_object(failure_path)
            if failure.get("problem") != problem:
                raise ValueError(f"dataset record key collision for {problem!r}")

            failed += 1
            unparseable += int(not failure_is_parseable(failure))
            reused += 1
            outcome = f"{failure.get('outcome', 'failed')} (cached)"
        elif partial_example_path.exists():
            validate_problem_record(partial_example_path, problem)
            proved += 1
            reused += 1
            outcome = "proved (partial)"
        elif partial_failure_path.exists():
            failure = read_object(partial_failure_path)
            if failure.get("problem") != problem:
                raise ValueError(f"dataset record key collision for {problem!r}")

            failed += 1
            unparseable += int(not failure_is_parseable(failure))
            reused += 1
            outcome = f"{failure.get('outcome', 'failed')} (partial)"
        else:
            pending.append(problem)
            continue

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
        partial_dir = partial_dir,
        tptp_root = tptp_root,

        step_limit = step_limit,
        timeout_seconds = timeout_seconds,
        sat_policy = sat_policy,
        num_workers = workers,
    ):
        if result.proved:
            proved += 1
        else:
            failed += 1

        unparseable += int(not result.parseable)
        report(result.problem, result.outcome)

    log(f"committing collected records to {output}")
    promote_partial_records(partial_examples_dir, examples_dir)
    promote_partial_records(partial_failures_dir, failures_dir)

    summary: dict[str, Any] = {
        "schema": AXIOM_DATASET_SCHEMA,
        "problems_requested": total,
        "problems_proved": proved,
        "problems_failed": failed,
        "problems_reused": reused,
        "problems_unparseable": unparseable,
    }

    write_json_atomic(output / "summary.json", summary)
    return summary

def collect_axiom_dataset_shard(
    problems: Sequence[str],
    *,
    output_dir: str | Path,
    shard_name: str,
    tptp_root: str | Path | None = None,

    step_limit: int = DEFAULT_STEP_LIMIT,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    sat_policy: str = ProverPolicy.SatResetCoP,
    num_workers: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Collect one independently writable, model-ready dataset shard.

    The public artifact is one atomic JSONL file named after the input
    directory. Per-problem cache records live below ``.cache`` so an
    interrupted SLURM job can resume without exposing a partial shard.
    """

    if not shard_name or Path(shard_name).name != shard_name:
        raise ValueError(f"invalid axiom dataset shard name: {shard_name!r}")

    output = Path(output_dir)
    cache = output / ".cache" / shard_name
    summary = collect_axiom_dataset(
        problems,
        output_dir = cache,
        tptp_root = tptp_root,

        step_limit = step_limit,
        timeout_seconds = timeout_seconds,
        sat_policy = sat_policy,
        num_workers = num_workers,
    )

    if summary.get(
        "problems_requested",
        0,
    ) > 0 and summary.get("problems_unparseable") == summary.get("problems_requested"):
        raise NoParseableProblemsError(
            f"none of the {summary['problems_requested']} .p files in "
            f"{shard_name!r} could be parsed as FOF or CNF"
        )

    metadata = read_object(cache / "metadata.json")
    shard = output / f"{shard_name}.jsonl"
    log(f"writing dataset shard {shard}")
    write_jsonl(shard, shard_rows(cache, metadata = metadata, summary = summary))
    log(f"saved dataset shard {shard}")
    return shard, summary

def shard_rows(
    cache: Path,
    *,
    metadata: Mapping[str, Any],
    summary: Mapping[str, object],
) -> Iterator[Mapping[str, Any]]:
    yield {
        "schema": AXIOM_DATASET_SHARD_SCHEMA,
        "record": "metadata",
        "collection": metadata.get("collection", {}),
        "summary": dict(summary),
    }

    for record in sorted((cache / "examples").glob("*.json")):
        yield read_object(record)

    for record in sorted((cache / "failures").glob("*.json")):
        yield read_object(record)

def collect_problems_parallel(
    problems: Sequence[str],
    *,
    tptp_root: str | Path | None,
    step_limit: int,

    timeout_seconds: float,
    sat_policy: str,
    num_workers: int | None = None,
) -> Iterator[CollectedAxiomProblem]:
    workers = determine_worker_count(len(problems), num_workers)
    if workers == 0:
        return

    arguments = (
        (problem, tptp_root, step_limit, timeout_seconds, sat_policy)
        for problem in problems
    )

    for result in supervised_collection(
        collect_one_problem,
        arguments,
        workers = workers,
        timeout = timeout_seconds,
    ):
        yield result.get("collected") or CollectedAxiomProblem(
            result["problem"],
            None,
            result["outcome"],
        )

def collect_problem_records_parallel(
    problems: Sequence[str],
    *,
    partial_dir: str | Path,
    tptp_root: str | Path | None,
    step_limit: int,

    timeout_seconds: float,
    sat_policy: str,
    num_workers: int | None = None,
) -> Iterator[CollectedAxiomRecord]:
    """Collect directly to durable partial files and return small receipts."""

    workers = determine_worker_count(len(problems), num_workers)
    if workers == 0:
        return

    arguments = (
        (
            problem,
            partial_dir,
            tptp_root,
            step_limit,
            timeout_seconds,
            sat_policy,
        )
        for problem in problems
    )

    for result in supervised_collection(
        collect_one_problem_to_partial,
        arguments,
        workers = workers,
        timeout = timeout_seconds,
    ):
        yield result.get("collected") or interrupted_record(partial_dir, result)

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

def interrupted_record(partial_dir, result):
    problem = result["problem"]
    key = problem_key(problem)
    partial = Path(partial_dir)
    example = partial / "examples" / f"{key}.json"
    failure = partial / "failures" / f"{key}.json"
    if example.is_file():
        validate_problem_record(example, problem)
        return CollectedAxiomRecord(problem, "proved", True)

    if failure.is_file():
        record = read_object(failure)
        return CollectedAxiomRecord(
            problem,
            record["outcome"],
            False,
            failure_is_parseable(record),
        )

    write_json_atomic(
        failure,
        {
            "schema": AXIOM_DATASET_SCHEMA,
            "problem": problem,
            "outcome": result["outcome"],
            "parseable": True,
        },
    )

    return CollectedAxiomRecord(problem, result["outcome"], False)

def collect_one_problem_to_partial(
    problem: str,
    partial_dir: str | Path,
    tptp_root: str | Path | None,
    step_limit: int,
    timeout_seconds: float,
    sat_policy: str,
) -> CollectedAxiomRecord:
    parseable = True
    try:
        example, outcome = collect_proof_example(
            problem,
            tptp_root = tptp_root,
            step_limit = step_limit,
            timeout_seconds = timeout_seconds,
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

    key = problem_key(problem)
    partial = Path(partial_dir)
    if example is None:
        write_json_atomic(
            partial / "failures" / f"{key}.json",
            {
                "schema": AXIOM_DATASET_SCHEMA,
                "problem": problem,
                "outcome": outcome,
                "parseable": parseable,
            },
        )

        return CollectedAxiomRecord(problem, outcome, False, parseable)

    write_json_atomic(
        partial / "examples" / f"{key}.json",
        axiom_training_example_to_json(example),
    )

    return CollectedAxiomRecord(problem, outcome, True, parseable)

def collect_one_problem(
    problem: str,
    tptp_root: str | Path | None,
    step_limit: int,
    timeout_seconds: float,
    sat_policy: str,
) -> CollectedAxiomProblem:
    parseable = True
    try:
        example, outcome = collect_proof_example(
            problem,
            tptp_root = tptp_root,
            step_limit = step_limit,
            timeout_seconds = timeout_seconds,
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
    path: str | Path
) -> tuple[list[AxiomTrainingExample], list[dict[str, str]], dict[str, Any]]:
    root = Path(path)
    log(f"discovering dataset records in {root}")
    if root.is_file():
        return load_axiom_dataset_shards((root,), dataset_path = root)

    shards = sorted(root.glob("*.jsonl"))
    if shards:
        if (root / "metadata.json").exists() or (root / "examples").exists():
            raise ValueError(
                f"mixed dataset formats in {root}: separate JSONL shards from legacy records"
            )

        return load_axiom_dataset_shards(shards, dataset_path = root)

    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"axiom dataset metadata not found: {metadata_path}")

    metadata = read_object(metadata_path)
    if metadata.get("schema") != AXIOM_DATASET_SCHEMA:
        raise ValueError(
            f"unsupported axiom dataset {root} (schema {metadata.get('schema')!r}); {OUTDATED_DATASET_HINT}"
        )

    examples = [
        axiom_training_example_from_json(read_object(record))
        for record in track_progress(sorted((root / "examples").glob("*.json")), "loading examples")
    ]

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

def load_axiom_dataset_shards(
    shards: Sequence[Path],
    *,
    dataset_path: Path,
) -> tuple[list[AxiomTrainingExample], list[dict[str, str]], dict[str, Any]]:
    examples_by_problem: dict[str, AxiomTrainingExample] = {}
    failures_by_problem: dict[str, dict[str, str]] = {}
    collection: dict[str, Any] | None = None

    for shard in track_progress(shards, "loading dataset shards", len(shards)):
        rows = iter(read_jsonl(shard))
        try:
            header = next(rows)
        except StopIteration:
            raise ValueError(f"empty axiom dataset shard: {shard}") from None

        if (
            header.get("schema") != AXIOM_DATASET_SHARD_SCHEMA
            or header.get("record") != "metadata"
            or not isinstance(header.get("collection"), Mapping)
        ):
            raise ValueError(
                f"unsupported axiom dataset shard {shard} (schema {header.get('schema')!r}); {OUTDATED_DATASET_HINT}"
            )

        shard_collection = dict(header["collection"])
        if collection is None:
            collection = shard_collection
        elif collection_settings(collection) != collection_settings(shard_collection):
            raise ValueError(f"axiom dataset shards have different collection settings: {shard}")

        for row in track_progress(rows, f"reading {shard.name}"):
            if row.get("schema") == AXIOM_DATASET_SCHEMA:
                problem = str(row.get("problem", ""))
                if problem and problem not in examples_by_problem:
                    failures_by_problem.setdefault(
                        problem,
                        {
                            "problem": problem,
                            "outcome": str(row.get("outcome", "failed")),
                        },
                    )

                continue

            example = axiom_training_example_from_json(row)
            failures_by_problem.pop(example.problem_path, None)
            examples_by_problem.setdefault(example.problem_path, example)

    if not examples_by_problem:
        raise RuntimeError(f"axiom dataset contains no training examples: {dataset_path}")

    metadata = {
        "schema": AXIOM_DATASET_SCHEMA,
        "collection": collection or {},
        "shards": [str(shard) for shard in shards],
    }

    return (
        list(examples_by_problem.values()),
        list(failures_by_problem.values()),
        metadata,
    )

def collection_settings(collection):
    return {key: value for key, value in collection.items() if key != "tptp_root"}

def problem_key(problem: str) -> str:
    return hashlib.sha256(problem.encode("utf-8")).hexdigest()

def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding = "utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON record must be an object: {path}")

    return dict(value)

def validate_problem_record(path: Path, problem: str):
    payload = read_object(path)
    if payload.get("problem_path") != problem:
        raise ValueError(f"dataset record key collision for {problem!r}")

    axiom_training_example_from_json(payload)

def failure_is_parseable(payload: Mapping[str, Any]) -> bool:
    value = payload.get("parseable")
    if isinstance(value, bool):
        return value
    # Backward compatibility for failure records written before parseability
    # was stored explicitly.

    return not str(payload.get("outcome", "")).startswith("TPTPParseError:")

def promote_partial_records(source: Path, destination: Path):
    for record in source.glob("*.json"):
        target = destination / record.name
        if target.exists():
            record.unlink()
        else:
            record.replace(target)

__all__ = [
    "AXIOM_DATASET_SCHEMA",
    "AXIOM_DATASET_SHARD_SCHEMA",

    "collect_axiom_dataset",
    "collect_axiom_dataset_shard",

    "CollectedAxiomProblem",
    "CollectedAxiomRecord",

    "collect_problem_records_parallel",
    "collect_problems_parallel",
    "load_axiom_dataset",

    "NoParseableProblemsError",
]
