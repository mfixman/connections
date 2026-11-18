# `collect`: proves problems in worker processes and appends one labelled row per problem to problems.csv.

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from connections.agent.sat import SATResetCoP
from connections.clausification import matrix_from_file
from connections.interaction.run import Problem, run_schedule
from connections.interaction.strategy import MatrixOptions, PolicyOptions, Strategy, StrategySchedule

from connections.interaction.szs import SUCCESS as success
from connections.parsing.tptp import TPTPParseError
from connections.recursion import deep_recursion

from .choices import ProverPolicy
from .data import sat_core_clause_ids, training_example_from_sat_core
from .dataset import axiom_dataset_schema, collection_settings
from .io import read_csv, write_csv
from .limits import CollectionTimeout, default_collection_step_limit, default_collection_timeout_s, wall_clock
from .logs import log

from .logs import progress as track_progress
from .parallel import determine_worker_count
from .problems_file import append_row, example_row, failure_row, header_fields, problem_fields, problem_name, repair_tail
from .search_workers import supervised_results
from .tptp import declared_non_refutable, find_tptp_root, load_tptp_problem, resolve_tptp_problem, selector_equality_axioms

class NoParseableProblemsError(RuntimeError):
    # Every requested problem failed TPTP parsing.
    pass

@dataclass(frozen = True, slots = True)
class CollectedProblem:
    problem: str
    outcome: str
    proved: bool
    parseable: bool
    row: dict[str, Any]

# The settings stored in every row; a resumed collection must match them. Native SATCoP also gets its options.
def requested_collection(output, *, tptp_root, step_limit, timeout_s, sat_policy, satcop_location, seed):
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

    from .satcop import binary_identity, label_semantics, native_binary

    binary = native_binary(satcop_location)
    collection.update(
        prover_backend = "rust-satcop",
        binary_sha256 = binary_identity(binary),
        label_semantics = label_semantics,
    )

    # Native SATCoP has no step limit, only the wall-clock timeout.
    collection.pop("step_limit")
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
        return path, problem_fields, {}

    repair_tail(path)
    return path, header_fields(path), finished_problems(path, collection)

# Rows store bare file names, so two problems of one collection cannot share one.
def require_unique_names(problems):
    counts = Counter(map(problem_name, problems))
    repeated = sorted(name for name, count in counts.items() if count > 1)
    if repeated:
        raise ValueError(f"problems in different directories share a file name: {repeated[:5]}")

def collect_axiom_dataset(
    problems,
    *,
    output_dir,
    tptp_root = None,
    step_limit = default_collection_step_limit,

    timeout_s = default_collection_timeout_s,
    sat_policy = ProverPolicy.SatResetCoP,
    num_workers = None,
    satcop_location = None,
    seed = 0,
) -> dict[str, Any]:
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

    path, fields, finished = open_problems_file(output, collection)
    reused = [finished[problem_name(problem)] for problem in problems if problem_name(problem) in finished]
    pending = [problem for problem in problems if problem_name(problem) not in finished]
    counts = Counter(
        proved = sum(outcome == "proved" for outcome, _ in reused),
        failed = sum(outcome != "proved" for outcome, _ in reused),
        unparseable = sum(not parseable for _, parseable in reused),
    )

    log(f"{len(reused)} problems already finished ({counts['proved']} proved); {len(pending)} to collect")
    last_report = time.monotonic()
    # Only this process appends results; workers never write the shared file.
    for result in collect_problems(
        pending,
        tptp_root = tptp_root,
        step_limit = step_limit,
        timeout_s = timeout_s,

        sat_policy = sat_policy,
        num_workers = num_workers,
        native_options = native_options,
        seed = seed,
    ):
        append_row(path, {**result.row, "collection": collection}, fields)
        counts["proved" if result.proved else "failed"] += 1
        counts["unparseable"] += not result.parseable
        processed = counts["proved"] + counts["failed"]
        if processed == len(problems) or time.monotonic() - last_report >= 60:
            log(
                f"[{processed}/{len(problems)}] {result.problem}: {result.outcome} "
                f"({counts['proved']} proved, {counts['failed']} failed, {len(reused)} reused)"
            )

            last_report = time.monotonic()

    if problems and counts["unparseable"] == len(problems):
        raise NoParseableProblemsError(f"none of the {len(problems)} .p files could be parsed as FOF or CNF")

    return {
        "schema": axiom_dataset_schema,
        "problems_requested": len(problems),
        "problems_proved": counts["proved"],
        "problems_failed": counts["failed"],

        "problems_reused": len(reused),
        "problems_unparseable": counts["unparseable"],
        "dataset_file": str(path),
    }

def collect_problems(problems, *, tptp_root, step_limit, timeout_s, sat_policy, num_workers, native_options, seed):
    workers = determine_worker_count(len(problems), num_workers)
    if workers == 0:
        return

    options = (tptp_root, step_limit, timeout_s, sat_policy, native_options, seed)
    requests = ((problem, *options) for problem in problems)
    for result in supervised_results(collect_one_problem, requests, workers = workers, timeout = timeout_s):
        collected = result.get("collected") or interrupted_problem(result)
        collected.row["seconds"] = round(result.get("problem_seconds", result["seconds"]), 3)
        yield collected

# A worker that timed out or died leaves only the supervisor's outcome.
def interrupted_problem(result):
    row = failure_row(result["problem"], result["outcome"], error = bool(result.get("error", False)))
    return CollectedProblem(result["problem"], result["outcome"], False, True, row)

# Runs in a worker; any error becomes a failure row, so that a resumed collection skips the problem.
def collect_one_problem(problem, tptp_root, step_limit, timeout_s, sat_policy, native_options, seed):
    # Timed in the worker, so that starting a new worker does not count.
    started = time.monotonic()
    parseable = True
    error = False
    try:
        example, outcome = collect_proof_example(
            problem,
            tptp_root = tptp_root,
            step_limit = step_limit,
            timeout_s = timeout_s,

            sat_policy = sat_policy,
            native_options = native_options,
            seed = seed,
        )
    except Exception as failure:
        example = None
        error = True
        parseable = not isinstance(failure, TPTPParseError)
        outcome = f"{type(failure).__name__}: {' '.join(str(failure).split())}"

    if example is None:
        row = failure_row(problem, outcome, parseable = parseable, error = error)
    else:
        row = example_row(problem, example)

    collected = CollectedProblem(problem, outcome, example is not None, parseable, row)
    return {"collected": collected, "problem_seconds": time.monotonic() - started}

# Returns the labelled example of a proof, or None with the reason; the budget covers parsing too.
@deep_recursion()
def collect_proof_example(
    problem,
    *,
    tptp_root = None,
    step_limit = default_collection_step_limit,

    timeout_s = default_collection_timeout_s,
    sat_policy = ProverPolicy.SatResetCoP,
    native_options = None,
    seed = 0,
):
    try:
        with wall_clock(timeout_s):
            path, root = resolve_tptp_problem(problem, tptp_root = tptp_root)
            declared_outcome = declared_non_refutable(path)
            if ProverPolicy(sat_policy) != ProverPolicy.SatCoP:
                return python_example(problem, path, root, declared_outcome, step_limit, timeout_s, seed)

            if declared_outcome is not None:
                return None, declared_outcome

            from .satcop import collect_native_example

            return collect_native_example(problem, tptp_root = root, timeout_s = timeout_s, seed = seed, **native_options)
    except CollectionTimeout:
        return None, "Timeout"

# The Python SATResetCoP of the Connections prover; its core comes from CaDiCaL's failed assumptions.
def python_example(problem, path, root, declared_outcome, step_limit, timeout_s, seed):
    source_dirs = () if root is None else (root,)
    if declared_outcome is not None:
        # Report parse errors even for problems we skip.
        matrix_from_file(path, mark_conjecture = True, source_file_dirs = source_dirs)
        return None, declared_outcome

    arguments = {"debug_sat_core": True, "seed": seed}
    strategy = Strategy(
        matrix = MatrixOptions(mark_conjecture = True, equality_axioms = selector_equality_axioms()),
        policy = PolicyOptions(policy_class = SATResetCoP, args = arguments),
    )

    agent = SATResetCoP(**arguments)
    result = run_schedule(
        Problem(path, logic = "classical", domain = "constant", source_file_dirs = source_dirs),
        agent = agent,
        schedule = StrategySchedule.single(strategy, steps = step_limit, timeout_seconds = timeout_s),
    )

    if result.szs_status not in success:
        return None, "unknown" if result.szs_status is None else result.szs_status.value

    # Loaded after the search, so that two matrices are never in memory at once.
    loaded = load_tptp_problem(problem, tptp_root = root)
    try:
        core = sat_core_clause_ids(agent.diagnostics())
    except ValueError as error:
        return None, f"invalid SAT core: {error}"

    check_sat_core_clauses(loaded.matrix, core, agent.diagnostics().get("sat_core_clause_texts"))
    example = training_example_from_sat_core(
        loaded.matrix,
        problem_path = loaded.requested_path,
        axiom_clause_ids = loaded.axiom_clause_ids,
        conjecture_clause_ids = loaded.conjecture_clause_ids,
        core_clause_ids = core,
    )

    return example, "proved"

# The labels are only right if the prover searched the matrix that gets labelled.
def check_sat_core_clauses(matrix, core, texts):
    if not isinstance(texts, list) or len(texts) != len(core):
        raise RuntimeError("SAT result is missing the clause texts of its core")

    for index, text in zip(core, texts, strict = True):
        if str(matrix.clauses[index]) != text:
            raise RuntimeError(
                f"SAT-core clause {index} is {text!r} in the proof matrix but {str(matrix.clauses[index])!r} in the labelled matrix"
            )
