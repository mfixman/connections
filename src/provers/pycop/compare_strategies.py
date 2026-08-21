from __future__ import annotations

import argparse
from collections import deque
import contextlib
import csv
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, cast

from connections.prover import ProblemSpec, Prover
from connections.runs import row_from_result
from provers.pycop.cli_common import (
    add_problem_arguments,
    add_split_arguments,
    determine_worker_count,
    finalize_split_arguments,
    resolved_problem_inputs,
    source_file_dirs,
)
from provers.pycop.policy_resolution import (
    PolicySelection,
    builtin_policy_names,
    resolve_policies,
    resolve_policy,
    schedule_for_policy,
)


SCHEMA = "connections.pycop-comparison.v1"
_SOLVED_STATUSES = frozenset(
    {"Theorem", "Unsatisfiable", "Satisfiable", "CounterSatisfiable"}
)


@dataclass(frozen=True, slots=True)
class ComparisonTask:
    path: str
    policy_spec: str
    policy_label: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare connections policies across TPTP problems"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="DIR_OR_FILE",
        help="Problem files, directories, or TPTP codenames (default: all TPTP problems)",
    )
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument(
        "--policies",
        "--strategies",
        nargs="+",
        dest="policies",
        help="Policy aliases or [LABEL=]package.module:PolicyClass specifications",
    )
    policy_group.add_argument(
        "--all-policies",
        "--all-strategies",
        action="store_true",
        dest="all_policies",
        help="Compare every built-in policy",
    )
    parser.add_argument(
        "--print-all-policies",
        "--print-all-strategies",
        action="store_true",
        help="Print built-in policy aliases and exit",
    )
    add_problem_arguments(parser)
    add_split_arguments(parser)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        metavar="N",
        help="Worker processes (default: SLURM allocation or CPU count)",
    )
    parser.add_argument(
        "--steps",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Show inference-action counts in the table",
    )
    parser.add_argument(
        "--time",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Show elapsed time in the table",
    )
    parser.add_argument("--csv", metavar="PATH", help="Final CSV path")
    parser.add_argument(
        "--partial-file",
        metavar="PATH",
        help="Manifest-validated JSONL checkpoint path",
    )
    parser.add_argument(
        "--name",
        metavar="NAME",
        help="Derive partial and CSV paths from an experiment name",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing partial run instead of resuming it",
    )
    parser.add_argument(
        "--color",
        "--colour",
        action="store_true",
        dest="color",
        help="Use ANSI colors in the human report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_all_policies:
        print("\n".join(builtin_policy_names()))
        return 0
    if args.num_workers is not None and args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if args.all_policies:
        args.policies = list(builtin_policy_names())
    if not args.policies:
        parser.error("pass --strategies POLICY [...] or --all-strategies")
    finalize_split_arguments(args, parser)

    try:
        selections = resolve_policies(args.policies)
        for selection in selections:
            schedule_for_policy(
                selection,
                settings=args.settings,
                backtrack=args.backtrack,
                steps=args.max_steps,
                timeout=args.timeout,
            )
        csv_path, partial_path = _output_paths(args)
        with resolved_problem_inputs(
            args.inputs,
            args=args,
            allow_raw=False,
            default_to_all_tptp=True,
        ) as problems:
            if not problems:
                raise ValueError("selected problem slice is empty")
            manifest = build_manifest(
                args,
                problems=problems,
                selections=selections,
                source_dirs=source_file_dirs(args),
            )
            existing = prepare_partial(
                partial_path,
                manifest=manifest,
                overwrite=args.overwrite,
            )
            tasks = tuple(
                ComparisonTask(
                    path=str(problem.resolve()),
                    policy_spec=specification,
                    policy_label=selection.label,
                )
                for problem in problems
                for specification, selection in zip(args.policies, selections, strict=True)
            )
            remaining = tuple(task for task in tasks if _task_key(task) not in existing)
            workers = determine_worker_count(len(remaining), args.num_workers)
            print(
                "Run: "
                f"partial={partial_path}  csv={csv_path}  tasks={len(tasks)}  "
                f"completed={len(existing)}  remaining={len(remaining)}  "
                f"workers={workers}  timeout={args.timeout:g}s",
                file=sys.stderr,
                flush=True,
            )
            new_results = run_task_matrix(
                remaining,
                settings=tuple(args.settings),
                backtrack=args.backtrack,
                logic=args.logic,
                domain=args.domain,
                source_dirs=tuple(str(path) for path in source_file_dirs(args)),
                max_steps=args.max_steps,
                timeout=args.timeout,
                workers=workers,
                on_result=lambda result: _record_result(
                    partial_path,
                    result,
                    show_steps=args.steps,
                    show_time=args.time,
                    color=args.color,
                ),
            )
            results = existing | {
                _result_key(result): result for result in new_results
            }
            ordered = tuple(results[_task_key(task)] for task in tasks)
        write_csv(csv_path, ordered)
        print_summary(ordered, selections)
        print(f"CSV: {csv_path}")
        return 1 if any(result["error_type"] is not None for result in ordered) else 0
    except (FileNotFoundError, ImportError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def run_task_matrix(
    tasks: tuple[ComparisonTask, ...],
    *,
    settings: tuple[str, ...],
    backtrack: str,
    logic: str,
    domain: str,
    source_dirs: tuple[str, ...],
    max_steps: int,
    timeout: float,
    workers: int,
    on_result: Callable[[dict[str, object]], object],
    grace_seconds: float = 5.0,
) -> tuple[dict[str, object], ...]:
    if not tasks:
        return ()
    context = cast(Any, _process_context())
    queued = deque(tasks)
    active: dict[Connection, tuple[ComparisonTask, mp.Process, float]] = {}
    completed: list[dict[str, object]] = []

    def start(task: ComparisonTask) -> None:
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_task_worker,
            args=(
                child,
                task,
                settings,
                backtrack,
                logic,
                domain,
                source_dirs,
                max_steps,
                timeout,
            ),
            daemon=True,
        )
        process.start()
        child.close()
        active[parent] = (task, process, time.monotonic() + timeout + grace_seconds)

    try:
        while queued and len(active) < workers:
            start(queued.popleft())
        while active:
            now = time.monotonic()
            for connection in tuple(active):
                task, process, deadline = active[connection]
                if deadline > now:
                    continue
                del active[connection]
                _terminate_process(process)
                connection.close()
                result = _timeout_result(task, timeout)
                completed.append(result)
                on_result(result)
            while queued and len(active) < workers:
                start(queued.popleft())
            if not active:
                break
            next_deadline = min(deadline for _, _, deadline in active.values())
            ready = wait(
                tuple(active),
                timeout=max(0.0, next_deadline - time.monotonic()),
            )
            for ready_item in ready:
                connection = cast(Connection, ready_item)
                task, process, _ = active.pop(connection)
                result = _receive_result(connection, process, task)
                connection.close()
                completed.append(result)
                on_result(result)
            while queued and len(active) < workers:
                start(queued.popleft())
    finally:
        for connection, (_, process, _) in active.items():
            _terminate_process(process)
            connection.close()
    return tuple(completed)


def _task_worker(
    connection: Connection,
    task: ComparisonTask,
    settings: tuple[str, ...],
    backtrack: str,
    logic: str,
    domain: str,
    source_dirs: tuple[str, ...],
    max_steps: int,
    timeout: float,
) -> None:
    _configure_worker_threads()
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                selection = resolve_policy(task.policy_spec)
                schedule = schedule_for_policy(
                    selection,
                    settings=list(settings),
                    backtrack=backtrack,
                    steps=max_steps,
                    timeout=timeout,
                )
                result = Prover().run(
                    ProblemSpec(
                        task.path,
                        logic=logic,
                        domain=domain,
                        source_file_dirs=source_dirs,
                    ),
                    schedule=schedule,
                )
                row = row_from_result(task.path, result)
        payload = {
            "record_type": "result",
            "problem": row.problem,
            "path": row.path,
            "policy": task.policy_label,
            "policy_spec": task.policy_spec,
            "status": row.status,
            "outcome": row.outcome,
            "solved": row.status in _SOLVED_STATUSES,
            "steps": row.steps,
            "inference_actions": row.inference_actions,
            "elapsed_seconds": row.elapsed_seconds,
            "error_type": row.error_type,
            "error_message": row.error_message,
        }
    except BaseException as exc:
        payload = _error_result(task, exc)
    try:
        connection.send(payload)
    finally:
        connection.close()


def build_manifest(
    args: argparse.Namespace,
    *,
    problems: tuple[Path, ...],
    selections: tuple[PolicySelection, ...],
    source_dirs: tuple[Path, ...],
) -> dict[str, object]:
    return {
        "record_type": "manifest",
        "schema": SCHEMA,
        "package_version": _package_version(),
        "code_fingerprint": _code_fingerprint(),
        "problems": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": _file_sha256(path),
            }
            for path in problems
        ],
        "policies": [
            {
                "label": selection.label,
                "reference": specification,
                "implementation_sha256": _policy_sha256(selection),
            }
            for specification, selection in zip(args.policies, selections, strict=True)
        ],
        "logic": args.logic,
        "domain": args.domain,
        "source_dirs": [str(path) for path in source_dirs],
        "settings": list(args.settings),
        "backtrack": args.backtrack,
        "max_steps": args.max_steps,
        "timeout": args.timeout,
        "split": args.split,
        "part": args.part,
    }


def prepare_partial(
    path: Path,
    *,
    manifest: dict[str, object],
    overwrite: bool,
) -> dict[tuple[str, str], dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        return {}
    stored_manifest, results = load_partial(path)
    if stored_manifest != manifest:
        raise RuntimeError(
            f"{path} belongs to a different code/configuration/problem set; "
            "use a new --name/--partial-file or pass --overwrite"
        )
    return results


def load_partial(
    path: Path,
) -> tuple[dict[str, object], dict[tuple[str, str], dict[str, object]]]:
    raw_lines = path.read_bytes().splitlines(keepends=True)
    if not raw_lines:
        raise RuntimeError(f"partial file is empty: {path}")
    records: list[dict[str, object]] = []
    for index, raw_line in enumerate(raw_lines):
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            is_truncated_tail = index == len(raw_lines) - 1 and not raw_line.endswith(b"\n")
            if is_truncated_tail:
                break
            raise RuntimeError(f"invalid JSONL record {index + 1} in {path}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"JSONL record {index + 1} in {path} is not an object")
        records.append(record)
    if not records or records[0].get("record_type") != "manifest":
        raise RuntimeError(f"partial file has no manifest: {path}")
    results: dict[tuple[str, str], dict[str, object]] = {}
    for record in records[1:]:
        if record.get("record_type") != "result":
            raise RuntimeError(f"unexpected partial record type in {path}")
        results[_result_key(record)] = record
    return records[0], results


def write_csv(path: Path, results: tuple[dict[str, object], ...]) -> None:
    fields = (
        "problem",
        "path",
        "policy",
        "policy_spec",
        "status",
        "outcome",
        "solved",
        "steps",
        "inference_actions",
        "elapsed_seconds",
        "error_type",
        "error_message",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fields})


def print_summary(
    results: tuple[dict[str, object], ...],
    selections: tuple[PolicySelection, ...],
) -> None:
    problem_count = len({str(result["path"]) for result in results})
    solved = {
        selection.label: sum(
            result["policy"] == selection.label and result["solved"] is True
            for result in results
        )
        for selection in selections
    }
    summary = "  ".join(
        f"{selection.label}={solved[selection.label]}/{problem_count}"
        for selection in selections
    )
    if len(selections) == 2:
        first, second = selections
        first_wins, second_wins, ties = _pairwise_wins(
            results,
            first.label,
            second.label,
        )
        summary += (
            f"  |  inference wins: {first.label}={first_wins}  "
            f"{second.label}={second_wins}  ties={ties}"
        )
    print(f"Solved: {summary}")


def _pairwise_wins(
    results: tuple[dict[str, object], ...],
    first: str,
    second: str,
) -> tuple[int, int, int]:
    by_problem: dict[str, dict[str, dict[str, object]]] = {}
    for result in results:
        by_problem.setdefault(str(result["path"]), {})[str(result["policy"])] = result
    first_wins = second_wins = ties = 0
    for policy_results in by_problem.values():
        left = policy_results[first]
        right = policy_results[second]
        left_solved = left["solved"] is True
        right_solved = right["solved"] is True
        if left_solved and not right_solved:
            first_wins += 1
        elif right_solved and not left_solved:
            second_wins += 1
        elif left_solved and right_solved:
            left_steps = _integer_value(left["inference_actions"])
            right_steps = _integer_value(right["inference_actions"])
            if left_steps < right_steps:
                first_wins += 1
            elif right_steps < left_steps:
                second_wins += 1
            else:
                ties += 1
        else:
            ties += 1
    return first_wins, second_wins, ties


def _record_result(
    partial_path: Path,
    result: dict[str, object],
    *,
    show_steps: bool,
    show_time: bool,
    color: bool,
) -> None:
    with partial_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(result, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    fields = [str(result["problem"]), str(result["policy"]), str(result["status"])]
    if show_steps:
        fields.append(f"{result['inference_actions']} steps")
    if show_time:
        fields.append(f"{_float_value(result['elapsed_seconds']):.3f}s")
    line = "  ".join(fields)
    if color and result["solved"] is True:
        line = f"\033[32m{line}\033[0m"
    elif color and result["error_type"] is not None:
        line = f"\033[31m{line}\033[0m"
    print(line, flush=True)


def _receive_result(
    connection: Connection,
    process: mp.Process,
    task: ComparisonTask,
) -> dict[str, object]:
    try:
        if connection.poll():
            result = cast(dict[str, object], connection.recv())
        else:
            result = _error_result(
                task,
                RuntimeError(f"worker exited with code {process.exitcode}"),
            )
    except EOFError:
        result = _error_result(
            task,
            RuntimeError(f"worker exited with code {process.exitcode}"),
        )
    process.join(timeout=1.0)
    if process.is_alive():
        _terminate_process(process)
    return result


def _terminate_process(process: mp.Process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)


def _timeout_result(task: ComparisonTask, timeout: float) -> dict[str, object]:
    return {
        "record_type": "result",
        "problem": Path(task.path).name,
        "path": task.path,
        "policy": task.policy_label,
        "policy_spec": task.policy_spec,
        "status": "Timeout",
        "outcome": "Timeout",
        "solved": False,
        "steps": 0,
        "inference_actions": 0,
        "elapsed_seconds": timeout,
        "error_type": None,
        "error_message": None,
    }


def _error_result(task: ComparisonTask, error: BaseException) -> dict[str, object]:
    return {
        "record_type": "result",
        "problem": Path(task.path).name,
        "path": task.path,
        "policy": task.policy_label,
        "policy_spec": task.policy_spec,
        "status": "Error",
        "outcome": "Error",
        "solved": False,
        "steps": 0,
        "inference_actions": 0,
        "elapsed_seconds": 0.0,
        "error_type": type(error).__name__,
        "error_message": "".join(traceback.format_exception_only(type(error), error)).strip(),
    }


def _task_key(task: ComparisonTask) -> tuple[str, str]:
    return task.path, task.policy_label


def _result_key(result: dict[str, object]) -> tuple[str, str]:
    return str(result["path"]), str(result["policy"])


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    input_name = "all" if not args.inputs else Path(args.inputs[0]).name
    suffix = "" if args.split == 1 else f"_{args.part}"
    stem = f"{input_name}{suffix}"
    if args.name:
        csv_path = Path(args.csv or Path("results") / args.name / f"{stem}.csv")
        partial_path = Path(
            args.partial_file or Path("partial") / f"{args.name}_{stem}.jsonl"
        )
        return csv_path, partial_path
    if not args.csv:
        raise ValueError("--csv is required unless --name is provided")
    csv_path = Path(args.csv)
    partial_path = Path(args.partial_file or f"{csv_path}.partial.jsonl")
    return csv_path, partial_path


def _package_version() -> str:
    try:
        return version("connections")
    except PackageNotFoundError:
        return "uninstalled"


def _code_fingerprint() -> dict[str, str]:
    source_root = Path(__file__).resolve().parents[2]
    project_root = source_root.parent
    source_hash = hashlib.sha256()
    for module_name in ("connections", "provers"):
        paths = (
            *(source_root / module_name).rglob("*.py"),
            *(source_root / module_name).rglob("*.json"),
        )
        for path in sorted(paths):
            source_hash.update(str(path.relative_to(source_root)).encode())
            source_hash.update(b"\0")
            source_hash.update(path.read_bytes())
            source_hash.update(b"\0")
    if not (project_root / ".git").exists():
        return {
            "kind": "package",
            "version": _package_version(),
            "source_sha256": source_hash.hexdigest(),
        }
    head = _git_output(project_root, "rev-parse", "HEAD")
    return {
        "kind": "git",
        "head": head.strip(),
        "source_sha256": source_hash.hexdigest(),
    }


def _policy_sha256(selection: PolicySelection) -> str:
    source_path = inspect.getsourcefile(selection.policy_class)
    if source_path is None:
        return "unavailable"
    return _file_sha256(Path(source_path))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _configure_worker_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"


def _integer_value(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected integer result value, got {value!r}")
    return value


def _float_value(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected numeric result value, got {value!r}")
    return float(value)


def _process_context() -> mp.context.BaseContext:
    methods = mp.get_all_start_methods()
    for method in ("spawn", "forkserver", "fork"):
        if method in methods:
            return mp.get_context(method)
    return mp.get_context(methods[0])


if __name__ == "__main__":
    raise SystemExit(main())
