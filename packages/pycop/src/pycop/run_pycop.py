from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any


from pycop.compare_strategies import ComparisonTask, run_task_matrix, run_task
from pycop.cli import configure_trace_loggers
from pycop.cli_common import (
    add_problem_arguments,
    add_split_arguments,
    determine_worker_count,
    finalize_split_arguments,
    resolved_problem_inputs,
    source_file_dirs,
)
from pycop.policy_resolution import builtin_policy_names, resolve_policy, schedule_for_policy


_SOLVED_STATUSES = frozenset({"Theorem", "Unsatisfiable", "Satisfiable", "CounterSatisfiable"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one connections policy over one or more TPTP problems"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="DIR_OR_FILE",
        help="Problem file, directory, TPTP codename, or one raw quoted problem",
    )
    parser.add_argument(
        "--policy",
        "--strategy",
        default="FirstActionIDPolicy",
        help="Proof policy alias or package.module:PolicyClass",
    )
    parser.add_argument(
        "--print-all-policies",
        "--print-all-strategies",
        action="store_true",
        help="Print built-in proof policy aliases and exit",
    )
    parser.add_argument(
        "--model-policy",
        metavar="POLICY",
        help="Finite-model policy: mace, sat-reset, or package.module:Class",
    )
    parser.add_argument(
        "--print-all-model-policies",
        action="store_true",
        help="Print built-in finite-model policy aliases and exit",
    )
    add_problem_arguments(parser)
    add_split_arguments(parser)
    parser.add_argument(
        "--model-max-domain",
        type=int,
        default=None,
        metavar="N",
        help="Stop after domain N (default in model mode: unbounded)",
    )
    parser.add_argument(
        "--model-tptp-dir",
        metavar="DIR",
        help="Write each finite interpretation to DIR/PROBLEM.model.p",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        metavar="N",
        help="Worker processes (default: SLURM allocation or CPU count)",
    )
    parser.add_argument(
        "--trace-search",
        "--print-trace",
        action="store_true",
        dest="trace_search",
        help="Print proof-search trace events",
    )
    parser.add_argument(
        "--trace-clausification",
        action="store_true",
        help="Print clausification trace events",
    )
    parser.add_argument("--out", metavar="PATH", help="Write result JSONL to PATH")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output/model files",
    )
    parser.add_argument(
        "--debug-sat-core", action="store_true", help="Report source-clause SAT cores"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args.model_max_steps = args.max_steps if _option_was_passed(raw_argv, "--max-steps") else None
    args.model_timeout = args.timeout if _option_was_passed(raw_argv, "--timeout") else None
    if args.print_all_policies:
        print("\n".join(builtin_policy_names()))
        return 0
    if args.print_all_model_policies:
        from connections.model_finding import builtin_model_policy_names

        print("\n".join(builtin_model_policy_names()))
        return 0
    if not args.inputs:
        parser.error("pass a problem file, directory, TPTP codename, or raw problem")
    if args.num_workers is not None and args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if args.model_max_domain is not None and args.model_max_domain <= 0:
        parser.error("--model-max-domain must be positive")
    finalize_split_arguments(args, parser)

    try:
        if args.model_policy is not None:
            return _run_model_mode(args)
        return _run_proof_mode(args)
    except (FileNotFoundError, ImportError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_proof_mode(args: argparse.Namespace) -> int:
    selection = resolve_policy(args.policy)
    configure_trace_loggers(search=args.trace_search, clausification=args.trace_clausification)
    schedule_for_policy(
        selection,
        settings=args.settings,
        backtrack=args.backtrack,
        steps=args.max_steps,
        timeout=args.timeout,
        debug_sat_core=args.debug_sat_core,
    )
    with resolved_problem_inputs(
        args.inputs,
        args=args,
        allow_raw=True,
        default_to_all_tptp=False,
    ) as problems:
        if not problems:
            raise ValueError("selected problem slice is empty")
        workers = determine_worker_count(len(problems), args.num_workers)
        tasks = tuple(ComparisonTask(str(path), args.policy, selection.label) for path in problems)
        options: dict[str, Any] = dict(
            settings=tuple(args.settings),
            backtrack=args.backtrack,
            logic=args.logic,
            domain=args.domain,
            source_dirs=tuple(map(str, source_file_dirs(args))),
            max_steps=args.max_steps,
            timeout=args.timeout,
            debug_sat_core=args.debug_sat_core,
        )
        if args.trace_search or args.trace_clausification:
            if workers != 1:
                raise ValueError("trace output requires --num-workers 1")
            records = tuple(run_task(task, **options) for task in tasks)
        else:
            completed = run_task_matrix(
                tasks, **options, workers=workers, on_result=lambda row: None
            )
            by_path = {row["path"]: row for row in completed}
            records = tuple(by_path[task.path] for task in tasks)
    _write_lines(
        tuple(json.dumps(row, sort_keys=True) for row in records),
        args.out,
        overwrite=args.overwrite,
    )
    if args.debug_sat_core:
        for row in records:
            print(f"SAT core for {row['problem']}: {row['diagnostics']}", file=sys.stderr)
    return int(any(row["error_type"] is not None for row in records))


@dataclass(frozen=True, slots=True)
class _ModelRunConfig:
    policy_specification: str
    policy_label: str
    source_roots: tuple[str, ...]
    timeout_seconds: float | None
    max_steps: int | None
    max_domain: int | None


def _run_model_mode(args: argparse.Namespace) -> int:
    from connections.model_finding import resolve_model_policy

    if args.logic != "classical":
        raise ValueError("model mode supports --logic classical only")
    if args.domain != "constant":
        raise ValueError("model mode supports --domain constant only")
    if args.settings:
        raise ValueError("--settings applies to proof policies, not model mode")
    if args.trace_search or args.trace_clausification:
        raise ValueError("proof-search trace options do not apply in model mode")
    selection = resolve_model_policy(args.model_policy)
    config = _ModelRunConfig(
        policy_specification=f"{selection.label}={selection.reference}",
        policy_label=selection.label,
        source_roots=tuple(str(path) for path in source_file_dirs(args)),
        timeout_seconds=args.model_timeout,
        max_steps=args.model_max_steps,
        max_domain=args.model_max_domain,
    )
    with resolved_problem_inputs(
        args.inputs,
        args=args,
        allow_raw=True,
        default_to_all_tptp=False,
    ) as problems:
        if not problems:
            raise ValueError("selected problem slice is empty")
        workers = determine_worker_count(len(problems), args.num_workers)
        if workers == 1:
            records = tuple(_run_model_problem(str(path), config) for path in problems)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                records = tuple(
                    executor.map(
                        _run_model_problem_task,
                        ((str(path), config) for path in problems),
                    )
                )
    lines = tuple(json.dumps(record, sort_keys=True) for record in records)
    _write_lines(lines, args.out, overwrite=args.overwrite)
    if args.model_tptp_dir is not None:
        _write_model_files(
            records,
            Path(args.model_tptp_dir),
            overwrite=args.overwrite,
        )
    return 1 if any(record["error_type"] is not None for record in records) else 0


def _run_model_problem_task(task: tuple[str, _ModelRunConfig]) -> dict[str, object]:
    return _run_model_problem(*task)


def _run_model_problem(path: str, config: _ModelRunConfig) -> dict[str, object]:
    from connections.model_finding import ModelFinder, ModelSearchBudget

    started = time.monotonic()
    try:
        result = ModelFinder(config.policy_specification).find_file(
            path,
            source_roots=config.source_roots,
            budget=ModelSearchBudget(
                timeout_seconds=config.timeout_seconds,
                max_steps=config.max_steps,
                max_domain=config.max_domain,
            ),
        )
        return result.to_json(problem=Path(path).name, path=path)
    except Exception as exc:
        return {
            "problem": Path(path).name,
            "path": path,
            "policy": config.policy_label,
            "status": "Error",
            "outcome": "Error",
            "solved": False,
            "steps": 0,
            "elapsed_seconds": time.monotonic() - started,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "model": None,
            "tptp_model": None,
            "debug": {"provisional": True, "bounds": []},
        }


def _write_model_files(
    records: tuple[dict[str, object], ...],
    directory: Path,
    *,
    overwrite: bool,
):
    directory.mkdir(parents=True, exist_ok=True)
    targets: list[tuple[Path, str]] = []
    for record in records:
        rendered = record.get("tptp_model")
        if not isinstance(rendered, str):
            continue
        target = directory / f"{Path(str(record['problem'])).stem}.model.p"
        if target.exists() and not overwrite:
            raise RuntimeError(f"{target} already exists; pass --overwrite")
        targets.append((target, rendered))
    for target, rendered in targets:
        target.write_text(rendered, encoding="utf-8")


def _write_lines(
    lines: tuple[str, ...],
    output: str | None,
    *,
    overwrite: bool,
):
    if output is None:
        for line in lines:
            print(line)
        return
    path = Path(output)
    if path.exists() and not overwrite:
        raise RuntimeError(f"{path} already exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _option_was_passed(arguments: list[str], option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in arguments)


if __name__ == "__main__":
    raise SystemExit(main())
