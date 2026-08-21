from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from connections.runs import RunRow, run_corpus
from provers.pycop.cli import configure_trace_loggers
from provers.pycop.cli_common import (
    add_problem_arguments,
    add_split_arguments,
    determine_worker_count,
    finalize_split_arguments,
    resolved_problem_inputs,
    source_file_dirs,
)
from provers.pycop.policy_resolution import (
    builtin_policy_names,
    resolve_policy,
    schedule_for_policy,
)


_SOLVED_STATUSES = frozenset(
    {"Theorem", "Unsatisfiable", "Satisfiable", "CounterSatisfiable"}
)


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
        help="Policy alias or package.module:PolicyClass",
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
        help="Replace an existing --out file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_all_policies:
        print("\n".join(builtin_policy_names()))
        return 0
    if not args.inputs:
        parser.error("pass a problem file, directory, TPTP codename, or raw problem")
    if args.num_workers is not None and args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    finalize_split_arguments(args, parser)

    try:
        selection = resolve_policy(args.policy)
        schedule = schedule_for_policy(
            selection,
            settings=args.settings,
            backtrack=args.backtrack,
            steps=args.max_steps,
            timeout=args.timeout,
        )
        configure_trace_loggers(
            search=args.trace_search,
            clausification=args.trace_clausification,
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
            if workers > 1 and (args.trace_search or args.trace_clausification):
                raise ValueError("trace output requires --num-workers 1")
            rows = tuple(
                run_corpus(
                    problems,
                    schedule=schedule,
                    logic=args.logic,
                    domain=args.domain,
                    source_file_dirs=source_file_dirs(args),
                    continue_on_error=True,
                    jobs=workers,
                    worker_threads=1,
                    retain_result=False,
                    yield_order="input",
                )
            )
        lines = tuple(
            json.dumps(_result_record(row, selection.label), sort_keys=True)
            for row in rows
        )
        _write_lines(lines, args.out, overwrite=args.overwrite)
        return 1 if any(row.error_type is not None for row in rows) else 0
    except (FileNotFoundError, ImportError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _result_record(row: RunRow, policy: str) -> dict[str, object]:
    return {
        "problem": row.problem,
        "path": row.path,
        "policy": policy,
        "status": row.status,
        "outcome": row.outcome,
        "solved": row.status in _SOLVED_STATUSES,
        "steps": row.steps,
        "inference_actions": row.inference_actions,
        "elapsed_seconds": row.elapsed_seconds,
        "error_type": row.error_type,
        "error_message": row.error_message,
    }


def _write_lines(
    lines: tuple[str, ...],
    output: str | None,
    *,
    overwrite: bool,
) -> None:
    if output is None:
        for line in lines:
            print(line)
        return
    path = Path(output)
    if path.exists() and not overwrite:
        raise RuntimeError(f"{path} already exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
